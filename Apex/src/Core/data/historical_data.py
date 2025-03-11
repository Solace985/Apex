import pandas as pd
import numpy as np
import os
import json
import time
import redis
import sqlite3
import asyncio
import requests
import io
import pickle
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps
from arch import arch_model  # For GARCH volatility modeling
import numba
from scipy import stats
import logging
import hashlib
import hmac

# Enhanced Apex integrations
from src.Core.trading.risk.risk_management import RiskManager
from src.Core.data.correlation_monitor import CorrelationMonitor
from src.Core.trading.execution.order_execution import OrderExecution
from src.Core.trading.strategies.strategy_orchestrator import StrategyOrchestrator
from src.Core.data.realtime.market_data import MarketDataAPI
from utils.logging.structured_logger import StructuredLogger
from src.ai.analysis.market_regime_classifier import MarketRegimeClassifier
from utils.analytics.monte_carlo_simulator import MonteCarloSimulator
from src.Core.data.realtime.websocket_handler import WebSocketHandler

# Security enhancements
from utils.helpers.validation import validate_symbol, sanitize_input, validate_date_range
from utils.helpers.error_handler import handle_exceptions
from src.Core.trading.security.security import encrypt_data, decrypt_data

# AI modules - not implementing them directly but using them appropriately
from src.ai.forecasting.technical_analysis import detect_anomalies
from src.ai.analysis.fundamental_analysis import get_fundamental_indicators

# Config imports
from Config.assets.asset_universe import load_asset_universe

# Constants
DATABASE_PATH = os.environ.get("HISTORICAL_DB_PATH", "data/historical.db")
CACHE_EXPIRY = 3600  # 1 hour in seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logger = StructuredLogger(__name__, log_level=LOG_LEVEL)

class RateLimiter:
    """Rate limiter for API calls to prevent exceeding quotas"""
    
    def __init__(self, calls_per_second: float = 5):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0
    
    async def wait(self):
        """Wait if necessary to comply with rate limits"""
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


class DataCache:
    """Advanced caching system for historical data with TTL and compression"""
    
    def __init__(self, ttl: int = CACHE_EXPIRY):
        self.ttl = ttl
        self.cache = {}
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", 6379)),
                password=os.environ.get("REDIS_PASSWORD", ""),
                ssl=os.environ.get("REDIS_SSL", "False").lower() == "true"
            )
            self.redis_client.ping()  # Test connection
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis cache initialization failed: {e}. Using in-memory cache.")
            self.redis_client = None
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from cache with TTL check"""
        # Try Redis first if available
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(f"hist_data:{key}")
                if cached_data:
                    metadata = self.redis_client.hgetall(f"hist_meta:{key}")
                    if metadata and time.time() - float(metadata.get(b'timestamp', 0)) < self.ttl:
                        return pd.read_msgpack(cached_data)
            except Exception as e:
                logger.warning(f"Redis cache retrieval failed: {e}. Falling back to in-memory cache.")
        
        # Fallback to in-memory
        if key in self.cache:
            timestamp, data = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
        
        return None
    
    def set(self, key: str, data: pd.DataFrame) -> None:
        """Set data in cache with TTL"""
        if data is None or data.empty:
            return
        
        # Try Redis first if available
        if self.redis_client:
            try:
                # Compress with msgpack for efficiency
                compressed_data = data.to_msgpack(compress='zlib')
                pipe = self.redis_client.pipeline()
                pipe.set(f"hist_data:{key}", compressed_data)
                pipe.hset(f"hist_meta:{key}", mapping={'timestamp': time.time(), 'rows': len(data)})
                pipe.expire(f"hist_data:{key}", self.ttl)
                pipe.expire(f"hist_meta:{key}", self.ttl)
                pipe.execute()
                return
            except Exception as e:
                logger.warning(f"Redis cache storage failed: {e}. Falling back to in-memory cache.")
        
        # Fallback to in-memory
        self.cache[key] = (time.time(), data)
        
        # Clean up expired entries occasionally
        if len(self.cache) > 100 and random.random() < 0.1:  # 10% chance when cache is large
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Remove expired entries from in-memory cache"""
        current_time = time.time()
        keys_to_remove = [
            k for k, (timestamp, _) in self.cache.items() 
            if current_time - timestamp >= self.ttl
        ]
        for k in keys_to_remove:
            del self.cache[k]


@numba.jit(nopython=True)
def _calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Optimized returns calculation with Numba"""
    returns = np.empty(len(prices) - 1)
    for i in range(1, len(prices)):
        returns[i-1] = (prices[i] / prices[i-1]) - 1
    return returns


class RetryWithBackoff:
    """Decorator for implementing retry logic with exponential backoff"""
    
    def __init__(self, max_retries: int = MAX_RETRIES, backoff_factor: float = 1.5,
                 exceptions: tuple = (requests.RequestException, ConnectionError)):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.exceptions = exceptions
    
    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            delay = RETRY_DELAY
            
            while retries <= self.max_retries:
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except self.exceptions as e:
                    retries += 1
                    if retries > self.max_retries:
                        logger.error(f"Maximum retries ({self.max_retries}) reached. Error: {str(e)}")
                        raise
                    
                    wait = delay * (self.backoff_factor ** (retries - 1))
                    logger.warning(f"Retry {retries}/{self.max_retries} after {wait:.2f}s. Error: {str(e)}")
                    await asyncio.sleep(wait)
            
            return None  # Should not reach here
        
        return wrapper


class HistoricalData:
    """Institutional-Grade Historical Data Management System"""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        """Initialize the historical data system with enhanced security and performance"""
        # Initialize core components
        self.db_path = db_path
        self.db_conn = self._initialize_database()
        self.cache = DataCache()
        self.rate_limiter = RateLimiter()
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        self.process_executor = ProcessPoolExecutor(max_workers=max(1, os.cpu_count() // 2))
        
        # Initialize integrated Apex components
        self.market_data_api = MarketDataAPI()
        self.regime_classifier = MarketRegimeClassifier()
        self.correlation_monitor = CorrelationMonitor()
        self.risk_manager = RiskManager()
        self.websocket_handler = WebSocketHandler()
        
        # Set up security and logging
        logger.info("HistoricalData initialized successfully")
    
    def _initialize_database(self) -> sqlite3.Connection:
        """Initialize and optimize the SQLite database with security measures"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Create database connection with proper security settings
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Apply security hardening
        conn.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging for concurrency
        conn.execute("PRAGMA synchronous = NORMAL")  # Balance between safety and speed
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        conn.execute("PRAGMA busy_timeout = 5000")  # 5 second timeout for busy database
        
        # Set up tables if they don't exist
        with conn:
            # Main historical prices table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                adjusted_close REAL,
                source TEXT,
                market_regime TEXT,
                volatility REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
            """)
            
            # Create index for faster queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON historical_prices(symbol, timestamp)")
            
            # Metadata table for data sources and quality
            conn.execute("""
            CREATE TABLE IF NOT EXISTS data_metadata (
                symbol TEXT NOT NULL,
                last_updated TEXT,
                source TEXT,
                quality_score REAL,
                anomalies_detected INTEGER,
                PRIMARY KEY(symbol)
            )
            """)
            
            # Volatility history table for risk management
            conn.execute("""
            CREATE TABLE IF NOT EXISTS volatility_history (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                historical_vol REAL,
                garch_vol REAL,
                ewma_vol REAL,
                regime TEXT,
                PRIMARY KEY(symbol, date)
            )
            """)
        
        return conn
    
    @handle_exceptions
    async def fetch_historical_data(self, symbol: str, start_date: str, end_date: str, 
                                   interval: str = "1d", include_regime: bool = False,
                                   include_volatility: bool = False) -> pd.DataFrame:
        """
        Enhanced multi-source data ingestion with security validation
        
        Args:
            symbol: Trading symbol (e.g., "AAPL", "BTC-USD")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval ("1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo")
            include_regime: Whether to include market regime classification
            include_volatility: Whether to include volatility calculations
        
        Returns:
            DataFrame with historical price data
        """
        # Security validation
        validate_symbol(symbol)
        sanitized_interval = sanitize_input(interval)
        validate_date_range(start_date, end_date)
        
        # Create a unique cache key
        cache_key = f"{symbol}_{start_date}_{end_date}_{sanitized_interval}"
        cache_key_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        # Check cache first
        cached_data = self.cache.get(cache_key_hash)
        if cached_data is not None:
            logger.debug(f"Cache hit for {symbol} from {start_date} to {end_date}")
            df = cached_data
        else:
            # Fetch from database first
            df = self._fetch_from_database(symbol, start_date, end_date, sanitized_interval)
            
            # If database doesn't have all the data, fetch from external sources
            if df.empty or self._needs_update(df, start_date, end_date):
                logger.info(f"Fetching {symbol} data from external sources")
                new_data = await self._fetch_from_sources(symbol, start_date, end_date, sanitized_interval)
                
                if not new_data.empty:
                    # Clean and process the data
                    new_data = await self._process_data(new_data, symbol)
                    
                    # Store in database
                    await self._store_in_database(new_data, symbol)
                    
                    # Merge with existing data if any
                    if not df.empty:
                        df = pd.concat([df, new_data]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                    else:
                        df = new_data
                    
                    # Update cache
                    self.cache.set(cache_key_hash, df)
        
        # Enhance with additional data if requested
        if include_regime and 'market_regime' not in df.columns:
            df = await self._add_market_regime(df)
        
        if include_volatility and 'volatility' not in df.columns:
            df = await self._add_volatility(df)
        
        return df
    
    def _needs_update(self, df: pd.DataFrame, start_date: str, end_date: str) -> bool:
        """Check if the data needs updating from external sources"""
        if df.empty:
            return True
        
        try:
            # Convert to datetime for comparison
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df_dates = pd.to_datetime(df['timestamp'])
            
            # Check if we have the full range
            df_start = df_dates.min()
            df_end = df_dates.max()
            
            return (df_start > start) or (df_end < end) or (len(df) < 0.9 * (end - start).days)
        except Exception as e:
            logger.warning(f"Error checking if data needs update: {str(e)}")
            return True
    
    def _fetch_from_database(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Fetch historical data from the local database"""
        try:
            query = """
            SELECT symbol, timestamp, open, high, low, close, volume, adjusted_close, 
                   market_regime, volatility, source
            FROM historical_prices
            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
            """
            
            df = pd.read_sql_query(
                query, 
                self.db_conn, 
                params=(symbol, start_date, end_date)
            )
            
            # Convert timestamp to datetime
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        except Exception as e:
            logger.error(f"Database fetch error: {str(e)}")
            return pd.DataFrame()
    
    @RetryWithBackoff()
    async def _fetch_from_sources(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """
        Multi-source data fetching with fallback mechanism and rate limiting
        """
        await self.rate_limiter.wait()
        
        # Try primary API first
        try:
            data = await self._fetch_from_primary_api(symbol, start_date, end_date, interval)
            if not data.empty:
                return data
        except Exception as e:
            logger.warning(f"Primary API failed: {e}. Trying fallbacks...")
        
        # Try fallback sources sequentially
        fallback_sources = [
            self._fetch_from_yahoo,
            self._fetch_from_alpha_vantage,
            self._fetch_from_iex
        ]
        
        for source_func in fallback_sources:
            try:
                await self.rate_limiter.wait()
                data = await source_func(symbol, start_date, end_date, interval)
                if not data.empty:
                    logger.info(f"Data fetched successfully from {source_func.__name__}")
                    return data
            except Exception as e:
                logger.warning(f"{source_func.__name__} failed: {str(e)}")
        
        logger.error(f"All data sources failed for {symbol} from {start_date} to {end_date}")
        return pd.DataFrame()
    
    async def _fetch_from_primary_api(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Fetch data from primary Apex MarketDataAPI"""
        # Convert to async if the API is synchronous
        result = await asyncio.to_thread(
            self.market_data_api.get_historical_prices,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        if not result:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(result)
        
        # Standardize column names if needed
        if 'date' in df.columns and 'timestamp' not in df.columns:
            df['timestamp'] = df['date']
            df.drop('date', axis=1, inplace=True)
        
        # Add source information
        df['source'] = 'primary_api'
        
        return df
    
    async def _fetch_from_yahoo(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        # Create URL and parameters
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
        params = {
            "period1": int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()),
            "period2": int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()),
            "interval": interval if interval in ["1d", "1wk", "1mo"] else "1d",
            "events": "history",
            "includeAdjustedClose": "true"
        }
        
        # Fetch data with proper error handling
        response = await asyncio.to_thread(
            requests.get,
            url,
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
        response.raise_for_status()
        
        # Parse CSV data
        df = pd.read_csv(io.StringIO(response.text))
        
        # Standardize column names
        if 'Date' in df.columns:
            df.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adjusted_close'
            }, inplace=True)
        
        # Add source information
        df['source'] = 'yahoo'
        
        return df
    
    async def _fetch_from_alpha_vantage(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Fetch data from Alpha Vantage API"""
        api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            logger.warning("Alpha Vantage API key not found in environment variables")
            return pd.DataFrame()
        
        # Map Apex intervals to Alpha Vantage intervals
        interval_map = {
            "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "60min", "1d": "daily", "1wk": "weekly", "1mo": "monthly"
        }
        av_interval = interval_map.get(interval, "daily")
        
        # Create URL for API request
        if av_interval in ["daily", "weekly", "monthly"]:
            function = f"TIME_SERIES_{av_interval.upper()}_ADJUSTED"
            url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize=full&apikey={api_key}"
        else:
            function = "TIME_SERIES_INTRADAY"
            url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={av_interval}&outputsize=full&apikey={api_key}"
        
        # Fetch data
        response = await asyncio.to_thread(requests.get, url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Extract the time series data
        if "Error Message" in data:
            logger.warning(f"Alpha Vantage API error: {data['Error Message']}")
            return pd.DataFrame()
        
        # Determine which key contains the time series data
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key:
                time_series_key = key
                break
        
        if not time_series_key:
            logger.warning("Could not find time series data in Alpha Vantage response")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        df.index.name = 'timestamp'
        df.reset_index(inplace=True)
        
        # Standardize column names
        column_map = {}
        for col in df.columns:
            if 'open' in col.lower():
                column_map[col] = 'open'
            elif 'high' in col.lower():
                column_map[col] = 'high'
            elif 'low' in col.lower():
                column_map[col] = 'low'
            elif 'close' in col.lower() and 'adjusted' not in col.lower():
                column_map[col] = 'close'
            elif 'volume' in col.lower():
                column_map[col] = 'volume'
            elif 'adjusted' in col.lower():
                column_map[col] = 'adjusted_close'
        
        df.rename(columns=column_map, inplace=True)
        
        # Make sure all required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        # Convert to numeric values
        for col in df.columns:
            if col != 'timestamp':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter by date range
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        # Add source information
        df['source'] = 'alpha_vantage'
        
        return df
    
    async def _fetch_from_iex(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Fetch data from IEX Cloud API"""
        api_key = os.environ.get("IEX_API_KEY")
        if not api_key:
            logger.warning("IEX Cloud API key not found in environment variables")
            return pd.DataFrame()
        
        # IEX only supports specific intervals
        if interval not in ["1d", "1m", "5m", "15m", "30m", "1h"]:
            interval = "1d"  # Default to daily if unsupported
        
        # Format the range parameter
        if interval == "1d":
            # For daily data, use date range
            start_obj = datetime.strptime(start_date, "%Y-%m-%d")
            end_obj = datetime.strptime(end_date, "%Y-%m-%d")
            days_diff = (end_obj - start_obj).days
            
            if days_diff <= 5:
                range_param = "5d"
            elif days_diff <= 30:
                range_param = "1m"
            elif days_diff <= 90:
                range_param = "3m"
            elif days_diff <= 180:
                range_param = "6m"
            elif days_diff <= 365:
                range_param = "1y"
            elif days_diff <= 730:
                range_param = "2y"
            else:
                range_param = "5y"
        else:
            # For intraday data, use day ranges
            range_param = "1d"  # Default to 1 day for intraday data
        
        # Create API URL
        base_url = "https://cloud.iexapis.com/stable"
        if interval == "1d":
            endpoint = f"/stock/{symbol}/chart/{range_param}"
        else:
            interval_min = interval[:-1]  # Remove the 'm' or 'h'
            if interval.endswith('h'):
                interval_min = str(int(interval_min) * 60)  # Convert hours to minutes
            endpoint = f"/stock/{symbol}/chart/{range_param}?chartInterval={interval_min}"
        
        url = f"{base_url}{endpoint}&token={api_key}"
        
        # Fetch data
        response = await asyncio.to_thread(requests.get, url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            logger.warning(f"No data returned from IEX for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Standardize column names
        column_map = {
            'date': 'timestamp',
            'minute': 'time',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'changeOverTime': 'change'
        }
        
        # Apply column mapping where columns exist
        for old_col, new_col in column_map.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Handle timestamp formatting
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns and 'time' in df.columns:
            # If we have separate date and time columns (for intraday)
            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            df.drop(['date', 'time'], axis=1, inplace=True)
        
        # Add adjusted close if missing
        if 'adjusted_close' not in df.columns:
            df['adjusted_close'] = df['close']
        
        # Add source information
        df['source'] = 'iex'
        
        # Filter by date range
        if 'timestamp' in df.columns:
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        return df
    
    async def _process_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process and clean historical data with advanced techniques"""
        if df.empty:
            return df
        
        # Basic cleaning
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        # Ensure data is properly sorted
        df = df.sort_values('timestamp')
        
        # Add adjusted_close if it doesn't exist
        if 'adjusted_close' not in df.columns:
            df['adjusted_close'] = df['close']
        
        # Validate data integrity
        df = df[(df['high'] >= df['low']) & (df['high'] >= df['open']) & 
                (df['high'] >= df['close']) & (df['low'] <= df['open']) & 
                (df['low'] <= df['close']) & (df['volume'] >= 0)]
        
        # Run anomaly detection in parallel process
        try:
            # We'll use the existing AI module but execute it in a separate process
            detected_anomalies = await asyncio.to_thread(
                detect_anomalies,
                df
            )
            
            # Remove anomalies
            if detected_anomalies is not None and len(detected_anomalies) > 0:
                logger.info(f"Detected {len(detected_anomalies)} anomalies in {symbol} data")
                df = df.drop(detected_anomalies.index)
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {str(e)}")
        
        return df
    
    async def _store_in_database(self, df: pd.DataFrame, symbol: str) -> None:
        """Store processed data in the database efficiently"""
        if df.empty:
            return
        
        # Prepare data for insertion
        records = []
        for _, row in df.iterrows():
            record = (
                symbol,
                row['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row['timestamp'], pd.Timestamp) else row['timestamp'],
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume']),
                float(row['adjusted_close']) if 'adjusted_close' in row else float(row['close']),
                row.get('source', 'unknown'),
                row.get('market_regime', None),
                row.get('volatility', None)
            )
            records.append(record)
        
        # Use executemany for batch insertion performance
        try:
            # Convert to records in a vectorized way
            records = list(zip(
                [symbol] * len(df),
                df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                df['open'].astype(float).tolist(),
                df['high'].astype(float).tolist(),
                df['low'].astype(float).tolist(),
                df['close'].astype(float).tolist(),
                df['volume'].astype(float).tolist(),
                df['adjusted_close' if 'adjusted_close' in df else 'close'].astype(float).tolist(),
                df.get('source', pd.Series(['unknown'] * len(df))).tolist(),
                df.get('market_regime', pd.Series([None] * len(df))).tolist(),
                df.get('volatility', pd.Series([None] * len(df))).tolist()
            ))
            
            # Use async executor to prevent blocking
            await asyncio.to_thread(
                self._batch_insert_data,
                records
            )
            
            # Update metadata table
            await self._update_metadata(symbol, len(records))
            
            logger.info(f"Successfully stored {len(records)} records for {symbol}")
        except Exception as e:
            logger.error(f"Database insertion error: {str(e)}")
            
    def _batch_insert_data(self, records: List[Tuple]) -> None:
        """Batch insert data into the database with optimized transaction handling"""
        insert_query = """
        INSERT OR REPLACE INTO historical_prices 
        (symbol, timestamp, open, high, low, close, volume, adjusted_close, source, market_regime, volatility)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Use a transaction for better performance
        with self.db_conn:
            self.db_conn.executemany(insert_query, records)
            
    async def _update_metadata(self, symbol: str, record_count: int) -> None:
        """Update metadata for data quality tracking"""
        try:
            # Get current metadata if exists
            query = "SELECT quality_score, anomalies_detected FROM data_metadata WHERE symbol = ?"
            cursor = self.db_conn.execute(query, (symbol,))
            result = cursor.fetchone()
            
            if result:
                quality_score, anomalies = result
            else:
                quality_score, anomalies = 0.95, 0  # Default values for new entries
            
            # Update metadata with current timestamp and quality info
            upsert_query = """
            INSERT OR REPLACE INTO data_metadata 
            (symbol, last_updated, source, quality_score, anomalies_detected)
            VALUES (?, ?, ?, ?, ?)
            """
            
            await asyncio.to_thread(
                self.db_conn.execute,
                upsert_query,
                (symbol, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'multi_source', quality_score, anomalies)
            )
        except Exception as e:
            logger.warning(f"Metadata update failed: {str(e)}")

    @lru_cache(maxsize=32)
    async def _add_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime classification using AI-powered analysis"""
        try:
            # Use the integrated regime classifier
            regime_data = await asyncio.to_thread(
                self.regime_classifier.classify_market_regimes,
                df
            )
            
            if regime_data is not None and isinstance(regime_data, dict):
                df['market_regime'] = regime_data.get('regimes', pd.Series([None] * len(df)))
                
                # Store regime data for future use
                if 'regimes' in regime_data and len(regime_data['regimes']) > 0:
                    self._update_regime_data(df['symbol'].iloc[0], regime_data)
        except Exception as e:
            logger.warning(f"Market regime classification failed: {str(e)}")
        
        return df

    def _update_regime_data(self, symbol: str, regime_data: Dict) -> None:
        """Update regime data in the database"""
        if not regime_data or 'regimes' not in regime_data:
            return
        
        try:
            # Create pairs of (date, regime) for batch insertion
            records = []
            for date, regime in regime_data['regimes'].items():
                records.append((symbol, date, regime))
            
            if records:
                # Update database with new regime data
                query = """
                INSERT OR REPLACE INTO market_regimes (symbol, date, regime)
                VALUES (?, ?, ?)
                """
                
                with self.db_conn:
                    self.db_conn.executemany(query, records)
        except Exception as e:
            logger.warning(f"Regime data update failed: {str(e)}")

    async def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility metrics using GARCH and EWMA models"""
        if len(df) < 30:  # Need sufficient data for volatility calculation
            return df
        
        try:
            # Calculate returns using numba-optimized function
            returns = _calculate_returns(df['close'].values)
            
            # Initialize volatility column
            df['volatility'] = np.nan
            
            # EWMA volatility (faster calculation for recent data)
            ewma_vol = await asyncio.to_thread(
                lambda: pd.Series(returns).ewm(span=20).std().values
            )
            
            # Pad to match original length
            padded_ewma = np.append([np.nan], ewma_vol)
            df['volatility'] = padded_ewma
            
            # For longer time series, use GARCH in a separate process to avoid blocking
            if len(df) > 100:
                # Run GARCH in separate process for performance
                vol_data = await self._run_garch_model(returns, df.index[1:])
                
                if vol_data is not None:
                    # Store volatility data for risk management
                    symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else "unknown"
                    await self._store_volatility_data(symbol, vol_data)
            
        except Exception as e:
            logger.warning(f"Volatility calculation failed: {str(e)}")
        
        return df

    async def _run_garch_model(self, returns: np.ndarray, index) -> Optional[pd.DataFrame]:
        """Run GARCH model in a separate process for optimal performance"""
        try:
            # Use process executor to avoid blocking the event loop
            model_result = await asyncio.get_event_loop().run_in_executor(
                self.process_executor,
                self._fit_garch_model,
                returns
            )
            
            if model_result is not None:
                # Create DataFrame with volatility data
                vol_df = pd.DataFrame({
                    'date': index,
                    'garch_vol': model_result
                })
                return vol_df
            return None
        except Exception as e:
            logger.warning(f"GARCH model failed: {str(e)}")
            return None

    def _fit_garch_model(self, returns: np.ndarray) -> Optional[np.ndarray]:
        """Fit GARCH model to return data"""
        try:
            # Simple GARCH(1,1) model for volatility
            model = arch_model(returns, vol='Garch', p=1, q=1)
            model_fit = model.fit(disp='off')
            
            # Extract conditional volatility
            return model_fit.conditional_volatility
        except Exception:
            return None

    async def _store_volatility_data(self, symbol: str, vol_data: pd.DataFrame) -> None:
        """Store volatility data for risk management"""
        if vol_data is None or vol_data.empty:
            return
        
        try:
            # Prepare data for insertion
            records = []
            for _, row in vol_data.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else row['date']
                records.append((
                    symbol,
                    date_str,
                    float(row.get('historical_vol', np.nan)),
                    float(row.get('garch_vol', np.nan)),
                    float(row.get('ewma_vol', np.nan)),
                    row.get('regime', None)
                ))
            
            # Batch insert volatility data
            if records:
                insert_query = """
                INSERT OR REPLACE INTO volatility_history
                (symbol, date, historical_vol, garch_vol, ewma_vol, regime)
                VALUES (?, ?, ?, ?, ?, ?)
                """
                
                await asyncio.to_thread(
                    self.db_conn.executemany,
                    insert_query,
                    records
                )
        except Exception as e:
            logger.warning(f"Volatility data storage failed: {str(e)}")

    async def aggregate_data(self, symbol: str, interval: str, 
                        start_date: str, end_date: str) -> pd.DataFrame:
        """
        Convert data to different timeframes (e.g., 1min to 5min, hourly, daily)
        
        Args:
            symbol: Trading symbol
            interval: Target interval for aggregation (5m, 15m, 30m, 1h, 1d, 1wk)
            start_date: Start date
            end_date: End date
        
        Returns:
            Aggregated DataFrame with the requested timeframe
        """
        validate_symbol(symbol)
        
        # Get the base data (using 1m for intraday, 1d for daily+)
        if interval in ["5m", "15m", "30m", "1h"]:
            base_interval = "1m"
        else:
            base_interval = "1d"
        
        # Fetch base data
        df = await self.fetch_historical_data(symbol, start_date, end_date, base_interval)
        
        if df.empty:
            return df
        
        # Map to pandas resampling rule
        resample_map = {
            "5m": "5T", "15m": "15T", "30m": "30T", "1h": "1H",
            "1d": "1D", "1wk": "1W", "1mo": "1M"
        }
        
        resample_rule = resample_map.get(interval)
        if not resample_rule:
            logger.warning(f"Unsupported interval: {interval}")
            return df
        
        # Set timestamp as index for resampling
        df = df.set_index('timestamp')
        
        # Perform resampling with optimized aggregations
        resampled = df.resample(resample_rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'adjusted_close': 'last'
        })
        
        # Reset index to get timestamp back as column
        resampled = resampled.reset_index()
        
        return resampled

    async def get_correlation_matrix(self, symbols: List[str], start_date: str, 
                                end_date: str, interval: str = "1d") -> pd.DataFrame:
        """
        Calculate correlation matrix for a list of symbols
        
        Args:
            symbols: List of trading symbols
            start_date: Start date
            end_date: End date
            interval: Data interval
        
        Returns:
            Correlation matrix DataFrame
        """
        # Validate inputs
        for symbol in symbols:
            validate_symbol(symbol)
        validate_date_range(start_date, end_date)
        
        # Get data for all symbols
        dfs = []
        for symbol in symbols:
            df = await self.fetch_historical_data(symbol, start_date, end_date, interval)
            if not df.empty:
                dfs.append(df[['timestamp', 'close']].rename(columns={'close': symbol}))
        
        if not dfs:
            return pd.DataFrame()
        
        # Merge all dataframes on timestamp
        merged = dfs[0]
        for df in dfs[1:]:
            merged = pd.merge(merged, df, on='timestamp', how='outer')
        
        # Drop timestamp after merging
        merged = merged.set_index('timestamp')
        
        # Calculate and return correlation matrix
        return merged.corr(method='pearson')

    async def get_seasonal_analysis(self, symbol: str, years: int = 5) -> Dict:
        """
        Perform seasonal analysis to detect patterns (day of week, month of year)
        
        Args:
            symbol: Trading symbol
            years: Number of years to analyze
        
        Returns:
            Dictionary with seasonal patterns
        """
        validate_symbol(symbol)
        
        # Calculate start date
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
        
        # Get daily data
        df = await self.fetch_historical_data(symbol, start_date, end_date, "1d")
        
        if df.empty:
            return {}
        
        # Add datetime components
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        
        # Calculate returns
        df['return'] = df['close'].pct_change()
        
        # Calculate seasonal patterns
        result = {}
        
        # Monthly returns
        monthly = df.groupby('month')['return'].agg(['mean', 'std', 'count'])
        result['monthly'] = monthly.to_dict()
        
        # Day of week returns
        day_of_week = df.groupby('day_of_week')['return'].agg(['mean', 'std', 'count'])
        result['day_of_week'] = day_of_week.to_dict()
        
        # Add statistical significance (t-test)
        overall_mean = df['return'].mean()
        
        for month in range(1, 13):
            month_data = df[df['month'] == month]['return']
            if len(month_data) > 30:  # Ensure enough data for significance
                t_stat, p_value = stats.ttest_1samp(month_data, overall_mean)
                monthly_dict = result['monthly'].get('mean', {})
                if isinstance(monthly_dict, dict):
                    monthly_dict[f'p_value_{month}'] = p_value
        
        return result

    async def export_data_to_file(self, symbol: str, start_date: str, end_date: str, 
                                interval: str = "1d", file_format: str = "csv") -> str:
        """
        Export historical data to a file (CSV, Parquet, JSON)
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            file_format: Export format (csv, parquet, json)
        
        Returns:
            Path to the exported file
        """
        validate_symbol(symbol)
        validate_date_range(start_date, end_date)
        
        # Get data
        df = await self.fetch_historical_data(symbol, start_date, end_date, interval)
        
        if df.empty:
            return ""
        
        # Create export directory if needed
        export_dir = "data/exports"
        os.makedirs(export_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{symbol}_{start_date}_{end_date}_{interval}_{timestamp}"
        
        # Export based on format
        if file_format.lower() == "csv":
            filepath = f"{export_dir}/{filename}.csv"
            df.to_csv(filepath, index=False)
        elif file_format.lower() == "parquet":
            filepath = f"{export_dir}/{filename}.parquet"
            table = pa.Table.from_pandas(df)
            pq.write_table(table, filepath, compression='snappy')
        elif file_format.lower() == "json":
            filepath = f"{export_dir}/{filename}.json"
            df.to_json(filepath, orient='records', date_format='iso')
        else:
            return ""
        
        return filepath

    async def close(self):
        """Properly close all connections and resources"""
        try:
            # Close database connection
            if hasattr(self, 'db_conn') and self.db_conn:
                self.db_conn.close()
            
            # Close Redis connection if available
            if hasattr(self, 'cache') and hasattr(self.cache, 'redis_client') and self.cache.redis_client:
                self.cache.redis_client.close()
            
            # Shut down executors
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
            
            if hasattr(self, 'process_executor'):
                self.process_executor.shutdown(wait=False)
            
            # Close WebSocket connections
            if hasattr(self, 'websocket_handler'):
                await self.websocket_handler.close_all()
                
            logger.info("HistoricalData resources closed successfully")
        except Exception as e:
            logger.error(f"Error closing resources: {str(e)}")