"""
Apex Trade History System - Neural Execution Optimizer
Specialized for ultra-low latency trade execution tracking with multi-tiered storage.
"""

import os
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from typing import Dict, Any, List, Deque, Optional, Tuple
from collections import deque
import hashlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rocksdb import Options, DB, WriteBatch
import redis

# Apex Core Imports
from Apex.utils.logging.structured_logger import StructuredLogger
from Apex.src.Core.trading.execution.order_execution import OrderExecutionManager
from Apex.src.Core.trading.risk.risk_management import RiskEngine
from Apex.src.ai.ensembles.meta_trader import MetaTrader
from Apex.src.ai.analysis.market_regime_classifier import MarketRegimeClassifier
from Apex.src.Core.trading.execution.market_impact import MarketImpactCalculator
from Apex.src.Core.trading.logging.decision_logger import DecisionLogger


class TradeHistory:
    """
    AI-Optimized Trade Execution Historian with Multi-Tiered Storage System
    Singleton pattern with thread-safe initialization.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
            
    def __init__(self, mode: str = 'prod'):
        if self._initialized:
            return
            
        # Core initialization with exception handling
        try:
            self._initialized = True
            self.logger = StructuredLogger("TradeHistory")
            self.mode = mode
            
            # System integration handles - no direct AI logic, only data routing
            self.risk_engine = RiskEngine()
            self.market_regime = MarketRegimeClassifier()
            self.execution_manager = OrderExecutionManager()
            self.market_impact = MarketImpactCalculator()
            self.meta_trader = MetaTrader()  # Only used for data passing, not execution
            self.decision_logger = DecisionLogger()  # Moved AI anomaly detection here
            
            # Multi-tier storage setup (Low latency → High durability)
            self._setup_storage_system()
            
            # Processing threads with adaptive batch sizing
            self._start_processing_threads()
            
            # Real-time metrics tracking for low-latency analytics
            self.current_metrics_cache = {
                'slippage': 0.0,
                'latency': 0.0,
                'impact': 0.0,
                'anomaly_rate': 0.0
            }
            self.last_metrics_update = time.time()
            
        except Exception as e:
            self.logger.critical(f"TradeHistory initialization failed", error=str(e))
            raise SystemExit("Critical dependency missing") 

    def _setup_storage_system(self):
        """Multi-tiered storage system with redundancy and failover"""
        # Tier 1: Ultra-low latency in-memory cache
        self.cache: Deque[Dict] = deque(maxlen=100000)
        
        # Tier 1.5: Redis cache for distributed access
        try:
            self.redis = redis.Redis(
                host=os.environ.get('REDIS_HOST', 'localhost'),
                port=int(os.environ.get('REDIS_PORT', 6379)),
                password=os.environ.get('REDIS_PASSWORD', None),
                socket_timeout=0.1  # 100ms max timeout for real-time usage
            )
            self.redis.ping()  # Verify connection
        except Exception as e:
            self.logger.warning(f"Redis connection failed, using fallback: {str(e)}")
            self.redis = None
            
        # Tier 2: Fast persistent storage with RocksDB
        rocksdb_options = Options()
        rocksdb_options.create_if_missing = True
        rocksdb_options.max_open_files = 300000
        rocksdb_options.write_buffer_size = 67108864  # 64MB
        rocksdb_options.max_write_buffer_number = 3
        rocksdb_options.target_file_size_base = 67108864  # 64MB
        rocksdb_options.compression = rocksdb_options.zstd_compression
        
        try:
            self.db = DB(os.path.join('Apex/data/trade_history.db'), rocksdb_options)
        except Exception as e:
            self.logger.critical(f"RocksDB initialization failed", error=str(e))
            raise SystemExit("Storage engine failure")
            
        # Tier 3: Analytical storage setup (Parquet)
        self.parquet_dir = 'Apex/data/trades'
        os.makedirs(self.parquet_dir, exist_ok=True)
        
        # Write queues with adaptive batch sizes
        self.write_queue = deque(maxlen=1000000)
        self.parquet_queue = deque(maxlen=100000)
        self.current_batch_size = 1000  # Dynamically adjusted based on system load
        
    def _start_processing_threads(self):
        """Initialize processing threads with proper exception handling"""
        # Real-time processing threads with error recovery
        self.thread_active = True
        self.writer_thread = threading.Thread(target=self._adaptive_batch_writer, daemon=True)
        self.metrics_thread = threading.Thread(target=self._update_metrics_loop, daemon=True)
        
        # Thread start with health check
        self.writer_thread.start()
        self.metrics_thread.start()
        
        # Verify threads are running
        if not self.writer_thread.is_alive() or not self.metrics_thread.is_alive():
            self.logger.critical("Failed to start processing threads")
            raise RuntimeError("Thread initialization failure")
            
    def log_trade(self, trade_data: Dict[str, Any]) -> str:
        """
        Nanosecond-precision trade logging with performance optimization
        Returns the trade_id for reference
        """
        # Generate trade ID with cryptographically secure method
        trade_id = self._generate_secure_id()
        
        # Market context with minimal computation
        market_context = {
            'market_regime': self.market_regime.current_regime,
            'liquidity_state': self.execution_manager.liquidity_state
        }
        
        # Core metadata without AI computation
        trade_metadata = {
            'trade_id': trade_id,
            'timestamp': time.time_ns(),
            'pre_trade_spread': self.execution_manager.current_spread,
            'ai_strategy_version': self.meta_trader.active_strategy_version,
            'execution_path_hash': self._hash_execution_path(trade_data.get('execution_path', {}))
        }
        
        # Create record with minimal overhead
        full_record = {**trade_data, **trade_metadata, **market_context}
        
        # Multi-tier write pipeline
        self._store_in_memory(full_record)
        self._queue_for_persistent_storage(full_record)
        
        # Send basic metrics to risk engine (without waiting for full analysis)
        if 'expected_price' in trade_data and 'execution_price' in trade_data:
            slippage = abs(trade_data['expected_price'] - trade_data['execution_price']) / trade_data['expected_price']
            asyncio.create_task(self.risk_engine.register_slippage(slippage))
        
        return trade_id
            
    def _store_in_memory(self, record: Dict[str, Any]):
        """Ultra-fast in-memory storage for real-time access"""
        # Memory cache for lowest latency
        self.cache.append(record)
        
        # Redis storage for distributed access if available
        if self.redis:
            try:
                # Store only essentials in Redis for performance
                redis_record = {
                    k: v for k, v in record.items() 
                    if k in ('trade_id', 'timestamp', 'symbol', 'execution_price', 'volume', 'slippage')
                }
                self.redis.hset(f"trade:{record['trade_id']}", mapping=redis_record)
                self.redis.expire(f"trade:{record['trade_id']}", 3600)  # 1 hour expiry
            except Exception as e:
                self.logger.warning(f"Redis write failed: {str(e)}")
    
    def _queue_for_persistent_storage(self, record: Dict[str, Any]):
        """Queue record for persistent storage with backpressure handling"""
        # Add to write queue for RocksDB
        self.write_queue.append(record)
        
        # Add to Parquet queue with adaptive batching
        self.parquet_queue.append(record)
        
        # Dynamic batching based on queue size
        queue_fullness = len(self.write_queue) / self.write_queue.maxlen
        if queue_fullness > 0.8:
            # Increase batch size when queue is getting full
            self.current_batch_size = min(5000, int(self.current_batch_size * 1.5))
        elif queue_fullness < 0.2:
            # Decrease batch size when queue is mostly empty
            self.current_batch_size = max(100, int(self.current_batch_size * 0.8))
    
    def _generate_secure_id(self) -> str:
        """Cryptographically secure ID generation with performance optimization"""
        # Use mix of time and random for uniqueness and security
        time_component = str(time.time_ns()).encode()
        random_component = os.urandom(8)
        
        # Fast SHA-256 hash for secure ID
        return hashlib.sha256(time_component + random_component).hexdigest()[:16]
    
    def _hash_execution_path(self, execution_path: Dict) -> str:
        """Fast hashing of execution path for efficient storage"""
        if not execution_path:
            return ""
            
        # Convert to sorted tuple of items for consistent hashing
        sorted_items = sorted(execution_path.items())
        return hashlib.sha256(str(sorted_items).encode()).hexdigest()[:8]
        
    def _adaptive_batch_writer(self):
        """
        Adaptive batch writer that adjusts write frequency based on system load
        """
        rocksdb_batch = []
        parquet_batch = []
        
        while self.thread_active:
            try:
                # Process RocksDB writes
                while self.write_queue and len(rocksdb_batch) < self.current_batch_size:
                    record = self.write_queue.popleft()
                    rocksdb_batch.append(record)
                
                # Process Parquet writes
                while self.parquet_queue and len(parquet_batch) < 5000:
                    record = self.parquet_queue.popleft()
                    parquet_batch.append(record)
                
                # Execute RocksDB writes if batch has data
                if rocksdb_batch:
                    self._write_rocksdb_batch(rocksdb_batch)
                    rocksdb_batch = []
                
                # Execute Parquet writes if batch is large enough
                if len(parquet_batch) >= 5000:
                    asyncio.run_coroutine_threadsafe(
                        self._write_parquet_batch(parquet_batch),
                        asyncio.get_event_loop()
                    )
                    parquet_batch = []
                
                # Adaptive sleep based on queue size
                sleep_time = 0.001  # Base sleep of 1ms
                if len(self.write_queue) > 10000:
                    sleep_time = 0.0001  # Reduce to 0.1ms when busy
                elif len(self.write_queue) < 100:
                    sleep_time = 0.01  # Increase to 10ms when idle
                
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Batch writer error: {str(e)}")
                time.sleep(0.1)  # Error recovery pause
                
                # Recovery mechanism - put data back in queue
                if rocksdb_batch:
                    self.write_queue.extendleft(reversed(rocksdb_batch))
                    rocksdb_batch = []
    
    def _write_rocksdb_batch(self, batch: List[Dict]):
        """Optimized RocksDB batch writing with error handling"""
        try:
            write_batch = WriteBatch()
            
            for record in batch:
                # Use trade_id and timestamp for efficient key structure
                key = f"{record['timestamp']}_{record['trade_id']}".encode()
                # Lean serialization - pandas is too heavy for this operation
                value = str(record).encode()
                write_batch.put(key, value)
                
            # Atomic write
            self.db.write(write_batch)
            
        except Exception as e:
            self.logger.error(f"RocksDB write failed: {str(e)}")
            # Re-queue on failure to prevent data loss
            self.write_queue.extendleft(reversed(batch))
            
    async def _write_parquet_batch(self, batch: List[Dict]):
        """Asynchronous Parquet writing with optimized partitioning"""
        try:
            # Convert batch to DataFrame
            df = pd.DataFrame(batch)
            
            # Convert timestamp to datetime for partitioning
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            
            # Time-based partitioning with hourly granularity
            for (year, month, day, hour), group in df.groupby([
                df['timestamp'].dt.year,
                df['timestamp'].dt.month,
                df['timestamp'].dt.day,
                df['timestamp'].dt.hour
            ]):
                # Create partition path
                partition_path = f"{self.parquet_dir}/{year}/{month:02d}/{day:02d}/{hour:02d}"
                os.makedirs(partition_path, exist_ok=True)
                
                # Create file name with unique identifier
                file_path = f"{partition_path}/trades_{time.time_ns()}.parquet"
                
                # Write to Parquet with compression
                table = pa.Table.from_pandas(group)
                pq.write_table(
                    table, 
                    file_path,
                    compression='zstd',
                    compression_level=3  # Balance between speed and compression
                )
                
        except Exception as e:
            self.logger.error(f"Parquet write failed: {str(e)}")
            # In case of failure, add back to queue
            self.parquet_queue.extend(batch)
            
    def _update_metrics_loop(self):
        """Efficiently updates real-time metrics for system monitoring"""
        while self.thread_active:
            try:
                # Only update if we have trades and sufficient time has passed
                if self.cache and time.time() - self.last_metrics_update >= 0.1:  # 100ms update interval
                    self._calculate_current_metrics()
                    self.last_metrics_update = time.time()
                    
                    # Send metrics to risk engine asynchronously
                    asyncio.run_coroutine_threadsafe(
                        self.risk_engine.sync_execution_metrics(self.current_metrics_cache),
                        asyncio.get_event_loop()
                    )
                
                time.sleep(0.01)  # 10ms sleep
                
            except Exception as e:
                self.logger.error(f"Metrics update error: {str(e)}")
                time.sleep(0.1)  # Recovery sleep
    
    def _calculate_current_metrics(self):
        """Vectorized calculation of execution metrics for performance"""
        # Use recent trades only (last 1000) for real-time metrics
        recent_trades = list(self.cache)[-1000:]
        
        if not recent_trades:
            return
            
        # Extract values using optimized list comprehensions
        slippage_values = [t.get('slippage', 0.0) for t in recent_trades]
        latency_values = [t.get('latency', 0.0) for t in recent_trades]
        impact_values = [t.get('market_impact', 0.0) for t in recent_trades]
        anomaly_values = [t.get('anomaly_score', 0.0) for t in recent_trades]
        
        # Vectorized calculations with numpy
        self.current_metrics_cache = {
            'slippage': float(np.mean(slippage_values)) if slippage_values else 0.0,
            'latency': float(np.percentile(latency_values, 95)) if latency_values else 0.0,
            'impact': float(np.sum(impact_values)) if impact_values else 0.0,
            'anomaly_rate': float(np.mean(np.array(anomaly_values) > 0.8)) if anomaly_values else 0.0
        }
    
    @lru_cache(maxsize=128)
    def get_execution_stats(self, symbol: str, lookback_seconds: int = 300) -> Dict[str, Any]:
        """
        Get execution statistics for a specific symbol with caching
        
        Args:
            symbol: Trading symbol to analyze
            lookback_seconds: How far back to analyze trades
            
        Returns:
            Dictionary of execution statistics
        """
        # Filter recent trades for the given symbol
        cutoff_time = time.time_ns() - (lookback_seconds * 1_000_000_000)
        symbol_trades = [
            t for t in self.cache 
            if t.get('symbol') == symbol and t.get('timestamp', 0) >= cutoff_time
        ]
        
        if not symbol_trades:
            return {
                'count': 0,
                'avg_slippage': 0.0,
                'fill_rate': 0.0,
                'avg_latency': 0.0
            }
        
        # Basic statistics calculation
        slippage_values = [t.get('slippage', 0.0) for t in symbol_trades]
        latency_values = [t.get('latency', 0.0) for t in symbol_trades]
        filled_count = sum(1 for t in symbol_trades if t.get('filled', False))
        
        # Compute statistics efficiently
        return {
            'count': len(symbol_trades),
            'avg_slippage': float(np.mean(slippage_values)) if slippage_values else 0.0,
            'fill_rate': filled_count / len(symbol_trades) if symbol_trades else 0.0,
            'avg_latency': float(np.mean(latency_values)) if latency_values else 0.0,
            'last_trade_time': max(t.get('timestamp', 0) for t in symbol_trades) if symbol_trades else 0
        }
        
    async def analyze_venue_performance(self, hours: int = 24) -> Dict[str, Dict[str, float]]:
        """
        Analyze execution venue performance over a time period
        
        Args:
            hours: Lookback period in hours
            
        Returns:
            Dictionary of venue statistics
        """
        # Get relevant trades from cache first for speed
        cutoff_time = time.time_ns() - (hours * 3600 * 1_000_000_000)
        recent_trades = [t for t in self.cache if t.get('timestamp', 0) >= cutoff_time]
        
        # If not enough data in cache, load from database
        if len(recent_trades) < 1000 and hours > 1:
            # This runs in a separate thread to avoid blocking
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._load_historical_trades, 
                    cutoff_time=cutoff_time, 
                    limit=10000
                )
                recent_trades = future.result()
        
        # Group trades by venue
        venues = {}
        for trade in recent_trades:
            venue = trade.get('execution_venue', 'unknown')
            if venue not in venues:
                venues[venue] = {
                    'trades': 0,
                    'slippage': [],
                    'latency': [],
                    'fills': 0
                }
            
            venues[venue]['trades'] += 1
            venues[venue]['slippage'].append(trade.get('slippage', 0.0))
            venues[venue]['latency'].append(trade.get('latency', 0.0))
            if trade.get('filled', False):
                venues[venue]['fills'] += 1
        
        # Calculate statistics
        result = {}
        for venue, data in venues.items():
            if data['trades'] == 0:
                continue
                
            result[venue] = {
                'trade_count': data['trades'],
                'avg_slippage': float(np.mean(data['slippage'])),
                'fill_rate': data['fills'] / data['trades'],
                'avg_latency_ms': float(np.mean(data['latency']) * 1000),  # Convert to ms
                'p95_latency_ms': float(np.percentile(data['latency'], 95) * 1000) if data['latency'] else 0.0
            }
            
        return result
    
    def _load_historical_trades(self, cutoff_time: int, limit: int = 10000) -> List[Dict]:
        """Load historical trades from RocksDB with optimized key-based retrieval"""
        try:
            trades = []
            count = 0
            
            # Convert cutoff_time to string format used in keys
            cutoff_str = str(cutoff_time)
            
            it = self.db.iterkeys()
            it.seek(cutoff_str.encode())
            
            for key in it:
                if count >= limit:
                    break
                    
                try:
                    value = self.db.get(key)
                    if value:
                        # Convert from string back to dict
                        trade_data = eval(value.decode())
                        trades.append(trade_data)
                        count += 1
                except Exception as inner_e:
                    self.logger.warning(f"Failed to parse trade record: {str(inner_e)}")
                    
            return trades
            
        except Exception as e:
            self.logger.error(f"Historical trade loading failed: {str(e)}")
            return []
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get latest execution metrics for monitoring"""
        return self.current_metrics_cache.copy()
    
    def shutdown(self):
        """Graceful shutdown with proper cleanup"""
        self.logger.info("Trade History shutdown initiated")
        self.thread_active = False
        
        # Wait for threads to complete
        if hasattr(self, 'writer_thread') and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=2.0)
            
        if hasattr(self, 'metrics_thread') and self.metrics_thread.is_alive():
            self.metrics_thread.join(timeout=1.0)
            
        # Flush remaining data
        if hasattr(self, 'write_queue') and self.write_queue:
            self._write_rocksdb_batch(list(self.write_queue))
            
        # Close database connections
        if hasattr(self, 'db'):
            self.db.close()
            
        if hasattr(self, 'redis') and self.redis:
            self.redis.close()
            
        self.logger.info("Trade History shutdown complete")


# Singleton instance for global access
trade_history = TradeHistory()

# Register shutdown handler
import atexit
atexit.register(trade_history.shutdown)