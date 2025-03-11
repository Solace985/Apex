import asyncio
import httpx
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Union
from collections import deque, defaultdict
import logging
from functools import lru_cache
import ujson
from concurrent.futures import ThreadPoolExecutor
import os

# Apex imports
from Apex.utils.helpers import validate_input, secure_api_call, rate_limiter
from Apex.utils.analytics.insider_data_cache import InsiderDataCache
from Apex.src.Core.fundamental.fundamental_engine import FundamentalAnalyzer
from Apex.src.ai.forecasting.sentiment_analysis import FinBertProcessor
from Apex.src.ai.reinforcement.q_learning import QLearningAgent
from Apex.src.ai.analysis.institutional_clusters import InsiderClusterModel
from Apex.src.Core.trading.risk.risk_engine import RiskEngine
from Apex.src.Core.trading.execution.order_execution import OrderExecutor
from Apex.src.Core.trading.strategies.strategy_orchestrator import StrategyOrchestrator
from Apex.utils.analytics.monte_carlo_simulator import MonteCarloSimulator
from Apex.src.Core.data.correlation_monitor import CorrelationMonitor
from Apex.src.Core.data.order_book_analyzer import OrderBookAnalyzer
from Apex.src.Core.data.trade_monitor import TradeMonitor
from Apex.src.Core.trading.logging.decision_logger import DecisionLogger
from Apex.src.Core.trading.security.security import SecurityValidator


class InstitutionalInsiderMonitor:
    """Enhanced institutional monitoring with multi-asset correlation and machine learning anomaly detection"""
    
    def __init__(self, config: Optional[Dict] = None):
        # Initialize core components
        self.fundamental = FundamentalAnalyzer()
        self.sentiment_analyzer = FinBertProcessor()
        self.rl_agent = QLearningAgent(state_size=7, action_size=5)  # Extended state space
        self.cluster_model = InsiderClusterModel()
        self.risk_engine = RiskEngine()
        self.order_executor = OrderExecutor()
        self.strategy_orchestrator = StrategyOrchestrator()
        self.data_cache = InsiderDataCache()
        self.monte_carlo = MonteCarloSimulator()
        self.correlation_monitor = CorrelationMonitor()
        self.order_book_analyzer = OrderBookAnalyzer()
        self.trade_monitor = TradeMonitor()
        self.decision_logger = DecisionLogger()
        self.security_validator = SecurityValidator()
        
        # Initialize optimized data structures
        self._init_datastructures()
        self._load_configurations(config)
        
        # Set up the ThreadPoolExecutor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("InstitutionalInsiderMonitor")

    def _init_datastructures(self):
        """Optimized data structures for real-time processing"""
        # High-performance deques with fixed size for O(1) operations
        self.insider_activity = deque(maxlen=2000)  # Extended history
        self.dark_pool_flows = defaultdict(lambda: deque(maxlen=500))  # Symbol-specific dark pool data
        self.crypto_whales = defaultdict(lambda: deque(maxlen=200))  # Token-specific whale data
        self.institutional_clusters = defaultdict(list)  # Market-wide institutional clustering
        
        # Pre-allocated numpy arrays for vectorized operations
        self.correlation_matrix = np.zeros((200, 200))  # Matrix for cross-asset correlation
        self.sentiment_history = np.zeros((100, 3))  # Store recent sentiment scores
        
        # Efficient lookups using sets and dictionaries
        self.watched_symbols = set()  # O(1) lookups for symbols under monitoring
        self.insider_scores = {}  # Historical effectiveness of insiders
        self.hedge_fund_positions = defaultdict(dict)  # Track major hedge fund positions

    def _load_configurations(self, config: Optional[Dict] = None):
        """Dynamic thresholds from config files with real-time updates"""
        from Apex.Config.assets.asset_universe import load_risk_parameters
        
        # Load default parameters if no config provided
        self.params = config or load_risk_parameters('insider_risk')
        
        # Performance critical parameters with defaults
        self.thresholds = {
            'dark_pool_threshold': self.params.get('dark_pool_threshold', 0.75),
            'insider_confidence': self.params.get('insider_confidence', 0.65),
            'crypto_correlation': self.params.get('crypto_correlation', 0.6),
            'sector_deviation': self.params.get('sector_deviation', 1.8),
            'execution_threshold': self.params.get('execution_threshold', 0.8),
            'sentiment_impact': self.params.get('sentiment_impact', 0.4),
            'anomaly_score': self.params.get('anomaly_score', 2.5),
            'position_scale': self.params.get('position_scale', 0.1),
            'market_cap_factor': self.params.get('market_cap_factor', 0.15),
            'hft_latency_ms': self.params.get('hft_latency_ms', 50),
        }
        
        # API endpoints with failover options
        self.endpoints = {
            "sec_primary": self.params.get("sec_api", "https://api.sec.gov/submissions/"),
            "sec_backup": self.params.get("sec_api_backup", "https://data.sec.gov/api/"),
            "dark_pool": self.params.get("dark_pool_api", "https://api.finra.org/darkpool/data"),
            "crypto": self.params.get("crypto_api", "https://api.whale-alert.io/v1/transactions"),
            "options_flow": self.params.get("options_api", "https://api.unusualwhales.com/flow"),
        }
        
        # Market regimes data for adaptive strategies
        self.market_regimes = self.params.get('market_regimes', {
            'high_volatility': {'dark_pool_threshold': 0.85, 'insider_confidence': 0.75},
            'low_volatility': {'dark_pool_threshold': 0.65, 'insider_confidence': 0.55},
            'earnings_season': {'dark_pool_threshold': 0.7, 'insider_confidence': 0.7},
            'crisis': {'dark_pool_threshold': 0.9, 'insider_confidence': 0.85},
        })

    async def monitor_insiders(self):
        """Parallel monitoring with priority-based execution"""
        self.logger.info("Starting insider monitoring system...")
        async with httpx.AsyncClient(timeout=30.0, limits=httpx.Limits(max_connections=20)) as client:
            while True:
                try:
                    # Determine current market regime for adaptive parameters
                    current_regime = self._determine_market_regime()
                    
                    # Adjust thresholds based on market regime
                    self._adjust_thresholds(current_regime)
                    
                    # Execute monitoring tasks in parallel with priority
                    results = await asyncio.gather(
                        self._process_sec_filings(client),
                        self._analyze_dark_pools(client),
                        self._track_crypto_whales(client),
                        self._monitor_options_flow(client),
                        return_exceptions=True  # Prevent one failure from stopping all monitors
                    )
                    
                    # Process any exceptions
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            self.logger.error(f"Error in monitoring task {i}: {str(result)}")
                    
                    # Update global correlation matrix across all asset classes
                    await self._update_cross_asset_correlation()
                    
                    # Optimize RL model with latest data
                    self._update_rl_models()
                    
                    # Calculate adaptive sleep interval based on market conditions
                    sleep_time = self._adaptive_sleep_interval()
                    await asyncio.sleep(sleep_time)
                    
                except Exception as e:
                    self.logger.error(f"Critical error in monitoring loop: {str(e)}")
                    await asyncio.sleep(5)  # Sleep briefly before retrying

    def _determine_market_regime(self) -> str:
        """Determine current market regime for adaptive parameters"""
        volatility = self.risk_engine.market_volatility()
        is_earnings = self.fundamental.earnings_imminent()
        market_stress = self.risk_engine.market_stress_index()
        
        if market_stress > 0.8:
            return 'crisis'
        elif volatility > 0.6:
            return 'high_volatility'
        elif is_earnings:
            return 'earnings_season'
        else:
            return 'low_volatility'

    def _adjust_thresholds(self, regime: str):
        """Adjust detection thresholds based on market regime"""
        if regime in self.market_regimes:
            regime_params = self.market_regimes[regime]
            for param, value in regime_params.items():
                if param in self.thresholds:
                    self.thresholds[param] = value

    async def _process_sec_filings(self, client: httpx.AsyncClient):
        """Parallel SEC filing processing with ML validation and adaptive clustering"""
        try:
            # Fetch SEC filings with fallback mechanism
            filings = await self._fetch_data_with_fallback(client, "sec")
            if not filings:
                return
            
            # Filter filings by relevance and security validation
            valid_filings = [f for f in filings if self._validate_sec_filing(f)]
            if not valid_filings:
                return
                
            # Convert to numpy array for vectorized processing
            filing_data = self._vectorize_filings(valid_filings)
            
            # Parallel model inference using ThreadPoolExecutor
            future_cluster = self.executor.submit(self.cluster_model.analyze, filing_data)
            future_sentiment = self.executor.submit(self.sentiment_analyzer.analyze_batch, 
                                                   [f.get('filing_text', '') for f in valid_filings])
            
            # Process results when available
            cluster_results = future_cluster.result()
            sentiment_results = future_sentiment.result()
            
            # Combined ML analysis with weighted features
            for filing, cluster, sentiment in zip(valid_filings, cluster_results, sentiment_results):
                anomaly_score = self._calculate_insider_anomaly_score(filing, cluster, sentiment)
                
                if anomaly_score > self.thresholds['anomaly_score']:
                    # Deep analysis of abnormal filing
                    await self._process_abnormal_filing(filing, anomaly_score, cluster, sentiment)
                    
                # Update insider history regardless of anomaly
                self._update_insider_history(filing, anomaly_score, sentiment)
                
        except Exception as e:
            self.logger.error(f"Error processing SEC filings: {str(e)}")
            raise

    async def _analyze_dark_pools(self, client: httpx.AsyncClient):
        """Dark pool trend analysis with pattern recognition and wash trade detection"""
        try:
            # Fetch dark pool trades
            trades = await self._fetch_data(client, "dark_pool")
            if not trades:
                return
                
            # Group by symbol for parallel processing
            symbol_trades = self._group_by_symbol(trades)
            
            # Analyze each symbol in parallel with ThreadPoolExecutor
            futures = []
            for symbol, symbol_trades in symbol_trades.items():
                futures.append(self.executor.submit(self._analyze_symbol_dark_pool, symbol, symbol_trades))
            
            # Process results and update strategies
            for future in futures:
                result = future.result()
                if result and result['confidence'] > self.thresholds['dark_pool_threshold']:
                    symbol, trend = result['symbol'], result
                    
                    # Cross-validate with order book imbalances
                    order_imbalance = await self.order_book_analyzer.get_imbalance(symbol)
                    
                    # If order book confirms dark pool signal, trigger strategy adjustment
                    if self._confirm_dark_pool_with_orderbook(trend, order_imbalance):
                        await self._trigger_strategy_adjustment(symbol, trend)
                    
                    # Store dark pool activity for future correlation
                    self.dark_pool_flows[symbol].append(trend)
                    
        except Exception as e:
            self.logger.error(f"Error analyzing dark pools: {str(e)}")
            raise

    def _analyze_symbol_dark_pool(self, symbol: str, trades: List[Dict]) -> Optional[Dict]:
        """Analyze dark pool trades for a specific symbol with pattern detection"""
        if not trades or len(trades) < 3:  # Need minimum data for pattern detection
            return None
            
        # Convert to numpy for vectorized calculations
        amounts = np.array([t.get('size', 0) for t in trades])
        prices = np.array([t.get('price', 0) for t in trades])
        timestamps = np.array([self._parse_timestamp(t.get('timestamp')) for t in trades])
        
        # Skip invalid data
        if len(amounts) == 0 or np.any(np.isnan(amounts)) or np.any(np.isnan(prices)):
            return None
            
        # Calculate temporal features for pattern detection
        time_diffs = np.diff(timestamps)
        periodic = np.std(time_diffs) < 0.3 * np.mean(time_diffs)
        
        # Detect accumulation vs distribution patterns
        price_amount_corr = np.corrcoef(prices, amounts)[0, 1] if len(prices) > 1 else 0
        
        # Wash trading detection - look for offsetting trades
        potential_wash = False
        if len(amounts) > 10:
            net_flows = np.convolve(amounts, [1, -1, -1, 1], mode='valid')
            potential_wash = np.any(np.abs(net_flows) < 0.01 * np.sum(np.abs(amounts)))
        
        # Detect sophisticated accumulation patterns (iceberg orders)
        iceberg_pattern = False
        if len(amounts) > 5:
            repeated_sizes = np.bincount(np.round(amounts).astype(int))
            if np.any(repeated_sizes > 3):  # Same size repeating multiple times
                iceberg_pattern = True
        
        # Calculate anomaly confidence for dark pool activity
        sector_avg = self.fundamental.get_sector_average_volume(symbol)
        relative_size = np.sum(amounts) / (sector_avg if sector_avg > 0 else 1)
        
        trend_type = 'accumulation' if price_amount_corr < -0.4 else 'distribution'
        
        # Confidence calculation with multiple factors
        confidence = min(1.0, (
            (0.4 * relative_size) +  # Volume significance
            (0.3 * abs(price_amount_corr)) +  # Pattern strength
            (0.2 * (1.0 if periodic else 0.0)) +  # Periodic buying/selling
            (0.1 * (1.0 if iceberg_pattern else 0.0))  # Iceberg detection
        ))
        
        # Reduce confidence if wash trading suspected
        if potential_wash:
            confidence *= 0.5
            trend_type = 'wash_trading'
        
        return {
            'symbol': symbol,
            'trend': trend_type,
            'confidence': confidence,
            'volume': float(np.sum(amounts)),
            'period': float(np.mean(time_diffs)) if len(time_diffs) > 0 else 0,
            'price_impact': float(np.max(prices) - np.min(prices)) / np.mean(prices) if len(prices) > 1 else 0,
            'wash_suspected': potential_wash,
            'iceberg_detected': iceberg_pattern,
            'price_amount_correlation': float(price_amount_corr) if not np.isnan(price_amount_corr) else 0,
            'timestamp': datetime.now().isoformat()
        }

    async def _track_crypto_whales(self, client: httpx.AsyncClient):
        """Blockchain whale tracking with stock correlation and stablecoin monitoring"""
        try:
            whale_txs = await self._fetch_data(client, "crypto")
            if not whale_txs:
                return
                
            # Focus on large transactions and stablecoin movements
            significant_txs = [tx for tx in whale_txs if 
                              self._is_significant_crypto_tx(tx)]
            
            for tx in significant_txs:
                # Find correlated stocks based on blockchain activity
                correlated_assets = await self._find_stock_correlation(tx)
                
                # Update risk models for correlated assets
                for asset in correlated_assets:
                    self._update_risk_model(asset['symbol'], tx['amount'], 'crypto_whale')
                    
                # Track stablecoin movements as liquidity indicators
                if tx.get('token_type') == 'stablecoin':
                    await self._analyze_stablecoin_flow(tx)
                    
                # Store whale transaction for pattern analysis
                self.crypto_whales[tx['token']].append(tx)
                
        except Exception as e:
            self.logger.error(f"Error tracking crypto whales: {str(e)}")
            raise

    def _is_significant_crypto_tx(self, tx: Dict) -> bool:
        """Determine if a crypto transaction is significant enough to analyze"""
        # Convert to USD for uniform comparison
        usd_value = tx.get('usd_value', tx.get('amount', 0) * tx.get('price', 0))
        
        # Different thresholds based on token type
        if tx.get('token_type') == 'major':  # BTC, ETH, etc.
            return usd_value > 1_000_000  # $1M+ for major tokens
        elif tx.get('token_type') == 'stablecoin':  # USDT, USDC, etc.
            return usd_value > 5_000_000  # $5M+ for stablecoins
        else:
            return usd_value > 500_000  # $500k+ for other tokens

    async def _analyze_stablecoin_flow(self, tx: Dict):
        """Analyze stablecoin flows as market liquidity indicators"""
        direction = tx.get('direction', '')
        if direction == 'to_exchange':
            # Liquidity moving to exchanges often precedes market activity
            self.risk_engine.update_liquidity_indicator('increasing')
        elif direction == 'from_exchange':
            # Liquidity leaving exchanges may indicate decreased trading interest
            self.risk_engine.update_liquidity_indicator('decreasing')
            
        # Check if stablecoin is moving to known trading wallets
        if tx.get('to_label') in self.data_cache.get_trading_wallets():
            # This might precede market buying
            self.strategy_orchestrator.flag_potential_buying('crypto')

    async def _monitor_options_flow(self, client: httpx.AsyncClient):
        """Track unusual options activity and correlate with insider/dark pool patterns"""
        try:
            options_data = await self._fetch_data(client, "options_flow")
            if not options_data:
                return
                
            # Focus on high delta and unusual volume options
            unusual_options = [opt for opt in options_data if 
                              self._is_unusual_option(opt)]
            
            # Group by underlying symbol
            symbol_options = self._group_by_symbol(unusual_options, key='underlying')
            
            for symbol, options in symbol_options.items():
                # Calculate options sentiment (bullish/bearish)
                sentiment = self._calculate_options_sentiment(options)
                
                # Cross-reference with insider activity
                insider_correlation = self._check_insider_options_correlation(symbol, options)
                
                if insider_correlation > self.thresholds['insider_confidence']:
                    # Strong correlation between options and insider activity
                    signal = {
                        'symbol': symbol,
                        'options_sentiment': sentiment,
                        'insider_correlation': insider_correlation,
                        'confidence': (sentiment['score'] + insider_correlation) / 2,
                        'source': 'options_insider_correlation',
                        'options_volume': sum(opt.get('volume', 0) for opt in options),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Trigger strategy adjustments for high-confidence signals
                    if signal['confidence'] > self.thresholds['execution_threshold']:
                        await self._trigger_strategy_adjustment(symbol, signal)
                        
        except Exception as e:
            self.logger.error(f"Error monitoring options flow: {str(e)}")
            raise

    def _is_unusual_option(self, option: Dict) -> bool:
        """Determine if an options trade is unusual and worth analyzing"""
        # Check for key indicators of unusual activity
        vol_oi_ratio = option.get('volume', 0) / max(1, option.get('open_interest', 1))
        is_high_delta = abs(option.get('delta', 0)) > 0.6
        is_large_premium = option.get('premium', 0) > 100000  # $100k+
        is_short_dated = option.get('dte', 30) < 14  # Two weeks or less
        
        # Different criteria for different option types
        if option.get('type') == 'call':
            return vol_oi_ratio > 3 and (is_high_delta or is_large_premium)
        else:  # put
            return vol_oi_ratio > 2 and (is_short_dated or is_large_premium)

    def _calculate_options_sentiment(self, options: List[Dict]) -> Dict:
        """Calculate overall sentiment from options flow"""
        # Separate calls and puts
        calls = [opt for opt in options if opt.get('type') == 'call']
        puts = [opt for opt in options if opt.get('type') == 'put']
        
        # Calculate weighted call/put ratio based on delta and premium
        call_weight = sum(opt.get('delta', 0.5) * opt.get('premium', 0) for opt in calls)
        put_weight = sum(abs(opt.get('delta', 0.5)) * opt.get('premium', 0) for opt in puts)
        
        # Determine sentiment direction and score
        total_weight = call_weight + put_weight
        if total_weight == 0:
            return {'direction': 'neutral', 'score': 0.5}
            
        call_ratio = call_weight / total_weight
        
        if call_ratio > 0.7:
            direction = 'strongly_bullish'
            score = min(1.0, call_ratio + 0.1)
        elif call_ratio > 0.55:
            direction = 'bullish'
            score = call_ratio
        elif call_ratio < 0.3:
            direction = 'strongly_bearish'
            score = min(1.0, (1 - call_ratio) + 0.1)
        elif call_ratio < 0.45:
            direction = 'bearish'
            score = 1 - call_ratio
        else:
            direction = 'neutral'
            score = 0.5
            
        return {
            'direction': direction,
            'score': score,
            'call_put_ratio': call_ratio,
            'call_weight': call_weight,
            'put_weight': put_weight
        }

    def _check_insider_options_correlation(self, symbol: str, options: List[Dict]) -> float:
        """Check correlation between options activity and insider trading"""
        # Get recent insider activity for this symbol
        recent_insiders = [
            f for f in self.insider_activity
            if f.get('symbol') == symbol and
            datetime.now() - self._parse_timestamp(f.get('timestamp')) < timedelta(days=5)
        ]
        
        if not recent_insiders:
            return 0.0
            
        # Calculate directional alignment
        insider_direction = 'bullish' if sum(f.get('net_amount', 0) for f in recent_insiders) > 0 else 'bearish'
        options_direction = self._calculate_options_sentiment(options)['direction']
        
        # Strong correlation if directions align
        if ('bullish' in insider_direction and 'bullish' in options_direction) or \
           ('bearish' in insider_direction and 'bearish' in options_direction):
            
            # Calculate temporal proximity - closer in time means stronger correlation
            closest_insider = min(recent_insiders, 
                                 key=lambda x: abs((datetime.now() - self._parse_timestamp(x.get('timestamp'))).total_seconds()))
            time_factor = max(0, 1 - (datetime.now() - self._parse_timestamp(closest_insider.get('timestamp'))).total_seconds() / (5 * 86400))
            
            # Calculate size correlation
            insider_size = sum(abs(f.get('net_amount', 0)) for f in recent_insiders)
            options_size = sum(opt.get('premium', 0) for opt in options)
            
            # Normalize by average trade sizes
            norm_insider = insider_size / max(1, self.fundamental.get_avg_insider_size(symbol))
            norm_options = options_size / max(1, self.fundamental.get_avg_options_premium(symbol))
            
            size_correlation = min(1.0, (norm_insider + norm_options) / 2)
            
            # Combined correlation score
            correlation = 0.5 + (0.3 * time_factor) + (0.2 * size_correlation)
            return min(1.0, correlation)
        
        # Directions don't align
        return 0.2  # Low baseline correlation

    async def _update_cross_asset_correlation(self):
        """Update correlation matrix across different asset classes"""
        # Get relevant symbols with activity
        active_symbols = set(list(self.dark_pool_flows.keys()) + 
                            [f.get('symbol') for f in self.insider_activity])
        
        if len(active_symbols) < 2:
            return
            
        # Request correlation matrix from correlation monitor
        correlation_matrix = await self.correlation_monitor.get_correlation_matrix(list(active_symbols))
        if correlation_matrix is not None:
            self.correlation_matrix = correlation_matrix
            
            # Detect clusters of correlated assets with insider activity
            clusters = self._detect_correlation_clusters()
            if clusters:
                for cluster in clusters:
                    self.logger.info(f"Detected correlated insider cluster: {cluster['symbols']}")
                    self.institutional_clusters[cluster['sector']] = cluster

    def _detect_correlation_clusters(self) -> List[Dict]:
        """Detect clusters of assets with high correlation and insider activity"""
        if self.correlation_matrix.size == 0:
            return []
            
        # Apply threshold to correlation matrix
        high_corr = self.correlation_matrix > 0.7
        
        # Find connected components (clusters)
        clusters = []
        visited = set()
        
        for i in range(high_corr.shape[0]):
            if i in visited:
                continue
                
            # Find all assets correlated with this one
            cluster = {i}
            stack = [i]
            
            while stack:
                node = stack.pop()
                visited.add(node)
                
                for j in range(high_corr.shape[1]):
                    if high_corr[node, j] and j not in visited:
                        cluster.add(j)
                        stack.append(j)
            
            # Only consider clusters with at least 3 assets
            if len(cluster) >= 3:
                # Get symbols and sector for this cluster
                symbols = [self.correlation_monitor.get_symbol_by_index(idx) for idx in cluster]
                sector = self.fundamental.get_common_sector(symbols)
                
                clusters.append({
                    'symbols': symbols,
                    'sector': sector,
                    'size': len(cluster),
                    'avg_correlation': float(np.mean(self.correlation_matrix[list(cluster), :][:, list(cluster)])),
                    'timestamp': datetime.now().isoformat()
                })
                
        return clusters

    async def _process_abnormal_filing(self, filing: Dict, anomaly_score: float, 
                                      cluster: Dict, sentiment: Dict):
        """Process abnormal insider filing with multi-factor analysis"""
        symbol = filing.get('symbol', '')
        if not symbol:
            return
            
        # Calculate market cap impact
        market_cap = await self.fundamental.get_market_cap(symbol)
        if market_cap:
            impact_pct = filing.get('transaction_value', 0) / market_cap
        else:
            impact_pct = 0
            
        # Adjust anomaly score by market cap impact
        adjusted_score = anomaly_score * (1 + self.thresholds['market_cap_factor'] * impact_pct)
        
        # Get order book data to check for potential front-running
        order_book = await self.order_book_analyzer.get_snapshot(symbol)
        front_running_detected = self._detect_front_running(order_book, filing)
        
        # Create filing signal with comprehensive metadata
        signal = {
            'symbol': symbol,
            'transaction_type': filing.get('transaction_type', ''),
            'insider_title': filing.get('insider_title', ''),
            'shares': filing.get('shares', 0),
            'price': filing.get('price', 0),
            'value': filing.get('transaction_value', 0),
            'impact_pct': impact_pct,
            'filing_date': filing.get('filing_date', ''),
            'transaction_date': filing.get('transaction_date', ''),
            'anomaly_score': anomaly_score,
            'adjusted_score': adjusted_score,
            'sentiment': sentiment.get('sentiment', 'neutral'),
            'sentiment_score': sentiment.get('score', 0.5),
            'cluster_deviation': cluster.get('size_deviation', 0),
            'sector_deviation': cluster.get('sector_deviation', 0),
            'front_running': front_running_detected,
            'market_cap': market_cap,
            'confidence': min(1.0, adjusted_score / 5.0),  # Normalize to 0-1
            'direction': 'buy' if filing.get('transaction_type', '').lower() in ['buy', 'purchase'] else 'sell',
            'timestamp': datetime.now().isoformat()
        }
        
        # Log decision for audit trail
        self.decision_logger.log_insider_signal(signal)
        
        # Execute trading strategy if confidence is high enough
        if signal['confidence'] > self.thresholds['execution_threshold']:
            # Verify with Monte Carlo simulation
            simulation = await self._simulate_insider_impact(signal)
            if simulation['expected_return'] > 0:
                # First, update risk parameters
                self.risk_engine.update_asset_risk(symbol, 'insider_activity', signal['confidence'])
                
                # Then trigger strategy adjustment
                await self._trigger_strategy_adjustment(symbol, signal)
            
        # Store signal in insider activity regardless of execution
        self.insider_activity.append(signal)
        
        # Update watched symbols for continuous monitoring
        self.watched_symbols.add(symbol)

    def _calculate_insider_anomaly_score(self, filing: Dict, cluster: Dict, sentiment: Dict) -> float:
        """Calculate comprehensive anomaly score for insider activity using ML-enhanced features"""
        # Extract key features for anomaly detection
        transaction_type = filing.get('transaction_type', '').lower()
        is_purchase = transaction_type in ['buy', 'purchase', 'acquire']
        insider_title = filing.get('insider_title', '').lower()
        transaction_value = filing.get('transaction_value', 0)
        shares = filing.get('shares', 0)
        
        # Initialize base score
        base_score = 0.0
        
        # Factor 1: Transaction size relative to historical patterns
        historical_avg = self.data_cache.get_avg_transaction_size(filing.get('symbol', ''))
        size_factor = min(3.0, transaction_value / max(1, historical_avg))
        
        # Factor a: High-level executives have more information advantage
        executive_weight = {
            'ceo': 2.0, 'chief': 1.8, 'president': 1.7, 'cfo': 1.7, 
            'director': 1.4, 'vp': 1.3, 'officer': 1.2
        }
        title_weight = 1.0
        for title, weight in executive_weight.items():
            if title in insider_title:
                title_weight = weight
                break
        
        # Factor 3: Buying is more significant than selling (executives sell for many reasons)
        direction_weight = 1.5 if is_purchase else 0.8
        
        # Factor 4: Cluster significance - how much does this deviate from sector-wide behavior
        cluster_deviation = cluster.get('sector_deviation', 0)
        cluster_weight = 1.0 + (0.5 * min(2.0, cluster_deviation))
        
        # Factor 5: Sentiment impact - align filing with market sentiment
        sentiment_score = sentiment.get('score', 0.5)
        sentiment_alignment = 1.0 + (self.thresholds['sentiment_impact'] * 
                                    (sentiment_score if is_purchase else 1 - sentiment_score))
        
        # Factor 6: Historical insider success rate
        insider_name = filing.get('insider_name', '')
        insider_success = self.insider_scores.get(insider_name, 0.5)
        success_weight = 1.0 + ((insider_success - 0.5) * 0.6)
        
        # Calculate combined anomaly score using weighted factors
        anomaly_score = (
            size_factor * 
            title_weight * 
            direction_weight * 
            cluster_weight * 
            sentiment_alignment * 
            success_weight
        )
        
        return anomaly_score

    def _update_insider_history(self, filing: Dict, anomaly_score: float, sentiment: Dict):
        """Update insider history and trading effectiveness metrics"""
        insider_name = filing.get('insider_name', '')
        if not insider_name:
            return
            
        # Store this filing in history
        simplified_filing = {
            'symbol': filing.get('symbol', ''),
            'insider_name': insider_name,
            'insider_title': filing.get('insider_title', ''),
            'transaction_type': filing.get('transaction_type', ''),
            'transaction_date': filing.get('transaction_date', ''),
            'filing_date': filing.get('filing_date', ''),
            'shares': filing.get('shares', 0),
            'price': filing.get('price', 0),
            'net_amount': filing.get('transaction_value', 0) * (1 if filing.get('transaction_type', '').lower() in ['buy', 'purchase'] else -1),
            'anomaly_score': anomaly_score,
            'sentiment': sentiment.get('sentiment', 'neutral'),
            'timestamp': datetime.now().isoformat()
        }
        
        self.insider_activity.append(simplified_filing)
        
        # Update insider success score if we have past trade outcomes
        if insider_name in self.insider_scores:
            # Will be updated by the performance evaluator callback
            pass
        else:
            # Initialize new insider with neutral score
            self.insider_scores[insider_name] = 0.5

    def _detect_front_running(self, order_book: Dict, filing: Dict) -> bool:
        """Detect potential front-running behavior in order book before insider transaction"""
        if not order_book or not filing:
            return False
            
        # Extract key metrics from order book
        bid_volume = sum(level['size'] for level in order_book.get('bids', []))
        ask_volume = sum(level['size'] for level in order_book.get('asks', []))
        
        # Calculate order book imbalance
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return False
            
        imbalance = abs(bid_volume - ask_volume) / total_volume
        
        # Different front-running patterns for buys vs sells
        transaction_type = filing.get('transaction_type', '').lower()
        is_purchase = transaction_type in ['buy', 'purchase', 'acquire']
        
        # For purchases, large ask imbalance suggests front-running
        # For sales, large bid imbalance suggests front-running
        if (is_purchase and ask_volume > 2 * bid_volume) or \
           (not is_purchase and bid_volume > 2 * ask_volume):
            
            # Check order timing relative to filing
            filing_time = self._parse_timestamp(filing.get('filing_date', ''))
            time_diff = (datetime.now() - filing_time).total_seconds() / 3600  # hours
            
            # Front-running typically happens just before filing publication
            if 0 < time_diff < 48:  # Within 48 hours of filing
                return True
                
        return False

    async def _simulate_insider_impact(self, signal: Dict) -> Dict:
        """Run Monte Carlo simulation to predict price impact of insider transaction"""
        symbol = signal.get('symbol', '')
        if not symbol:
            return {'expected_return': 0, 'confidence': 0}
            
        # Get historical insider transactions impact data
        historical_data = await self.data_cache.get_insider_impact_history(symbol)
        
        # Run Monte Carlo simulation
        simulation_results = await self.monte_carlo.simulate_insider_impact(
            symbol=symbol,
            transaction_type=signal.get('transaction_type', ''),
            insider_title=signal.get('insider_title', ''),
            impact_pct=signal.get('impact_pct', 0),
            historical_data=historical_data,
            num_simulations=1000  # Use fewer simulations for lower latency
        )
        
        return simulation_results

    async def _trigger_strategy_adjustment(self, symbol: str, signal: Dict):
        """Trigger strategy adjustment based on insider or dark pool signal"""
        # Create unified signal format for strategy orchestrator
        direction = signal.get('direction', signal.get('trend', 'neutral'))
        confidence = signal.get('confidence', 0.5)
        source = signal.get('source', 'insider')
        
        # Calculate position sizing based on confidence and risk parameters
        position_size = self._calculate_position_size(symbol, confidence, source)
        
        # Create strategy adjustment signal
        strategy_signal = {
            'symbol': symbol,
            'direction': direction,
            'confidence': confidence,
            'source': source,
            'position_size': position_size,
            'volatility_adjustment': await self.risk_engine.get_volatility_adjustment(symbol),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log decision for audit
        self.decision_logger.log_strategy_signal(strategy_signal)
        
        # Use security validator before executing strategy
        if not await self.security_validator.validate_security(symbol):
            self.logger.warning(f"Security validation failed for {symbol}. Skipping strategy adjustment.")
            return
        
        # Send signal to strategy orchestrator
        await self.strategy_orchestrator.adjust_strategy(strategy_signal)
        
        # Update real-time monitoring
        self.watched_symbols.add(symbol)

    def _calculate_position_size(self, symbol: str, confidence: float, source: str) -> float:
        """Calculate optimal position size based on signal confidence and risk parameters"""
        # Base position size from confidence
        base_size = confidence * self.thresholds['position_scale']
        
        # Adjust based on signal source
        source_weights = {
            'insider': 1.0,
            'dark_pool': 0.9,
            'options_insider_correlation': 1.1,
            'crypto_whale': 0.7,
            'insider_cluster': 1.2
        }
        source_weight = source_weights.get(source, 0.8)
        
        # Apply market regime adjustment
        regime = self._determine_market_regime()
        regime_adjustment = 1.0
        if regime == 'crisis':
            regime_adjustment = 0.5  # Reduce position size in crisis
        elif regime == 'high_volatility':
            regime_adjustment = 0.7  # Reduce in high volatility
        
        # Calculate final position size with risk constraints
        position_size = base_size * source_weight * regime_adjustment
        
        # Apply risk constraints from risk engine
        max_position = self.risk_engine.get_max_position_size(symbol)
        
        return min(position_size, max_position)

    def _adaptive_sleep_interval(self) -> float:
        """Calculate adaptive sleep interval based on market conditions"""
        base_interval = 5.0  # Base interval in seconds
        
        # Adjust for market hours (more frequent during market hours)
        current_hour = datetime.now().hour
        is_market_hours = 9 <= current_hour < 16  # 9:30AM-4PM EST approximation
        
        # Adjust for volatility
        volatility = self.risk_engine.market_volatility()
        
        # Calculate final interval
        if is_market_hours:
            # During market hours, based on volatility
            interval = base_interval * max(0.5, 1.0 - volatility)
        else:
            # Outside market hours, less frequent
            interval = base_interval * 3.0
            
        # Enforce minimum interval for rate limiting
        return max(1.0, interval)

    def _parse_timestamp(self, timestamp_str) -> datetime:
        """Parse timestamp string to datetime with error handling"""
        if not timestamp_str:
            return datetime.now()
            
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            try:
                return datetime.strptime(timestamp_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                return datetime.now()

    def _group_by_symbol(self, items: List[Dict], key: str = 'symbol') -> Dict[str, List[Dict]]:
        """Group a list of dictionaries by symbol key"""
        result = defaultdict(list)
        for item in items:
            symbol = item.get(key, '')
            if symbol:
                result[symbol].append(item)
        return result

    def _vectorize_filings(self, filings: List[Dict]) -> np.ndarray:
        """Convert filings to numpy array for vectorized processing"""
        # Extract key numerical features
        features = []
        for filing in filings:
            # Create feature vector for ML models
            feature = [
                filing.get('transaction_value', 0),
                1.0 if filing.get('transaction_type', '').lower() in ['buy', 'purchase'] else 0.0,
                filing.get('shares', 0),
                filing.get('price', 0),
                filing.get('executive_level', 0),  # Numeric representation of executive level
                (datetime.now() - self._parse_timestamp(filing.get('transaction_date', ''))).days,
                (self._parse_timestamp(filing.get('filing_date', '')) - 
                 self._parse_timestamp(filing.get('transaction_date', ''))).days
            ]
            features.append(feature)
            
        return np.array(features)

    async def _fetch_data_with_fallback(self, client: httpx.AsyncClient, data_type: str) -> List[Dict]:
        """Fetch data with fallback to backup endpoint"""
        if data_type == "sec":
            # Try primary endpoint first
            data = await self._fetch_data(client, "sec_primary")
            if data:
                return data
                
            # Fallback to backup endpoint
            return await self._fetch_data(client, "sec_backup")
        else:
            # For other data types, use standard fetch
            return await self._fetch_data(client, data_type)

    @rate_limiter
    async def _fetch_data(self, client: httpx.AsyncClient, endpoint_type: str) -> List[Dict]:
        """Fetch data from API with rate limiting and security"""
        if endpoint_type not in self.endpoints:
            self.logger.error(f"Unknown endpoint type: {endpoint_type}")
            return []
            
        endpoint = self.endpoints[endpoint_type]
        
        try:
            # Use secure API call wrapper
            response = await secure_api_call(client, "GET", endpoint)
            
            if response.status_code == 200:
                # Parse response with faster ujson
                data = ujson.loads(response.content)
                
                # Validate response data
                if not validate_input(data):
                    self.logger.warning(f"Invalid data received from {endpoint_type}")
                    return []
                    
                return data.get('data', []) if isinstance(data, dict) else data
            else:
                self.logger.warning(f"API error from {endpoint_type}: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error fetching {endpoint_type} data: {str(e)}")
            return []

    def _validate_sec_filing(self, filing: Dict) -> bool:
        """Validate SEC filing for relevance and security"""
        # Check for required fields
        required_fields = ['symbol', 'transaction_type', 'shares', 'price']
        if not all(filing.get(field) for field in required_fields):
            return False
            
        # Validate transaction size
        if filing.get('shares', 0) <= 0 or filing.get('price', 0) <= 0:
            return False
            
        # Calculate and add transaction value if not present
        if 'transaction_value' not in filing:
            filing['transaction_value'] = filing.get('shares', 0) * filing.get('price', 0)
            
        # Validate symbol against allowlist
        symbol = filing.get('symbol', '')
        if not self.security_validator.is_valid_symbol(symbol):
            return False
            
        return True

    def _confirm_dark_pool_with_orderbook(self, dark_pool_trend: Dict, order_imbalance: float) -> bool:
        """Confirm dark pool signal with order book imbalance"""
        # Dark pool trend direction
        trend_type = dark_pool_trend.get('trend', 'neutral')
        
        # For accumulation (buying), expect positive order imbalance (more bids)
        if trend_type == 'accumulation' and order_imbalance > 0.2:
            return True
            
        # For distribution (selling), expect negative order imbalance (more asks)
        if trend_type == 'distribution' and order_imbalance < -0.2:
            return True
            
        # If suspected wash trading, less likely to be confirmed by order book
        if dark_pool_trend.get('wash_suspected', False):
            return False
            
        return False

    async def _find_stock_correlation(self, crypto_tx: Dict) -> List[Dict]:
        """Find correlated stocks based on blockchain activity"""
        token = crypto_tx.get('token', '')
        if not token:
            return []
            
        # Get stocks with blockchain exposure
        blockchain_stocks = await self.fundamental.get_blockchain_stocks()
        
        # Filter relevant stocks based on token
        relevant_stocks = []
        
        # For major tokens like BTC, ETH, look at all blockchain stocks
        if token.lower() in ['btc', 'bitcoin', 'eth', 'ethereum']:
            relevant_stocks = blockchain_stocks
        else:
            # For other tokens, find specific correlations
            for stock in blockchain_stocks:
                if token.lower() in stock.get('token_correlations', []):
                    relevant_stocks.append(stock)
        
        # Calculate correlation strength based on transaction size
        usd_value = crypto_tx.get('usd_value', 0)
        for stock in relevant_stocks:
            # Calculate correlation coefficient (0-1)
            corr = min(1.0, (usd_value / 10_000_000) * stock.get('correlation_factor', 0.5))
            stock['correlation'] = corr
            
        # Return stocks with significant correlation
        return [s for s in relevant_stocks if s.get('correlation', 0) > self.thresholds['crypto_correlation']]

    def _update_rl_models(self):
        """Update reinforcement learning models with latest market data"""
        # Gather state features for RL model
        states = []
        for symbol in self.watched_symbols:
            # Create feature vector for RL state
            try:
                # Market features
                volatility = self.risk_engine.get_volatility(symbol)
                volume = self.trade_monitor.get_relative_volume(symbol)
                sentiment = self.sentiment_analyzer.get_latest_sentiment(symbol)
                
                # Insider features
                insider_score = self._get_avg_insider_score(symbol)
                dark_pool_signal = self._get_latest_dark_pool_signal(symbol)
                
                # Create state vector
                state = np.array([
                    volatility,
                    volume,
                    sentiment.get('score', 0.5),
                    insider_score,
                    dark_pool_signal,
                    self.risk_engine.market_stress_index(),
                    self.risk_engine.vix_percentile()
                ])
                
                # Update RL agent
                self.rl_agent.update_state(symbol, state)
                states.append((symbol, state))
                
            except Exception as e:
                self.logger.error(f"Error updating RL model for {symbol}: {str(e)}")
        
        # Batch update RL models for efficiency
        if states:
            self.rl_agent.batch_update(states)

    def _get_avg_insider_score(self, symbol: str) -> float:
        """Get average insider anomaly score for a symbol"""
        recent_insiders = [
            f for f in self.insider_activity
            if f.get('symbol') == symbol and
            datetime.now() - self._parse_timestamp(f.get('timestamp')) < timedelta(days=30)
        ]
        
        if not recent_insiders:
            return 0.5
            
        return np.mean([f.get('anomaly_score', 0) for f in recent_insiders]) / 5.0  # Normalize to 0-1

    def _get_latest_dark_pool_signal(self, symbol: str) -> float:
        """Get latest dark pool signal strength for a symbol"""
        if symbol not in self.dark_pool_flows or not self.dark_pool_flows[symbol]:
            return 0.0
            
        # Get most recent dark pool signal
        latest = self.dark_pool_flows[symbol][-1]
        
        # Convert to signed signal (-1 to 1)
        if latest.get('trend') == 'accumulation':
            return latest.get('confidence', 0.5)
        elif latest.get('trend') == 'distribution':
            return -latest.get('confidence', 0.5)
        else:
            return 0.0

    def register_callbacks(self, performance_evaluator=None):
        """Register callbacks for integration with other modules"""
        # Register performance evaluator for insider score updates
        if performance_evaluator:
            performance_evaluator.register_insider_monitor(self)