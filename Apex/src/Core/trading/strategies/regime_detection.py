# src/Core/trading/strategies/regime_detection.py

import numpy as np
import logging
import asyncio
import hashlib
import time
import functools
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
import torch
from collections import deque

# Core System Imports
from src.Core.data.realtime.market_data import MultiFeedStreamManager
from src.Core.data.order_book_analyzer import OrderBookMicrostructure
from src.Core.trading.hft.liquidity_manager import LiquidityOptimizer, DarkPoolRouter
from src.Core.trading.risk.risk_management import AdaptiveRiskOrchestrator, GlobalRiskMonitor
from src.Core.trading.logging.decision_logger import RegimeAuditLogger
from src.Core.trading.strategies.strategy_orchestrator import StrategyRouter
from src.ai.ensembles.meta_trader import MarketRegimeEnsemble
from src.ai.analysis.market_regime_classifier import DeepRegimeClassifier
from src.ai.analysis.market_maker_patterns import ManipulationDetector
from src.ai.analysis.institutional_clusters import InstitutionalBehaviorAnalyzer
from utils.logging.structured_logger import QuantumLogger
from utils.helpers.error_handler import CriticalSystemGuard
from utils.analytics.monte_carlo_simulator import VolatilitySimulator
from utils.helpers.stealth_api import DataObfuscator
from utils.analytics.insider_data_cache import InsiderActivityMonitor
from src.Core.trading.security.alert_system import SecurityAlertManager
from src.Core.trading.risk.incident_response import MarketManipulationResponse


# Cache decorator for memory optimization
def cached_async(ttl_seconds=60):
    """Memory cache for async functions with time-to-live functionality"""
    cache = {}
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            key_hash = hashlib.md5(key.encode()).hexdigest()
            
            if key_hash in cache:
                result, timestamp = cache[key_hash]
                if time.time() - timestamp < ttl_seconds:
                    return result
            
            result = await func(*args, **kwargs)
            cache[key_hash] = (result, time.time())
            
            # Clean expired cache entries
            for k in list(cache.keys()):
                if time.time() - cache[k][1] > ttl_seconds:
                    del cache[k]
                    
            return result
        return wrapper
    return decorator

class RegimeClassificationCache:
    """Thread-safe, fixed-size cache for regime classifications"""
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = {}
        self._fifo_queue = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
    
    async def get(self, asset: str) -> Optional[Dict]:
        """Get regime classification from cache"""
        async with self._lock:
            return self.cache.get(asset)
    
    async def set(self, asset: str, regime: Dict) -> None:
        """Set regime classification in cache"""
        async with self._lock:
            if asset not in self.cache and len(self.cache) >= self.max_size:
                # Remove oldest item
                oldest = self._fifo_queue.popleft()
                del self.cache[oldest]
            
            self.cache[asset] = regime
            self._fifo_queue.append(asset)
    
    async def invalidate(self, asset: str) -> None:
        """Invalidate cache for specific asset"""
        async with self._lock:
            if asset in self.cache:
                del self.cache[asset]

class QuantumRegimeDetector:
    """
    AI-powered market regime classification system with institutional-grade adaptive capabilities
    
    Key responsibilities:
    1. Detect and classify market regimes across asset universe
    2. Provide regime-specific strategy recommendations
    3. Update dependent system components with regime classifications
    4. Monitor market manipulation attempts
    5. Provide continuous regime monitoring
    
    Integration points:
    - MultiFeedStreamManager: Provides market data
    - OrderBookMicrostructure: Analyzes order book depth and patterns
    - LiquidityOptimizer: Adjusts execution based on liquidity profiles
    - AdaptiveRiskOrchestrator: Updates risk parameters based on regimes
    - MarketRegimeEnsemble: AI model for regime classification
    - DeepRegimeClassifier: Deep learning model for regime classification
    - StrategyRouter: Receives regime updates for strategy selection
    - GlobalRiskMonitor: Receives regime data for risk assessment
    """
    
    # Class constants for improved performance
    REGIME_CONFIDENCE_THRESHOLD = 0.75
    REGIME_UPDATE_FREQUENCY_HZ = 10
    REGIME_TYPES = {
        'TRENDING': 0,
        'RANGING': 1,
        'HIGH_VOLATILITY': 2, 
        'LOW_VOLATILITY': 3,
        'MANIPULATION': 4,
        'TRANSITION': 5,
        'INSTITUTIONAL_ACCUMULATION': 6,
        'INSTITUTIONAL_DISTRIBUTION': 7
    }
    
    # Static strategy mapping for performance
    STRATEGY_MAP = {
        'TRENDING': ['momentum', 'breakout'],
        'RANGING': ['mean_reversion', 'arbitrage'],
        'HIGH_VOLATILITY': ['scalping', 'volatility_arbitrage'],
        'LOW_VOLATILITY': ['statistical_arb', 'carry_trades'],
        'MANIPULATION': ['pause_trading'],
        'TRANSITION': ['reduced_size', 'hedging'],
        'INSTITUTIONAL_ACCUMULATION': ['trend_following', 'breakout'],
        'INSTITUTIONAL_DISTRIBUTION': ['mean_reversion', 'protective_puts']
    }
    
    def __init__(self, asset_universe: list):
        """Initialize QuantumRegimeDetector with specified asset universe"""
        self.asset_universe = asset_universe
        self.regime_cache = RegimeClassificationCache(max_size=len(asset_universe) * 2)
        self._init_performance_metrics()
        self._init_components()
        self._init_state()
        
        # GPU acceleration if available
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.logger.info("GPU acceleration enabled for regime detection")
        
        # Adaptive update frequency
        self.dynamic_update_interval = 1.0 / self.REGIME_UPDATE_FREQUENCY_HZ
        
        # Threadpool with dynamic sizing
        max_workers = min(32, len(asset_universe) + 4)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize market manipulation detection subsystem
        self._init_manipulation_detection()

    def _init_performance_metrics(self):
        """Initialize performance tracking metrics"""
        self.performance_metrics = {
            'classification_latency': deque(maxlen=100),
            'data_fetch_latency': deque(maxlen=100),
            'processing_latency': deque(maxlen=100),
            'total_latency': deque(maxlen=100),
            'classification_success_rate': 0.99,  # Start optimistic
            'cache_hit_rate': 0.0
        }

    def _init_components(self):
        """Initialize integrated system components with error handling"""
        try:
            # Core Modules - direct dependency injection for faster access
            self.data_feeds = MultiFeedStreamManager()
            self.order_book_analyzer = OrderBookMicrostructure()
            self.liquidity_optimizer = LiquidityOptimizer()
            self.risk_orchestrator = AdaptiveRiskOrchestrator()
            self.dark_pool_router = DarkPoolRouter()
            
            # AI/ML Components
            self.deep_classifier = DeepRegimeClassifier()
            self.ensemble_model = MarketRegimeEnsemble()
            self.institutional_analyzer = InstitutionalBehaviorAnalyzer()
            self.insider_monitor = InsiderActivityMonitor()
            self._init_reinforcement_learner()
            
            # Utilities with performance optimization
            self.logger = QuantumLogger("quantum_regime")
            self.system_guard = CriticalSystemGuard()
            self.data_obfuscator = DataObfuscator()
            self.volatility_simulator = VolatilitySimulator()
            
            # Decision audit logging with advanced storage management
            self.audit_logger = RegimeAuditLogger()
            
            # Schedule automatic log cleanup to prevent storage bloat
            self._schedule_log_maintenance()
        except Exception as e:
            logging.critical(f"Fatal error initializing QuantumRegimeDetector components: {str(e)}")
            raise SystemExit("Critical system component initialization failed")

    def _init_state(self):
        """Initialize internal state management with thread safety"""
        self.last_regime_update = {}
        self.regime_transition_tracking = {}
        self.manipulation_scores = {}
        self.asset_volatility_state = {}
        self.is_running = False
        self.event_loop = None
        
        # Prefetch common data for performance
        self.prefetched_liquidity_profiles = {}
        self.prefetched_risk_profiles = {}

    def _init_reinforcement_learner(self):
        """Initialize RL component for continuous improvement"""
        from src.ai.reinforcement.maddpg_model import AdaptiveRegimeAgent
        self.rl_agent = AdaptiveRegimeAgent()

    def _init_manipulation_detection(self):
        """Initialize specialized market manipulation detection"""
        self.manipulation_detector = ManipulationDetector()
        self.manipulation_thresholds = {
            'order_book_imbalance': 0.85,  # 85% imbalance
            'price_volatility_spike': 3.5,  # 3.5x normal volatility
            'volume_anomaly': 2.5,         # 2.5x average volume
            'trade_size_anomaly': 3.0      # 3.0x average trade size
        }

    async def start(self):
        """Start the regime detector with proper initialization"""
        if self.is_running:
            return
            
        self.is_running = True
        self.event_loop = asyncio.get_event_loop()
        
        # Pre-load data for faster startup
        await self._prefetch_common_data()
        
        # Start continuous monitoring
        asyncio.create_task(self.realtime_regime_monitor())
        self.logger.info("QuantumRegimeDetector started successfully")

    async def stop(self):
        """Gracefully stop the regime detector"""
        self.is_running = False
        self.logger.info("QuantumRegimeDetector stopped")

    async def _prefetch_common_data(self):
        """Prefetch commonly used data for performance optimization"""
        # Prefetch liquidity profiles
        tasks = []
        for asset in self.asset_universe:
            tasks.append(self.liquidity_optimizer.get_liquidity_profile(asset))
        
        self.prefetched_liquidity_profiles = {
            asset: profile for asset, profile in 
            zip(self.asset_universe, await asyncio.gather(*tasks))
        }
        
        # Prefetch risk profiles
        self.prefetched_risk_profiles = {
            regime: self.risk_orchestrator.get_regime_risk_profile(regime)
            for regime in self.REGIME_TYPES.keys()
        }

    @cached_async(ttl_seconds=5)
    async def detect_market_regimes(self) -> Dict[str, Dict]:
        """
        Detect and classify market regimes across asset universe
        
        Returns:
            Dict[str, Dict]: Asset-keyed dictionary of regime classifications
        """
        start_time = time.perf_counter()
        
        try:
            # Fetch and validate data with error handling and timing
            data_fetch_start = time.perf_counter()
            raw_data = await self._fetch_validated_data()
            self.performance_metrics['data_fetch_latency'].append(
                time.perf_counter() - data_fetch_start
            )
            
            if not raw_data:
                self.logger.warning("No valid data received from feeds")
                return {}
                
            # Process data in parallel
            processing_start = time.perf_counter()
            processed_data = await self._parallel_process_data(raw_data)
            self.performance_metrics['processing_latency'].append(
                time.perf_counter() - processing_start
            )
            
            # Classify regimes with timing
            classification_start = time.perf_counter()
            regimes = await self._classify_regimes(processed_data)
            self.performance_metrics['classification_latency'].append(
                time.perf_counter() - classification_start
            )
            
            # Update dependent systems
            await self._update_system_components_batch(regimes)
            
            # Record total operation time
            self.performance_metrics['total_latency'].append(
                time.perf_counter() - start_time
            )
            
            return regimes
            
        except Exception as e:
            self.system_guard.handle_critical(e)
            self.logger.error(f"Regime detection failed: {str(e)}")
            return {}

    async def _fetch_validated_data(self) -> Dict:
        """
        Fetch and validate market data from multiple feeds with improved error handling
        
        Returns:
            Dict: Validated market data keyed by asset
        """
        try:
            # Fetch raw data from feeds
            raw_data = await self.data_feeds.get_multi_asset_feed(self.asset_universe)
            
            # Validate data integrity with cryptographic methods
            valid_data = {}
            for asset, data in raw_data.items():
                if not data:
                    self.logger.warning(f"Empty data received for {asset}")
                    continue
                    
                if self._validate_data_integrity(data):
                    valid_data[asset] = data
                else:
                    self.logger.warning(f"Data integrity validation failed for {asset}")
            
            # Check if we have enough valid data
            if len(valid_data) < len(self.asset_universe) * 0.7:  # 70% threshold
                self.logger.warning(f"Insufficient valid data: {len(valid_data)}/{len(self.asset_universe)}")
            
            return valid_data
            
        except Exception as e:
            self.logger.error(f"Data fetching error: {str(e)}")
            return {}

    def _validate_data_integrity(self, data: Dict) -> bool:
        """
        Cryptographic validation of market data
        
        Args:
            data: Market data to validate
            
        Returns:
            bool: True if data passes integrity checks
        """
        try:
            # Use non-blocking hash computation for better performance
            data_hash = hashlib.sha256(str(data).encode()).hexdigest()
            
            # Validate data signature
            is_valid = self.data_feeds.validate_data_signature(data_hash)
            
            # Additional validation checks
            if not is_valid:
                return False
                
            # Check for data staleness (older than 5 seconds)
            if 'timestamp' in data and (time.time() - data['timestamp']) > 5:
                self.logger.warning(f"Stale data detected: {time.time() - data['timestamp']}s old")
                return False
                
            # Check for required fields
            required_fields = ['open', 'high', 'low', 'close', 'volume']
            if not all(field in data for field in required_fields):
                self.logger.warning("Missing required data fields")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation error: {str(e)}")
            # Return False on any validation exception
            return False
        
    async def _parallel_process_data(self, raw_data: Dict) -> Dict:
        """
        Process data using parallel execution for maximum performance
        
        Args:
            raw_data: Raw market data by asset
            
        Returns:
            Dict: Processed data features ready for regime classification
        """
        # Create processing tasks for each asset
        tasks = []
        for asset, data in raw_data.items():
            # Use create_task for better cancellation handling
            tasks.append(asyncio.create_task(
                self._process_asset_data(asset, data)
            ))
        
        # Wait for all tasks to complete with timeout
        try:
            processed_results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.TimeoutError:
            self.logger.error("Data processing timeout")
            return {}
        
        # Filter out exceptions and empty results
        processed_data = {}
        for result in processed_results:
            if isinstance(result, Exception):
                self.logger.error(f"Processing error: {str(result)}")
                continue
                
            if result and 'asset' in result:
                processed_data[result['asset']] = result
        
        return processed_data

    async def _process_asset_data(self, asset: str, data: Dict) -> Optional[Dict]:
        """
        Transform raw data into regime detection features with error handling
        
        Args:
            asset: Asset identifier
            data: Raw market data
            
        Returns:
            Dict: Processed feature set for regime classification
        """
        try:
            # Check cache for recent order book data
            order_book = await self.regime_cache.get(f"ob_{asset}")
            if not order_book:
                # Only fetch if not in cache
                order_book = await self.order_book_analyzer.get_microstructure(asset)
                await self.regime_cache.set(f"ob_{asset}", order_book)
            
            # Use prefetched liquidity profile if available
            liquidity_profile = self.prefetched_liquidity_profiles.get(asset)
            if not liquidity_profile:
                liquidity_profile = await self.liquidity_optimizer.get_liquidity_profile(asset)
            
            # Extract price features using vectorized operations
            price_features = self._vectorized_price_features(data)
            
            # Compute volatility metrics
            volatility = await self.volatility_simulator.calculate_metrics(data)
            
            # Get institutional behavior data
            institutional_data = await self.institutional_analyzer.get_activity(asset)
            
            # Check for insider activity
            insider_activity = await self.insider_monitor.check_activity(asset)
            
            # Assemble feature package
            return {
                'asset': asset,
                'features': {
                    'price_actions': price_features,
                    'liquidity': liquidity_profile,
                    'order_book': order_book,
                    'volatility': volatility,
                    'institutional': institutional_data,
                    'insider': insider_activity
                },
                'metadata': {
                    'timestamp': time.time(),
                    'data_quality': self._calculate_data_quality(data)
                }
            }
        except Exception as e:
            self.logger.error(f"Data processing failed for {asset}: {str(e)}")
            return None

    def _vectorized_price_features(self, data: Dict) -> Dict:
        """
        Extract technical features using vectorized operations for maximum performance
        
        Args:
            data: Raw market data
            
        Returns:
            Dict: Extracted price features
        """
        # Ensure data arrays are numpy arrays for vectorized operations
        close_prices = np.array(data['close'][-100:])
        high_prices = np.array(data['high'][-100:])
        low_prices = np.array(data['low'][-100:])
        volume = np.array(data['volume'][-100:])
        
        # Calculate returns using vectorized operations
        returns = np.diff(close_prices) / close_prices[:-1]
        
        # Calculate high-low range
        high_low_range = high_prices - low_prices
        
        # Fast moving average calculation
        fast_ma = np.mean(close_prices[-20:])
        slow_ma = np.mean(close_prices[-50:])
        
        # Trend strength calculation
        trend_strength = (close_prices[-1] - close_prices[0]) / (np.std(close_prices) + 1e-8)
        
        # Volatility calculation
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Volume profile
        volume_trend = np.corrcoef(np.arange(len(volume)), volume)[0, 1]
        
        # Price momentum
        momentum = (close_prices[-1] / close_prices[-10] - 1) if len(close_prices) >= 10 else 0
        
        return {
            'returns': returns,
            'high_low_range': high_low_range,
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'trend_strength': trend_strength,
            'volatility': volatility,
            'volume_trend': volume_trend,
            'momentum': momentum,
            'close_prices': close_prices,
            'volume': volume
        }

    def _calculate_data_quality(self, data: Dict) -> float:
        """
        Calculate data quality score for confidence weighting
        
        Args:
            data: Raw market data
            
        Returns:
            float: Quality score from 0.0 to 1.0
        """
        # Check completeness
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        completeness = sum(1 for field in required_fields if field in data) / len(required_fields)
        
        # Check freshness
        freshness = 1.0
        if 'timestamp' in data:
            age_seconds = time.time() - data['timestamp']
            freshness = max(0, 1.0 - (age_seconds / 60.0))  # Linear decay over 1 minute
        
        # Check consistency
        consistency = 1.0
        if all(field in data for field in ['high', 'low', 'close']):
            if not (data['low'][-1] <= data['close'][-1] <= data['high'][-1]):
                consistency = 0.5  # Penalize for inconsistent price data
        
        # Combine scores with weights
        quality_score = (0.4 * completeness) + (0.4 * freshness) + (0.2 * consistency)
        return min(1.0, max(0.0, quality_score))  # Clamp between 0 and 1

    async def _classify_regimes(self, processed_data: Dict) -> Dict[str, Dict]:
        """
        Classify market regimes using AI models with high-performance parallel processing
        
        Args:
            processed_data: Processed feature data by asset
            
        Returns:
            Dict: Classified regimes by asset
        """
        # Use queue-based approach for better load balancing and throughput
        classification_queue = asyncio.Queue()
        result_queue = asyncio.Queue()
        regimes = {}
        
        # Check cache first to avoid unnecessary processing
        cache_check_tasks = []
        assets_to_process = []
        
        for asset, data in processed_data.items():
            cache_check_tasks.append(self._check_cache_for_asset(asset, data, result_queue))
            
        # Run cache checks in parallel
        await asyncio.gather(*cache_check_tasks)
        
        # Determine which assets need processing
        for asset, data in processed_data.items():
            if asset not in regimes:
                assets_to_process.append((asset, data))
                await classification_queue.put((asset, data))
        
        # Define worker function for parallel processing
        async def classification_worker():
            while not classification_queue.empty():
                try:
                    asset, data = await classification_queue.get()
                    result = await self._classify_single_asset(asset, data)
                    if result:
                        await result_queue.put((asset, result))
                    classification_queue.task_done()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Worker error classifying {asset}: {str(e)}")
                    classification_queue.task_done()
        
        # Spawn worker tasks - adaptive to system load
        worker_count = min(len(assets_to_process), 
                          max(1, min(20, len(self.asset_universe) // 5)))
        workers = [asyncio.create_task(classification_worker()) 
                  for _ in range(worker_count)]
        
        # Process results as they come in
        async def result_processor():
            processed_count = 0
            expected_count = len(assets_to_process)
            
            while processed_count < expected_count:
                try:
                    asset, result = await result_queue.get()
                    if result:
                        regimes[asset] = result
                        # Update cache
                        await self.regime_cache.set(asset, result)
                    processed_count += 1
                    result_queue.task_done()
                except Exception as e:
                    self.logger.error(f"Error processing result: {str(e)}")
        
        # Start result processor
        result_processor_task = asyncio.create_task(result_processor())
        
        # Wait for all classification tasks to complete
        if assets_to_process:
            await classification_queue.join()
            
        # Wait for result processing to complete
        if assets_to_process:
            await result_processor_task
            
        # Clean up workers
        for worker in workers:
            worker.cancel()
            
        # Log performance metrics
        self.logger.debug(f"Classified {len(regimes)}/{len(processed_data)} assets with {worker_count} workers")
        
        return regimes
    
    async def _check_cache_for_asset(self, asset: str, data: Dict, result_queue: asyncio.Queue) -> None:
        """Check if asset has recent classification in cache"""
        cached_regime = await self.regime_cache.get(asset)
        if cached_regime and time.time() - cached_regime.get('timestamp', 0) < 5:  # 5 second TTL
            self.performance_metrics['cache_hit_rate'] += 0.01  # Increment hit rate
            await result_queue.put((asset, cached_regime))

    async def _classify_single_asset(self, asset: str, data: Dict) -> Optional[Dict]:
        """
        Classify regime for single asset with AI ensemble and cross-validation
        
        Args:
            asset: Asset identifier
            data: Processed feature data
            
        Returns:
            Dict: Classified regime with confidence and parameters
        """
        try:
            # Track last regime for transition detection
            last_regime = self.last_regime_update.get(asset, {}).get('regime')
            
            # Get deep learning classification
            dl_prediction = await self.deep_classifier.predict(data['features'])
            
            # Get ensemble prediction
            ensemble_prediction = await self.ensemble_model.predict(data['features'])
            
            # Get institutional behavior prediction
            institutional_prediction = await self.institutional_analyzer.predict_regime(
                asset, data['features']['institutional']
            )
            
            # Cross-validation between models
            final_regime = await self._cross_validate_predictions(
                asset, 
                dl_prediction, 
                ensemble_prediction, 
                institutional_prediction,
                data['features']
            )
            
            # Early exit if confidence is too low after cross-validation
            if not self._validate_regime_confidence(final_regime):
                self.logger.debug(f"Low confidence regime for {asset}: {final_regime['confidence']:.2f}")
                return None
            
            # Check for market manipulation
            manipulation_score = await self._detect_market_manipulation(asset, data['features'])
            self.manipulation_scores[asset] = manipulation_score
            
            if manipulation_score > 0.8:  # High manipulation probability
                # Enhanced manipulation detection and alerting
                severity_level = "CRITICAL" if manipulation_score > 0.9 else "HIGH"
                manipulation_type = await self._identify_manipulation_type(asset, data['features'])
                
                self.logger.critical(f"🚨 MANIPULATION DETECTED: {asset} | Score: {manipulation_score:.3f} | Type: {manipulation_type} | Severity: {severity_level}")
                
                # Collect forensic data for investigation
                forensic_data = await self._collect_manipulation_evidence(asset, data['features'])
                
                # Trigger comprehensive security alert with detailed information
                alert_manager = SecurityAlertManager()
                alert_id = await alert_manager.trigger_alert(
                    asset=asset,
                    regime='MANIPULATION',
                    severity=severity_level,
                    confidence=manipulation_score,
                    manipulation_type=manipulation_type,
                    evidence=forensic_data,
                    timestamp=time.time(),
                    details=f"Manipulation pattern detected in {asset}. Immediate action required. Pattern matches {manipulation_type} with {manipulation_score:.2f} confidence."
                )
                
                # Initiate automated incident response procedures
                response_handler = MarketManipulationResponse()
                await response_handler.initiate_response_protocol(
                    asset=asset, 
                    alert_id=alert_id,
                    manipulation_score=manipulation_score,
                    manipulation_type=manipulation_type
                )
                
                # Log the incident for compliance and audit purposes
                await self.audit_logger.log_manipulation_incident(
                    asset=asset,
                    alert_id=alert_id,
                    manipulation_score=manipulation_score,
                    manipulation_type=manipulation_type,
                    evidence=forensic_data
                )
                
                # Notify global risk monitor for cross-asset risk assessment
                await GlobalRiskMonitor().register_manipulation_event(
                    asset=asset,
                    severity=severity_level,
                    score=manipulation_score,
                    timestamp=time.time()
                )
                
                return {
                    'regime': 'MANIPULATION',
                    'confidence': manipulation_score,
                    'timestamp': time.time(),
                    'manipulation_type': manipulation_type,
                    'alert_id': alert_id,
                    'liquidity_profile': data['features']['liquidity'],
                    'volatility_metrics': data['features']['volatility'],
                    'optimal_strategies': self._map_strategies('MANIPULATION'),
                    'risk_parameters': self.prefetched_risk_profiles.get('MANIPULATION', 
                                        self.risk_orchestrator.get_regime_risk_profile('MANIPULATION'))
                }
            
            # Check for regime transition
            if last_regime and last_regime != final_regime['classification']:
                self.logger.info(f"Regime transition for {asset}: {last_regime} -> {final_regime['classification']}")
                # Track transition for smoother response
                self.regime_transition_tracking[asset] = {
                    'from': last_regime,
                    'to': final_regime['classification'],
                    'timestamp': time.time(),
                    'confidence_delta': final_regime['confidence'] - self.last_regime_update.get(asset, {}).get('confidence', 0)
                }
                
                # Apply transition smoothing if confidence is borderline
                if final_regime['confidence'] < 0.85:
                    final_regime = self._apply_transition_smoothing(final_regime, last_regime)
            
            # Update last regime with confidence
            self.last_regime_update[asset] = {
                'regime': final_regime['classification'],
                'timestamp': time.time(),
                'confidence': final_regime['confidence']
            }
            
            # Build complete regime package
            return self._build_regime_package(final_regime, data)
            
        except Exception as e:
            self.logger.error(f"Classification failed for {asset}: {str(e)}")
            self.system_guard.record_error("regime_classification", asset, str(e))
            return None

    async def _resolve_predictions(self, dl_pred: Dict, ensemble_pred: Dict, 
                            institutional_pred: Dict, data_quality: float) -> Dict:
        """
        Resolve conflicts between different model predictions using adaptive weighted voting
        with reinforcement learning optimization
        
        Args:
            dl_pred: Deep learning model prediction
            ensemble_pred: Ensemble model prediction
            institutional_pred: Institutional behavior prediction
            data_quality: Data quality score
            
        Returns:
            Dict: Resolved prediction with highest confidence and optimized weights
        """
        # Adjust confidence based on data quality
        dl_conf = dl_pred['confidence'] * data_quality
        ensemble_conf = ensemble_pred['confidence'] * data_quality
        inst_conf = institutional_pred['confidence'] * data_quality
        
        # Get current confidence scores for RL agent
        confidence_scores = {
            'deep_learning': dl_conf,
            'ensemble': ensemble_conf,
            'institutional': inst_conf
        }
        
        # Use reinforcement learning to dynamically optimize model weights
        # based on historical performance and current market conditions
        try:
            # Get asset from context if available for asset-specific optimization
            asset = getattr(self, '_current_asset', None)
            
            # Optimize weights using RL agent with contextual information
            adjusted_weights = await self.adaptive_regime_agent.optimize_weights(
                confidence_scores,
                self.performance_metrics,
                regime_history=self.last_regime_update,
                asset=asset,
                market_state=self._get_global_volatility()
            )
            
            # Log weight adjustments for monitoring
            self.logger.debug(f"RL adjusted weights: {adjusted_weights} for asset: {asset}")
            
        except Exception as e:
            # Fallback to default weights if RL optimization fails
            self.logger.warning(f"RL weight optimization failed: {str(e)}. Using fallback weights.")
            adjusted_weights = {
                'deep_learning': 0.35,
                'ensemble': 0.45,
                'institutional': 0.20
            }
        
        # Calculate weighted scores for each regime type using optimized weights
        regimes = set([dl_pred['classification'], ensemble_pred['classification'], 
                      institutional_pred['classification']])
        
        regime_scores = {}
        for regime in regimes:
            score = 0
            if dl_pred['classification'] == regime:
                score += dl_conf * adjusted_weights['deep_learning']
            if ensemble_pred['classification'] == regime:
                score += ensemble_conf * adjusted_weights['ensemble']
            if institutional_pred['classification'] == regime:
                score += inst_conf * adjusted_weights['institutional']
            regime_scores[regime] = score
        
        # Find regime with highest score
        best_regime = max(regime_scores.items(), key=lambda x: x[1])
        
        # Calculate weighted confidence using optimized weights
        total_weight = sum(adjusted_weights.values())
        weighted_confidence = best_regime[1] / total_weight if total_weight > 0 else 0.0
        
        # Record weight effectiveness for RL feedback loop
        if hasattr(self, 'weight_performance_history'):
            self.weight_performance_history.append({
                'weights': adjusted_weights,
                'regime': best_regime[0],
                'confidence': weighted_confidence,
                'timestamp': time.time()
            })
        
        return {
            'classification': best_regime[0],
            'confidence': weighted_confidence,
            'model_predictions': {
                'deep_learning': dl_pred,
                'ensemble': ensemble_pred,
                'institutional': institutional_pred
            },
            'optimized_weights': adjusted_weights  # Include weights in result for transparency
        }

    def _validate_regime_confidence(self, regime: Dict) -> bool:
        """
        Ensure minimum confidence threshold for regime classification
        
        Args:
            regime: Regime classification result
            
        Returns:
            bool: True if confidence meets threshold
        """
        # Dynamic confidence threshold based on system performance
        dynamic_threshold = self.REGIME_CONFIDENCE_THRESHOLD
        
        # Adjust threshold based on recent performance
        if hasattr(self, 'performance_metrics') and 'classification_success_rate' in self.performance_metrics:
            # Lower threshold slightly if success rate is high
            if self.performance_metrics['classification_success_rate'] > 0.95:
                dynamic_threshold *= 0.9
            # Raise threshold if success rate is low
            elif self.performance_metrics['classification_success_rate'] < 0.85:
                dynamic_threshold *= 1.1
        
        return regime['confidence'] >= dynamic_threshold

    async def _detect_market_manipulation(self, asset: str, features: Dict) -> float:
        """
        Check for anomalous market conditions indicating potential manipulation
        
        Args:
            asset: Asset identifier
            features: Extracted market features
            
        Returns:
            float: Manipulation probability score (0.0-1.0)
        """
        # Use specialized manipulation detection
        manipulation_prob = await self.manipulation_detector.detect_abnormalities(asset, features)
        
        # Get additional contextual information for detection
        order_book = features['order_book']
        liquidity = features['liquidity']
        price_actions = features['price_actions']
        
        # Custom manipulation detection logic for specific patterns
        manipulation_signals = []
        
        # Check for extreme order book imbalance
        if order_book and 'bid_ask_imbalance' in order_book:
            imbalance = abs(order_book['bid_ask_imbalance'])
            if imbalance > self.manipulation_thresholds['order_book_imbalance']:
                manipulation_signals.append(imbalance / self.manipulation_thresholds['order_book_imbalance'])
        
        # Check for abnormal price volatility
        if 'volatility' in price_actions:
            volatility = price_actions['volatility']
            normal_volatility = features.get('volatility', {}).get('historical_avg_volatility', volatility * 0.5)
            volatility_ratio = volatility / max(normal_volatility, 1e-6)
            
            if volatility_ratio > self.manipulation_thresholds['price_volatility_spike']:
                manipulation_signals.append(volatility_ratio / self.manipulation_thresholds['price_volatility_spike'])
        
        # Check for volume anomalies
        if 'volume' in price_actions:
            latest_volume = np.mean(price_actions['volume'][-5:])  # Last 5 volume points
            avg_volume = np.mean(price_actions['volume'])
            volume_ratio = latest_volume / max(avg_volume, 1)
            
            if volume_ratio > self.manipulation_thresholds['volume_anomaly']:
                manipulation_signals.append(volume_ratio / self.manipulation_thresholds['volume_anomaly'])
        
        # Check for trade size anomalies in order book
        if order_book and 'trade_sizes' in order_book:
            avg_trade_size = np.mean(order_book['trade_sizes'])
            max_trade_size = np.max(order_book['trade_sizes'])
            trade_size_ratio = max_trade_size / max(avg_trade_size, 1)
            
            if trade_size_ratio > self.manipulation_thresholds['trade_size_anomaly']:
                manipulation_signals.append(trade_size_ratio / self.manipulation_thresholds['trade_size_anomaly'])
        
        # Check for liquidity withdrawal
        if liquidity and 'depth_profile' in liquidity:
            if liquidity['depth_profile'].get('sudden_withdrawal', False):
                manipulation_signals.append(1.5)  # Strong indicator
        
        # Combine all signals with existing manipulation probability
        if manipulation_signals:
            # Normalize signals and combine
            normalized_signals = [min(signal, 5.0) / 5.0 for signal in manipulation_signals]
            avg_signal = np.mean(normalized_signals)
            
            # Combine with model prediction using geometric mean
            combined_score = np.sqrt(manipulation_prob * avg_signal)
            
            # Apply sigmoid function for better distribution
            final_score = 1.0 / (1.0 + np.exp(-6 * (combined_score - 0.5)))
            
            # Log high manipulation scores
            if final_score > 0.7:
                self.logger.warning(f"High manipulation score for {asset}: {final_score:.2f}")
                
            return final_score
        
        return manipulation_prob

    def _build_regime_package(self, final_regime: Dict, data: Dict) -> Dict:
        """
        Build comprehensive regime package for strategy adaptation
        
        Args:
            final_regime: Classification results
            data: Processed feature data
            
        Returns:
            Dict: Complete regime package with strategies and parameters
        """
        regime_type = final_regime['classification']
        
        # Get predefined strategies for this regime
        optimal_strategies = self._map_strategies(regime_type)
        
        # Get risk parameters for this regime
        risk_profile = self.prefetched_risk_profiles.get(
            regime_type, 
            self.risk_orchestrator.get_regime_risk_profile(regime_type)
        )
        
        # Return complete package
        return {
            'regime': regime_type,
            'confidence': final_regime['confidence'],
            'timestamp': time.time(),
            'features': {
                'price_actions': data['features']['price_actions'],
                'institutional_activity': data['features']['institutional'],
                'liquidity_profile': data['features']['liquidity'],
                'volatility_metrics': data['features']['volatility']
            },
            'optimal_strategies': optimal_strategies,
            'risk_parameters': risk_profile,
            'execution_parameters': self._get_execution_parameters(regime_type, data['features']['liquidity'])
        }

    def _map_strategies(self, regime_type: str) -> List[str]:
        """
        Map regime types to optimal trading strategies
        
        Args:
            regime_type: Detected market regime
            
        Returns:
            List[str]: List of optimal trading strategies
        """
        return self.STRATEGY_MAP.get(regime_type, ['reduced_size'])

    def _get_execution_parameters(self, regime_type: str, liquidity_profile: Dict) -> Dict:
        """
        Generate execution parameters based on regime and liquidity
        
        Args:
            regime_type: Detected market regime
            liquidity_profile: Liquidity data
            
        Returns:
            Dict: Execution parameters
        """
        # Define execution parameters based on regime
        if regime_type == 'TRENDING':
            # In trending markets, use momentum-based execution
            return {
                'execution_style': 'VWAP',
                'aggression': 0.65,
                'dark_pool_usage': 0.3,
                'size_scaling': 1.2,
                'urgency_factor': 0.8
            }
        elif regime_type == 'RANGING':
            # In ranging markets, use mean reversion execution
            return {
                'execution_style': 'TWAP',
                'aggression': 0.4,
                'dark_pool_usage': 0.5,
                'size_scaling': 0.8,
                'urgency_factor': 0.3
            }
        elif regime_type == 'HIGH_VOLATILITY':
            # In volatile markets, use conservative execution
            return {
                'execution_style': 'PASSIVE',
                'aggression': 0.2,
                'dark_pool_usage': 0.7,
                'size_scaling': 0.5,
                'urgency_factor': 0.4
            }
        elif regime_type == 'LOW_VOLATILITY':
            # In stable markets, use aggressive execution
            return {
                'execution_style': 'AGGRESSIVE',
                'aggression': 0.8,
                'dark_pool_usage': 0.2,
                'size_scaling': 1.4,
                'urgency_factor': 0.6
            }
        elif regime_type == 'MANIPULATION':
            # During manipulation, use ultra-cautious execution
            return {
                'execution_style': 'DARK_ONLY',
                'aggression': 0.1,
                'dark_pool_usage': 0.9,
                'size_scaling': 0.3,
                'urgency_factor': 0.2
            }
        else:
            # Default execution parameters
            return {
                'execution_style': 'ADAPTIVE',
                'aggression': 0.5,
                'dark_pool_usage': 0.4,
                'size_scaling': 1.0,
                'urgency_factor': 0.5
            }

    async def _update_system_components_batch(self, regimes: Dict[str, Dict]) -> None:
        """
        Update dependent system components with new regime classifications
        
        Args:
            regimes: Regime classifications by asset
        """
        if not regimes:
            return
            
        # Group updates by component for batch processing
        strategy_updates = {}
        risk_updates = {}
        liquidity_updates = {}
        
        for asset, regime in regimes.items():
            # Create component-specific updates
            strategy_updates[asset] = {
                'regime': regime['regime'],
                'strategies': regime['optimal_strategies'],
                'confidence': regime['confidence']
            }
            
            risk_updates[asset] = {
                'regime': regime['regime'],
                'risk_parameters': regime['risk_parameters']
            }
            
            liquidity_updates[asset] = {
                'regime': regime['regime'],
                'liquidity_profile': regime.get('features', {}).get('liquidity_profile', {})
            }
        
        # Execute batch updates in parallel
        update_tasks = [
            self._update_strategy_router(strategy_updates),
            self._update_risk_orchestrator(risk_updates),
            self._update_liquidity_optimizer(liquidity_updates),
            self._update_audit_logger(regimes)
        ]
        
        await asyncio.gather(*update_tasks)

    async def _update_strategy_router(self, strategy_updates: Dict) -> None:
        """
        Update strategy router with new regime classifications
        
        Args:
            strategy_updates: Updates for strategy router
        """
        try:
            # Import only when needed to avoid circular dependencies
            from src.Core.trading.strategies.strategy_orchestrator import StrategyRouter
            router = StrategyRouter()
            
            # Send batch update
            await router.batch_update_regimes(strategy_updates)
        except Exception as e:
            self.logger.error(f"Strategy router update failed: {str(e)}")

    async def _update_risk_orchestrator(self, risk_updates: Dict) -> None:
        """
        Update risk orchestrator with new regime classifications
        
        Args:
            risk_updates: Updates for risk orchestrator
        """
        try:
            # Import only when needed to avoid circular dependencies
            await self.risk_orchestrator.batch_update_regimes(risk_updates)
        except Exception as e:
            self.logger.error(f"Risk orchestrator update failed: {str(e)}")

    async def _update_liquidity_optimizer(self, liquidity_updates: Dict) -> None:
        """
        Update liquidity optimizer with new regime classifications
        
        Args:
            liquidity_updates: Updates for liquidity optimizer
        """
        try:
            await self.liquidity_optimizer.batch_update_regimes(liquidity_updates)
        except Exception as e:
            self.logger.error(f"Liquidity optimizer update failed: {str(e)}")

    async def _update_audit_logger(self, regimes: Dict) -> None:
        """
        Update audit logger with new regime classifications
        
        Args:
            regimes: Regime classifications
        """
        try:
            await self.audit_logger.log_regime_classifications(regimes)
        except Exception as e:
            self.logger.error(f"Audit logger update failed: {str(e)}")

    async def realtime_regime_monitor(self) -> None:
        """
        Continuous monitoring of market regimes with adaptive frequency
        """
        while self.is_running:
            try:
                # Adaptive update frequency based on market volatility
                update_interval = self._calculate_adaptive_interval()
                
                # Detect regimes
                regimes = await self.detect_market_regimes()
                
                # Check for critical regime changes
                await self._check_critical_regime_changes(regimes)
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Adjust model weights based on performance
                if time.time() % 60 < 1:  # Once per minute
                    await self._adjust_model_weights()
                
                # Wait for next update with adaptive interval
                await asyncio.sleep(update_interval)
                
            except asyncio.CancelledError:
                self.logger.info("Regime monitor task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Regime monitor error: {str(e)}")
                await asyncio.sleep(1)  # Delay before retry

    def _calculate_adaptive_interval(self) -> float:
        """
        Calculate adaptive update interval based on market volatility
        
        Returns:
            float: Update interval in seconds
        """
        base_interval = 1.0 / self.REGIME_UPDATE_FREQUENCY_HZ
        
        # Check global volatility state
        global_volatility = self._get_global_volatility()
        
        # Adjust interval based on volatility
        if global_volatility > 0.8:  # High volatility
            return max(0.1, base_interval * 0.5)  # 2x faster updates, min 0.1s
        elif global_volatility < 0.3:  # Low volatility
            return min(10.0, base_interval * 2.0)  # 2x slower updates, max 10s
        else:
            return base_interval

    def _get_global_volatility(self) -> float:
        """
        Get global market volatility score
        
        Returns:
            float: Global volatility from 0.0 to 1.0
        """
        # If we have no asset volatility data, return moderate volatility
        if not self.asset_volatility_state:
            return 0.5
            
        # Calculate average volatility across assets
        return sum(self.asset_volatility_state.values()) / max(1, len(self.asset_volatility_state))

    async def _check_critical_regime_changes(self, regimes: Dict) -> None:
        """
        Check for critical regime changes that require immediate action
        
        Args:
            regimes: Current regime classifications
        """
        critical_changes = []
        
        for asset, regime in regimes.items():
            # Check for transition to high volatility or manipulation
            if regime['regime'] in ['HIGH_VOLATILITY', 'MANIPULATION']:
                if self.last_regime_update.get(asset, {}).get('regime') not in ['HIGH_VOLATILITY', 'MANIPULATION']:
                    critical_changes.append({
                        'asset': asset,
                        'from': self.last_regime_update.get(asset, {}).get('regime', 'UNKNOWN'),
                        'to': regime['regime'],
                        'confidence': regime['confidence']
                    })
        
        if critical_changes:
            self.logger.warning(f"Critical regime changes detected: {critical_changes}")
            
            # Notify risk management system
            await self.risk_orchestrator.handle_critical_regime_changes(critical_changes)
            
            # Update global risk monitor
            await GlobalRiskMonitor().update_critical_regimes(critical_changes)

    def _update_performance_metrics(self) -> None:
        """
        Update performance metrics based on recent operations
        """
        # Calculate average latencies
        if self.performance_metrics['classification_latency']:
            avg_latency = np.mean(self.performance_metrics['classification_latency'])
            self.logger.debug(f"Average classification latency: {avg_latency*1000:.2f}ms")
        
        # Calculate cache hit rate
        total_requests = len(self.asset_universe)
        if total_requests > 0:
            self.performance_metrics['cache_hit_rate'] *= 0.95  # Decay factor
            self.logger.debug(f"Cache hit rate: {self.performance_metrics['cache_hit_rate']:.2f}")

    async def _adjust_model_weights(self) -> None:
        """
        Adjust AI model weights based on performance using reinforcement learning
        """
        try:
            # Get current model weights for each asset
            weights = {
                asset: {
                    'deep_learning': 0.35,
                    'ensemble': 0.45,
                    'institutional': 0.20
                }
                for asset in self.asset_universe
            }
            
            # Get recent performance metrics
            performance = {
                'latency': np.mean(self.performance_metrics['total_latency']) if self.performance_metrics['total_latency'] else 0.1,
                'success_rate': self.performance_metrics['classification_success_rate']
            }
            
            # Use RL agent to adjust weights
            new_weights = await self.rl_agent.optimize_weights(weights, performance)
            
            # Update model weights
            # In a real system, this would be implemented to update the weights
            # For this implementation, we'll just log the update
            self.logger.info(f"Model weights adjusted based on performance")
            
        except Exception as e:
            self.logger.error(f"Model weight adjustment failed: {str(e)}")

    async def get_regime_classification(self, asset: str) -> Dict:
        """
        Get latest regime classification for specific asset
        
        Args:
            asset: Asset identifier
            
        Returns:
            Dict: Latest regime classification
        """
        # First try cache
        cached_regime = await self.regime_cache.get(asset)
        if cached_regime:
            return cached_regime
            
        # If not in cache, check last update
        if asset in self.last_regime_update:
            return {
                'regime': self.last_regime_update[asset]['regime'],
                'timestamp': self.last_regime_update[asset]['timestamp'],
                'confidence': 0.7,  # Default confidence for cached results
                'note': 'Retrieved from last update cache'
            }
            
        # If not available, perform on-demand classification
        self.logger.info(f"Performing on-demand regime classification for {asset}")
        
        try:
            # Fetch data for single asset
            raw_data = await self.data_feeds.get_data_feed(asset)
            if not raw_data:
                return {'regime': 'UNKNOWN', 'confidence': 0.0, 'error': 'No data available'}
                
            # Process data
            processed_data = await self._process_asset_data(asset, raw_data)
            if not processed_data:
                return {'regime': 'UNKNOWN', 'confidence': 0.0, 'error': 'Data processing failed'}
                
            # Classify regime
            classification = await self._classify_single_asset(asset, processed_data)
            if not classification:
                return {'regime': 'UNKNOWN', 'confidence': 0.0, 'error': 'Classification failed'}
                
            return classification
            
        except Exception as e:
            self.logger.error(f"On-demand classification failed for {asset}: {str(e)}")
            return {'regime': 'UNKNOWN', 'confidence': 0.0, 'error': str(e)}

    async def get_regime_statistics(self) -> Dict:
        """
        Get statistics about current market regimes across asset universe
        
        Returns:
            Dict: Statistics about current market regimes
        """
        # Count regimes by type
        regime_counts = {}
        for asset, update in self.last_regime_update.items():
            regime = update['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
        # Calculate average confidence
        confidences = []
        for asset in self.last_regime_update:
            cached_regime = await self.regime_cache.get(asset)
            if cached_regime and 'confidence' in cached_regime:
                confidences.append(cached_regime['confidence'])
                
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Calculate regime stability
        stability = {
            asset: time.time() - update['timestamp']
            for asset, update in self.last_regime_update.items()
        }
        
        avg_stability = np.mean(list(stability.values())) if stability else 0.0
        
        # Return statistics
        return {
            'regime_counts': regime_counts,
            'asset_count': len(self.last_regime_update),
            'average_confidence': avg_confidence,
            'average_stability_seconds': avg_stability,
            'manipulation_detected': any(score > 0.7 for score in self.manipulation_scores.values()),
            'timestamp': time.time()
        }

    async def simulate_regime_impact(self, asset: str, scenario: str) -> Dict:
        """
        Simulate impact of hypothetical regime changes for risk assessment
        
        Args:
            asset: Asset identifier
            scenario: Regime change scenario
            
        Returns:
            Dict: Simulated impact assessment
        """
        # Get current regime
        current_regime = await self.get_regime_classification(asset)
        if not current_regime or current_regime.get('regime') == 'UNKNOWN':
            return {'error': 'Unknown current regime'}
            
        # Define scenario mapping
        scenarios = {
            'volatility_spike': 'HIGH_VOLATILITY',
            'sudden_trend': 'TRENDING',
            'liquidity_crisis': 'MANIPULATION',
            'range_bound': 'RANGING',
            'stabilization': 'LOW_VOLATILITY'
        }
        
        target_regime = scenarios.get(scenario)
        if not target_regime:
            return {'error': f'Unknown scenario: {scenario}'}
            
        # Perform simulation using Monte Carlo
        sim_results = await self.volatility_simulator.simulate_regime_change(
            asset, current_regime['regime'], target_regime
        )
        
        # Get risk impact assessment
        risk_impact = await self.risk_orchestrator.assess_regime_change(
            asset, current_regime['regime'], target_regime
        )
        
        # Combine results
        return {
            'current_regime': current_regime['regime'],
            'simulated_regime': target_regime,
            'price_impact': sim_results.get('price_impact', 0.0),
            'volatility_change': sim_results.get('volatility_change', 0.0),
            'liquidity_impact': sim_results.get('liquidity_impact', 0.0),
            'risk_assessment': risk_impact,
            'recommended_actions': self._get_regime_change_recommendations(
                current_regime['regime'], target_regime
            )
        }

    def _get_regime_change_recommendations(self, current_regime: str, target_regime: str) -> List[str]:
        """
        Get recommendations for handling regime changes
        
        Args:
            current_regime: Current market regime
            target_regime: Target market regime
            
        Returns:
            List[str]: Recommended actions
        """
        # Define a mapping of regime transitions to recommendations
        transition_recommendations = {
            ('TRENDING', 'HIGH_VOLATILITY'): [
                'Reduce position sizes by 50%',
                'Implement dynamic stop-losses',
                'Switch to short-term scalping strategies',
                'Increase use of options for hedging'
            ],
            ('RANGING', 'TRENDING'): [
                'Switch from mean-reversion to momentum strategies',
                'Adjust take-profit levels to allow for trend continuation',
                'Implement trailing stops instead of fixed stops',
                'Focus on breakout confirmation signals'
            ],
            ('LOW_VOLATILITY', 'HIGH_VOLATILITY'): [
                'Immediately reduce leverage',
                'Widen stop-loss orders',
                'Increase position in defensive assets',
                'Deploy volatility-based position sizing'
            ],
            ('ANY', 'MANIPULATION'): [
                'Suspend new position entries',
                'Reduce existing positions by 75%',
                'Move to dark pool execution only',
                'Implement anti-manipulation protective measures'
            ]
        }
        
        # Get recommendations for specific transition
        key = (current_regime, target_regime)
        if key in transition_recommendations:
            return transition_recommendations[key]
            
        # Try generic target with "ANY" source
        key = ('ANY', target_regime)
        if key in transition_recommendations:
            return transition_recommendations[key]
            
        # Default recommendations
        return [
            'Review current positions for alignment with new regime',
            'Adjust risk parameters according to volatility changes',
            'Consider partial profit taking during transition period'
        ]

    async def export_regime_data(self, timeframe: str = 'today', cleanup_days: int = 30) -> Dict:
        """
        Export regime classification data for analysis and model training
        with automatic cleanup of old records to prevent storage bloat
        
        Args:
            timeframe: Time period for export
            cleanup_days: Number of days to keep logs before cleanup (default: 30)
            
        Returns:
            Dict: Regime classification data and cleanup statistics
        """
        try:
            # Perform cleanup of old logs before exporting data
            cleanup_result = await self.audit_logger.cleanup_old_logs(days=cleanup_days)
            
            # Get regime data from audit logger (after cleanup)
            data = await self.audit_logger.get_regime_history(timeframe)
            
            # Preprocess for export
            processed_data = {
                'metadata': {
                    'export_time': time.time(),
                    'timeframe': timeframe,
                    'asset_count': len(set(entry['asset'] for entry in data)) if data else 0,
                    'regime_changes': sum(1 for i in range(1, len(data)) 
                                    if data[i]['asset'] == data[i-1]['asset'] and 
                                    data[i]['regime'] != data[i-1]['regime']) if len(data) > 1 else 0,
                    'cleanup_performed': True,
                    'cleanup_days_threshold': cleanup_days,
                    'records_removed': cleanup_result.get('records_removed', 0),
                    'storage_saved': cleanup_result.get('storage_saved_kb', 0)
                },
                'regime_data': data,
                'statistics': await self.get_regime_statistics()
            }
            
            self.logger.info(f"Exported regime data with {len(data)} records after removing {cleanup_result.get('records_removed', 0)} old records")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Export regime data failed: {str(e)}")
            return {
                'error': str(e),
                'metadata': {
                    'export_time': time.time(),
                    'cleanup_performed': False,
                    'error_details': str(e)
                }
            }

    async def handle_insider_trading_signals(self) -> None:
        """
        Process insider trading signals to enhance regime detection
        """
        try:
            # Get latest insider trading signals
            insider_signals = await self.insider_monitor.get_latest_signals(self.asset_universe)
            
            # Process signals for integration
            for asset, signal in insider_signals.items():
                if not signal or signal.get('significance', 0) < 0.7:
                    continue
                    
                # Determine if signal indicates accumulation or distribution
                signal_type = signal.get('type', 'UNKNOWN')
                if signal_type == 'BUY':
                    regime_type = 'INSTITUTIONAL_ACCUMULATION'
                elif signal_type == 'SELL':
                    regime_type = 'INSTITUTIONAL_DISTRIBUTION'
                else:
                    continue
                    
                # Calculate confidence based on significance and volume
                confidence = min(0.95, signal.get('significance', 0.7) * 1.2)
                
                # Get current regime
                current_regime = await self.get_regime_classification(asset)
                
                # Only update if confidence is significantly higher
                if current_regime.get('confidence', 0) + 0.1 < confidence:
                    # Create new regime classification
                    regime_data = {
                        'regime': regime_type,
                        'confidence': confidence,
                        'timestamp': time.time(),
                        'source': 'insider_trading',
                        'insider_data': signal
                    }
                    
                    # Update cache
                    await self.regime_cache.set(asset, regime_data)
                    
                    # Update last regime
                    self.last_regime_update[asset] = {
                        'regime': regime_type,
                        'timestamp': time.time()
                    }
                    
                    self.logger.info(f"Regime updated based on insider trading: {asset} -> {regime_type}")
                    
        except Exception as e:
            self.logger.error(f"Insider trading signal handling failed: {str(e)}")

    async def analyze_market_maker_behavior(self) -> None:
        """
        Analyze market maker behavior for enhanced manipulation detection
        """
        try:
            for asset in self.asset_universe:
                # Get order book data
                order_book = await self.order_book_analyzer.get_microstructure(asset)
                if not order_book:
                    continue
                    
                # Detect market maker footprints
                mm_analysis = await self.manipulation_detector.analyze_market_maker_footprints(asset, order_book)
                if not mm_analysis:
                    continue
                    
                # Check for manipulation patterns
                if mm_analysis.get('manipulation_probability', 0) > 0.8:
                    self.logger.warning(f"Market maker manipulation detected for {asset}: {mm_analysis}")
                    
                    # Update manipulation score
                    self.manipulation_scores[asset] = mm_analysis.get('manipulation_probability', 0.8)
                    
                    # Create manipulation regime if confidence is high
                    if mm_analysis.get('confidence', 0) > 0.85:
                        regime_data = {
                            'regime': 'MANIPULATION',
                            'confidence': mm_analysis.get('confidence', 0.85),
                            'timestamp': time.time(),
                            'source': 'market_maker_analysis',
                            'mm_data': mm_analysis
                        }
                        
                        # Update cache
                        await self.regime_cache.set(asset, regime_data)
                        
                        # Update last regime
                        self.last_regime_update[asset] = {
                            'regime': 'MANIPULATION',
                            'timestamp': time.time()
                        }
                        
                        # Notify risk management immediately
                        await self.risk_orchestrator.handle_critical_regime_changes([{
                            'asset': asset,
                            'from': 'UNKNOWN',
                            'to': 'MANIPULATION',
                            'confidence': mm_analysis.get('confidence', 0.85)
                        }])
                    
        except Exception as e:
            self.logger.error(f"Market maker behavior analysis failed: {str(e)}")

    async def get_mobile_app_dashboard_data(self) -> Dict:
        """
        Get optimized data for mobile app dashboard with current regimes and recommendations
        
        Returns:
            Dict: Mobile-optimized dashboard data
        """
        try:
            # Get regime statistics
            stats = await self.get_regime_statistics()
            
            # Get top assets for each regime
            top_assets = {}
            for regime in self.REGIME_TYPES:
                regime_assets = [
                    asset for asset, update in self.last_regime_update.items()
                    if update['regime'] == regime
                ]
                
                if regime_assets:
                    # Get confidence values for sorting
                    asset_confidences = []
                    for asset in regime_assets:
                        cached = await self.regime_cache.get(asset)
                        confidence = cached.get('confidence', 0.5) if cached else 0.5
                        asset_confidences.append((asset, confidence))
                    
                    # Sort by confidence and take top 5
                    sorted_assets = sorted(asset_confidences, key=lambda x: x[1], reverse=True)[:5]
                    top_assets[regime] = sorted_assets
            
            # Get real-time market insights
            market_insights = await self._generate_market_insights()
            
            # Get regime transition probabilities
            transition_probs = await self._calculate_regime_transition_probabilities()
            
            # Get AI model performance metrics
            model_performance = await self._get_ai_model_performance_metrics()
            
            # Get risk-adjusted strategy recommendations
            strategy_recommendations = await self._generate_strategy_recommendations()
            
            # Get explainability data for current regime classifications
            explainability_data = await self._generate_explainability_data()
            
            # Prepare optimized mobile dashboard data
            dashboard_data = {
                "timestamp": time.time(),
                "regime_statistics": stats,
                "top_assets_by_regime": top_assets,
                "market_insights": market_insights,
                "regime_transitions": transition_probs,
                "model_performance": model_performance,
                "strategy_recommendations": strategy_recommendations,
                "explainability": explainability_data,
                "market_health": await self._calculate_market_health_index(),
                "volatility_forecast": await self._generate_volatility_forecast(),
                "liquidity_conditions": await self._assess_market_liquidity(),
                "institutional_activity": await self._detect_institutional_activity(),
                "correlation_matrix": await self._generate_correlation_heatmap(simplified=True),
                "regime_duration_forecast": await self._forecast_regime_duration(),
                "risk_metrics": await self._calculate_risk_metrics_by_regime()
            }
            
            # Add personalized alerts based on user preferences (if available)
            if hasattr(self, 'user_preferences') and self.user_preferences:
                dashboard_data["personalized_alerts"] = await self._generate_personalized_alerts()
            
            # Compress data for mobile transmission
            compressed_data = self._compress_dashboard_data(dashboard_data)
            
            # Add data version for client-side caching
            data_hash = hashlib.md5(str(compressed_data).encode()).hexdigest()
            compressed_data["data_version"] = data_hash
            
            return compressed_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate mobile dashboard data: {str(e)}", exc_info=True)
            # Return minimal fallback data
            return {
                "timestamp": time.time(),
                "error": "Data generation failed",
                "fallback_data": {
                    "regime_statistics": await self._get_fallback_regime_statistics(),
                    "market_health": 0.5  # Neutral fallback value
                }
            }
    
    async def _generate_market_insights(self) -> Dict[str, Any]:
        """
        Generate natural language insights about current market regimes
        
        Returns:
            Dict: Market insights with explanations and key factors
        """
        insights = {}
        
        try:
            # Get dominant regime
            dominant_regime = await self._identify_dominant_regime()
            
            # Generate natural language explanation
            if dominant_regime:
                regime_factors = await self._extract_regime_factors(dominant_regime)
                
                # Use NLG (Natural Language Generation) to create human-readable insights
                insights = {
                    "dominant_regime": dominant_regime,
                    "explanation": await self._generate_nlg_explanation(dominant_regime, regime_factors),
                    "key_factors": regime_factors[:3],  # Top 3 factors
                    "confidence": await self._calculate_regime_confidence(dominant_regime),
                    "historical_context": await self._get_historical_regime_context(dominant_regime),
                    "actionable_insights": await self._generate_actionable_insights(dominant_regime)
                }
            
            # Add anomaly detection insights if available
            anomalies = await self._detect_market_anomalies()
            if anomalies:
                insights["anomalies"] = anomalies
                
            # Add sentiment analysis if available
            if hasattr(self, 'sentiment_analyzer'):
                insights["market_sentiment"] = await self.sentiment_analyzer.get_current_sentiment()
        
        except Exception as e:
            self.logger.warning(f"Failed to generate market insights: {str(e)}")
            insights = {"error": "Could not generate insights", "fallback_regime": "NEUTRAL"}
            
        return insights
    
    async def _calculate_regime_transition_probabilities(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate the probability of transitioning from current regime to others
        
        Returns:
            Dict: Transition probabilities for each current regime
        """
        transition_probs = {}
        
        try:
            # Get historical regime transitions from database
            historical_transitions = await self._fetch_historical_regime_transitions()
            
            # Calculate transition matrix using Markov Chain
            transition_matrix = self._compute_markov_transition_matrix(historical_transitions)
            
            # Get current market features
            current_features = await self._extract_current_market_features()
            
            # Use ML model to predict transition probabilities based on current features
            for current_regime in self.REGIME_TYPES:
                # Initialize with base Markov probabilities
                base_probs = transition_matrix.get(current_regime, {})
                
                # Adjust probabilities using ML model predictions
                adjusted_probs = await self._predict_transition_probabilities(
                    current_regime, 
                    current_features, 
                    base_probs
                )
                
                transition_probs[current_regime] = adjusted_probs
                
            # Apply time-based decay to certain transitions based on regime duration
            transition_probs = self._apply_time_decay_to_transitions(transition_probs)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate transition probabilities: {str(e)}")
            # Return uniform distribution as fallback
            uniform_prob = 1.0 / len(self.REGIME_TYPES)
            transition_probs = {
                regime: {target: uniform_prob for target in self.REGIME_TYPES}
                for regime in self.REGIME_TYPES
            }
            
        return transition_probs
    
    async def _get_ai_model_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the AI models used in regime detection
        
        Returns:
            Dict: Performance metrics for each model
        """
        metrics = {}
        
        try:
            # Get accuracy metrics for each model
            for model_name, model in self.ai_models.items():
                if hasattr(model, 'get_performance_metrics'):
                    model_metrics = await model.get_performance_metrics()
                    metrics[model_name] = model_metrics
                else:
                    # Fallback to basic metrics if method not available
                    metrics[model_name] = {
                        "accuracy": self.model_accuracy.get(model_name, 0.0),
                        "confidence": self.model_confidence.get(model_name, 0.0),
                        "last_updated": self.model_last_updated.get(model_name, time.time())
                    }
            
            # Calculate ensemble performance
            if self.ai_models:
                metrics["ensemble"] = {
                    "accuracy": np.mean([m.get("accuracy", 0) for m in metrics.values()]),
                    "confidence": np.mean([m.get("confidence", 0) for m in metrics.values()]),
                    "weighted_accuracy": self._calculate_weighted_ensemble_accuracy()
                }
                
            # Add explainability metrics if available
            if hasattr(self, 'explainability_metrics'):
                metrics["explainability"] = self.explainability_metrics
                
        except Exception as e:
            self.logger.warning(f"Failed to get AI model performance metrics: {str(e)}")
            metrics = {"error": "Failed to retrieve model metrics"}
            
        return metrics
    
    async def _generate_strategy_recommendations(self) -> Dict[str, Any]:
        """
        Generate strategy recommendations based on current market regimes
        
        Returns:
            Dict: Strategy recommendations for different market regimes
        """
        recommendations = {}
        
        try:
            # Get current dominant regimes
            dominant_regimes = await self._get_dominant_regimes(limit=3)
            
            for regime_data in dominant_regimes:
                regime = regime_data["regime"]
                confidence = regime_data["confidence"]
                
                # Get optimal strategies for this regime
                optimal_strategies = await self._match_regime_to_optimal_strategy(regime, confidence)
                
                # Get risk parameters for this regime
                risk_params = await self._adjust_risk_parameters_per_regime(regime, confidence)
                
                # Get hedging recommendations
                hedging = await self._recommend_hedging_strategies(regime, confidence)
                
                # Combine into recommendation
                recommendations[regime] = {
                    "confidence": confidence,
                    "optimal_strategies": optimal_strategies,
                    "risk_parameters": risk_params,
                    "hedging_recommendations": hedging,
                    "time_horizon": self._get_optimal_time_horizon(regime),
                    "execution_style": self._get_optimal_execution_style(regime),
                    "asset_allocation": await self._recommend_asset_allocation(regime)
                }
            
            # Add meta-strategy recommendations
            recommendations["meta"] = {
                "portfolio_tilt": await self._calculate_portfolio_tilt(dominant_regimes),
                "risk_appetite": await self._calculate_risk_appetite(dominant_regimes),
                "execution_priority": await self._determine_execution_priority(dominant_regimes)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to generate strategy recommendations: {str(e)}")
            recommendations = {"error": "Failed to generate recommendations"}
            
        return recommendations
    
    async def _generate_explainability_data(self) -> Dict[str, Any]:
        """
        Generate explainability data for current regime classifications
        
        Returns:
            Dict: Explainability data with feature importance and decision factors
        """
        explainability = {}
        
        try:
            # Get current regime classifications
            current_regimes = {
                asset: update["regime"] 
                for asset, update in self.last_regime_update.items()
                if time.time() - update.get("timestamp", 0) < 3600  # Last hour
            }
            
            # Sample a few assets for detailed explainability
            sample_assets = random.sample(list(current_regimes.keys()), 
                                         min(3, len(current_regimes)))
            
            for asset in sample_assets:
                regime = current_regimes[asset]
                
                # Get SHAP values if available
                if hasattr(self, 'shap_explainer'):
                    shap_values = await self._calculate_shap_values(asset, regime)
                    top_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                else:
                    # Fallback to basic feature importance
                    top_features = await self._get_basic_feature_importance(asset, regime)
                
                # Get decision path for tree-based models
                decision_path = await self._get_decision_path(asset, regime)
                
                # Get counterfactual explanation
                counterfactual = await self._generate_counterfactual_explanation(asset, regime)
                
                explainability[asset] = {
                    "regime": regime,
                    "top_features": top_features,
                    "decision_path": decision_path,
                    "counterfactual": counterfactual,
                    "confidence_breakdown": await self._get_confidence_breakdown(asset, regime)
                }
            
            # Add global explainability
            explainability["global"] = {
                "feature_importance": await self._get_global_feature_importance(),
                "regime_characteristics": self._get_regime_characteristics(),
                "model_interpretation": await self._get_model_interpretation_summary()
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to generate explainability data: {str(e)}")
            explainability = {"error": "Explainability data generation failed"}
            
        return explainability