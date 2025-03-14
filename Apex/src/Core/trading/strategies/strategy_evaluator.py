# src/Core/trading/strategies/strategy_evaluator.py

import numpy as np
import asyncio
import hashlib
import torch
import time
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Event
import logging
from dataclasses import dataclass

# Core System Imports
from src.Core.trading.strategies.strategy_orchestrator import StrategyOrchestrator
from src.Core.trading.risk.risk_management import AdaptiveRiskEngine
from src.Core.data.realtime.market_data import UnifiedDataFeed
from src.ai.ensembles.meta_trader import StrategyOptimizer
from src.ai.reinforcement.maddpg_model import StrategyAdaptationLearner
from utils.analytics.monte_carlo_simulator import AdversarialScenarioTester
from utils.logging.structured_logger import QuantumLogger
from utils.helpers.error_handler import CriticalSystemGuard
from utils.helpers.stealth_api import EvaluationObfuscator
from metrics.performance_metrics import StrategyMetricsCalculator
from Core.trading.security.blacklist import StrategyBlacklist
from src.Core.trading.logging.decision_logger import StrategyAuditLogger
from src.Core.data.realtime.websocket_manager import WebSocketManager
from src.Core.trading.execution.market_impact import MarketImpactAnalyzer
from src.Core.data.order_book_analyzer import OrderBookAnalyzer
from src.Core.trading.hft.liquidity_manager import LiquidityManager

@dataclass
class EvaluationResult:
    """Immutable data structure for strategy evaluation results"""
    asset: str
    timestamp: datetime
    score: float
    risk_score: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    signature: str
    parameters: Dict[str, Any]
    liquidity_score: float
    market_impact_estimate: float
    model_confidence: float

class QuantumStrategyEvaluator:
    """Institutional-Grade Strategy Evaluation Engine with AI-Driven Optimization"""
    
    # Configurable parameters
    EVALUATION_FREQUENCY_HZ = 20  # 20Hz update frequency (50ms)
    HIGH_PRIORITY_FREQUENCY_HZ = 100  # 100Hz for high-priority assets (10ms)
    MIN_ACCEPTABLE_SCORE = 0.8
    MAX_ACCEPTABLE_RISK = 0.3
    CACHE_TIMEOUT_SECONDS = 30
    MODEL_RETRAINING_INTERVAL_MINUTES = 60
    HEALTH_CHECK_INTERVAL_SECONDS = 10
    
    def __init__(self, asset_universe: list, config: Dict[str, Any] = None):
        """
        Initialize the strategy evaluator with configuration options
        
        Args:
            asset_universe: List of assets to evaluate
            config: Optional configuration parameters to override defaults
        """
        self.asset_universe = asset_universe
        self.config = config or {}
        self.last_evaluation: Dict[str, EvaluationResult] = {}
        self.evaluation_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self.high_priority_assets: Set[str] = set()
        self._init_components()
        self._init_state()
        self._init_locks_and_events()
        
        # Initialize health monitoring
        self._last_health_check = time.time()
        self._health_status = {"status": "healthy", "last_check": self._last_health_check}
        self._register_system_hooks()
        
        # Load hardware-optimized models
        self._load_optimized_models()
        
        # Log initialization
        self.logger.info(f"QuantumStrategyEvaluator initialized with {len(asset_universe)} assets")

    def _init_components(self):
        """Initialize integrated system components with proper dependency injection"""
        # Core Modules
        self.data_feed = UnifiedDataFeed()
        self.strategy_orchestrator = StrategyOrchestrator()
        self.risk_engine = AdaptiveRiskEngine()
        self.metrics_calculator = StrategyMetricsCalculator()
        
        # Market Structure Components
        self.liquidity_manager = LiquidityManager()
        self.order_book_analyzer = OrderBookAnalyzer()
        self.market_impact_analyzer = MarketImpactAnalyzer()
        
        # AI/ML Components
        self.strategy_optimizer = StrategyOptimizer()
        self.adaptation_learner = StrategyAdaptationLearner()
        self.scenario_tester = AdversarialScenarioTester()
        
        # Communication Components
        self.websocket_manager = WebSocketManager()
        
        # Utilities
        self.logger = QuantumLogger("quantum_evaluator")
        self.system_guard = CriticalSystemGuard()
        self.obfuscator = EvaluationObfuscator()
        self.blacklist = StrategyBlacklist()
        self.audit_logger = StrategyAuditLogger()
        
        # Thread pool with adaptive sizing
        cpu_count = os.cpu_count() or 8
        self._executor = ThreadPoolExecutor(max_workers=max(8, cpu_count * 2))
        
        # GPU Acceleration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Memory-mapped arrays for ultra-fast evaluation results
        self._init_memory_mapped_storage()

    def _init_memory_mapped_storage(self):
        """Initialize memory-mapped arrays for high-performance data sharing"""
        try:
            # Create memory-mapped arrays for sharing evaluation results across processes
            # without serialization overhead
            assets_count = len(self.asset_universe)
            self.shared_scores = np.memmap('strategy_scores.dat', dtype=np.float32, 
                                          mode='w+', shape=(assets_count, 5))
            self.shared_parameters = {}  # Will store parameter hashes for quick lookups
            
            # Zero out initial values
            self.shared_scores.fill(0)
            self.shared_scores.flush()
            
        except Exception as e:
            self.logger.warning(f"Memory mapping failed, falling back to standard memory: {str(e)}")
            # Fallback to standard numpy arrays if memory mapping fails
            assets_count = len(self.asset_universe)
            self.shared_scores = np.zeros((assets_count, 5), dtype=np.float32)

    def _init_locks_and_events(self):
        """Initialize synchronization primitives for thread safety"""
        self._evaluation_lock = Lock()
        self._cache_lock = Lock()
        self._model_lock = Lock()
        self._emergency_shutdown_event = Event()
        self._retraining_event = Event()

    def _init_state(self):
        """Initialize evaluator state variables"""
        self.last_update_time = time.time()
        self.last_retraining_time = time.time()
        self.evaluation_stats = {
            "total_evaluations": 0,
            "approved_strategies": 0,
            "rejected_strategies": 0,
            "average_latency_ms": 0,
            "peak_latency_ms": 0
        }
        
        # Strategy version control
        self.strategy_versions = {}
        self.strategy_history = {}

    def _register_system_hooks(self):
        """Register system-wide hooks for events"""
        # Register for market regime change notifications
        self.data_feed.register_regime_change_callback(self._handle_regime_change)
        
        # Register for risk threshold changes
        self.risk_engine.register_thresholds_callback(self._handle_risk_change)
        
        # Register for emergency notifications
        self.system_guard.register_emergency_callback(self.emergency_shutdown)

    def _load_optimized_models(self):
        """Load hardware-optimized models based on available accelerators"""
        try:
            if self.device.type == "cuda":
                # Use TorchScript models optimized for GPU
                self.logger.info(f"Loading GPU-optimized models on {torch.cuda.get_device_name(0)}")
                self.strategy_scoring_model = torch.jit.load('models/strategy_scoring.pt').to(self.device)
                self.risk_prediction_model = torch.jit.load('models/risk_prediction.pt').to(self.device)
                
                # Enable CUDA graphs for repeated inference patterns
                self._setup_cuda_graphs()
            else:
                # Use CPU-optimized models
                self.logger.info("Loading CPU-optimized models")
                self.strategy_scoring_model = torch.jit.load('models/strategy_scoring_cpu.pt')
                self.risk_prediction_model = torch.jit.load('models/risk_prediction_cpu.pt')
                
            # Load fallback models for emergencies
            self._load_fallback_models()
            
        except Exception as e:
            self.logger.error(f"Failed to load optimized models: {str(e)}")
            self.system_guard.handle_operational_error(e)
            raise RuntimeError(f"Critical initialization failure: {str(e)}")

    def _setup_cuda_graphs(self):
        """Set up CUDA graphs for faster repeated inference if GPU is available"""
        if self.device.type != "cuda":
            return
            
        try:
            # Create static input shapes for CUDA graphs
            self.static_input = torch.zeros(1, 5, device=self.device)
            
            # Capture CUDA graph for strategy scoring model
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g):
                self.static_output = self.strategy_scoring_model(self.static_input)
                
            self.logger.info("CUDA graphs initialized for faster inference")
        except Exception as e:
            self.logger.warning(f"CUDA graph setup failed, falling back to standard inference: {str(e)}")

    def _load_fallback_models(self):
        """Load lightweight fallback models for emergency situations"""
        # These models have fewer parameters and are designed for reliability over accuracy
        self.fallback_scoring_model = torch.jit.load('models/fallback_scoring.pt')
        self.fallback_risk_model = torch.jit.load('models/fallback_risk.pt')

    async def evaluate_strategies(self) -> Dict[str, Dict]:
        """
        Perform full strategy evaluation cycle with AI optimization
        
        Returns:
            Dict[str, Dict]: Dictionary of evaluated strategies with their metrics
        """
        start_time = time.time()
        evaluation_results = {}
        
        try:
            # Check if retraining is needed before proceeding
            await self._check_retraining_needed()
            
            # Get market data with validation
            market_data = await self._get_validated_data()
            if not market_data:
                self.logger.warning("No valid market data received, skipping evaluation cycle")
                return {}
                
            # Calculate performance metrics for all strategies
            metrics_future = asyncio.create_task(
                self._calculate_metrics(market_data)
            )
            
            # Get current market liquidity conditions in parallel
            liquidity_future = asyncio.create_task(
                self.liquidity_manager.get_current_conditions(list(market_data.keys()))
            )
            
            # Await and combine results
            strategy_performance = await metrics_future
            liquidity_conditions = await liquidity_future
            
            # Add liquidity data to performance metrics
            for asset, data in strategy_performance.items():
                if asset in liquidity_conditions:
                    data['liquidity_score'] = liquidity_conditions[asset]['score']
                    data['market_depth'] = liquidity_conditions[asset]['depth']
                else:
                    data['liquidity_score'] = 0.5  # Default value
                    data['market_depth'] = 0.0
            
            # Run AI validation and adversarial testing
            validated_strategies = await self._run_ai_validation(strategy_performance)
            
            # Run post-evaluation processes only if we have valid strategies
            if validated_strategies:
                # Process order book implications
                order_book_impact = await self._analyze_order_book_impact(validated_strategies)
                
                # Merge order book impact data
                for asset, impact in order_book_impact.items():
                    if asset in validated_strategies:
                        validated_strategies[asset]['order_book_impact'] = impact
                
                # Update system components with new evaluations
                await self._update_system_components(validated_strategies)
                
                # Cache results for future reference
                self._cache_evaluation_results(validated_strategies)
                
                # Update evaluation statistics
                self._update_evaluation_stats(start_time, validated_strategies)
                
                evaluation_results = validated_strategies
            
            # Check system health periodically
            await self._periodic_health_check()
            
            return evaluation_results
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Strategy evaluation failed in {elapsed_ms:.2f}ms: {str(e)}")
            self.system_guard.handle_operational_error(e)
            
            # Attempt recovery with cached results if available
            return await self._attempt_recovery_from_cache()

    async def _attempt_recovery_from_cache(self) -> Dict[str, Dict]:
        """Attempt to recover from cache if evaluation fails"""
        self.logger.info("Attempting recovery from evaluation cache")
        
        current_time = time.time()
        valid_cached_results = {}
        
        with self._cache_lock:
            for asset, timestamp in self.cache_timestamps.items():
                # Use cached results that are less than CACHE_TIMEOUT_SECONDS old
                if current_time - timestamp < self.CACHE_TIMEOUT_SECONDS and asset in self.evaluation_cache:
                    valid_cached_results[asset] = self.evaluation_cache[asset]
        
        if valid_cached_results:
            self.logger.info(f"Recovered {len(valid_cached_results)} strategies from cache")
            return valid_cached_results
        else:
            self.logger.warning("No valid cached results available")
            return {}

    async def _get_validated_data(self) -> Dict:
        """
        Fetch and validate input data from multiple sources with cryptographic validation
        
        Returns:
            Dict: Validated market data keyed by asset
        """
        # Get raw data with timeout to prevent blocking
        try:
            raw_data = await asyncio.wait_for(
                self.data_feed.get_strategy_evaluation_data(),
                timeout=0.5  # 500ms timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning("Data feed request timed out, using cached data if available")
            # Try to use cached data from previous evaluations
            return self._get_cached_market_data()
        
        # Validate each data point cryptographically and structurally
        validated_data = {}
        validation_tasks = []
        
        for asset, data in raw_data.items():
            validation_task = asyncio.create_task(self._validate_asset_data(asset, data))
            validation_tasks.append((asset, validation_task))
        
        # Process all validations concurrently
        for asset, task in validation_tasks:
            try:
                is_valid = await task
                if is_valid:
                    validated_data[asset] = raw_data[asset]
                else:
                    self.logger.warning(f"Data validation failed for {asset}, skipping")
            except Exception as e:
                self.logger.error(f"Validation error for {asset}: {str(e)}")
        
        if not validated_data:
            self.logger.warning("No valid data points after validation")
        
        return validated_data

    def _get_cached_market_data(self) -> Dict:
        """Get cached market data when real-time data is unavailable"""
        cached_data = {}
        with self._cache_lock:
            current_time = time.time()
            for asset in self.asset_universe:
                if asset in self.evaluation_cache and current_time - self.cache_timestamps.get(asset, 0) < self.CACHE_TIMEOUT_SECONDS:
                    # Extract just the market data portion from cached evaluation
                    if 'market_data' in self.evaluation_cache[asset]:
                        cached_data[asset] = self.evaluation_cache[asset]['market_data']
        
        return cached_data

    async def _validate_asset_data(self, asset: str, data: Dict) -> bool:
        """
        Validate a single asset's data with cryptographic and structural checks
        
        Args:
            asset: Asset symbol
            data: Market data for the asset
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        # Structure validation
        required_fields = ['ohlcv', 'volume_profile', 'trade_history', 'signature']
        if not all(field in data for field in required_fields):
            self.logger.warning(f"Missing required fields for {asset}")
            return False
            
        # Size validation to prevent DOS attacks
        if len(str(data)) > 1_000_000:  # 1MB limit
            self.logger.warning(f"Data size exceeded limit for {asset}")
            return False
        
        # Cryptographic validation
        try:
            # Remove signature before hashing
            data_copy = {k: v for k, v in data.items() if k != 'signature'}
            data_hash = hashlib.sha256(str(data_copy).encode()).hexdigest()
            
            # Validate with data feed's public key
            return await self.data_feed.validate_data_signature(asset, data_hash, data['signature'])
        except Exception as e:
            self.logger.error(f"Cryptographic validation failed for {asset}: {str(e)}")
            return False

    async def _calculate_metrics(self, market_data: Dict) -> Dict:
        """
        Calculate performance metrics using dedicated module
        
        Args:
            market_data: Validated market data
            
        Returns:
            Dict: Strategy performance metrics
        """
        # Get current risk parameters
        risk_parameters = self.risk_engine.current_risk_parameters()
        
        # Get current market regime
        market_regime = await self.data_feed.get_current_market_regime()
        
        # Calculate metrics with risk adjustment and market regime awareness
        metrics = await self.metrics_calculator.compute_strategy_metrics(
            market_data,
            risk_parameters,
            market_regime=market_regime
        )
        
        # Add signature to each strategy's metrics
        for asset, data in metrics.items():
            # Store original market data for caching
            data['market_data'] = market_data[asset]
            
            # Add cryptographic signature to prevent tampering
            strategy_hash = hashlib.sha256(str(data['parameters']).encode()).hexdigest()
            data['signature'] = strategy_hash
            
            # Add version tracking
            data['version'] = self.strategy_versions.get(asset, 1)
            
            # Add timestamp
            data['timestamp'] = datetime.utcnow()
        
        return metrics

    async def _run_ai_validation(self, strategy_data: Dict) -> Dict:
        """
        Execute AI-powered strategy validation pipeline with adversarial testing
        
        Args:
            strategy_data: Strategy performance metrics
            
        Returns:
            Dict: Validated strategies
        """
        # Organize assets by priority
        high_priority_assets = [asset for asset in strategy_data.keys() if asset in self.high_priority_assets]
        normal_priority_assets = [asset for asset in strategy_data.keys() if asset not in self.high_priority_assets]
        
        # Process high priority assets first
        validated_strategies = {}
        
        # Process high priority assets
        high_priority_tasks = [
            self._validate_single_strategy(asset, strategy_data[asset], priority=True)
            for asset in high_priority_assets
        ]
        high_priority_results = await asyncio.gather(*high_priority_tasks)
        
        for asset, result in zip(high_priority_assets, high_priority_results):
            if result:
                validated_strategies[asset] = strategy_data[asset]
        
        # Process normal priority assets
        normal_priority_tasks = [
            self._validate_single_strategy(asset, strategy_data[asset])
            for asset in normal_priority_assets
        ]
        normal_priority_results = await asyncio.gather(*normal_priority_tasks)
        
        for asset, result in zip(normal_priority_assets, normal_priority_results):
            if result:
                validated_strategies[asset] = strategy_data[asset]
        
        return validated_strategies

    async def _validate_single_strategy(self, asset: str, data: Dict, priority: bool = False) -> bool:
        """
        Validate strategy through multiple AI-driven checks
        
        Args:
            asset: Asset symbol
            data: Strategy performance data
            priority: Whether this is a high-priority asset
            
        Returns:
            bool: True if strategy is valid, False otherwise
        """
        try:
            # Start with quick blacklist check
            if await self.blacklist.is_blacklisted(asset, data):
                self.logger.info(f"Strategy for {asset} is blacklisted, skipping validation")
                return False
                
            # Cryptographic strategy validation
            if not self._validate_strategy_signature(data):
                self.logger.warning(f"Strategy signature validation failed for {asset}")
                return False
                
            # Adversarial testing - simulate extreme market conditions
            if not await self.scenario_tester.test_strategy(data):
                self.logger.info(f"Strategy for {asset} failed adversarial testing")
                return False
                
            # Check liquidity conditions
            if 'liquidity_score' in data and data['liquidity_score'] < 0.3:
                self.logger.info(f"Strategy for {asset} has insufficient liquidity ({data['liquidity_score']})")
                return False
                
            # Risk validation
            risk_assessment = await self.risk_engine.assess_strategy(data)
            if risk_assessment['score'] < 0.7:
                self.logger.info(f"Strategy for {asset} failed risk assessment ({risk_assessment['score']})")
                return False
                
            # AI model validation
            if not await self._neural_validation(data):
                self.logger.info(f"Strategy for {asset} failed neural validation")
                return False
                
            # All checks passed
            data['validation_timestamp'] = datetime.utcnow().isoformat()
            data['priority'] = priority
            return True
            
        except Exception as e:
            self.logger.error(f"Validation failed for {asset}: {str(e)}")
            return False

    def _validate_strategy_signature(self, data: Dict) -> bool:
        """
        Validate strategy cryptographic signature
        
        Args:
            data: Strategy data
            
        Returns:
            bool: True if signature is valid, False otherwise
        """
        try:
            # Extract parameters for hashing
            parameters = data.get('parameters', {})
            provided_signature = data.get('signature', '')
            
            # Compute hash of parameters
            computed_hash = hashlib.sha256(str(parameters).encode()).hexdigest()
            
            # Compare computed hash with provided signature
            return computed_hash == provided_signature
        except Exception as e:
            self.logger.error(f"Strategy signature validation error: {str(e)}")
            return False

    async def _neural_validation(self, data: Dict) -> bool:
        """
        GPU-accelerated neural validation of strategies
        
        Args:
            data: Strategy data
            
        Returns:
            bool: True if strategy passes neural validation, False otherwise
        """
        # Check for emergency shutdown
        if self._emergency_shutdown_event.is_set():
            return False
            
        # Extract relevant features for model input
        try:
            # Prepare input features
            inputs = self._prepare_tensor_inputs(data)
            
            # Execute neural model validation
            with torch.no_grad():
                if self.device.type == "cuda" and hasattr(self, 'g'):
                    # Use CUDA graph for faster inference
                    self.static_input.copy_(inputs)
                    self.g.replay()
                    score = self.static_output.item()
                    
                    # Run second model for risk prediction
                    risk_pred = self.risk_prediction_model(inputs).item()
                else:
                    # Standard inference
                    score = self.strategy_scoring_model(inputs).item()
                    risk_pred = self.risk_prediction_model(inputs).item()
                
            # Add scores to data for reference
            data['score'] = score
            data['risk'] = risk_pred
            
            # Strategy passes if score is high enough and risk is low enough
            return score > self.MIN_ACCEPTABLE_SCORE and risk_pred < self.MAX_ACCEPTABLE_RISK
            
        except Exception as e:
            self.logger.error(f"Neural validation error: {str(e)}")
            # Attempt fallback model if main model fails
            return await self._fallback_neural_validation(data)

    async def _fallback_neural_validation(self, data: Dict) -> bool:
        """
        Fallback neural validation using simpler models
        
        Args:
            data: Strategy data
            
        Returns:
            bool: True if strategy passes fallback validation, False otherwise
        """
        try:
            # Use CPU for fallback regardless of device
            inputs = self._prepare_tensor_inputs(data, use_cpu=True)
            
            # Run fallback models
            with torch.no_grad():
                score = self.fallback_scoring_model(inputs).item()
                risk_pred = self.fallback_risk_model(inputs).item()
            
            # Add scores to data
            data['score'] = score
            data['risk'] = risk_pred
            data['used_fallback'] = True
            
            # Log fallback usage
            self.logger.warning(f"Used fallback neural validation with score={score:.2f}, risk={risk_pred:.2f}")
            
            # Use slightly more lenient thresholds for fallback
            return score > 0.75 and risk_pred < 0.35
            
        except Exception as e:
            self.logger.error(f"Fallback neural validation error: {str(e)}")
            return False

    def _prepare_tensor_inputs(self, data: Dict, use_cpu: bool = False) -> torch.Tensor:
        """
        Convert strategy data to optimized tensor format
        
        Args:
            data: Strategy data
            use_cpu: Whether to force CPU usage
            
        Returns:
            torch.Tensor: Tensor representation of strategy features
        """
        # Extract relevant features
        features = np.array([
            data.get('sharpe_ratio', 0),
            data.get('max_drawdown', 0),
            data.get('profit_factor', 0),
            data.get('win_rate', 0),
            data.get('risk_score', 0)
        ], dtype=np.float32)
        
        # Convert to tensor and move to appropriate device
        tensor = torch.from_numpy(features).unsqueeze(0)
        if not use_cpu and self.device.type == "cuda":
            return tensor.to(self.device)
        return tensor

    async def _analyze_order_book_impact(self, strategies: Dict) -> Dict:
        """
        Analyze order book impact of strategies
        
        Args:
            strategies: Validated strategies
            
        Returns:
            Dict: Order book impact metrics by asset
        """
        # Get current order book snapshots
        assets = list(strategies.keys())
        order_book_snapshots = await self.order_book_analyzer.get_current_snapshots(assets)
        
        # Calculate impact for each strategy
        impact_results = {}
        
        for asset, data in strategies.items():
            # Skip if no order book data
            if asset not in order_book_snapshots:
                continue
                
            # Get position size from strategy
            position_size = data.get('parameters', {}).get('position_size', 0)
            
            # Calculate market impact
            impact = await self.market_impact_analyzer.calculate_impact(
                asset, 
                position_size, 
                order_book_snapshots[asset]
            )
            
            impact_results[asset] = {
                'slippage_estimate': impact['slippage'],
                'price_impact': impact['price_impact'],
                'liquidity_score': impact['liquidity_score']
            }
        
        return impact_results

    async def _update_system_components(self, strategies: Dict):
        """
        Update all dependent system components with new evaluations
        
        Args:
            strategies: Validated strategies
        """
        # Convert strategies to required format
        strategy_weights = {asset: data['score'] for asset, data in strategies.items()}
        
        # Update strategy version tracking
        for asset in strategies:
            self.strategy_versions[asset] = self.strategy_versions.get(asset, 0) + 1
        
        # Execute updates concurrently
        update_tasks = [
            # Update strategy orchestrator
            self.strategy_orchestrator.update_strategy_weights(strategy_weights),
            
            # Update risk engine
            self.risk_engine.adjust_parameters_based_on_performance(strategies),
            
            # Update reinforcement learner
            self.adaptation_learner.incorporate_evaluations(strategies),
            
            # Log evaluations
            self._log_evaluation_results(strategies)
        ]
        
        # Wait for all updates to complete
        await asyncio.gather(*update_tasks)
        
        # Publish updates to websockets for real-time monitoring
        await self._publish_evaluation_updates(strategies)

    async def _publish_evaluation_updates(self, strategies: Dict):
        """
        Publish evaluation updates to websocket clients
        
        Args:
            strategies: Validated strategies
        """
        # Prepare update message
        update_message = {
            'type': 'strategy_evaluation',
            'timestamp': datetime.utcnow().isoformat(),
            'strategies': {}
        }
        
        # Add obfuscated strategy data
        for asset, data in strategies.items():
            update_message['strategies'][asset] = {
                'score': data['score'],
                'risk': data['risk'],
                'version': data['version'],
                'timestamp': data['validation_timestamp']
            }
        
        # Publish update
        await self.websocket_manager.broadcast_message('strategy_updates', update_message)

    async def _log_evaluation_results(self, strategies: Dict):
        """
        Log evaluation results to multiple systems
        
        Args:
            strategies: Validated strategies
        """
        log_tasks = []
        
        for asset, data in strategies.items():
            # Prepare log data
            log_data = {
                'asset': asset,
                'score': data['score'],
                'risk': data['risk'],
                'parameters': self.obfuscator.obfuscate_parameters(data['parameters']),
                'timestamp': datetime.utcnow().isoformat(),
                'version': data['version']
            }
            
            # Structured logging
            self.logger.log_strategy_evaluation(**log_data)
            
            # Audit logging
            audit_task = self.audit_logger.log_strategy_evaluation(
                asset, 
                data['score'], 
                data['risk'],
                data['version'],
                self.obfuscator.generate_hash(data)
            )
            log_tasks.append(audit_task)
        
        # Wait for all audit logs to complete
        await asyncio.gather(*log_tasks)
        
        # Update metrics database asynchronously
        asyncio.create_task(self._update_metrics_database(strategies))
    
    async def _update_metrics_database(self, strategies: Dict):
        """Update metrics database with evaluation results for historical tracking"""
        try:
            # Prepare batch data for efficient database insertion
            timestamp = datetime.utcnow()
            batch_data = []
            
            for asset, data in strategies.items():
                entry = {
                    'asset': asset,
                    'timestamp': timestamp,
                    'score': data['score'],
                    'risk': data['risk'],
                    'version': data['version'],
                    'sharpe_ratio': data.get('sharpe_ratio', 0),
                    'max_drawdown': data.get('max_drawdown', 0),
                    'win_rate': data.get('win_rate', 0),
                    'profit_factor': data.get('profit_factor', 0),
                    'liquidity_score': data.get('liquidity_score', 0)
                }
                batch_data.append(entry)
            
            # Store data in time-series database (implementation handled elsewhere)
            await self.strategy_orchestrator.store_evaluation_metrics(batch_data)
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics database: {str(e)}")
    
    def _cache_evaluation_results(self, strategies: Dict):
        """
        Cache evaluation results for future reference
        
        Args:
            strategies: Validated strategies
        """
        with self._cache_lock:
            current_time = time.time()
            for asset, data in strategies.items():
                # Store in memory cache with timestamp
                self.evaluation_cache[asset] = data
                self.cache_timestamps[asset] = current_time
                
                # Store last evaluation result
                self.last_evaluation[asset] = EvaluationResult(
                    asset=asset,
                    timestamp=datetime.utcnow(),
                    score=data['score'],
                    risk_score=data['risk'],
                    sharpe_ratio=data.get('sharpe_ratio', 0),
                    max_drawdown=data.get('max_drawdown', 0),
                    win_rate=data.get('win_rate', 0),
                    profit_factor=data.get('profit_factor', 0),
                    signature=data.get('signature', ''),
                    parameters=data.get('parameters', {}),
                    liquidity_score=data.get('liquidity_score', 0),
                    market_impact_estimate=data.get('order_book_impact', {}).get('price_impact', 0),
                    model_confidence=data.get('model_confidence', 0)
                )
    
    def _update_evaluation_stats(self, start_time: float, strategies: Dict):
        """
        Update evaluation statistics for monitoring
        
        Args:
            start_time: Start time of evaluation in seconds
            strategies: Validated strategies
        """
        # Calculate latency
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Update evaluation statistics atomically
        with self._evaluation_lock:
            self.evaluation_stats['total_evaluations'] += len(strategies)
            self.evaluation_stats['approved_strategies'] += len(strategies)
            
            # Update average latency with exponential moving average
            if self.evaluation_stats['average_latency_ms'] == 0:
                self.evaluation_stats['average_latency_ms'] = elapsed_ms
            else:
                self.evaluation_stats['average_latency_ms'] = (
                    0.9 * self.evaluation_stats['average_latency_ms'] + 0.1 * elapsed_ms
                )
            
            # Update peak latency if current latency is higher
            if elapsed_ms > self.evaluation_stats['peak_latency_ms']:
                self.evaluation_stats['peak_latency_ms'] = elapsed_ms
        
        # Log performance metrics if latency is high
        if elapsed_ms > 50:  # 50ms threshold
            self.logger.warning(f"High evaluation latency: {elapsed_ms:.2f}ms")
    
    async def _check_retraining_needed(self):
        """Check if model retraining is needed and initiate if required"""
        current_time = time.time()
        minutes_since_retraining = (current_time - self.last_retraining_time) / 60
        
        # Check if retraining interval has elapsed
        if minutes_since_retraining >= self.MODEL_RETRAINING_INTERVAL_MINUTES:
            # Set retraining event
            self._retraining_event.set()
            
            # Start retraining in background without waiting
            asyncio.create_task(self._retrain_models())
    
    async def _retrain_models(self):
        """Retrain AI models with latest data"""
        try:
            self.logger.info("Starting model retraining")
            
            # Acquire model lock to prevent concurrent retraining
            if not self._model_lock.acquire(blocking=False):
                self.logger.info("Model retraining already in progress, skipping")
                return
                
            try:
                # Get training data
                training_data = await self._gather_training_data()
                
                # Retrain models
                await self.strategy_optimizer.retrain_models(training_data)
                
                # Update model timestamp
                self.last_retraining_time = time.time()
                
                # Reload optimized models
                self._load_optimized_models()
                
                self.logger.info("Model retraining completed successfully")
            finally:
                # Release model lock
                self._model_lock.release()
                # Clear retraining event
                self._retraining_event.clear()
                
        except Exception as e:
            self.logger.error(f"Model retraining failed: {str(e)}")
            self._retraining_event.clear()
    
    async def _gather_training_data(self) -> Dict:
        """Gather data for model retraining"""
        # Get historical strategy performance
        strategy_history = await self.strategy_orchestrator.get_strategy_history(
            lookback_days=30  # Last 30 days of data
        )
        
        # Get market regime data
        market_regimes = await self.data_feed.get_market_regime_history(
            lookback_days=30
        )
        
        # Get risk profile history
        risk_profiles = await self.risk_engine.get_risk_profile_history(
            lookback_days=30
        )
        
        # Combine all data
        return {
            'strategy_history': strategy_history,
            'market_regimes': market_regimes,
            'risk_profiles': risk_profiles
        }
    
    async def _periodic_health_check(self):
        """Perform periodic health check"""
        current_time = time.time()
        
        # Run health check at specified interval
        if current_time - self._last_health_check >= self.HEALTH_CHECK_INTERVAL_SECONDS:
            await self._run_health_check()
            self._last_health_check = current_time
    
    async def _run_health_check(self):
        """Run comprehensive health check on evaluator"""
        try:
            # Check component health
            component_status = {
                'data_feed': await self.data_feed.check_health(),
                'risk_engine': await self.risk_engine.check_health(),
                'strategy_optimizer': await self.strategy_optimizer.check_health(),
                'websocket_manager': await self.websocket_manager.check_health()
            }
            
            # Check memory usage
            memory_usage = self._check_memory_usage()
            
            # Check thread pool health
            thread_pool_status = self._check_thread_pool()
            
            # Check model health
            model_health = await self._check_model_health()
            
            # Update health status
            self._health_status = {
                'status': all(s.get('healthy', False) for s in component_status.values()) and 
                          model_health['healthy'] and 
                          thread_pool_status['healthy'],
                'last_check': time.time(),
                'components': component_status,
                'memory_usage': memory_usage,
                'thread_pool': thread_pool_status,
                'models': model_health
            }
            
            # Log health status
            if not self._health_status['status']:
                self.logger.warning(f"Health check failed: {self._health_status}")
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            self._health_status = {
                'status': False,
                'last_check': time.time(),
                'error': str(e)
            }
    
    def _check_memory_usage(self) -> Dict:
        """Check memory usage of evaluator"""
        try:
            # Get process memory info
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                'rss': memory_info.rss / (1024 * 1024),  # RSS in MB
                'vms': memory_info.vms / (1024 * 1024),  # VMS in MB
                'percent': process.memory_percent(),
                'healthy': process.memory_percent() < 80  # Less than 80% is healthy
            }
        except ImportError:
            return {
                'error': 'psutil not available',
                'healthy': True  # Assume healthy if can't check
            }
        except Exception as e:
            return {
                'error': str(e),
                'healthy': False
            }
    
    def _check_thread_pool(self) -> Dict:
        """Check thread pool health"""
        try:
            # Check thread pool stats
            return {
                'workers': self._executor._max_workers,
                'tasks': len(self._executor._pending_work_items) if hasattr(self._executor, '_pending_work_items') else 0,
                'healthy': not self._executor._shutdown
            }
        except Exception as e:
            return {
                'error': str(e),
                'healthy': False
            }
    
    async def _check_model_health(self) -> Dict:
        """Check AI model health"""
        try:
            # Run simple inference test
            test_input = torch.zeros(1, 5, device=self.device)
            
            with torch.no_grad():
                start_time = time.time()
                _ = self.strategy_scoring_model(test_input)
                inference_time = (time.time() - start_time) * 1000  # ms
            
            return {
                'inference_time_ms': inference_time,
                'device': str(self.device),
                'healthy': inference_time < 10  # Less than 10ms is healthy
            }
        except Exception as e:
            return {
                'error': str(e),
                'healthy': False
            }
    
    async def evaluate_strategy(self, asset: str, parameters: Dict = None) -> Optional[EvaluationResult]:
        """
        Evaluate a single strategy with optional parameter override
        
        Args:
            asset: Asset symbol
            parameters: Optional parameter override
            
        Returns:
            Optional[EvaluationResult]: Evaluation result or None if not available
        """
        # Check cache first
        with self._cache_lock:
            if asset in self.last_evaluation:
                result = self.last_evaluation[asset]
                # If parameters match or no parameters provided, return cached result
                if parameters is None or parameters == result.parameters:
                    return result
        
        # Get existing parameters if not provided
        if parameters is None:
            current_strategy = await self.strategy_orchestrator.get_strategy(asset)
            if current_strategy:
                parameters = current_strategy.get('parameters', {})
            else:
                self.logger.warning(f"No existing strategy found for {asset}")
                return None
        
        # Force strategy evaluation
        try:
            # Get market data
            market_data = await self.data_feed.get_asset_data(asset)
            if not market_data:
                self.logger.warning(f"No market data available for {asset}")
                return None
                
            # Calculate metrics
            metrics = await self.metrics_calculator.compute_single_strategy_metrics(
                asset, market_data, parameters
            )
            
            # Run validation
            is_valid = await self._validate_single_strategy(asset, metrics)
            
            if is_valid:
                # Create evaluation result
                result = EvaluationResult(
                    asset=asset,
                    timestamp=datetime.utcnow(),
                    score=metrics['score'],
                    risk_score=metrics['risk'],
                    sharpe_ratio=metrics.get('sharpe_ratio', 0),
                    max_drawdown=metrics.get('max_drawdown', 0),
                    win_rate=metrics.get('win_rate', 0),
                    profit_factor=metrics.get('profit_factor', 0),
                    signature=metrics.get('signature', ''),
                    parameters=parameters,
                    liquidity_score=metrics.get('liquidity_score', 0),
                    market_impact_estimate=0,  # Will be calculated later
                    model_confidence=metrics.get('model_confidence', 0)
                )
                
                # Cache result
                with self._cache_lock:
                    self.last_evaluation[asset] = result
                
                return result
            else:
                self.logger.info(f"Strategy validation failed for {asset}")
                return None
                
        except Exception as e:
            self.logger.error(f"Strategy evaluation failed for {asset}: {str(e)}")
            return None
    
    async def set_high_priority_assets(self, assets: List[str]):
        """
        Set high priority assets for faster evaluation frequency
        
        Args:
            assets: List of asset symbols to prioritize
        """
        # Update high priority assets atomically
        self.high_priority_assets = set(assets)
        self.logger.info(f"Updated high priority assets: {assets}")
    
    async def get_evaluation_stats(self) -> Dict:
        """
        Get evaluation statistics
        
        Returns:
            Dict: Current evaluation statistics
        """
        with self._evaluation_lock:
            # Create a copy to avoid race conditions
            stats = dict(self.evaluation_stats)
            
        # Add health status
        stats['health'] = self._health_status['status']
        stats['last_health_check'] = self._health_status['last_check']
        stats['active_assets'] = len(self.last_evaluation)
        stats['high_priority_assets'] = len(self.high_priority_assets)
        stats['cache_size'] = len(self.evaluation_cache)
        
        # Add performance metrics
        stats['model_device'] = str(self.device)
        stats['cuda_available'] = torch.cuda.is_available()
        
        return stats
    
    async def get_strategy_versions(self) -> Dict[str, int]:
        """
        Get current strategy versions
        
        Returns:
            Dict[str, int]: Dictionary of asset to strategy version
        """
        return dict(self.strategy_versions)
    
    async def get_evaluation_history(self, asset: str, lookback: int = 10) -> List[Dict]:
        """
        Get historical evaluations for an asset
        
        Args:
            asset: Asset symbol
            lookback: Number of historical entries to retrieve
            
        Returns:
            List[Dict]: List of historical evaluations
        """
        # This would typically query a time-series database
        # Here we'll simulate the response structure
        try:
            history = await self.audit_logger.get_evaluation_history(
                asset=asset,
                limit=lookback
            )
            
            # Format for API response
            formatted_history = []
            for entry in history:
                formatted_history.append({
                    'timestamp': entry['timestamp'],
                    'score': entry['score'],
                    'risk': entry['risk'],
                    'version': entry['version'],
                    'sharpe_ratio': entry.get('sharpe_ratio', 0),
                    'win_rate': entry.get('win_rate', 0)
                })
            
            return formatted_history
        except Exception as e:
            self.logger.error(f"Failed to get evaluation history for {asset}: {str(e)}")
            return []
    
    async def _handle_regime_change(self, new_regime: str):
        """
        Handle market regime changes
        
        Args:
            new_regime: New market regime identifier
        """
        self.logger.info(f"Market regime changed to {new_regime}, triggering re-evaluation")
        
        # Update all strategies with new regime
        await self.strategy_optimizer.update_market_regime(new_regime)
        
        # Trigger immediate retraining
        self._retraining_event.set()
        
        # Mark all cached evaluations as stale
        with self._cache_lock:
            self.cache_timestamps = {}  # Clear all timestamps to force re-evaluation
    
    async def _handle_risk_change(self, risk_thresholds: Dict):
        """
        Handle risk threshold changes
        
        Args:
            risk_thresholds: New risk thresholds
        """
        self.logger.info(f"Risk thresholds updated: {risk_thresholds}")
        
        # Update internal risk thresholds
        self.MAX_ACCEPTABLE_RISK = risk_thresholds.get('max_risk', self.MAX_ACCEPTABLE_RISK)
        self.MIN_ACCEPTABLE_SCORE = risk_thresholds.get('min_score', self.MIN_ACCEPTABLE_SCORE)
        
        # Update strategy optimizer
        await self.strategy_optimizer.update_risk_thresholds(risk_thresholds)
    
    def emergency_shutdown(self, reason: str = "Unknown"):
        """
        Emergency shutdown procedure
        
        Args:
            reason: Reason for emergency shutdown
        """
        self.logger.critical(f"EMERGENCY SHUTDOWN INITIATED: {reason}")
        
        # Set shutdown event
        self._emergency_shutdown_event.set()
        
        # Switch to fallback models
        self._use_fallback_models()
        
        # Notify other components
        self.system_guard.notify_emergency(
            component="strategy_evaluator",
            reason=reason,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Log to audit system
        asyncio.create_task(
            self.audit_logger.log_emergency_event(
                component="strategy_evaluator",
                reason=reason
            )
        )
    
    def _use_fallback_models(self):
        """Switch to fallback models during emergency"""
        self.logger.warning("Switching to fallback models")
        
        try:
            # Copy fallback models to main models
            self.strategy_scoring_model = self.fallback_scoring_model
            self.risk_prediction_model = self.fallback_risk_model
            
            # Move to CPU if needed
            if self.device.type == "cuda":
                self.device = torch.device("cpu")
                self.strategy_scoring_model = self.strategy_scoring_model.to("cpu")
                self.risk_prediction_model = self.risk_prediction_model.to("cpu")
                
        except Exception as e:
            self.logger.error(f"Failed to switch to fallback models: {str(e)}")
    
    async def reset_emergency_state(self):
        """Reset emergency state after resolution"""
        self.logger.info("Resetting emergency state")
        
        # Clear emergency event
        self._emergency_shutdown_event.clear()
        
        # Reload optimized models
        self._load_optimized_models()
        
        # Log recovery
        await self.audit_logger.log_emergency_resolution(
            component="strategy_evaluator",
            timestamp=datetime.utcnow().isoformat()
        )
    
    async def run_mobile_evaluation(self, asset: str, parameters: Dict) -> Dict:
        """
        Run lightweight evaluation for mobile clients
        
        Args:
            asset: Asset symbol
            parameters: Strategy parameters
            
        Returns:
            Dict: Evaluation results optimized for mobile
        """
        # Get evaluation with potential parameter override
        result = await self.evaluate_strategy(asset, parameters)
        
        if not result:
            return {
                'status': 'error',
                'message': f'Strategy evaluation failed for {asset}'
            }
        
        # Format for mobile client (reduced payload size)
        mobile_result = {
            'asset': asset,
            'score': round(result.score, 3),
            'risk': round(result.risk_score, 3),
            'sharpe': round(result.sharpe_ratio, 2),
            'drawdown': round(result.max_drawdown, 2),
            'winRate': round(result.win_rate, 2),
            'timestamp': result.timestamp.isoformat(),
            'version': self.strategy_versions.get(asset, 1)
        }
        
        return {
            'status': 'success',
            'data': mobile_result
        }
    
    async def get_dashboard_metrics(self) -> Dict:
        """
        Get metrics formatted for the web dashboard
        
        Returns:
            Dict: Dashboard metrics
        """
        # Get evaluation stats
        stats = await self.get_evaluation_stats()
        
        # Get top performing strategies
        top_strategies = []
        for asset, result in self.last_evaluation.items():
            if result:
                top_strategies.append({
                    'asset': asset,
                    'score': result.score,
                    'risk': result.risk_score,
                    'sharpe': result.sharpe_ratio,
                    'timestamp': result.timestamp.isoformat()
                })
        
        # Sort by score, descending
        top_strategies.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top 5
        top_strategies = top_strategies[:5]
        
        # Format for dashboard
        dashboard_metrics = {
            'strategies': {
                'total': stats['total_evaluations'],
                'approved': stats['approved_strategies'],
                'rejected': stats['rejected_strategies'],
                'topPerformers': top_strategies
            },
            'performance': {
                'avgLatency': stats['average_latency_ms'],
                'peakLatency': stats['peak_latency_ms']
            },
            'system': {
                'health': stats['health'],
                'lastCheck': stats['last_health_check'],
                'device': stats['model_device']
            }
        }
        asha
        return dashboard_metrics
    
    # Integration with institutional-grade risk frameworks
    async def export_strategy_data_for_compliance(self, start_date: str, end_date: str) -> Dict:
        """
        Export strategy evaluation data for compliance reporting
        
        Args:
            start_date: Start date in ISO format
            end_date: End date in ISO format
            
        Returns:
            Dict: Compliance report data
        """
        try:
            # Convert dates to datetime objects
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            
            # Get evaluation history from audit log
            evaluation_history = await self.audit_logger.get_evaluation_history_between_dates(
                start_date=start_dt,
                end_date=end_dt
            )
            
            # Calculate risk metrics
            risk_metrics = await self.risk_engine.calculate_aggregate_risk_metrics(
                start_date=start_dt,
                end_date=end_dt
            )
            
            # Generate compliance report
            compliance_report = {
                'period': {
                    'start': start_date,
                    'end': end_date
                },
                'evaluations': {
                    'total': len(evaluation_history),
                    'byAsset': {}
                },
                'risk': risk_metrics,
                'signatures': {}
            }
            
            # Organize by asset
            for eval_entry in evaluation_history:
                asset = eval_entry['asset']
                
                if asset not in compliance_report['evaluations']['byAsset']:
                    compliance_report['evaluations']['byAsset'][asset] = []
                
                compliance_report['evaluations']['byAsset'][asset].append({
                    'timestamp': eval_entry['timestamp'].isoformat(),
                    'score': eval_entry['score'],
                    'risk': eval_entry['risk'],
                    'version': eval_entry['version']
                })
            
            # Generate cryptographic signatures for report integrity
            report_hash = hashlib.sha256(str(compliance_report).encode()).hexdigest()
            compliance_report['signatures']['report_hash'] = report_hash
            
            return compliance_report
            
        except Exception as e:
            self.logger.error(f"Failed to generate compliance report: {str(e)}")
            return {
                'status': 'error',
                'message': f'Compliance report generation failed: {str(e)}'
            }
    
    async def start(self):
        """Start the evaluator service"""
        self.logger.info("Starting QuantumStrategyEvaluator")
        
        # Initialize components
        await self.data_feed.connect()
        await self.websocket_manager.connect()
        
        # Reset emergency state if set
        if self._emergency_shutdown_event.is_set():
            await self.reset_emergency_state()
        
        # Run initial health check
        await self._run_health_check()
        
        # Log startup
        self.logger.info("QuantumStrategyEvaluator started successfully")
    
    async def stop(self):
        """Stop the evaluator service"""
        self.logger.info("Stopping QuantumStrategyEvaluator")
        
        # Disconnect from external services
        await self.data_feed.disconnect()
        await self.websocket_manager.disconnect()
        
        # Shutdown thread pool
        self._executor.shutdown(wait=True)
        
        # Close memory-mapped files
        if hasattr(self, 'shared_scores') and isinstance(self.shared_scores, np.memmap):
            self.shared_scores._mmap.close()
        
        # Log shutdown
        self.logger.info("QuantumStrategyEvaluator stopped successfully")

    # Mobile app API integration methods
    async def get_mobile_portfolio_summary(self, user_id: str) -> Dict:
        """
        Get portfolio summary for mobile app
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict: Portfolio summary optimized for mobile
        """
        try:
            # Get user's portfolio
            portfolio = await self.strategy_orchestrator.get_user_portfolio(user_id)
            
            # Get evaluations for portfolio assets
            portfolio_evaluations = {}
            evaluation_tasks = []
            
            for asset in portfolio['assets']:
                task = asyncio.create_task(self.evaluate_strategy(asset['symbol']))
                evaluation_tasks.append((asset['symbol'], task))
            
            # Gather all evaluations
            for asset_symbol, task in evaluation_tasks:
                try:
                    result = await task
                    if result:
                        portfolio_evaluations[asset_symbol] = {
                            'score': round(result.score, 2),
                            'risk': round(result.risk_score, 2)
                        }
                except Exception:
                    # Skip failed evaluations
                    continue
            
            # Calculate portfolio metrics
            total_value = sum(asset['value'] for asset in portfolio['assets'])
            weighted_score = 0
            weighted_risk = 0
            
            for asset in portfolio['assets']:
                symbol = asset['symbol']
                weight = asset['value'] / total_value if total_value > 0 else 0
                
                if symbol in portfolio_evaluations:
                    weighted_score += portfolio_evaluations[symbol]['score'] * weight
                    weighted_risk += portfolio_evaluations[symbol]['risk'] * weight
            
            # Format for mobile app
            return {
                'portfolio': {
                    'totalValue': round(total_value, 2),
                    'score': round(weighted_score, 2),
                    'risk': round(weighted_risk, 2),
                    'assets': [
                        {
                            'symbol': asset['symbol'],
                            'value': round(asset['value'], 2),
                            'allocation': round(asset['value'] / total_value * 100 if total_value > 0 else 0, 1),
                            'evaluation': portfolio_evaluations.get(asset['symbol'], {'score': 0, 'risk': 0})
                        }
                        for asset in portfolio['assets']
                    ]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get mobile portfolio summary: {str(e)}")
            return {
                'status': 'error',
                'message': f'Portfolio summary generation failed: {str(e)}'
            }
    
    # Institutional API methods
    async def perform_institutional_batch_evaluation(self, strategies: List[Dict]) -> Dict:
        """
        Perform batch evaluation for institutional clients
        
        Args:
            strategies: List of strategies to evaluate
            
        Returns:
            Dict: Batch evaluation results
        """
        # Track performance for institutional clients
        start_time = time.time()
        results = {}
        failed = []
        
        # Process in parallel
        evaluation_tasks = []
        for strategy in strategies:
            if 'asset' not in strategy or 'parameters' not in strategy:
                failed.append({
                    'id': strategy.get('id', 'unknown'),
                    'reason': 'Missing required fields'
                })
                continue
                
            task = asyncio.create_task(
                self.evaluate_strategy(strategy['asset'], strategy['parameters'])
            )
            evaluation_tasks.append((strategy.get('id', strategy['asset']), task))
        
        # Process results in parallel
        results = {}
        failed = []
        
        # Process in parallel
        results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        
        # Handle results
        for strategy_id, result in zip(strategies, results):
            if isinstance(result, Exception):
                failed.append({
                    'id': strategy_id,
                    'reason': str(result)
                })
            else:
                results[strategy_id] = result
                
        # Return results
        return {
            'success': len(results),
            'failed': failed,
            'results': results
        }
        
