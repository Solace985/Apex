# apex/src/Core/trading/strategies/trend/momentum_breakout.py

import numpy as np
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import time
import functools
import uuid

# Core System Integration
from src.Core.data.realtime.market_data import UnifiedMarketFeed
from src.Core.data.processing.data_parser import validate_market_data, DataValidationError
from src.Core.trading.risk.risk_management import AdaptiveRiskController
from src.Core.trading.hft.liquidity_manager import LiquidityOptimizer
from src.Core.trading.strategies.strategy_orchestrator import StrategyOrchestrator
from src.ai.ensembles.meta_trader import MetaStrategyOptimizer
from src.ai.analysis.market_regime_classifier import MarketRegimeClassifier
from src.ai.forecasting.order_flow import OrderFlowAnalyzer
from src.Core.data.realtime.order_book_analyzer import OrderBookDepthAnalyzer
from src.utils.analytics.monte_carlo_simulator import MonteCarloVaR
from src.ai.reinforcement.q_learning.agent import QStrategyOptimizer
from src.ai.analysis.institutional_clusters import InstitutionalClusterAnalyzer
from src.Core.trading.execution.market_impact import MarketImpactAnalyzer
from src.Core.trading.execution.order_execution import OrderExecutionManager
from src.Core.trading.security.alert_system import SecurityMonitor
from src.Core.data.asset_validator import validate_asset_tradability

# Shared Utilities
from src.utils.logging.structured_logger import StructuredLogger
from src.utils.helpers.error_handler import ErrorHandler, StrategyError
from src.utils.helpers.stealth_api import StealthAPIManager
from src.Core.trading.logging.decision_logger import DecisionMetadata
from src.utils.analytics.insider_data_cache import InsiderDataCache

# Constants
STRATEGY_NAME = "momentum_breakout"
STRATEGY_VERSION = "2.0.0"
MAX_PARALLEL_ASSETS = 100
CACHE_EXPIRY_SECONDS = 30
HEARTBEAT_INTERVAL_SECONDS = 5
CRITICAL_LATENCY_THRESHOLD_MS = 50
EMERGENCY_TIMEOUT_SECONDS = 3
EXECUTION_PRIORITY = 1  # Higher priority (0-10 scale)
MAX_RETRY_ATTEMPTS = 3
SIGNAL_TTL_SECONDS = 5  # Time-to-live for signal caching
HEALTH_CHECK_INTERVAL_SECONDS = 60

# Initialize structured logger
logger = StructuredLogger(STRATEGY_NAME)

# Initialize StealthAPIManager as a singleton
stealth_manager = StealthAPIManager()

class MomentumBreakoutStrategy:
    """
    Quantum-Resistant Momentum Breakout System with Multi-Asset Adaptive Learning
    Fully integrated with Apex's AI ecosystem and risk framework
    """
    
    # Thread-safe shared state cache with TTL management
    _shared_state = {}
    _parameter_cache = {}
    _signal_cache = {}
    _model_cache = {}
    _regime_cache = {}
    _last_health_check = datetime.utcnow()
    _last_latency_report = datetime.utcnow()
    _execution_latencies = []
    
    def __init__(self, assets: List[str], config: Dict[str, Any]):
        """
        Initialize the momentum breakout strategy with proper integration to other Apex components.
        
        Args:
            assets: List of asset symbols to monitor for breakout opportunities
            config: Configuration parameters for the strategy
        """
        # Use shared state for resource optimization
        self.__dict__ = self._shared_state
        
        # Only initialize components if not already done
        if not hasattr(self, 'initialized'):
            self.assets = assets
            self.config = self._validate_config(config)
            self.adaptive_params = self._load_adaptive_parameters()
            
            # System startup timestamp
            self.startup_time = datetime.utcnow()
            self.last_model_update = datetime.utcnow()
            self.strategy_id = str(uuid.uuid4())
            
            # Initialize performance metrics
            self.performance_metrics = {
                'signals_generated': 0,
                'avg_execution_time_ms': 0,
                'last_execution_times': [],
                'false_positives': 0,
                'true_positives': 0,
                'signal_accuracy': 0.0,
                'mean_signal_latency_ms': 0.0
            }
            
            # Integrated System Components - Asynchronous initialization
            self.loop = asyncio.get_event_loop()
            self.loop.run_until_complete(self._initialize_components())
            
            # Thread pool for parallel asset processing with dynamic sizing
            self.executor = ThreadPoolExecutor(max_workers=min(len(assets), MAX_PARALLEL_ASSETS))
            
            # Signal memoization cache with TTL
            self.signal_cache_ttl = {}
            
            # Asset prioritization using dynamic scoring
            self.asset_scores = {asset: 1.0 for asset in assets}
            self.asset_volatility = {asset: 0.0 for asset in assets}
            self.asset_liquidity = {asset: 0.0 for asset in assets}
            self.asset_regime = {asset: "unknown" for asset in assets}
            
            # Breakout detection parameters (dynamic, asset-specific)
            self.breakout_thresholds = {asset: self._calculate_initial_thresholds(asset) for asset in assets}
            
            # Security and redundancy
            self.failsafe_triggered = False
            self.emergency_shutdown = False
            self.last_heartbeat = datetime.utcnow()
            self.component_health = {
                'market_feed': True,
                'risk_controller': True,
                'order_execution': True,
                'ai_components': True
            }
            
            # Start the heartbeat and monitoring tasks
            self.heartbeat_task = self.loop.create_task(self._system_heartbeat())
            self.health_check_task = self.loop.create_task(self._periodic_health_check())
            
            # State tracking for signals and parameters
            self.last_signal = {asset: "HOLD" for asset in assets}
            self.parameter_history = []
            self.signal_explanations = {}
            
            # Precomputed lookback windows for optimization
            self.lookback_windows = {}
            self._precompute_lookback_windows()
            
            # System is now initialized
            self.initialized = True
            
            # Log successful initialization
            logger.info(f"{STRATEGY_NAME} initialized with {len(assets)} assets", 
                        metadata={"version": STRATEGY_VERSION, "assets": len(assets), "strategy_id": self.strategy_id})
    
    def _calculate_initial_thresholds(self, asset: str) -> Dict[str, Any]:
        """Calculate initial breakout thresholds for an asset"""
        return {
            'atr_multiplier': self.adaptive_params['volatility_multiplier'],
            'volume_threshold': self.adaptive_params['volume_threshold'],
            'momentum_threshold': self.adaptive_params['momentum_threshold'],
            'lookback': self.adaptive_params['lookback'],
            'last_update': datetime.utcnow()
        }
    
    def _precompute_lookback_windows(self) -> None:
        """Precompute common lookback windows for optimization"""
        lookbacks = set()
        lookbacks.add(self.adaptive_params['lookback'])
        lookbacks.add(self.adaptive_params['dynamic_lookback'])
        lookbacks.add(self.adaptive_params['volume_lookback'])
        
        # Add standard technical analysis lookback periods
        for period in [5, 10, 20, 50, 100, 200]:
            lookbacks.add(period)
        
        self.lookback_windows = {period: np.arange(period) for period in lookbacks}
    
    async def _initialize_components(self) -> None:
        """Initialize all required components asynchronously"""
        try:
            init_tasks = [
                self._init_market_data(),
                self._init_risk_components(),
                self._init_ai_components(),
                self._init_execution_components(),
                self._init_security_components()
            ]
            results = await asyncio.gather(*init_tasks, return_exceptions=True)
            
            # Check for initialization errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    component_name = ['market_data', 'risk_components', 'ai_components', 
                                     'execution_components', 'security_components'][i]
                    logger.error(f"Failed to initialize {component_name}", 
                                metadata={"error": str(result), "strategy": STRATEGY_NAME})
                    
                    # Report critical initialization error
                    if hasattr(self, 'security_monitor'):
                        await self.security_monitor.report_incident(
                            'initialization_failure',
                            {"component": component_name, "strategy": STRATEGY_NAME, "error": str(result)}
                        )
        except Exception as e:
            logger.critical(f"Critical error during component initialization: {e}", 
                          metadata={"error": str(e), "strategy": STRATEGY_NAME})
            raise RuntimeError(f"Failed to initialize {STRATEGY_NAME} strategy: {e}")
    
    async def _init_market_data(self) -> None:
        """Initialize market data components"""
        self.market_feed = UnifiedMarketFeed()
        self.book_analyzer = OrderBookDepthAnalyzer()
        
        # Initialize components with timeout protection
        await asyncio.wait_for(self.market_feed.initialize(), timeout=EMERGENCY_TIMEOUT_SECONDS)
        await asyncio.wait_for(self.book_analyzer.initialize(), timeout=EMERGENCY_TIMEOUT_SECONDS)
        
        # Pre-subscribe to all asset data feeds to minimize latency
        for asset in self.assets:
            await self.market_feed.subscribe(asset)
    
    async def _init_risk_components(self) -> None:
        """Initialize risk management components"""
        self.risk_controller = AdaptiveRiskController()
        self.liquidity_optimizer = LiquidityOptimizer()
        
        # Pre-load risk parameters to avoid latency during trading
        await asyncio.wait_for(self.risk_controller.preload_parameters(), timeout=EMERGENCY_TIMEOUT_SECONDS)
        await asyncio.wait_for(self.liquidity_optimizer.initialize(), timeout=EMERGENCY_TIMEOUT_SECONDS)
        
        # Register strategy with risk controller for exposure tracking
        await self.risk_controller.register_strategy(STRATEGY_NAME, self.strategy_id)
    
    async def _init_ai_components(self) -> None:
        """Initialize AI and decision making components"""
        # Initialize components with error handling
        try:
            self.strategy_orchestrator = StrategyOrchestrator()
            self.meta_optimizer = MetaStrategyOptimizer()
            self.regime_classifier = MarketRegimeClassifier()
            self.orderflow_analyzer = OrderFlowAnalyzer()
            self.q_optimizer = QStrategyOptimizer()
            self.institutional_analyzer = InstitutionalClusterAnalyzer()
            
            # AI model pre-loading to minimize latency (with timeouts)
            await asyncio.wait_for(self.meta_optimizer.preload_models(), timeout=EMERGENCY_TIMEOUT_SECONDS * 2)
            await asyncio.wait_for(self.regime_classifier.preload_models(), timeout=EMERGENCY_TIMEOUT_SECONDS)
            await asyncio.wait_for(self.orderflow_analyzer.initialize(), timeout=EMERGENCY_TIMEOUT_SECONDS)
            await asyncio.wait_for(self.q_optimizer.initialize(), timeout=EMERGENCY_TIMEOUT_SECONDS)
            await asyncio.wait_for(self.institutional_analyzer.initialize(), timeout=EMERGENCY_TIMEOUT_SECONDS)
            
            # Register this strategy with the orchestrator
            await self.strategy_orchestrator.register_strategy(
                strategy_name=STRATEGY_NAME,
                priority=EXECUTION_PRIORITY,
                assets=self.assets,
                strategy_id=self.strategy_id
            )
        except Exception as e:
            logger.error(f"AI component initialization error: {e}", 
                        metadata={"error": str(e), "strategy": STRATEGY_NAME})
            raise
    
    async def _init_execution_components(self) -> None:
        """Initialize execution-related components"""
        self.market_impact_analyzer = MarketImpactAnalyzer()
        self.order_execution_manager = OrderExecutionManager()
        
        await asyncio.wait_for(self.market_impact_analyzer.initialize(), timeout=EMERGENCY_TIMEOUT_SECONDS)
        await asyncio.wait_for(self.order_execution_manager.initialize(), timeout=EMERGENCY_TIMEOUT_SECONDS)
    
    async def _init_security_components(self) -> None:
        """Initialize security and validation components"""
        self.security_monitor = SecurityMonitor()
        self.insider_cache = InsiderDataCache()
        
        await asyncio.wait_for(self.security_monitor.initialize(), timeout=EMERGENCY_TIMEOUT_SECONDS)
        await asyncio.wait_for(self.insider_cache.initialize(), timeout=EMERGENCY_TIMEOUT_SECONDS)
        
        # Register strategy with security monitoring
        await self.security_monitor.register_strategy(STRATEGY_NAME, self.strategy_id)
    
    async def _system_heartbeat(self) -> None:
        """Continuous system health monitoring"""
        while True:
            try:
                self.last_heartbeat = datetime.utcnow()
                
                # Check for component health (with timeout protection)
                health_checks = await asyncio.gather(
                    asyncio.wait_for(self.market_feed.check_health(), timeout=EMERGENCY_TIMEOUT_SECONDS),
                    asyncio.wait_for(self.risk_controller.check_health(), timeout=EMERGENCY_TIMEOUT_SECONDS),
                    asyncio.wait_for(self.strategy_orchestrator.check_health(), timeout=EMERGENCY_TIMEOUT_SECONDS),
                    asyncio.wait_for(self.security_monitor.check_health(), timeout=EMERGENCY_TIMEOUT_SECONDS),
                    return_exceptions=True
                )
                
                # Check for failures and update component health status
                components = ['market_feed', 'risk_controller', 'strategy_orchestrator', 'security_monitor']
                for i, check in enumerate(health_checks):
                    component = components[i]
                    is_healthy = not isinstance(check, Exception) and (isinstance(check, dict) and check.get('healthy', False))
                    self.component_health[component] = is_healthy
                    
                    if not is_healthy:
                        logger.error(f"Component failure detected: {component}", 
                                    metadata={"error": str(check) if isinstance(check, Exception) else check})
                        
                        # Notify security system
                        await self.security_monitor.report_incident(
                            'component_failure', 
                            {"component": component, "strategy": STRATEGY_NAME}
                        )
                
                # Emergency shutdown if critical components are unhealthy
                critical_failure = not (self.component_health['market_feed'] and self.component_health['risk_controller'])
                if critical_failure and not self.emergency_shutdown:
                    logger.critical("Critical component failure - triggering emergency shutdown", 
                                  metadata={"strategy": STRATEGY_NAME, "health_status": self.component_health})
                    self.emergency_shutdown = True
                    await self.security_monitor.report_incident(
                        'emergency_shutdown', 
                        {"strategy": STRATEGY_NAME, "reason": "critical_component_failure"}
                    )
                
                # Clean expired caches
                self._clean_expired_caches()
                
                # Wait for next heartbeat
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
                
            except Exception as e:
                logger.error(f"Heartbeat system error: {e}", 
                            metadata={"error": str(e)})
                await asyncio.sleep(1)  # Sleep shorter time on error
    
    async def _periodic_health_check(self) -> None:
        """Perform periodic comprehensive health checks"""
        while True:
            try:
                # Wait for the next health check interval
                await asyncio.sleep(HEALTH_CHECK_INTERVAL_SECONDS)
                
                # Check for high latency
                if self._execution_latencies:
                    avg_latency = sum(self._execution_latencies) / len(self._execution_latencies)
                    if avg_latency > CRITICAL_LATENCY_THRESHOLD_MS:
                        logger.warning(f"High execution latency detected: {avg_latency:.2f}ms", 
                                      metadata={"latency": avg_latency, "threshold": CRITICAL_LATENCY_THRESHOLD_MS})
                        
                        # Report high latency incident
                        await self.security_monitor.report_incident(
                            'high_latency', 
                            {"strategy": STRATEGY_NAME, "latency_ms": avg_latency}
                        )
                    
                    # Reset latency measurements
                    self._execution_latencies = []
                
                # Check if AI models need updating
                time_since_update = (datetime.utcnow() - self.last_model_update).total_seconds()
                if time_since_update > 3600:  # Update models hourly
                    await self._update_ai_models()
                
                # Update asset scoring based on performance
                await self._update_asset_prioritization()
                
                # Update the last health check timestamp
                self._last_health_check = datetime.utcnow()
                
            except Exception as e:
                logger.error(f"Health check error: {e}", metadata={"error": str(e)})
    
    async def _update_ai_models(self) -> None:
        """Update AI models with latest performance data"""
        try:
            # Get strategy performance data for model retraining
            performance_data = {
                'accuracy': self.performance_metrics['signal_accuracy'],
                'false_positives': self.performance_metrics['false_positives'],
                'true_positives': self.performance_metrics['true_positives'],
                'latency': self.performance_metrics['mean_signal_latency_ms']
            }
            
            # Update Q-learning model with recent performance
            if hasattr(self, 'q_optimizer'):
                await self.q_optimizer.update_model(STRATEGY_NAME, performance_data)
            
            # Update meta-strategy optimizer with performance data
            if hasattr(self, 'meta_optimizer'):
                await self.meta_optimizer.update_strategy_weights(STRATEGY_NAME, performance_data)
            
            # Update timestamp
            self.last_model_update = datetime.utcnow()
            logger.info("AI models updated successfully", 
                      metadata={"strategy": STRATEGY_NAME, "performance": performance_data})
            
        except Exception as e:
            logger.error(f"Failed to update AI models: {e}", 
                        metadata={"error": str(e)})
    
    async def _update_asset_prioritization(self) -> None:
        """Update asset scoring based on signal performance"""
        try:
            # Calculate dynamic asset scores based on performance, liquidity, and volatility
            for asset in self.assets:
                # Get recent performance for this asset
                asset_performance = await self.strategy_orchestrator.get_asset_performance(STRATEGY_NAME, asset)
                
                # Update asset volatility from market data
                self.asset_volatility[asset] = await self._get_asset_volatility(asset)
                
                # Update asset liquidity from liquidity optimizer
                self.asset_liquidity[asset] = await self.liquidity_optimizer.get_asset_liquidity(asset)
                
                # Update market regime classification
                self.asset_regime[asset] = await self._get_market_regime(asset)
                
                # Calculate composite score based on multiple factors
                volatility_score = min(1.0, self.asset_volatility[asset] / 0.02)  # Normalize volatility
                liquidity_score = min(1.0, self.asset_liquidity[asset])
                performance_score = min(1.0, asset_performance.get('win_rate', 0.5))
                
                # Weighted composite score
                self.asset_scores[asset] = (
                    0.4 * volatility_score + 
                    0.3 * liquidity_score + 
                    0.3 * performance_score
                )
            
            # Sort assets by score for prioritization
            self.prioritized_assets = sorted(
                self.assets, 
                key=lambda x: self.asset_scores.get(x, 0.0), 
                reverse=True
            )
            
            logger.info("Asset prioritization updated", 
                      metadata={"top_assets": self.prioritized_assets[:5]})
            
        except Exception as e:
            logger.error(f"Failed to update asset prioritization: {e}", 
                        metadata={"error": str(e)})
    
    def _clean_expired_caches(self) -> None:
        """Clean expired items from caches"""
        now = datetime.utcnow()
        
        # Clean signal cache
        expired = [k for k, v in self.signal_cache_ttl.items() if v < now]
        for key in expired:
            self.signal_cache_ttl.pop(key, None)
            if key in self._signal_cache:
                self._signal_cache.pop(key, None)
        
        # Clean regime cache
        expired = [k for k, v in self._regime_cache.items() 
                  if 'timestamp' in v and v['timestamp'] < now - timedelta(seconds=CACHE_EXPIRY_SECONDS)]
        for key in expired:
            self._regime_cache.pop(key, None)
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize configuration parameters"""
        required_params = [
            'version', 'var_confidence', 'var_threshold', 'max_exposure', 
            'max_slippage', 'institutional_threshold', 'allowed_regimes', 
            'allowed_parameters'
        ]
        
        # Ensure all required parameters exist
        for param in required_params:
            if param not in config:
                logger.error(f"Missing required configuration parameter: {param}")
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate version
        if config['version'] != STRATEGY_VERSION:
            logger.warning(f"Version mismatch in configuration. Expected {STRATEGY_VERSION}, got {config['version']}")
            config['version'] = STRATEGY_VERSION
        
        # Validate numerical parameters
        for param in ['var_confidence', 'var_threshold', 'max_exposure', 'max_slippage', 'institutional_threshold']:
            if not isinstance(config[param], (int, float)) or config[param] <= 0:
                logger.error(f"Invalid {param} value: {config[param]}")
                raise ValueError(f"Invalid {param} value: {config[param]}")
        
        # Validate allowed regimes
        if not isinstance(config['allowed_regimes'], dict) or not config['allowed_regimes']:
            logger.error("Invalid allowed_regimes configuration")
            raise ValueError("Invalid allowed_regimes configuration")
        
        # Validate allowed parameters
        if not isinstance(config['allowed_parameters'], dict) or not config['allowed_parameters']:
            logger.error("Invalid allowed_parameters configuration")
            raise ValueError("Invalid allowed_parameters configuration")
        
        # Ensure additional security parameters
        security_params = {
            'max_trade_size': 0.05,  # Default to 5% of portfolio
            'max_drawdown': 0.10,    # Default to 10% max drawdown
            'position_timeout': 1800  # Default to 30 minutes
        }
        
        # Add missing security parameters with defaults
        for param, default in security_params.items():
            if param not in config:
                config[param] = default
                logger.info(f"Added missing security parameter {param} with default value {default}")
        
        return config
    
    async def _load_adaptive_parameters(self) -> Dict[str, Any]:
        """Load optimized parameters for the strategy from AI models"""
        # First check if we have cached parameters
        cached_params = self._parameter_cache.get('adaptive_params')
        if cached_params:
            return cached_params
        
        # Default parameters
        default_params = {
            'lookback': 20,
            'dynamic_lookback': 100,
            'volume_lookback': 10,
            'volatility_multiplier': 0.5,
            'liquidity_requirement': 0.8,
            'momentum_threshold': 0.6,
            'volume_threshold': 2.0,
            'regime_weight': 0.7,
            'orderflow_weight': 0.6,
            'institutional_weight': 0.5,
            'max_position_size': 0.05,  # 5% of portfolio
            'use_order_book': True,
            'use_reinforcement': True,
            'use_regime_classifier': True,
            'breakout_confirmation_time': 2,  # Time in seconds to confirm breakout
            'fake_breakout_filter': True,
            'min_volume_spike': 1.5       # Minimum volume spike for breakout confirmation
        }
        
        # Try to load from Q-learning optimizer
        try:
            if hasattr(self, 'q_optimizer'):
                q_params = await self.q_optimizer.get_optimized_parameters(STRATEGY_NAME)
                if q_params and isinstance(q_params, dict):
                    # Merge with defaults, keeping constraints
                    for k, v in q_params.items():
                        if k in default_params:
                            # Apply constraints from config
                            if k in self.config['allowed_parameters']:
                                constraints = self.config['allowed_parameters'][k]
                                v = np.clip(v, constraints['min'], constraints['max'])
                            default_params[k] = v
            
            # Also check meta_optimizer for ensemble recommendations
            if hasattr(self, 'meta_optimizer'):
                meta_params = await self.meta_optimizer.get_strategy_parameters(STRATEGY_NAME)
                if meta_params and isinstance(meta_params, dict):
                    # Only use meta_optimizer params for specific adaptive values
                    for k in ['volatility_multiplier', 'momentum_threshold', 'volume_threshold']:
                        if k in meta_params:
                            # Apply constraints
                            if k in self.config['allowed_parameters']:
                                constraints = self.config['allowed_parameters'][k]
                                meta_params[k] = np.clip(meta_params[k], constraints['min'], constraints['max'])
                            default_params[k] = meta_params[k]
        
        except Exception as e:
            logger.warning(f"Failed to load AI parameters: {e}", 
                          metadata={"error": str(e)})
        
        # Cache the parameters
        self._parameter_cache['adaptive_params'] = default_params
        self._parameter_cache['last_update'] = datetime.utcnow()
        
        # Log parameter loading
        logger.info("Adaptive parameters loaded", 
                  metadata={"params": {k: v for k, v in default_params.items() if k in 
                                     ['lookback', 'volatility_multiplier', 'momentum_threshold']}})
        
        return default_params

    async def compute_signals(self, asset: str) -> Dict[str, Any]:
        """
        Quantum-inspired breakout detection with real-time adaptive learning
        Returns structured decision metadata for system integration
        
        Args:
            asset: The asset symbol to analyze
            
        Returns:
            Dict containing the signal and associated metadata
        """
        start_time = time.time()
        
        try:
            # Check if we have a cached signal that's still valid
            cache_key = f"{asset}_{int(start_time / SIGNAL_TTL_SECONDS)}"
            if cache_key in self._signal_cache:
                return self._signal_cache[cache_key]
            
            # Check if the system is in emergency shutdown
            if self.emergency_shutdown:
                return self._create_hold_signal(asset, "emergency_shutdown")
            
            # Check if any component is unhealthy
            if not all(self.component_health.values()):
                unhealthy = [k for k, v in self.component_health.items() if not v]
                return self._create_hold_signal(asset, f"unhealthy_components: {unhealthy}")
            
            # Pre-validation phase - Fast checks before proceeding with analysis
            # 1. Check if the strategy is permitted to execute by the orchestrator
            permitted = await self.strategy_orchestrator.check_strategy_allocation(STRATEGY_NAME, asset)
            if not permitted:
                return self._create_hold_signal(asset, "strategy_orchestrator_denied")
            
            # 2. Check asset tradability - fast regulatory & security checks
            tradable = await validate_asset_tradability(asset)
            if not tradable:
                return self._create_hold_signal(asset, "asset_not_tradable")
            
            # 3. Check for basic liquidity - fast check before detailed analysis
            basic_liquidity = await self._check_basic_liquidity(asset)
            if not basic_liquidity:
                return self._create_hold_signal(asset, "insufficient_basic_liquidity")
            
            # Get market regime to adapt breakout parameters
            market_regime = await self._get_market_regime(asset)
            
            # Skip if the current regime is not in allowed regimes
            if market_regime not in self.config['allowed_regimes']:
                return self._create_hold_signal(asset, f"disallowed_market_regime: {market_regime}")
            
            # Adapt strategy parameters based on market regime (optimized for latency)
            adjusted_params = await self._adapt_parameters_for_regime(asset, market_regime)
            
            # Parallel analysis tasks - execute in parallel to minimize latency
            analysis_tasks = [
                self._get_market_data(asset, adjusted_params['lookback']),
                self._analyze_order_book(asset),
                self._check_institutional_activity(asset),
                self._analyze_order_flow(asset)
            ]
            
            # Execute analysis tasks in parallel
            market_data, order_book_analysis, institutional_analysis, order_flow_analysis = await asyncio.gather(*analysis_tasks)
            
            # Calculate breakout signals using vectorized computation
            breakout_signal, signal_metadata = self._calculate_breakout_signal(
                asset, 
                market_data, 
                adjusted_params,
                order_book_analysis,
                institutional_analysis,
                order_flow_analysis,
                market_regime
            )
            
            # Risk management checks - must be sequential for dependency reasons
            risk_approval = await self._check_risk_constraints(asset, breakout_signal, signal_metadata)
            if not risk_approval['approved']:
                return self._create_hold_signal(asset, risk_approval['reason'])
            
            # Liquidity validation (detailed check after signal generation)
            liquidity_approval = await self._check_liquidity_for_execution(asset, breakout_signal, signal_metadata)
            if not liquidity_approval['approved']:
                return self._create_hold_signal(asset, liquidity_approval['reason'])
            
            # Market impact analysis
            impact_analysis = await self._analyze_market_impact(asset, breakout_signal, signal_metadata)
            if impact_analysis['excessive_impact']:
                return self._create_hold_signal(asset, "excessive_market_impact")
            
            # Execute through order_execution_manager
            execution_start = time.time()
            execution_data = {
                'asset': asset,
                'signal': breakout_signal,
                'position_size': signal_metadata['recommended_size'],
                'expected_impact': impact_analysis['expected_impact'],
                'strategy_id': self.strategy_id,
                'parameters': adjusted_params,
                'timestamp': datetime.utcnow().isoformat(),
                'market_regime': market_regime,
                'orderbook_state': order_book_analysis['summary'],
                'institutional_activity': institutional_analysis['activity_level']
            }
            
            # Create structured decision metadata for logging and analysis
            decision_metadata = DecisionMetadata(
                strategy=STRATEGY_NAME,
                asset=asset,
                signal=breakout_signal,
                confidence=signal_metadata['confidence'],
                factors=signal_metadata['decision_factors'],
                risk_metrics=risk_approval['metrics'],
                execution_params=execution_data,
                timestamp=datetime.utcnow()
            )
            
            # Set signal expiry (TTL)
            self.signal_cache_ttl[cache_key] = datetime.utcnow() + timedelta(seconds=SIGNAL_TTL_SECONDS)
            
            # Cache the signal
            signal_result = {
                'asset': asset,
                'signal': breakout_signal,
                'metadata': signal_metadata,
                'decision': decision_metadata,
                'timestamp': datetime.utcnow().isoformat()
            }
            self._signal_cache[cache_key] = signal_result
            
            # Track performance metrics
            execution_time_ms = (time.time() - start_time) * 1000
            self._execution_latencies.append(execution_time_ms)
            self.performance_metrics['last_execution_times'].append(execution_time_ms)
            self.performance_metrics['signals_generated'] += 1
            
            # Log decision for AI training and analysis
            logger.info(f"Generated {breakout_signal} signal for {asset}", 
                      metadata={"signal": breakout_signal, "asset": asset, 
                                "execution_time_ms": execution_time_ms})
            
            # Return the signal result
            return signal_result
            
        except Exception as e:
            error_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Error generating signal for {asset}: {str(e)}", 
                        metadata={"asset": asset, "error": str(e), "time_ms": error_time_ms})
            
            # Report error to security monitor
            if hasattr(self, 'security_monitor'):
                asyncio.create_task(self.security_monitor.report_incident(
                    'signal_generation_error',
                    {"asset": asset, "strategy": STRATEGY_NAME, "error": str(e)}
                ))
            
            # Return a safe HOLD signal on error
            return self._create_hold_signal(asset, f"error: {str(e)}")
    
    def _create_hold_signal(self, asset: str, reason: str) -> Dict[str, Any]:
        """Create a HOLD signal with reason metadata"""
        return {
            'asset': asset,
            'signal': "HOLD",
            'metadata': {
                'reason': reason,
                'confidence': 0.0,
                'timestamp': datetime.utcnow().isoformat()
            },
            'decision': None,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _get_market_data(self, asset: str, lookback: int) -> Dict[str, np.ndarray]:
        """
        Get market data for breakout analysis using vectorized operations
        Returns structured numpy arrays for fast computation
        """
        try:
            # Use cached lookback window for optimization
            lookback_indices = self.lookback_windows.get(lookback, np.arange(lookback))
            
            # Get market data with timeout protection
            market_data = await asyncio.wait_for(
                self.market_feed.get_market_data(asset, lookback), 
                timeout=EMERGENCY_TIMEOUT_SECONDS
            )
            
            # Validate market data integrity
            try:
                validate_market_data(market_data)
            except DataValidationError as e:
                logger.warning(f"Market data validation failed for {asset}: {e}")
                # Use fallback data retrieval method
                market_data = await asyncio.wait_for(
                    self.market_feed.get_market_data_fallback(asset, lookback),
                    timeout=EMERGENCY_TIMEOUT_SECONDS
                )
                validate_market_data(market_data)
            
            # Extract and convert to numpy arrays for vectorized computation
            close_prices = np.array(market_data['close'])
            high_prices = np.array(market_data['high'])
            low_prices = np.array(market_data['low'])
            volumes = np.array(market_data['volume'])
            timestamps = np.array(market_data['timestamp'])
            
            # Compute key metrics using vectorized operations
            # 1. Average True Range (ATR) for volatility assessment
            true_ranges = np.zeros(lookback)
            true_ranges[0] = high_prices[0] - low_prices[0]  # First value initialization
            
            # Vectorized computation of true ranges
            h_l = high_prices[1:] - low_prices[1:]
            h_pc = np.abs(high_prices[1:] - close_prices[:-1])
            l_pc = np.abs(low_prices[1:] - close_prices[:-1])
            
            # Element-wise maximum
            true_ranges[1:] = np.maximum(h_l, np.maximum(h_pc, l_pc))
            
            # Exponential weighted moving average for ATR
            atr = np.zeros(lookback)
            atr[0] = true_ranges[0]
            for i in range(1, lookback):
                atr[i] = (atr[i-1] * 13 + true_ranges[i] * 1) / 14  # 14-period EMA
            
            # 2. Price momentum using percentage change
            momentum = np.zeros(lookback)
            momentum[1:] = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]
            
            # 3. Volume relative to moving average
            volume_ma = np.zeros(lookback)
            for i in range(lookback):
                if i < 10:  # For first 10 periods
                    volume_ma[i] = np.mean(volumes[:i+1])
                else:
                    volume_ma[i] = np.mean(volumes[i-9:i+1])  # 10-period MA
            
            volume_ratio = np.zeros(lookback)
            volume_ratio[10:] = volumes[10:] / volume_ma[10:]
            
            # 4. Price levels for resistance detection
            sma20 = np.zeros(lookback)
            for i in range(lookback):
                if i < 20:
                    sma20[i] = np.mean(close_prices[:i+1])
                else:
                    sma20[i] = np.mean(close_prices[i-19:i+1])
            
            # 5. Resistance levels using rolling max
            resistance_levels = np.zeros(lookback)
            for i in range(lookback):
                if i < 20:
                    resistance_levels[i] = np.max(high_prices[:i+1])
                else:
                    resistance_levels[i] = np.max(high_prices[i-19:i+1])
            
            # Return all computed metrics for breakout analysis
            return {
                'close': close_prices,
                'high': high_prices,
                'low': low_prices,
                'volume': volumes,
                'timestamp': timestamps,
                'atr': atr,
                'momentum': momentum,
                'volume_ratio': volume_ratio,
                'sma20': sma20,
                'resistance_levels': resistance_levels
            }
            
        except Exception as e:
            logger.error(f"Error getting market data for {asset}: {e}", 
                        metadata={"error": str(e)})
            # Return empty data on error
            empty_array = np.array([])
            return {
                'close': empty_array,
                'high': empty_array,
                'low': empty_array,
                'volume': empty_array,
                'timestamp': empty_array,
                'atr': empty_array,
                'momentum': empty_array,
                'volume_ratio': empty_array,
                'sma20': empty_array,
                'resistance_levels': empty_array
            }
    
    async def _analyze_order_book(self, asset: str) -> Dict[str, Any]:
        """Analyze order book depth and structure for breakout confirmation"""
        try:
            # Get order book data with timeout protection
            order_book = await asyncio.wait_for(
                self.book_analyzer.analyze_order_book(asset), 
                timeout=EMERGENCY_TIMEOUT_SECONDS / 2  # Shorter timeout for critical path
            )
            
            # Extract key metrics for breakout analysis
            buy_pressure = order_book.get('buy_pressure', 0.0)
            sell_pressure = order_book.get('sell_pressure', 0.0)
            spread = order_book.get('spread', 0.0)
            depth_imbalance = order_book.get('depth_imbalance', 0.0)
            top_level_volume = order_book.get('top_level_volume', 0.0)
            
            # Calculate buy-sell pressure ratio (avoid division by zero)
            if sell_pressure > 0:
                pressure_ratio = buy_pressure / sell_pressure
            else:
                pressure_ratio = 1.0 if buy_pressure > 0 else 0.0
            
            # Create summary for decision making
            summary = {
                'pressure_ratio': pressure_ratio,
                'spread': spread,
                'depth_imbalance': depth_imbalance,
                'buy_pressure': buy_pressure,
                'sell_pressure': sell_pressure,
                'top_level_volume': top_level_volume
            }
            
            return {
                'raw_data': order_book,
                'summary': summary,
                'pressure_ratio': pressure_ratio,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Order book analysis failed for {asset}: {e}", 
                          metadata={"error": str(e)})
            
            # Return default neutral values on failure
            return {
                'raw_data': {},
                'summary': {
                    'pressure_ratio': 1.0,
                    'spread': 0.0,
                    'depth_imbalance': 0.0,
                    'buy_pressure': 0.0,
                    'sell_pressure': 0.0,
                    'top_level_volume': 0.0
                },
                'pressure_ratio': 1.0,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _check_institutional_activity(self, asset: str) -> Dict[str, Any]:
        """
        Check for institutional activity signals
        Returns activity level and classified patterns
        """
        try:
            # Check cache first for recently computed results
            cache_key = f"inst_{asset}_{int(time.time() / CACHE_EXPIRY_SECONDS)}"
            if cache_key in self._signal_cache:
                return self._signal_cache[cache_key]
            
            # Get institutional activity data with timeout protection
            institutional_data = await asyncio.wait_for(
                self.institutional_analyzer.analyze_asset(asset),
                timeout=EMERGENCY_TIMEOUT_SECONDS
            )
            
            # Check insider data cache for recent activity
            insider_data = await self.insider_cache.get_recent_activity(asset)
            
            # Combine institutional and insider data
            activity_level = institutional_data.get('activity_level', 0.0)
            if insider_data and 'activity_score' in insider_data:
                # Weight insider activity more heavily
                activity_level = 0.7 * activity_level + 0.3 * insider_data['activity_score']
            
            # Determine if activity is above threshold
            above_threshold = activity_level >= self.config['institutional_threshold']
            
            # Create result object
            result = {
                'activity_level': activity_level,
                'above_threshold': above_threshold,
                'patterns': institutional_data.get('patterns', []),
                'directional_bias': institutional_data.get('directional_bias', 'neutral'),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Cache the result
            self._signal_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.warning(f"Institutional activity check failed for {asset}: {e}", 
                          metadata={"error": str(e)})
            
            # Return default neutral values on failure
            return {
                'activity_level': 0.0,
                'above_threshold': False,
                'patterns': [],
                'directional_bias': 'neutral',
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _analyze_order_flow(self, asset: str) -> Dict[str, Any]:
        """
        Analyze order flow patterns to detect smart money movements
        Returns directional bias and confidence
        """
        try:
            # Get order flow data with timeout protection
            order_flow = await asyncio.wait_for(
                self.orderflow_analyzer.analyze_order_flow(asset),
                timeout=EMERGENCY_TIMEOUT_SECONDS
            )
            
            # Extract key metrics
            buying_pressure = order_flow.get('buying_pressure', 0.0)
            selling_pressure = order_flow.get('selling_pressure', 0.0)
            delta = buying_pressure - selling_pressure
            delta_normalized = np.tanh(delta)  # Normalize to [-1, 1] range
            
            # Determine directional bias
            if delta_normalized > 0.2:
                direction = "bullish"
                strength = min(1.0, delta_normalized)
            elif delta_normalized < -0.2:
                direction = "bearish"
                strength = min(1.0, abs(delta_normalized))
            else:
                direction = "neutral"
                strength = 0.0
            
            # Process aggressive orders
            aggressive_buys = order_flow.get('aggressive_buys', 0)
            aggressive_sells = order_flow.get('aggressive_sells', 0)
            total_aggressives = aggressive_buys + aggressive_sells
            
            if total_aggressives > 0:
                aggression_ratio = aggressive_buys / total_aggressives
            else:
                aggression_ratio = 0.5  # Neutral if no aggressive orders
            
            return {
                'direction': direction,
                'strength': strength,
                'delta': delta,
                'buying_pressure': buying_pressure,
                'selling_pressure': selling_pressure,
                'aggressive_buys': aggressive_buys,
                'aggressive_sells': aggressive_sells,
                'aggression_ratio': aggression_ratio,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Order flow analysis failed for {asset}: {e}", 
                          metadata={"error": str(e)})
            
            # Return default neutral values on failure
            return {
                'direction': 'neutral',
                'strength': 0.0,
                'delta': 0.0,
                'buying_pressure': 0.0,
                'selling_pressure': 0.0,
                'aggressive_buys': 0,
                'aggressive_sells': 0,
                'aggression_ratio': 0.5,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _get_market_regime(self, asset: str) -> str:
        """
        Get current market regime classification
        Uses cached results when available to reduce latency
        """
        try:
            # Check cache first
            cache_key = f"regime_{asset}_{int(time.time() / CACHE_EXPIRY_SECONDS)}"
            if cache_key in self._regime_cache:
                return self._regime_cache[cache_key]['regime']
            
            # If we have the regime classifier available
            if hasattr(self, 'regime_classifier'):
                # Get regime classification with timeout protection
                regime_data = await asyncio.wait_for(
                    self.regime_classifier.classify_regime(asset),
                    timeout=EMERGENCY_TIMEOUT_SECONDS
                )
                
                regime = regime_data.get('regime', 'unknown')
                confidence = regime_data.get('confidence', 0.0)
                
                # Cache the result
                self._regime_cache[cache_key] = {
                    'regime': regime,
                    'confidence': confidence,
                    'timestamp': datetime.utcnow()
                }
                
                return regime
            else:
                # Return default regime if classifier not available
                return "normal"
            
        except Exception as e:
            logger.warning(f"Market regime classification failed for {asset}: {e}", 
                          metadata={"error": str(e)})
            
            # Return default regime on failure
            return "normal"
    
    async def _adapt_parameters_for_regime(self, asset: str, regime: str) -> Dict[str, Any]:
        """
        Adapt strategy parameters based on market regime
        Returns optimized parameters for current conditions
        """
        # Start with base parameters
        params = self.adaptive_params.copy()
        
        # Get regime-specific parameter adjustments
        regime_adjustments = self.config['allowed_regimes'].get(regime, {})
        
        # Apply regime-specific adjustments
        for key, adjustment in regime_adjustments.items():
            if key in params:
                params[key] = adjustment
        
        # Get asset-specific volatility
        volatility = await self._get_asset_volatility(asset)
        
        # Adapt momentum threshold based on volatility
        if volatility > 0:
            # Higher volatility needs higher threshold to filter noise
            volatility_factor = min(2.0, max(0.5, volatility / 0.01))  # Normalize to reasonable range
            params['momentum_threshold'] *= volatility_factor
            
            # Adjust lookback for volatile markets
            if regime in ['volatile', 'crisis']:
                params['lookback'] = max(5, int(params['lookback'] / 2))
            
            # Adjust ATR multiplier for volatility
            params['volatility_multiplier'] *= 1.0 / volatility_factor
        
        # If using Q-learning optimization and asset-specific parameters are available
        if hasattr(self, 'q_optimizer'):
            try:
                # Get asset-specific optimized parameters with timeout
                asset_params = await asyncio.wait_for(
                    self.q_optimizer.get_asset_parameters(asset, STRATEGY_NAME),
                    timeout=EMERGENCY_TIMEOUT_SECONDS / 2
                )
                
                # Apply asset-specific parameters if available
                if asset_params and isinstance(asset_params, dict):
                    for k, v in asset_params.items():
                        if k in params:
                            # Apply constraints from config
                            if k in self.config['allowed_parameters']:
                                constraints = self.config['allowed_parameters'][k]
                                v = np.clip(v, constraints['min'], constraints['max'])
                            params[k] = v
            except Exception as e:
                logger.debug(f"Failed to get asset-specific parameters: {e}")
        
        return params
    
    def _calculate_breakout_signal(self, 
                                  asset: str, 
                                  market_data: Dict[str, np.ndarray],
                                  params: Dict[str, Any],
                                  order_book_analysis: Dict[str, Any],
                                  institutional_analysis: Dict[str, Any],
                                  order_flow_analysis: Dict[str, Any],
                                  market_regime: str) -> Tuple[str, Dict[str, Any]]:
        """
        Calculate momentum breakout signals using vectorized computation
        Returns signal direction and detailed metadata
        """
        # Avoid computation if market data is empty
        if len(market_data.get('close', [])) == 0:
            return "HOLD", {"reason": "insufficient_market_data", "confidence": 0.0}
        
        try:
            # Extract data arrays for vectorized calculation
            close = market_data['close']
            high = market_data['high']
            low = market_data['low']
            volume = market_data['volume']
            atr = market_data['atr']
            resistance_levels = market_data['resistance_levels']
            
            # Current values (most recent data point)
            current_close = close[-1]
            current_volume = volume[-1]
            current_atr = atr[-1]
            
            # Calculate key momentum indicators
            # 1. Price crossing above resistance level
            resistance_level = resistance_levels[-1]
            resistance_cross = current_close > resistance_level
            
            # 2. Volume confirmation (current volume vs recent average)
            if len(volume) >= 10:
                avg_volume = np.mean(volume[-10:-1])  # Average of last 10 periods excluding current
                volume_surge = current_volume > (params['volume_threshold'] * avg_volume)
            else:
                # If not enough volume history
                volume_surge = current_volume > (params['volume_threshold'] * np.mean(volume))
            
            # 3. Momentum confirmation
            if len(close) >= 3:
                recent_momentum = (close[-1] - close[-3]) / close[-3]
                momentum_strong = recent_momentum > params['momentum_threshold']
            else:
                momentum_strong = False
            
            # 4. ATR-based breakout threshold
            breakout_threshold = current_atr * params['volatility_multiplier']
            price_movement = current_close - np.mean(close[-3:-1])  # Current vs 2-period average
            strong_movement = abs(price_movement) > breakout_threshold
            
            # 5. Order book confirmation
            buy_pressure_strong = order_book_analysis['summary']['pressure_ratio'] > 1.2
            
            # 6. Order flow confirmation
            order_flow_bullish = order_flow_analysis['direction'] == 'bullish' and order_flow_analysis['strength'] > 0.5
            order_flow_bearish = order_flow_analysis['direction'] == 'bearish' and order_flow_analysis['strength'] > 0.5
            
            # 7. Institutional activity confirmation
            institutional_active = institutional_analysis['above_threshold']
            institutional_direction = institutional_analysis['directional_bias']
            
            # Calculate signal strength using decision factors
            decision_factors = {
                'resistance_cross': float(resistance_cross),
                'volume_surge': float(volume_surge),
                'momentum_strong': float(momentum_strong),
                'strong_movement': float(strong_movement),
                'buy_pressure_strong': float(buy_pressure_strong),
                'order_flow_bullish': float(order_flow_bullish),
                'institutional_active': float(institutional_active)
            }
            
            # Calculate confidence score (weighted average of factors)
            weights = {
                'resistance_cross': 0.25,
                'volume_surge': 0.15,
                'momentum_strong': 0.15,
                'strong_movement': 0.15,
                'buy_pressure_strong': 0.10,
                'order_flow_bullish': 0.10,
                'institutional_active': 0.10
            }
            
            # Calculate weighted confidence
            confidence = sum(decision_factors[k] * weights[k] for k in weights)
            
            # Filter fake breakouts by applying strict confirmation rules
            fake_breakout = False
            if params['fake_breakout_filter']:
                # Check for common fake breakout patterns
                if volume_surge and not buy_pressure_strong:
                    fake_breakout = True
                if resistance_cross and not momentum_strong:
                    fake_breakout = True
                if institutional_active and institutional_direction == 'bearish' and resistance_cross:
                    fake_breakout = True
            
            # Determine long/short signal
            long_signal = (
                resistance_cross and 
                volume_surge and 
                momentum_strong and 
                not fake_breakout and
                confidence > 0.6
            )
            
            short_signal = (
                strong_movement and 
                volume_surge and 
                order_flow_bearish and
                confidence > 0.6
            )
            
            # Determine final signal
            if long_signal:
                signal = "BUY"
            elif short_signal:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            # Calculate recommended position size based on volatility
            position_size = self._calculate_position_size(asset, current_atr, confidence, params)
            
            # Create detailed signal metadata
            signal_metadata = {
                'confidence': confidence,
                'decision_factors': decision_factors,
                'atr': float(current_atr),
                'resistance_level': float(resistance_level),
                'volume_ratio': float(current_volume / np.mean(volume[-10:])) if len(volume) >= 10 else 1.0,
                'momentum': float(recent_momentum) if len(close) >= 3 else 0.0,
                'fake_breakout_detected': fake_breakout,
                'market_regime': market_regime,
                'recommended_size': position_size,
                'order_flow_direction': order_flow_analysis['direction'],
                'institutional_bias': institutional_analysis['directional_bias']
            }
            
            return signal, signal_metadata
            
        except Exception as e:
            logger.error(f"Error calculating breakout signal: {e}", 
                        metadata={"asset": asset, "error": str(e)})
            
            # Return safe HOLD signal on error
            return "HOLD", {"reason": f"calculation_error: {str(e)}", "confidence": 0.0}
    
    def _calculate_position_size(self, 
                                asset: str, 
                                atr: float, 
                                confidence: float, 
                                params: Dict[str, Any]) -> float:
        """
        Calculate position size based on ATR, confidence and strategy parameters
        Returns position size as fraction of portfolio
        """
        # Base position size from parameters
        base_size = params['max_position_size']
        
        # Scale by confidence
        confidence_factor = confidence  # Linear scaling
        
        # Inverse scale by volatility (more volatile = smaller position)
        if atr > 0:
            normalized_atr = min(3.0, atr / 0.01)  # Normalize ATR
            volatility_factor = 1.0 / normalized_atr
        else:
            volatility_factor = 1.0
        
        # Additional scaling for certain market regimes
        regime_factor = 1.0
        if asset in self.asset_regime:
            if self.asset_regime[asset] in ['volatile', 'crisis']:
                regime_factor = 0.5  # Half position size in volatile regimes
            elif self.asset_regime[asset] in ['bear']:
                regime_factor = 0.7  # 70% position size in bear markets
        
        # Calculate final position size
        position_size = base_size * confidence_factor * volatility_factor * regime_factor
        
        # Apply max constraint from config
        max_size = self.config['max_exposure']
        position_size = min(position_size, max_size)
        
        return position_size
    
    async def _check_risk_constraints(self, 
                                     asset: str, 
                                     signal: str, 
                                     metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if the trade meets risk management constraints
        Returns approval status and risk metrics
        """
        try:
            # Skip checks for HOLD signals
            if signal == "HOLD":
                return {
                    'approved': True,
                    'reason': "hold_signal_approved",
                    'metrics': {}
                }
            
            # Prepare trade data for risk controller
            trade_data = {
                'asset': asset,
                'direction': signal,
                'size': metadata['recommended_size'],
                'confidence': metadata['confidence'],
                'strategy': STRATEGY_NAME,
                'strategy_id': self.strategy_id
            }
            
            # Get risk approval with timeout protection
            risk_approval = await asyncio.wait_for(
                self.risk_controller.check_trade(trade_data),
                timeout=EMERGENCY_TIMEOUT_SECONDS / 2  # Critical path needs shorter timeout
            )
            
            # Calculate Value-at-Risk if not provided by risk controller
            if 'var' not in risk_approval['metrics']:
                # Initialize Monte Carlo simulator for VaR calculation
                mc_simulator = MonteCarloVaR()
                var_result = await mc_simulator.calculate_var(
                    asset, 
                    position_size=metadata['recommended_size'],
                    confidence_level=self.config['var_confidence']
                )
                risk_approval['metrics']['var'] = var_result['var']
            
            # Check if VaR is within acceptable threshold
            if risk_approval['metrics'].get('var', 0) > self.config['var_threshold']:
                return {
                    'approved': False,
                    'reason': "var_threshold_exceeded",
                    'metrics': risk_approval['metrics']
                }
            
            return risk_approval
            
        except Exception as e:
            logger.error(f"Risk constraint check failed: {e}", 
                        metadata={"asset": asset, "error": str(e)})
            
            # Return disapproval on error (safe approach)
            return {
                'approved': False,
                'reason': f"risk_check_error: {str(e)}",
                'metrics': {}
            }
    
    async def _check_liquidity_for_execution(self, 
                                           asset: str, 
                                           signal: str, 
                                           metadata: Dict[str, Any]) -> Dict[str, bool]:
        """
        Check if there's sufficient liquidity for trade execution
        Returns approval status and liquidity metrics
        """
        try:
            # Skip checks for HOLD signals
            if signal == "HOLD":
                return {
                    'approved': True,
                    'reason': "hold_signal_approved"
                }
            
            # Prepare liquidity check request
            liquidity_request = {
                'asset': asset,
                'size': metadata['recommended_size'],
                'expected_impact': 0.01  # Default impact estimate
            }
            
            # Get liquidity approval with timeout protection
            liquidity_approval = await asyncio.wait_for(
                self.liquidity_optimizer.check_liquidity(liquidity_request),
                timeout=EMERGENCY_TIMEOUT_SECONDS / 2
            )
            
            # Additional check against adaptive parameters
            if liquidity_approval['liquidity_score'] < self.adaptive_params['liquidity_requirement']:
                return {
                    'approved': False,
                    'reason': "insufficient_liquidity_for_adaptive_params"
                }
            
            return {
                'approved': liquidity_approval['sufficient_liquidity'],
                'reason': liquidity_approval.get('reason', "liquidity_check_complete")
            }
            
        except Exception as e:
            logger.error(f"Liquidity check failed: {e}", 
                        metadata={"asset": asset, "error": str(e)})
            
            # Return disapproval on error (safe approach)
            return {
                'approved': False,
                'reason': f"liquidity_check_error: {str(e)}",
                'metrics': {}
            }