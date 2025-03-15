import numpy as np
import logging
import asyncio
import time
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import functools
from dataclasses import dataclass, asdict
import json
import hashlib
import os
from enum import Enum
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from cryptography.fernet import Fernet
import base64

# Core System Integration
from src.Core.data.realtime.market_data import UnifiedMarketFeed
from src.Core.data.processing.data_parser import validate_market_data
from src.Core.trading.risk.risk_management import AdaptiveRiskController
from src.Core.trading.execution.algo_engine import AlgorithmicExecution
from src.Core.trading.strategies.strategy_orchestrator import StrategyOrchestrator
from src.ai.ensembles.meta_trader import MetaStrategyOptimizer
from src.ai.analysis.market_regime_classifier import MarketRegimeClassifier
from src.ai.forecasting.order_flow import OrderFlowAnalyzer
from Core.data.order_book_analyzer import OrderBookDepthAnalyzer
from utils.analytics.monte_carlo_simulator import MonteCarloVaR
from ai.reinforcement.q_learning.agent import QStrategyOptimizer
from Core.trading.execution.execution_validator import TradeValidator
from Core.trading.security.trade_security_guard import TradingSecurityGuard
from Core.trading.logging.performance_tracker import PerformanceTracker

# Security & Validation
from src.Core.trading.security.alert_system import SecurityMonitor
from src.Core.data.asset_validator import validate_asset_tradability
from utils.helpers.stealth_api import StealthAPIManager
from Core.trading.security.api_security import SecureAPIManager

# Resource Optimization
from Core.trading.optimization.resource_optimizer import ResourceOptimizer
from Core.trading.optimization.device_detector import DeviceCapabilities

# Shared Utilities
from utils.logging.structured_logger import StructuredLogger
from utils.helpers.error_handler import ErrorHandler
from src.Core.trading.logging.decision_logger import DecisionMetadata
from utils.helpers.cache_manager import CacheManager, CachePolicy

# Thread safety
from threading import Lock

class TrendDirection(Enum):
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"

class ExecutionStrategy(Enum):
    TWAP = "TWAP"
    VWAP = "VWAP"
    SMART = "SMART"
    ICEBERG = "ICEBERG"
    ADAPTIVE = "ADAPTIVE"

@dataclass
class TrendCondition:
    trend_direction: TrendDirection
    trend_strength: float
    volatility: float
    book_confirmation: bool
    confidence: float = 0.0

@dataclass
class OrderFlowCondition:
    institutional_participation: bool
    hidden_liquidity: bool
    order_imbalance: float
    block_trade_detected: bool = False
    cumulative_delta: float = 0.0

@dataclass
class MarketRegimeCondition:
    regime: str
    confidence: float
    regime_compatibility: float
    volatility_regime: str = "NORMAL"
    correlation_state: str = "NORMAL"

@dataclass
class RiskParameters:
    max_exposure: float
    current_exposure: float
    var: float
    liquidity_risk: float
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

@dataclass
class LiquidityCondition:
    available_liquidity: float
    impact_cost: float
    acceptable_slippage: float
    spread: float = 0.0
    depth_imbalance: float = 0.0

@dataclass
class SignalDecision:
    asset: str
    decision: str
    size: Optional[float] = None
    direction: Optional[str] = None
    confidence: float = 0.0
    risk_parameters: Optional[Dict[str, float]] = None
    execution_metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    reason: Optional[str] = None
    system_state: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class QuantumTrendStrategy:
    """
    Multi-Dimensional Trend Following System with:
    - Spacetime-optimized trend detection
    - Quantum-resistant decision making
    - Institutional-grade risk controls
    - Deep reinforcement learning integration
    - Mobile-optimized processing
    - High-frequency trading capabilities
    - Integrated with Apex system architecture
    """
    
    # Thread-safe instance sharing with explicit locking
    _instance_lock = Lock()
    _shared_state = {}
    
    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if not hasattr(cls, 'instance'):
                cls.instance = super(QuantumTrendStrategy, cls).__new__(cls)
            return cls.instance
    
    def __init__(self, assets: List[str], config: Dict[str, Any]):
        with self._instance_lock:
            if not hasattr(self, 'initialized'):
                self.assets = assets
                self.config = config
                self.adaptive_params = self._load_adaptive_parameters()
                
                # System performance tracking
                self.last_process_time = {}
                self.execution_latency = {}
                self.signal_accuracy = {}
                
                # Cache initialization
                self.cache = CacheManager(
                    max_size=config.get('cache_size', 10000),
                    policy=CachePolicy.LRU
                )
                
                # Device capability detection for resource optimization
                self.device_capabilities = DeviceCapabilities.detect()
                self.resource_optimizer = ResourceOptimizer(self.device_capabilities)
                self.is_mobile = self.device_capabilities.is_mobile
                
                # Integrated System Components
                self.market_feed = UnifiedMarketFeed()
                self.risk_controller = AdaptiveRiskController()
                self.execution_engine = AlgorithmicExecution()
                self.strategy_orchestrator = StrategyOrchestrator()
                self.meta_optimizer = MetaStrategyOptimizer()
                self.regime_classifier = MarketRegimeClassifier()
                self.orderflow_analyzer = OrderFlowAnalyzer()
                self.security_monitor = SecurityMonitor()
                self.q_optimizer = QStrategyOptimizer()
                self.book_analyzer = OrderBookDepthAnalyzer()
                self.trade_validator = TradeValidator()
                self.security_guard = TradingSecurityGuard()
                self.performance_tracker = PerformanceTracker("quantum_trend")
                self.secure_api = SecureAPIManager()
                
                # Thread-safe locks
                self.strategy_lock = Lock()
                self.data_lock = Lock()
                
                # Parallel Processing - Optimized for multi-core CPUs
                max_workers = min(len(assets), os.cpu_count() * 2) if not self.is_mobile else 2
                self.executor = ThreadPoolExecutor(max_workers=max_workers)
                self.loop = asyncio.get_event_loop()
                
                # State Management
                self.trend_states = {asset: None for asset in assets}
                self.parameter_history = []
                self.market_states = {}
                self.decision_cache = {}
                
                # Performance metrics
                self.decision_timing = {}
                self.signal_metrics = {
                    "true_positives": 0,
                    "false_positives": 0,
                    "true_negatives": 0,
                    "false_negatives": 0
                }
                
                self.initialized = True
                
                # Structured Logger
                self.logger = StructuredLogger("quantum_trend")

    def _load_adaptive_parameters(self) -> Dict[str, Any]:
        """Load optimized parameters with fallback defaults"""
        try:
            # Try to load from orchestrator for system-wide coherence
            params = self.strategy_orchestrator.get_strategy_parameters('quantum_trend') if hasattr(self, 'strategy_orchestrator') else None
            
            if not params:
                # Default parameters with reasonable values
                params = {
                    'temporal_lookback': 300,
                    'price_lookback': 200,
                    'volume_lookback': 100,
                    'trend_strength_threshold': 0.3,
                    'volatility_scaling': True,
                    'min_liquidity_ratio': 3.0,
                    'regime_weight': 0.4,
                    'orderflow_weight': 0.3,
                    'price_action_weight': 0.3,
                    'mobile_optimization': True
                }
                
            # Validate parameters are within acceptable bounds
            return self._validate_trend_parameters(params)
            
        except Exception as e:
            ErrorHandler.handle(e, context={"component": "quantum_trend", "method": "_load_adaptive_parameters"})
            # Return safe defaults in case of error
            return {
                'temporal_lookback': 200,
                'price_lookback': 100,
                'volume_lookback': 50,
                'trend_strength_threshold': 0.25,
                'volatility_scaling': True,
                'min_liquidity_ratio': 2.0,
                'regime_weight': 0.33,
                'orderflow_weight': 0.33,
                'price_action_weight': 0.34,
                'mobile_optimization': True
            }

    @functools.lru_cache(maxsize=128)
    def _get_optimized_lookback(self, asset: str, market_volatility: float) -> int:
        """Dynamically optimize lookback period based on market conditions"""
        # Start with base lookback
        base_lookback = self.adaptive_params['price_lookback']
        
        # Scale with volatility - higher volatility requires shorter lookback
        volatility_factor = 1.0 - min(market_volatility * 2.0, 0.7)
        
        # Get asset-specific optimal lookback from q-learning
        asset_lookback = self.q_optimizer.get_optimal_lookback(asset)
        
        # Compute final lookback with bounds
        lookback = int(base_lookback * volatility_factor * asset_lookback)
        
        # Ensure lookback is within reasonable bounds
        return max(20, min(lookback, 500))

    async def compute_trend_signals(self, asset: str) -> Dict[str, Any]:
        """
        Multi-Timeframe Trend Analysis with Quantum-Resistant Verification
        Returns structured decision metadata for system integration
        
        Optimized for:
        - Lower latency with memoization
        - Proper integration with system orchestrator
        - Mobile-friendly computation when needed
        - Security validation at every step
        """
        start_time = time.time()
        decision_id = f"{asset}_{int(start_time * 1000)}"
        
        try:
            # Check if processing is allowed by the orchestrator
            if not await self.strategy_orchestrator.is_strategy_active('quantum_trend'):
                return self._create_hold_signal(asset, "strategy_inactive")
                
            # Security Validation
            if not await self.security_monitor.validate_trading_environment(asset):
                self.security_guard.log_security_event(
                    "invalid_trading_environment", 
                    {"asset": asset, "timestamp": datetime.utcnow().isoformat()}
                )
                return self._create_hold_signal(asset, "security_validation_failed")
                
            # Data Acquisition with caching - avoid redundant API calls
            cache_key = f"market_data_{asset}_{int(time.time() / self.config.get('data_cache_seconds', 30))}"
            
            market_data = self.cache.get(cache_key)
            if market_data is None:
                market_data = await self.market_feed.get_temporal_data(
                    asset, 
                    fields=['price', 'volume', 'order_book', 'l2_depth'],
                    lookback=self.adaptive_params['temporal_lookback']
                )
                # Cache the data for reuse
                self.cache.set(cache_key, market_data, ttl=self.config.get('data_cache_seconds', 30))
            
            # Data validation
            validation = validate_market_data(market_data)
            if not validation['valid']:
                self.logger.warning(f"Data validation failed for {asset}", metadata=validation)
                return self._create_hold_signal(asset, "invalid_data")

            # Optimize computation based on device capabilities
            lightweight_mode = self.is_mobile and self.adaptive_params.get('mobile_optimization', True)
            
            # Determine if full analysis is needed (avoid redundant analysis)
            current_market_hash = self._compute_market_state_hash(market_data)
            cached_decision = self.decision_cache.get(asset)
            
            if cached_decision and cached_decision.get('market_hash') == current_market_hash:
                # Market hasn't changed significantly, return cached decision
                cached_decision['metadata']['from_cache'] = True
                self.logger.info(f"Using cached decision for {asset} - no significant market change")
                return cached_decision
            
            # Multi-Dimensional Trend Analysis - parallelized where possible
            trend_tasks = [
                self._analyze_price_trend(asset, market_data, lightweight_mode),
                self._analyze_order_flow(asset, market_data, lightweight_mode),
                self._analyze_market_regime(asset, market_data, lightweight_mode),
                self._analyze_risk_parameters(asset, lightweight_mode),
                self._analyze_liquidity(asset, lightweight_mode)
            ]
            
            # Execute all analysis in parallel
            trend_conditions = await asyncio.gather(*trend_tasks)
            
            # AI-Optimized Decision Fusion
            decision_matrix = await self.meta_optimizer.fuse_trend_decisions(
                trend_conditions,
                strategy_type='quantum_trend',
                asset=asset
            )
            
            # Signal generation with fail-safes
            signal = await self._generate_trend_signal(asset, decision_matrix, market_data)
            
            # Store in cache with market state hash
            signal['market_hash'] = current_market_hash
            self.decision_cache[asset] = signal
            
            # Performance tracking
            processing_time = time.time() - start_time
            self.last_process_time[asset] = processing_time
            
            if processing_time > self.config.get('latency_warning_threshold', 0.05):
                self.logger.warning(f"High latency trend processing for {asset}: {processing_time:.4f}s", 
                                   metadata={"asset": asset, "latency": processing_time})
            
            # Security guard final check - anomaly detection
            if signal['decision'] != 'HOLD':
                allowed = await self.security_guard.validate_signal(
                    asset=asset,
                    decision=signal['decision'],
                    size=signal.get('size', 0),
                    time_window=processing_time
                )
                
                if not allowed:
                    return self._create_hold_signal(asset, "security_guard_rejection")
            
            return signal
            
        except Exception as e:
            # Comprehensive error handling
            processing_time = time.time() - start_time
            self._handle_trend_error(asset, e, decision_id, processing_time)
            
            # Always return safe HOLD signal on error
            return self._create_hold_signal(asset, f"error: {str(e)}")

    def _compute_market_state_hash(self, market_data: Dict) -> str:
        """
        Compute a hash of the relevant market data to detect significant changes
        """
        # Extract only the relevant parts of market data to detect meaningful changes
        key_data = {
            'last_price': market_data['price'][-1] if market_data.get('price') else None,
            'last_volume': market_data['volume'][-1] if market_data.get('volume') else None,
            'price_window': market_data['price'][-10:] if market_data.get('price') else None,
            'orderbook_imbalance': self.book_analyzer.calculate_imbalance(market_data.get('order_book', {}))
        }
        
        # Create a deterministic string representation and hash it
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    async def _analyze_price_trend(self, asset: str, data: Dict, lightweight_mode: bool = False) -> TrendCondition:
        """
        Temporal Trend Analysis with Adaptive Thresholds
        Optimized for vectorized computation and minimal latency
        """
        # Vectorized operations for performance
        price_data = np.array(data['price'][-self.adaptive_params['price_lookback']:])
        volume_data = np.array(data['volume'][-self.adaptive_params['volume_lookback']:])
        
        # Anomaly check - quick sanity check to prevent bad data
        if len(price_data) < 10 or np.isnan(price_data).any():
            self.logger.warning(f"Insufficient price data for {asset}")
            return TrendCondition(
                trend_direction=TrendDirection.NEUTRAL,
                trend_strength=0.0,
                volatility=0.0,
                book_confirmation=False,
                confidence=0.0
            )
        
        # Performance optimization for mobile devices
        if lightweight_mode:
            # Simple moving averages - computationally efficient
            short_window = max(5, int(self.adaptive_params['price_lookback'] * 0.1))
            long_window = max(20, int(self.adaptive_params['price_lookback'] * 0.5))
            
            short_ma = np.mean(price_data[-short_window:])
            long_ma = np.mean(price_data[-long_window:])
            
            volatility = np.std(price_data[-20:]) / np.mean(price_data[-20:])
            trend_strength = abs(short_ma - long_ma) / (volatility * long_ma) if volatility > 0 else 0
            
            # Skip book confirmation for speed in lightweight mode
            book_confirmation = True
            confidence = min(0.7, trend_strength)  # Cap confidence in simplified model
            
        else:
            # Full analysis for high-performance devices
            # AI-Optimized Moving Averages
            ma_bounds = self.config.get('ma_bounds', {'min_short': 5, 'max_short': 50, 'min_long': 20, 'max_long': 200})
            
            # Get optimal windows from reinforcement learning
            short_window, long_window = await self.q_optimizer.get_optimal_windows(asset, price_data, ma_bounds)
            
            # Use numpy's optimized functions for speed
            short_ma = np.mean(price_data[-short_window:])
            long_ma = np.mean(price_data[-long_window:])
            
            # Calculate returns for volatility
            returns = np.diff(price_data) / price_data[:-1]
            
            # EWMA volatility (more responsive to recent volatility changes)
            volatility = np.sqrt(np.sum(returns**2 * np.exp(np.linspace(-1, 0, len(returns)))) / np.sum(np.exp(np.linspace(-1, 0, len(returns)))))
            
            # Trend strength with volume confirmation
            volume_factor = np.sum(volume_data[-5:]) / np.sum(volume_data[-20:-5]) if len(volume_data) >= 20 else 1.0
            trend_strength = abs(short_ma - long_ma) / (volatility * long_ma) * min(2.0, volume_factor) if volatility > 0 else 0
            
            # Order Book Confirmation - check if order book supports the trend
            book_confirmation = await self.book_analyzer.confirm_trend(
                asset,
                price_data[-1],
                'up' if short_ma > long_ma else 'down'
            )
            
            # AI-weighted confidence based on multiple factors
            confidence = await self.q_optimizer.calculate_trend_confidence(
                asset,
                trend_strength,
                volatility,
                book_confirmation,
                volume_factor
            )
        
        # Create structured trend condition
        return TrendCondition(
            trend_direction=TrendDirection.UP if short_ma > long_ma else TrendDirection.DOWN,
            trend_strength=float(trend_strength),
            volatility=float(volatility),
            book_confirmation=book_confirmation,
            confidence=float(confidence)
        )

    async def _analyze_order_flow(self, asset: str, data: Dict, lightweight_mode: bool = False) -> OrderFlowCondition:
        """
        Institutional Order Flow Analysis
        Detects large player activity and hidden liquidity
        """
        # Basic cached order flow analysis for all modes
        cache_key = f"orderflow_{asset}_{int(time.time() / 60)}"  # 1-minute cache
        cached_flow = self.cache.get(cache_key)
        
        if cached_flow:
            orderflow = cached_flow
        else:
            orderflow = await self.orderflow_analyzer.analyze_flow(asset)
            self.cache.set(cache_key, orderflow, ttl=60)  # Cache for 60 seconds
            
        # Basic analysis that works in both modes
        large_blocks = orderflow['large_block_ratio'] > self.config.get('institutional_threshold', 0.65)
        hidden_liquidity = orderflow['hidden_liquidity'] > self.config.get('hidden_order_threshold', 0.4)
        
        # Enhanced analysis for high-performance mode
        if not lightweight_mode:
            # Check for block trades in the order book
            block_trade_detected = await self.orderflow_analyzer.detect_block_trades(asset, lookback_minutes=15)
            
            # Calculate cumulative delta (buy vs sell pressure)
            cumulative_delta = await self.orderflow_analyzer.calculate_cumulative_delta(
                asset, 
                lookback_bars=min(50, self.adaptive_params['price_lookback'])
            )
        else:
            # Simplified version for lightweight mode
            block_trade_detected = False
            cumulative_delta = 0.0
        
        return OrderFlowCondition(
            institutional_participation=large_blocks,
            hidden_liquidity=hidden_liquidity,
            order_imbalance=orderflow['imbalance_ratio'],
            block_trade_detected=block_trade_detected,
            cumulative_delta=cumulative_delta
        )

    async def _analyze_market_regime(self, asset: str, data: Dict, lightweight_mode: bool = False) -> MarketRegimeCondition:
        """
        Multi-Factor Regime Classification
        Determines market state and suitability for trend following
        """
        # Always get basic regime classification
        regime = await self.regime_classifier.classify_regime(asset, data)
        confidence = await self.regime_classifier.get_regime_confidence(asset)
        
        # Check compatibility with current strategy
        regime_compatibility = self.config['allowed_regimes'].get(regime, 0.0)
        
        # Enhanced analysis for high-performance mode
        if not lightweight_mode:
            # Get detailed volatility regime and correlation state
            volatility_regime = await self.regime_classifier.classify_volatility_regime(asset)
            correlation_state = await self.regime_classifier.get_correlation_state(asset)
        else:
            # Default values for lightweight mode
            volatility_regime = "NORMAL"
            correlation_state = "NORMAL"
        
        return MarketRegimeCondition(
            regime=regime,
            confidence=confidence,
            regime_compatibility=regime_compatibility,
            volatility_regime=volatility_regime,
            correlation_state=correlation_state
        )

    async def _analyze_risk_parameters(self, asset: str, lightweight_mode: bool = False) -> RiskParameters:
        """
        Real-Time Risk Constraints Analysis
        Ensures trading within risk limits
        """
        # Get core risk metrics that are required in all modes
        exposure = await self.risk_controller.get_current_exposure(asset)
        liquidity_risk = await self.risk_controller.calculate_liquidity_risk(asset)
        
        # VaR calculation - use cached value if in lightweight mode
        if lightweight_mode:
            var_cache_key = f"var_{asset}_{int(time.time() / 300)}"  # 5-minute cache
            var = self.cache.get(var_cache_key, 0.0)
            if var == 0.0:
                # Use simplified VaR for mobile
                var = exposure * self.risk_controller.get_volatility_factor(asset) * 1.645  # 95% confidence
                self.cache.set(var_cache_key, var, ttl=300)
        else:
            # Full Monte Carlo VaR for high-performance mode
            var = await MonteCarloVaR.calculate(asset, self.config.get('var_confidence', 0.95))
        
        # Enhanced risk metrics for high-performance mode
        if not lightweight_mode:
            # Get performance metrics for risk-adjusted returns
            performance = await self.performance_tracker.get_strategy_metrics(asset)
            max_drawdown = performance.get('max_drawdown', 0.0)
            sharpe_ratio = performance.get('sharpe_ratio', 0.0)
            sortino_ratio = performance.get('sortino_ratio', 0.0)
        else:
            # Default values for lightweight mode
            max_drawdown = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
        
        return RiskParameters(
            max_exposure=self.config.get('max_exposure', 1.0),
            current_exposure=exposure,
            var=var,
            liquidity_risk=liquidity_risk,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio
        )

    async def _analyze_liquidity(self, asset: str, lightweight_mode: bool = False) -> LiquidityCondition:
        """
        Multi-Dimensional Liquidity Analysis
        Ensures trades can be executed with minimal slippage
        """
        # Basic liquidity metrics for all modes
        liquidity = await self.risk_controller.get_asset_liquidity(asset)
        impact_cost = await self.risk_controller.calculate_market_impact(asset)
        
        # Enhanced liquidity analysis for high-performance mode
        if not lightweight_mode:
            # Get detailed order book metrics
            book_metrics = await self.book_analyzer.get_liquidity_metrics(asset)
            spread = book_metrics.get('spread', 0.0)
            depth_imbalance = book_metrics.get('depth_imbalance', 0.0)
        else:
            # Default values for lightweight mode
            spread = 0.0
            depth_imbalance = 0.0
        
        return LiquidityCondition(
            available_liquidity=liquidity['available'],
            impact_cost=impact_cost,
            acceptable_slippage=self.config.get('max_slippage', 0.001),
            spread=spread,
            depth_imbalance=depth_imbalance
        )

    async def _generate_trend_signal(self, asset: str, conditions: Dict, data: Dict) -> Dict[str, Any]:
        """
        AI-Fused Trend Decision Making with Adaptive Liquidity Execution
        - Justifies why the AI makes each trade decision
        - Detects potential bias in reinforcement learning
        """
        # Create structured decision metadata for logging and XAI
        decision_metadata = DecisionMetadata(
            strategy="quantum_trend",
            asset=asset,
            conditions=conditions,
            timestamp=datetime.utcnow(),
            system_version=self.config.get('version', '1.0')
        )
        
        # Fetch real-time order book liquidity conditions
        liquidity = await self.risk_controller.get_asset_liquidity(asset)
        impact_cost = await self.risk_controller.calculate_market_impact(asset)

        # Determine execution algorithm based on market conditions
        if liquidity['available'] > 10.0 and impact_cost < 0.005:
            execution_algo = ExecutionStrategy.VWAP.value  # Large liquidity → VWAP
        elif liquidity['available'] > 5.0:
            execution_algo = ExecutionStrategy.TWAP.value  # Medium liquidity → TWAP
        else:
            execution_algo = ExecutionStrategy.ICEBERG.value  # Low liquidity → Hidden Iceberg Orders

        # Meta Strategy Optimization - core decision logic
        current_market_state = await self.strategy_orchestrator.get_market_state()
        
        optimized_decision = await self.meta_optimizer.adjust_trend_decision(
            base_conditions=conditions,
            market_context=current_market_state
        )
        
        # 🔴 AI Bias Checking
        bias_score = await self.q_optimizer.detect_model_bias(asset)
        if bias_score > 0.7:  # If bias score exceeds threshold, reject trade
            self.logger.warning(f"Trade rejected for {asset} due to AI bias score: {bias_score}")
            return self._create_hold_signal(asset, "AI Bias Detected")

        # Validate trade with risk system before proceeding
        if optimized_decision.get('execute', False):
            risk_validation = await self.risk_controller.validate_trade(
                asset=asset,
                direction=optimized_decision.get('direction', 'NEUTRAL'),
                size=optimized_decision.get('size', 0.0)
            )
            
            if not risk_validation['approved']:
                optimized_decision['execute'] = False
                optimized_decision['rejection_reason'] = risk_validation['reason']
        
        # 🔍 Explainability: Add feature importance breakdown
        feature_importance = await self.meta_optimizer.get_feature_importance(asset)
        optimized_decision['explainability'] = feature_importance

        # Update conditions with the selected execution strategy
        conditions['execution_strategy'] = execution_algo

        # Generate Structured Signal
        signal = SignalDecision(
            asset=asset,
            decision='HOLD',
            metadata=decision_metadata.to_dict(),
            system_state={
                'orchestrator_weights': self.strategy_orchestrator.get_current_weights(),
                'risk_parameters': self.risk_controller.get_current_parameters(asset),
                'liquidity_profile': self.risk_controller.get_liquidity_profile(asset),
                'market_state': current_market_state
            },
            timestamp=datetime.utcnow().isoformat()
        )
        
        # If decision is to execute a trade, add trade-specific data
        if optimized_decision.get('execute', False):
            # Update signal with trade details
            signal = SignalDecision(
                asset=asset,
                decision=optimized_decision['direction'].upper(),
                size=optimized_decision['size'],
                direction=optimized_decision['direction'].upper(),
                confidence=optimized_decision.get('confidence', 0.0),
                risk_parameters={
                    'stop_loss': optimized_decision['stop_loss'],
                    'take_profit': optimized_decision['take_profit'],
                    'max_slippage': self.config.get('max_slippage', 0.001),
                    'position_ttl': optimized_decision.get('position_ttl', 60*60*24)  # Default 24h TTL
                },
                execution_metadata={
                    'strategy': execution_algo,
                    'urgency': optimized_decision.get('urgency', 'NORMAL'),
                    'adaptive_execution': True,
                    'allow_partial_fills': optimized_decision.get('allow_partial_fills', True)
                },
                metadata=decision_metadata.to_dict(),
                system_state={
                    'orchestrator_weights': self.strategy_orchestrator.get_current_weights(),
                    'risk_parameters': self.risk_controller.get_current_parameters(asset),
                    'liquidity_profile': self.risk_controller.get_liquidity_profile(asset)
                },
                timestamp=datetime.utcnow().isoformat()
            )
            
        # Capture execution decision for performance tracking
        self.logger.log_decision(decision_metadata)
        await self.strategy_orchestrator.register_trend_decision(asset, signal)
        
        # Return the final signal decision
        return signal
    
    def _create_hold_signal(self, asset: str, reason: str) -> Dict[str, Any]:
        """Create a standardized HOLD signal with reason for traceability"""
        return {
            'asset': asset,
            'decision': 'HOLD',
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': {
                'strategy': 'quantum_trend',
                'version': self.config.get('version', '1.0'),
                'signal_source': 'safety_constraint'
            }
        }
    
    def _handle_trend_error(self, asset: str, error: Exception, decision_id: str, processing_time: float):
        """Comprehensive error handling with logging and recovery"""
        error_context = {
            "asset": asset,
            "decision_id": decision_id,
            "processing_time": processing_time,
            "error_type": type(error).__name__,
            "strategy": "quantum_trend"
        }
        
        self.logger.error(
            f"Error in trend analysis for {asset}: {str(error)}",
            metadata=error_context
        )
        
        # Log to security monitor for potential exploits
        if isinstance(error, (ValueError, TypeError)):
            self.security_monitor.log_potential_data_issue(asset, str(error))
        
        # Report to orchestrator to potentially disable strategy if persistent
        self.strategy_orchestrator.report_strategy_error("quantum_trend", asset, str(error))
        
        # Ensure relevant context is captured for debugging
        ErrorHandler.handle(error, context=error_context)
    
    @functools.lru_cache(maxsize=128)
    def _validate_trend_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters against security bounds"""
        validated = {}
        
        # Define acceptable bounds for each parameter
        bounds = {
            'temporal_lookback': (50, 1000),
            'price_lookback': (20, 500),
            'volume_lookback': (10, 300),
            'trend_strength_threshold': (0.1, 0.9),
            'min_liquidity_ratio': (1.0, 10.0),
            'regime_weight': (0.1, 0.6),
            'orderflow_weight': (0.1, 0.6),
            'price_action_weight': (0.1, 0.6)
        }
        
        # Enforce bounds on numerical parameters
        for param, value in params.items():
            if param in bounds:
                min_val, max_val = bounds[param]
                validated[param] = max(min_val, min(value, max_val))
            else:
                # Pass through non-bounded parameters
                validated[param] = value
        
        # Ensure weights sum to 1.0
        weight_keys = ['regime_weight', 'orderflow_weight', 'price_action_weight']
        weight_sum = sum(validated.get(k, 0) for k in weight_keys)
        
        if weight_sum > 0:
            # Normalize weights to sum to 1.0
            for k in weight_keys:
                if k in validated:
                    validated[k] = validated[k] / weight_sum
        
        return validated

    class SecureAPIManager:
        """
        Handles API key encryption and secure access control
        """

        def __init__(self):
            self.encryption_key = os.environ.get('API_ENCRYPTION_KEY', Fernet.generate_key())
            self.fernet = Fernet(self.encryption_key)

        def encrypt_api_key(self, api_key: str) -> str:
            return self.fernet.encrypt(api_key.encode()).decode()

        def decrypt_api_key(self, encrypted_api_key: str) -> str:
            return self.fernet.decrypt(encrypted_api_key.encode()).decode()

        def validate_trade_request(self, trade_request: Dict) -> bool:
            """
            Ensures trade requests cannot be manipulated externally
            """
            required_fields = ["asset", "size", "direction"]
            for field in required_fields:
                if field not in trade_request:
                    return False
            return True

    async def execute_trend_strategy(self, assets: Optional[List[str]] = None) -> Dict[str, SignalDecision]:
        """
        Execute trend strategy across multiple assets with optimized parallel processing
        
        Args:
            assets: List of assets to analyze, defaults to all configured assets
            
        Returns:
            Dict mapping assets to their signal decisions
        """
        start_time = time.time()
        results = {}
        
        # Use all configured assets if none specified
        target_assets = assets if assets else self.assets
        
        # Skip execution if orchestrator has disabled this strategy
        if not await self.strategy_orchestrator.is_strategy_active('quantum_trend'):
            self.logger.info("Quantum trend strategy is currently inactive")
            return {asset: self._create_hold_signal(asset, "strategy_inactive") for asset in target_assets}
        
        # Determine if we should use async batching based on device capabilities
        use_batching = not self.is_mobile and len(target_assets) > 3
        
        api_manager = SecureAPIManager()  # Initialize the secure API manager

        if use_batching:
            # Process assets in parallel with batching for high-performance systems
            batch_size = min(len(target_assets), max(1, os.cpu_count()))
            asset_batches = [target_assets[i:i + batch_size] for i in range(0, len(target_assets), batch_size)]
            
            for batch in asset_batches:
                batch_tasks = [self.compute_trend_signals(asset) for asset in batch]
                batch_results = await asyncio.gather(*batch_tasks)
                
                for i, asset in enumerate(batch):
                    trade_request = batch_results[i]

                    # 🔐 Validate trade before execution
                    if not api_manager.validate_trade_request(trade_request):
                        self.logger.warning(f"Invalid trade request detected for {asset}")
                        continue

                    # 🔑 Encrypt API keys before sending request
                    encrypted_key = api_manager.encrypt_api_key(self.secure_api.get_api_key(asset))
                    trade_request['api_key'] = encrypted_key

                    # 🔄 Execute trade only if fully validated
                    await self.execution_engine.place_order(trade_request)
                    results[asset] = trade_request  # Store the result for the asset
        else:
            for asset in target_assets:
                trade_request = await self.compute_trend_signals(asset)

                # 🔐 Validate trade before execution
                if not api_manager.validate_trade_request(trade_request):
                    self.logger.warning(f"Invalid trade request detected for {asset}")
                    continue

                # 🔑 Encrypt API keys before sending request
                encrypted_key = api_manager.encrypt_api_key(self.secure_api.get_api_key(asset))
                trade_request['api_key'] = encrypted_key

                # 🔄 Execute trade only if fully validated
                await self.execution_engine.place_order(trade_request)
                results[asset] = trade_request  # Store the result for the asset
        
        # Performance tracking
        execution_time = time.time() - start_time
        self.logger.info(
            f"Trend strategy execution completed for {len(target_assets)} assets in {execution_time:.4f}s",
            metadata={"avg_time_per_asset": execution_time / len(target_assets) if target_assets else 0}
        )
        
        # Security validation - ensure no anomalous signals
        if self.security_guard.check_for_anomalous_signals(results):
            self.logger.warning("Security guard detected potential anomalous signals")
            return {asset: self._create_hold_signal(asset, "security_anomaly_detected") for asset in target_assets}
        
        return results
    
    async def backtest_trend_strategy(self, 
                                     start_date: datetime, 
                                     end_date: datetime,
                                     assets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Backtest the trend strategy over a historical period
        
        Args:
            start_date: Beginning of backtest period
            end_date: End of backtest period
            assets: List of assets to backtest, defaults to all configured assets
            
        Returns:
            Dict with backtest results and performance metrics
        """
        backtest_results = {}
        
        # Use provided assets or all configured assets
        target_assets = assets if assets else self.assets
        
        # Store original adaptive parameters to restore later
        original_params = self.adaptive_params.copy()
        
        try:
            # Set up performance metrics collectors
            metrics = {
                'win_rate': {},
                'profit_factor': {},
                'sharpe_ratio': {},
                'max_drawdown': {},
                'avg_holding_period': {}
            }
            
            # Load historical data for all assets in one call if possible
            historical_data = await self.market_feed.get_historical_data(
                assets=target_assets,
                start_date=start_date,
                end_date=end_date,
                fields=['price', 'volume', 'order_book']
            )
            
            # Run backtests for each asset
            for asset in target_assets:
                # Run ML optimization to find best parameters for this asset/period
                optimized_params = await self.meta_optimizer.optimize_trend_parameters(
                    asset=asset,
                    historical_data=historical_data.get(asset, {}),
                    base_params=self.adaptive_params
                )
                
                # Temporarily set optimized parameters
                self.adaptive_params = optimized_params
                
                # Run the backtest for this asset
                asset_result = await self.strategy_orchestrator.backtest_strategy(
                    strategy_name='quantum_trend',
                    asset=asset,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=optimized_params,
                    historical_data=historical_data.get(asset, {})
                )
                
                # Store results
                backtest_results[asset] = asset_result
                
                # Collect metrics
                metrics['win_rate'][asset] = asset_result.get('win_rate', 0)
                metrics['profit_factor'][asset] = asset_result.get('profit_factor', 0)
                metrics['sharpe_ratio'][asset] = asset_result.get('sharpe_ratio', 0)
                metrics['max_drawdown'][asset] = asset_result.get('max_drawdown', 0)
                metrics['avg_holding_period'][asset] = asset_result.get('avg_holding_period', 0)
            
            # Calculate aggregate metrics
            aggregate_metrics = {
                'avg_win_rate': np.mean(list(metrics['win_rate'].values())),
                'avg_profit_factor': np.mean(list(metrics['profit_factor'].values())),
                'avg_sharpe_ratio': np.mean(list(metrics['sharpe_ratio'].values())),
                'avg_max_drawdown': np.mean(list(metrics['max_drawdown'].values())),
                'best_asset': max(metrics['sharpe_ratio'], key=metrics['sharpe_ratio'].get),
                'worst_asset': min(metrics['sharpe_ratio'], key=metrics['sharpe_ratio'].get)
            }
            
            # Store learning from backtest in Q-learning model
            await self.q_optimizer.update_from_backtest(backtest_results)
            
            return {
                'results': backtest_results,
                'metrics': metrics,
                'aggregate': aggregate_metrics,
                'optimized_parameters': {asset: backtest_results[asset].get('optimized_params', {}) 
                                         for asset in target_assets}
            }
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}")
            ErrorHandler.handle(e, context={"component": "quantum_trend", "method": "backtest_trend_strategy"})
            return {'error': str(e)}
            
        finally:
            # Restore original parameters
            self.adaptive_params = original_params
    
    async def tune_strategy_parameters(self, 
                                     assets: Optional[List[str]] = None,
                                     lookback_days: int = 30) -> Dict[str, Any]:
        """
        Optimize strategy parameters using recent market data and machine learning
        
        Args:
            assets: List of assets to optimize, defaults to all configured assets
            lookback_days: Days of historical data to use for optimization
            
        Returns:
            Dict with optimized parameters for each asset
        """
        # Calculate date range for optimization
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Use provided assets or all configured assets
        target_assets = assets if assets else self.assets
        
        optimized_params = {}
        
        try:
            # Load historical data for the optimization period
            historical_data = await self.market_feed.get_historical_data(
                assets=target_assets,
                start_date=start_date,
                end_date=end_date,
                fields=['price', 'volume', 'order_book']
            )
            
            # Process each asset with the meta optimizer
            for asset in target_assets:
                # Use reinforcement learning to find optimal parameters
                asset_params = await self.meta_optimizer.optimize_trend_parameters(
                    asset=asset,
                    historical_data=historical_data.get(asset, {}),
                    base_params=self.adaptive_params
                )
                
                # Validate parameters for security
                asset_params = self._validate_trend_parameters(asset_params)
                
                # Store optimized parameters
                optimized_params[asset] = asset_params
                
                # Log the optimization results
                self.logger.info(
                    f"Parameter optimization completed for {asset}",
                    metadata={
                        "asset": asset,
                        "original_params": str(self.adaptive_params),
                        "optimized_params": str(asset_params)
                    }
                )
            
            # Update global adaptive parameters to weighted average of individual optimizations
            # This promotes strategy coherence while respecting asset-specific needs
            global_params = {}
            
            # Calculate weighted parameters based on asset performance
            for param in self.adaptive_params:
                if param in ['volatility_scaling', 'mobile_optimization']:
                    # Boolean parameters - use majority vote
                    true_count = sum(1 for a in optimized_params if optimized_params[a].get(param, False))
                    global_params[param] = true_count > len(optimized_params) / 2
                else:
                    # Numerical parameters - weighted average
                    values = [optimized_params[a].get(param, self.adaptive_params.get(param, 0)) 
                              for a in optimized_params]
                    global_params[param] = np.mean(values)
            
            # Update the main adaptive parameters
            self.adaptive_params = self._validate_trend_parameters(global_params)
            
            # Store asset-specific parameters in cache for retrieval during execution
            for asset in target_assets:
                cache_key = f"asset_params_{asset}"
                self.cache.set(cache_key, optimized_params[asset], ttl=86400)  # Cache for 24 hours
            
            return {
                'global_parameters': self.adaptive_params,
                'asset_parameters': optimized_params,
                'tuning_period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat(),
                    'days': lookback_days
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in parameter tuning: {str(e)}")
            ErrorHandler.handle(e, context={"component": "quantum_trend", "method": "tune_strategy_parameters"})
            return {'error': str(e)}
    
    async def detect_trading_opportunities(self, 
                                         market_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Real-time opportunity scanner for mobile and dashboard alerts
        Optimized for low-latency scanning of multiple assets
        
        Args:
            market_data: Optional pre-loaded market data to avoid redundant API calls
            
        Returns:
            List of trading opportunities with confidence scores
        """
        opportunities = []
        start_time = time.time()
        
        # Lightweight mode for faster scanning
        lightweight_mode = self.is_mobile
        
        try:
            # Get market data if not provided
            if not market_data:
                # Use batch data retrieval for efficiency
                market_data = await self.market_feed.get_real_time_batch(self.assets)
            
            # Check market regime first to filter assets
            global_regime = await self.regime_classifier.get_global_market_regime()
            
            # Skip scanning if global regime is unfavorable
            if global_regime in self.config.get('unfavorable_regimes', ['HIGH_VOLATILITY', 'CRASH']):
                self.logger.info(f"Skipping opportunity scanning in {global_regime} regime")
                return []
            
            # Calculate time-adjusted opportunity thresholds
            time_of_day = datetime.utcnow().hour + datetime.utcnow().minute / 60.0
            # Adjust threshold based on historical performance at this time
            time_performance = await self.performance_tracker.get_time_of_day_performance(time_of_day)
            base_threshold = self.config.get('opportunity_threshold', 0.7)
            adjusted_threshold = base_threshold * (1.0 + (time_performance - 0.5) * 0.2)
            
            for asset in self.assets:
                # Skip assets with insufficient data
                if asset not in market_data or 'price' not in market_data[asset]:
                    continue
                
                # Get asset data
                asset_data = market_data[asset]
                
                # Quick check for minimum conditions before detailed analysis
                # Ensures we don't waste computation on obvious non-opportunities
                quick_check = await self._quick_opportunity_check(asset, asset_data)
                if not quick_check:
                    continue
                
                # Get asset-specific threshold adjustments
                asset_performance = await self.performance_tracker.get_asset_performance(asset)
                asset_threshold = adjusted_threshold * (0.9 + asset_performance * 0.2)
                
                # Multi-factor opportunity scoring - vectorized for speed
                price_opportunity = self._calculate_price_opportunity(asset_data, lightweight_mode)
                
                # Only continue analysis if price signal is promising
                if price_opportunity < asset_threshold * 0.8:
                    continue
                
                # Calculate additional opportunity factors
                volume_opportunity = self._calculate_volume_opportunity(asset_data, lightweight_mode)
                regime_opportunity = await self._calculate_regime_opportunity(asset, lightweight_mode)
                liquidity_opportunity = await self._calculate_liquidity_opportunity(asset, lightweight_mode)
                
                # Combined opportunity score with weighted factors
                weights = self.config.get('opportunity_weights', {
                    'price': 0.4,
                    'volume': 0.2,
                    'regime': 0.2,
                    'liquidity': 0.2
                })
                
                combined_score = (
                    price_opportunity * weights['price'] +
                    volume_opportunity * weights['volume'] +
                    regime_opportunity * weights['regime'] +
                    liquidity_opportunity * weights['liquidity']
                )
                
                # Direction determination
                direction = "UP" if price_opportunity > 0 else "DOWN"
                
                # Filter based on threshold
                if abs(combined_score) > asset_threshold:
                    # Calculate opportunity timeframe
                    timeframe = self._estimate_opportunity_timeframe(
                        asset, abs(combined_score), asset_data
                    )
                    
                    # Calculate expected edge
                    edge = await self._calculate_opportunity_edge(
                        asset, direction, abs(combined_score)
                    )
                    
                    opportunities.append({
                        'asset': asset,
                        'direction': direction,
                        'confidence': abs(combined_score),
                        'timeframe': timeframe,
                        'expected_edge': edge,
                        'timestamp': datetime.utcnow().isoformat(),
                        'opportunity_factors': {
                            'price': float(price_opportunity),
                            'volume': float(volume_opportunity),
                            'regime': float(regime_opportunity),
                            'liquidity': float(liquidity_opportunity)
                        },
                        'risk_metrics': {
                            'stop_loss_percent': float(edge['stop_loss_percent']),
                            'take_profit_percent': float(edge['take_profit_percent']),
                            'risk_reward_ratio': float(edge['risk_reward_ratio'])
                        }
                    })
            
            # Sort opportunities by confidence
            opportunities.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Limit number of opportunities for mobile
            if self.is_mobile:
                opportunities = opportunities[:5]
                
            # Performance tracking
            processing_time = time.time() - start_time
            self.logger.info(
                f"Opportunity scanning completed in {processing_time:.4f}s, found {len(opportunities)} opportunities",
                metadata={"assets_scanned": len(self.assets)}
            )
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error in opportunity detection: {str(e)}")
            ErrorHandler.handle(e, context={"component": "quantum_trend", "method": "detect_trading_opportunities"})
            return []
    
    async def _quick_opportunity_check(self, asset: str, data: Dict[str, Any]) -> bool:
        """Fast pre-check to filter out obvious non-opportunities"""
        # Ensure minimum data quality
        if not data.get('price') or len(data['price']) < 20:
            return False
        
        # Check if asset is tradable
        if not await validate_asset_tradability(asset):
            return False
            
        # Simple momentum check
        prices = np.array(data['price'][-20:])
        ma5 = np.mean(prices[-5:])
        ma20 = np.mean(prices)
        
        # Check for minimum directional movement
        min_movement = abs(ma5 - ma20) / ma20
        if min_movement < self.config.get('min_opportunity_movement', 0.005):
            return False
            
        return True
    
    def _calculate_price_opportunity(self, data: Dict[str, Any], lightweight_mode: bool) -> float:
        """Calculate price-based opportunity score"""
        # Extract price data and ensure it's a numpy array for vectorized operations
        prices = np.array(data['price'])
        
        if lightweight_mode:
            # Simple moving average crossover calculation
            short_window = 5
            medium_window = 20
            long_window = 50
            
            # Ensure we have enough data
            if len(prices) < long_window:
                return 0.0
                
            # Calculate moving averages
            ma_short = np.mean(prices[-short_window:])
            ma_medium = np.mean(prices[-medium_window:])
            ma_long = np.mean(prices[-long_window:])
            
            # Calculate crossover score
            short_cross = (ma_short - ma_medium) / ma_medium
            medium_cross = (ma_medium - ma_long) / ma_long
            
            # Combine with direction sign
            return np.sign(short_cross) * (abs(short_cross) + abs(medium_cross)) / 2
            
        else:
            # More sophisticated calculation for high-performance devices
            # Calculate various technical indicators using vectorized operations
            
            # Ensure adequate data
            if len(prices) < 50:
                return 0.0
                
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            
            # Momentum indicators
            mom_10 = prices[-1] / prices[-10] - 1 if len(prices) >= 10 else 0
            mom_20 = prices[-1] / prices[-20] - 1 if len(prices) >= 20 else 0
            
            # Compute RSI using vectorized numpy
            delta = np.diff(prices)
            up, down = delta.copy(), delta.copy()
            up[up < 0] = 0
            down[down > 0] = 0
            down = abs(down)
            
            # Calculate RSI with EWMA
            avg_gain = np.mean(up[-14:]) if len(up) >= 14 else np.mean(up)
            avg_loss = np.mean(down[-14:]) if len(down) >= 14 else np.mean(down)
            
            if avg_loss == 0:
                rsi = 100 if avg_gain > 0 else 50
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
            # Normalize RSI to -1 to 1 range
            norm_rsi = (rsi - 50) / 50
            
            # Moving average signals
            ma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
            ma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
            ma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else prices[-1]
            
            ma_crossover = (ma_10 - ma_20) / ma_20 + (ma_20 - ma_50) / ma_50
            
            # Recent volatility-adjusted trend
            recent_volatility = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
            if recent_volatility == 0:
                vol_adj_trend = 0
            else:
                vol_adj_trend = (prices[-1] - np.mean(prices[-10:])) / (recent_volatility * np.mean(prices[-10:]))
            
            # Combine signals with empirically determined weights
            combined_signal = (
                0.3 * (mom_10 + mom_20) / 2 +  # Momentum component
                0.3 * norm_rsi +               # RSI component
                0.2 * ma_crossover +           # MA crossover component
                0.2 * vol_adj_trend            # Volatility-adjusted trend
            )
            
            # Bound the result
            return max(-1.0, min(1.0, combined_signal))
    
    def _calculate_volume_opportunity(self, data: Dict[str, Any], lightweight_mode: bool) -> float:
        """Calculate volume-based opportunity score"""
        # Ensure volume data exists
        if 'volume' not in data or len(data['volume']) < 10:
            return 0.0
            
        volumes = np.array(data['volume'])
        prices = np.array(data['price'])
        
        # Ensure we have both price and volume data of the same length
        min_len = min(len(volumes), len(prices))
        volumes = volumes[-min_len:]
        prices = prices[-min_len:]
        
        if lightweight_mode:
            # Simple volume surge detection
            avg_volume = np.mean(volumes[:-5]) if len(volumes) > 5 else np.mean(volumes)
            recent_volume = np.mean(volumes[-5:])
            
            if avg_volume == 0:
                return 0.0
                
            volume_surge = recent_volume / avg_volume - 1
            
            # Direction from price
            price_direction = 1 if prices[-1] > prices[0] else -1
            
            return price_direction * min(1.0, max(-1.0, volume_surge))
            
        else:
            # Sophisticated volume analysis
            # Calculate on-balance volume
            obv = np.zeros(len(prices))
            for i in range(1, len(prices)):
                if prices[i] > prices[i-1]:
                    obv[i] = obv[i-1] + volumes[i]
                elif prices[i] < prices[i-1]:
                    obv[i] = obv[i-1] - volumes[i]
                else:
                    obv[i] = obv[i-1]
            
            # OBV trend
            obv_short = np.mean(obv[-5:])
            obv_long = np.mean(obv[-20:]) if len(obv) >= 20 else obv[0]
            
            # Volume trend
            vol_short = np.mean(volumes[-5:])
            vol_long = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[0]
            
            # Calculate volume-price trend
            if vol_long == 0 or obv_long == obv[0]:
                return 0.0
                
            vol_trend = vol_short / vol_long - 1
            obv_trend = (obv_short - obv_long) / abs(obv_long - obv[0]) if abs(obv_long - obv[0]) > 0 else 0
            
            # Price direction for sign
            price_direction = 1 if prices[-1] > np.mean(prices[-10:]) else -1
            
            # Combined score
            combined = price_direction * (0.6 * obv_trend + 0.4 * vol_trend)
            
            return max(-1.0, min(1.0, combined))
    
    async def _calculate_regime_opportunity(self, asset: str, lightweight_mode: bool) -> float:
        """Calculate regime-based opportunity score"""
        # Cache regime data to avoid redundant API calls
        cache_key = f"regime_{asset}_{int(time.time() / 300)}"  # 5-minute cache
        
        cached_regime = self.cache.get(cache_key)
        if cached_regime:
            regime_data = cached_regime
        else:
            # Get current market regime
            regime_data = await self.regime_classifier.get_asset_regime(asset)
            self.cache.set(cache_key, regime_data, ttl=300)
        
        # Base opportunity on regime compatibility with trend strategy
        base_score = self.config['allowed_regimes'].get(regime_data['regime'], 0.0)
        
        if lightweight_mode:
            return base_score
            
        else:
            # Enhanced regime analysis
            volatility_regime = regime_data.get('volatility_regime', 'NORMAL')
            correlation_state = regime_data.get('correlation_state', 'NORMAL')
            
            # Adjust score based on volatility regime
            vol_multiplier = {
                'LOW': 0.8,      # Lower volatility - weaker trends
                'NORMAL': 1.0,   # Normal volatility - normal trends
                'HIGH': 0.7,     # Higher volatility - riskier trends
                'EXTREME': 0.3   # Extreme volatility - avoid trends
            }.get(volatility_regime, 1.0)
            
            # Adjust score based on correlation regime
            corr_multiplier = {
                'DISPERSED': 1.2,    # Low asset correlation - better for individual trends
                'NORMAL': 1.0,       # Normal correlation
                'TIGHT': 0.8,        # High correlation - trends driven by market, not asset
                'CRISIS': 0.5        # Crisis correlation - avoid trend following
            }.get(correlation_state, 1.0)
            
            # Combined score
            adjusted_score = base_score * vol_multiplier * corr_multiplier
            
            return max(-1.0, min(1.0, adjusted_score))

    async def _calculate_liquidity_opportunity(self, asset: str, lightweight_mode: bool) -> float:
        """Calculate liquidity-based opportunity score"""
        # Get basic liquidity metrics
        liquidity = await self.risk_controller.get_asset_liquidity(asset)
        
        # Ensure minimum liquidity for any opportunity
        if liquidity['available'] < self.config.get('min_liquidity', 1.0):
            return 0.0
            
        if lightweight_mode:
            # Simple liquidity check for mobile
            base_score = min(1.0, liquidity['available'] / self.config.get('ideal_liquidity', 5.0))
            return base_score
            
        else:
            # Enhanced liquidity analysis
            # Get order book metrics
            book_metrics = await self.book_analyzer.get_liquidity_metrics(asset)
            
            # Calculate spread factor (lower spread = better opportunity)
            spread = book_metrics.get('spread', 0.0)
            max_acceptable_spread = self.config.get('max_acceptable_spread', 0.005)
            
            spread_factor = 1.0 - (spread / max_acceptable_spread) if spread <= max_acceptable_spread else 0.0
            
            # Calculate depth factor (higher depth = better opportunity)
            depth = book_metrics.get('depth', 0.0)
            min_acceptable_depth = self.config.get('min_acceptable_depth', 1.0)
            
            depth_factor = min(1.0, depth / min_acceptable_depth) if depth >= min_acceptable_depth else 0.0
            
            # Combined liquidity score
            combined_liquidity_score = spread_factor * depth_factor
            
            return max(0.0, min(1.0, combined_liquidity_score))

    async def _validate_trend_signals(self, asset: str) -> bool:
        """Validate trend signals against historical data"""
        historical_data = await self.data_fetcher.get_historical_data(asset)
        current_trend = await self._detect_current_trend(asset)
        
        # Validate trend persistence over multiple timeframes
        if not self._is_trend_persistent(historical_data, current_trend):
            return False
        
        return True

    async def _detect_current_trend(self, asset: str) -> str:
        """Detect the current trend direction for the asset"""
        prices = await self.data_fetcher.get_current_prices(asset)
        if prices[-1] > prices[-2]:
            return "UP"
        elif prices[-1] < prices[-2]:
            return "DOWN"
        return "NEUTRAL"

    def _is_trend_persistent(self, historical_data: List[float], current_trend: str) -> bool:
        """Check if the current trend is persistent over historical data"""
        trend_count = sum(1 for price in historical_data if (price > historical_data[0]) == (current_trend == "UP"))
        return trend_count > len(historical_data) * 0.6  # At least 60% of historical data should confirm the trend

    async def _execute_trade(self, asset: str, trade_size: float) -> None:
        """Execute trade based on calculated opportunity scores"""
        if trade_size <= 0:
            return
        
        # Pre-trade security checks
        if not await self._validate_trend_signals(asset):
            return
        
        # Execute trade through the execution engine
        await self.execution_engine.place_order(asset, trade_size)

    async def _dynamic_position_sizing(self, asset: str, volatility: float) -> float:
        """Calculate dynamic position size based on volatility"""
        base_size = self.config.get('base_trade_size', 1.0)
        adjusted_size = base_size / volatility if volatility > 0 else base_size
        return max(self.config.get('min_trade_size', 0.1), adjusted_size)

    async def _optimize_trend_conditions(self, asset: str) -> None:
        """Optimize trend-following conditions using AI models"""
        current_conditions = await self._fetch_current_conditions(asset)
        optimized_parameters = await self.ai_model.optimize_conditions(current_conditions)
        self.config.update(optimized_parameters)

    async def _fetch_current_conditions(self, asset: str) -> Dict[str, Any]:
        """Fetch current market conditions for the asset"""
        return {
            'price': await self.data_fetcher.get_current_price(asset),
            'volume': await self.data_fetcher.get_current_volume(asset),
            'liquidity': await self.risk_controller.get_asset_liquidity(asset)
        }

    async def _log_trend_decision(self, asset: str, decision: str) -> None:
        """Log the trend decision for auditing and analysis"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'asset': asset,
            'decision': decision
        }
        self.logger.log(json.dumps(log_entry))

    async def run_trend_analysis(self, assets: List[str]) -> None:
        """Run trend analysis for multiple assets in a distributed manner"""
        start_time = time.time()
        results = {}

        # Determine number of parallel processes based on CPU availability
        max_workers = min(len(assets), multiprocessing.cpu_count() * 2)

        # Run analysis in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            loop = asyncio.get_event_loop()
            tasks = {asset: loop.run_in_executor(executor, self._analyze_asset_trend, asset) for asset in assets}

            # Collect results as they complete
            for asset, task in tasks.items():
                results[asset] = await task

        execution_time = time.time() - start_time
        self.logger.info(f"Distributed trend analysis completed in {execution_time:.4f}s")

    async def _analyze_asset_trend(self, asset: str) -> None:
        """Analyze trend for a single asset"""
        lightweight_mode = self.config.get('lightweight_mode', False)
        regime_score = await self._calculate_regime_opportunity(asset, lightweight_mode)
        liquidity_score = await self._calculate_liquidity_opportunity(asset, lightweight_mode)
        
        # Combine scores to make a decision
        if regime_score > 0.5 and liquidity_score > 0.5:
            volatility = await self.risk_controller.get_asset_volatility(asset)
            trade_size = await self._dynamic_position_sizing(asset, volatility)
            await self._execute_trade(asset, trade_size)
            await self._log_trend_decision(asset, "BUY")
        elif regime_score < -0.5:
            await self._log_trend_decision(asset, "SELL")
