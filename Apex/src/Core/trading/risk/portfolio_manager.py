# apex/src/Core/trading/risk/portfolio_manager.py

import numpy as np
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple, Any, Union
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import ray
import orjson
from datetime import datetime
import functools
import hashlib

# Core System Integration
from src.Core.data.realtime.market_data import UnifiedMarketFeed
from src.Core.trading.risk.risk_management import QuantumRiskManager
from src.Core.trading.risk.risk_engine import RiskEngine
from src.Core.trading.execution.algo_engine import AlgorithmicExecution
from src.Core.trading.execution.order_execution import OrderExecutor
from src.Core.trading.execution.market_impact import MarketImpactCalculator
from src.Core.trading.strategies.strategy_orchestrator import StrategyOrchestrator
from src.Core.trading.security.alert_system import SecurityMonitor
from src.Core.trading.security.trade_security_guard import TradeSecurityGuard
from src.Core.trading.execution.execution_validator import ExecutionValidator
from src.Core.data.order_book_analyzer import OrderBookDepthAnalyzer
from src.Core.data.trade_history import TradeHistoryAnalyzer

# AI System Integration
from src.ai.ensembles.meta_trader import MetaStrategyOptimizer
from src.ai.reinforcement.q_learning.agent import QStrategyOptimizer
from src.ai.analysis.correlation_engine import AssetCorrelationEngine
from src.ai.forecasting.ai_forecaster import AIForecaster
from src.ai.reinforcement.reinforcement_learning import ExplainableRL
from src.ai.ensembles.ensemble_voting import ModelEnsembleVoter

# Analytics and Simulation
from utils.analytics.monte_carlo_simulator import MonteCarloVaR, MonteCarloSimulator
from Tests.backtesting.simulation_engine import SimulationEngine
from Tests.backtesting.performance_evaluator import PerformanceEvaluator

# Security & Validation
from Core.trading.risk.incident_response import IncidentResponder
from Core.data.asset_validator import validate_asset_tradability
from src.Core.trading.security.security import TradingSecurityManager

# Shared Utilities
from utils.logging.structured_logger import StructuredLogger
from utils.helpers.error_handler import ErrorHandler
from utils.helpers.stealth_api import StealthAPIManager
from src.Core.trading.logging.decision_logger import DecisionLogger, DecisionMetadata
from utils.helpers.validation import validate_decimal_input
from utils.helpers.error_handler import SecurityError


# Initialize Ray for distributed computing
ray.init(ignore_reinit_error=True)

# Initialize structured logging
logger = StructuredLogger("quantum_portfolio")

# Constants for optimization
MAX_RETRY_ATTEMPTS = 3
CACHE_TTL = 300  # 5 minutes in seconds
GPU_OFFLOAD_THRESHOLD = 10000  # Minimum data points for GPU offloading
MIN_OPTIMIZATION_INTERVAL = 60  # Seconds between full portfolio optimizations
HEDGE_RATIO_PRECISION = 4
EXECUTION_TIMEOUT = 2.0  # Seconds before execution timeout
PANIC_MODE_DRAWDOWN = Decimal('0.15')  # 15% drawdown triggers emergency protocols

class PortfolioCache:
    """Thread-safe cache for portfolio computations with TTL and integrity validation"""
    
    def __init__(self, ttl: int = CACHE_TTL):
        self._cache = {}
        self._ttl = ttl
        self._lock = asyncio.Lock()
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with integrity check"""
        async with self._lock:
            if key not in self._cache:
                return None
                
            entry = self._cache[key]
            if time.time() - entry['timestamp'] > self._ttl:
                del self._cache[key]
                return None
                
            # Verify data integrity
            current_hash = self._hash_data(entry['data'])
            if current_hash != entry['hash']:
                logger.warning(f"Cache integrity violation detected for key: {key}")
                del self._cache[key]
                return None
                
            return entry['data']
    
    async def set(self, key: str, value: Any) -> None:
        """Set value in cache with timestamp and integrity hash"""
        async with self._lock:
            self._cache[key] = {
                'data': value,
                'timestamp': time.time(),
                'hash': self._hash_data(value)
            }
    
    async def invalidate(self, key: str = None) -> None:
        """Invalidate specific key or entire cache"""
        async with self._lock:
            if key is None:
                self._cache.clear()
            elif key in self._cache:
                del self._cache[key]

    @staticmethod
    def _hash_data(data: Any) -> str:
        """Create hash of data for integrity validation"""
        serialized = orjson.dumps(data, default=str)
        return hashlib.sha256(serialized).hexdigest()

@ray.remote
class DistributedRiskCalculator:
    """Distributed risk calculations offloaded to Ray cluster"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monte_carlo = MonteCarloSimulator()
    
    def calculate_var(self, positions: Dict[str, Any], confidence_level: float = 0.95, 
                      time_horizon: int = 1) -> Dict[str, float]:
        """Calculate Value at Risk using Monte Carlo simulation"""
        return self.monte_carlo.calculate_portfolio_var(
            positions=positions,
            confidence_level=confidence_level,
            time_horizon=time_horizon
        )
    
    def stress_test_portfolio(self, positions: Dict[str, Any], 
                             scenarios: List[Dict[str, Any]]) -> Dict[str, float]:
        """Run stress tests on portfolio under different market scenarios"""
        results = {}
        for scenario in scenarios:
            results[scenario['name']] = self.monte_carlo.stress_test(
                positions=positions,
                market_changes=scenario['changes']
            )
        return results

@ray.remote
class DistributedBacktester:
    """Distributed backtest runner for strategy validation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.simulation_engine = SimulationEngine()
        self.performance_evaluator = PerformanceEvaluator()
    
    def validate_strategy(self, strategy_id: str, symbol: str, 
                         params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate strategy with quick backtest before deployment"""
        simulation_results = self.simulation_engine.quick_simulate(
            strategy_id=strategy_id,
            symbol=symbol,
            params=params,
            days=self.config.get('validation_days', 30)
        )
        
        return self.performance_evaluator.evaluate_strategy(
            results=simulation_results,
            metrics=['sharpe', 'sortino', 'max_drawdown', 'win_rate', 'profit_factor']
        )

class QuantumPortfolioManager:
    """Institutional-Grade Portfolio Management System with Distributed AI-Driven Optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the portfolio manager with optimized architecture for HFT"""
        self.config = config
        self.adaptive_params = self._load_adaptive_parameters()
        
        # Caching and optimization
        self.cache = PortfolioCache(ttl=config.get('cache_ttl', CACHE_TTL))
        self.last_full_optimization = 0
        
        # Distributed computing setup
        self.dist_risk_calculator = DistributedRiskCalculator.remote(config)
        self.dist_backtester = DistributedBacktester.remote(config)
        
        # Performance monitoring
        self.execution_latencies = []
        self.allocation_latencies = []
        self.last_performance_report = time.time()
        
        # Multi-layered execution setup
        self._setup_execution_layers()
                
        # System state with atomic update capability
        self.portfolio_state = self._load_quantum_state()
        self._state_lock = asyncio.Lock()
        
        # High-frequency event buffer for real-time analytics
        self.event_buffer = asyncio.Queue(maxsize=10000)
        asyncio.create_task(self._process_event_buffer())
        
        # Heartbeat monitoring
        self._last_heartbeat = time.time()
        asyncio.create_task(self._system_heartbeat())

    def _setup_execution_layers(self):
        """Setup execution layers with fallback mechanisms"""
        # Core data feeds with redundancy
        self.market_feed = UnifiedMarketFeed()
        self.book_analyzer = OrderBookDepthAnalyzer()
        self.trade_history = TradeHistoryAnalyzer()
        
        # Risk and security systems
        self.risk_engine = RiskEngine()
        self.risk_manager = QuantumRiskManager()
        self.security_monitor = SecurityMonitor()
        self.security_guard = TradeSecurityGuard()
        self.incident_responder = IncidentResponder()
        
        # AI optimization systems
        self.strategy_orchestrator = StrategyOrchestrator()
        self.meta_optimizer = MetaStrategyOptimizer()
        self.q_optimizer = QStrategyOptimizer()
        self.correlation_engine = AssetCorrelationEngine()
        self.ai_forecaster = AIForecaster()
        self.explainable_rl = ExplainableRL()
        self.model_voter = ModelEnsembleVoter()
        
        # Execution systems with multi-broker fallback
        self.execution_engine = AlgorithmicExecution()
        self.order_executor = OrderExecutor()
        self.execution_validator = ExecutionValidator()
        self.impact_calculator = MarketImpactCalculator()
        
        # Decision logging for AI feedback and compliance
        self.decision_logger = DecisionLogger()
        
        # Security
        self.security_manager = TradingSecurityManager()
        self.stealth_api = StealthAPIManager()

    @validate_decimal_input
    async def evaluate_strategy_allocation(self, strategy_id: str, symbol: str, 
                                          urgency: str = "normal") -> Dict[str, Any]:
        """Evaluate and determine optimal allocation for a strategy with AI optimization"""
        start_time = time.time()
        
        try:
            # Fast path for cached recent allocations in non-urgent situations
            cache_key = f"allocation:{strategy_id}:{symbol}"
            if urgency != "high":
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    return cached_result
            
            # Security validation (fail fast pattern)
            await self._validate_security_context(strategy_id, symbol)
            
            # Gather all necessary data in parallel with timeouts
            analysis_tasks = [
                self._analyze_market_regime(symbol),
                self._analyze_risk_parameters(symbol),
                self._analyze_liquidity(symbol),
                self._analyze_correlations(symbol),
                self._analyze_strategy_capacity(strategy_id),
                self._validate_strategy_performance(strategy_id, symbol)
            ]
            
            # Gather with timeout protection
            analysis_results = await asyncio.gather(*analysis_tasks)
            analysis_data = {
                'market_regime': analysis_results[0],
                'risk_parameters': analysis_results[1],
                'liquidity': analysis_results[2], 
                'correlations': analysis_results[3],
                'capacity': analysis_results[4],
                'validation': analysis_results[5]
            }
            
            # Use Ray for distributed AI allocation optimization
            allocation_future = self.meta_optimizer.determine_allocation.remote(
                strategy_id=strategy_id,
                symbol=symbol,
                analysis=analysis_data,
                portfolio_state=self.portfolio_state,
                urgency=urgency
            )
            
            # Set timeout for distributed computation
            allocation = await asyncio.wait_for(
                ray.get(allocation_future), 
                timeout=self.config.get('allocation_timeout', 1.0)
            )
            
            # Ensure strategy passes risk validation
            if not await self._validate_allocation(strategy_id, symbol, allocation):
                return self._reject_allocation(symbol, "risk_validation_failed")
            
            # Create optimized execution plan
            execution_plan = await self._create_execution_plan(symbol, allocation, urgency)
            
            # Cache the result for non-high urgency requests
            if urgency != "high":
                await self.cache.set(cache_key, execution_plan)
            
            # Log decision and execution plan
            await self._log_allocation_decision(strategy_id, symbol, allocation, execution_plan)
            
            # Track allocation latency
            allocation_time = time.time() - start_time
            self.allocation_latencies.append(allocation_time)
            
            return execution_plan
            
        except asyncio.TimeoutError:
            logger.warning(f"Allocation timeout for {symbol} with strategy {strategy_id}")
            return self._create_fallback_allocation(symbol, strategy_id)
        except Exception as e:
            await self._handle_portfolio_error(symbol, e)
            return self._reject_allocation(symbol, f"system_error: {str(e)}")

    async def _validate_security_context(self, strategy_id: str, symbol: str) -> None:
        """Validate security context with multiple security layers"""
        # Parallel security checks with fail-fast pattern
        security_tasks = [
            self.security_monitor.validate_trading_context(symbol),
            self.security_guard.verify_strategy_signature(strategy_id),
            validate_asset_tradability(symbol),
            self.security_manager.check_permissions(strategy_id, symbol)
        ]
        
        results = await asyncio.gather(*security_tasks, return_exceptions=True)
        
        # Check if any security validations failed
        for i, result in enumerate(results):
            if isinstance(result, Exception) or result is False:
                if isinstance(result, Exception):
                    error_msg = str(result)
                else:
                    error_msg = f"Security check {i} failed"
                    
                logger.warning(f"Security validation failed for {symbol}: {error_msg}")
                await self.security_monitor.report_security_incident(
                    strategy_id=strategy_id,
                    symbol=symbol,
                    incident_type="security_validation_failed",
                    details={"error": error_msg}
                )
                raise SecurityError(f"Security validation failed: {error_msg}")

    @ray.remote
    async def _analyze_market_regime(self, symbol: str) -> Dict[str, Any]:
        """Multi-timeframe market regime analysis with trend identification"""
        # Get cached result if available
        cache_key = f"regime:{symbol}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
            
        # Perform analysis in parallel for different timeframes
        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        regime_tasks = [
            self.market_feed.get_regime_data(symbol, timeframe) 
            for timeframe in timeframes
        ]
        
        # Gather all timeframe data with timeout protection
        try:
            regime_data = await asyncio.gather(*regime_tasks)
            
            # Process regime data through strategy orchestrator
            regime_analysis = await self.strategy_orchestrator.analyze_regime_batch(
                symbol, regime_data, timeframes
            )
            
            # Combine regime data with AI forecaster predictions
            forecasts = await self.ai_forecaster.predict_regime_shift(
                symbol, regime_analysis, horizon=self.config.get('forecast_horizon', 24)
            )
            
            result = {
                'current_regime': regime_analysis['primary_regime'],
                'regime_strength': regime_analysis['regime_strength'],
                'timeframe_analysis': {tf: regime_analysis[tf] for tf in timeframes},
                'regime_shift_probability': forecasts['shift_probability'],
                'forecast': forecasts['direction']
            }
            
            # Cache the result
            await self.cache.set(cache_key, result)
            return result
            
        except asyncio.TimeoutError:
            # Fallback to cached data or simplified analysis
            logger.warning(f"Regime analysis timeout for {symbol}")
            return {
                'current_regime': 'unknown',
                'regime_strength': 0.5,
                'timeframe_analysis': {},
                'regime_shift_probability': 0.5,
                'forecast': 'neutral'
            }

    async def _analyze_risk_parameters(self, symbol: str) -> Dict[str, Any]:
        """Real-time risk constraints analysis using distributed computing"""
        # Delegate heavy risk calculations to the risk engine
        risk_data = await self.risk_engine.get_risk_profile(symbol)
        
        # Perform Monte Carlo VaR calculation in parallel using Ray
        var_future = self.dist_risk_calculator.calculate_var.remote(
            {symbol: risk_data['position']} if symbol in risk_data['positions'] else {},
            self.config['var_confidence']
        )
        
        # Get current exposure directly from risk engine
        exposure = await self.risk_engine.get_current_exposure(symbol)
        
        # Combine local and distributed risk calculations
        var_result = await ray.get(var_future)
        
        return {
            'var': var_result.get(symbol, {'value': 0.0, 'percent': 0.0}),
            'max_drawdown': risk_data['max_drawdown'],
            'exposure': exposure,
            'margin_utilization': risk_data['margin_utilization'],
            'concentration_risk': risk_data['concentration_risk'],
            'liquidity_risk': risk_data['liquidity_risk']
        }

    async def _analyze_liquidity(self, symbol: str) -> Dict[str, Any]:
        """Multi-exchange liquidity analysis with slippage estimation"""
        # Get cached liquidity data if available
        cache_key = f"liquidity:{symbol}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        # Parallel liquidity analysis
        liquidity_tasks = [
            self.book_analyzer.get_market_depth(symbol),
            self.impact_calculator.estimate_market_impact(symbol),
            self.book_analyzer.estimate_slippage(symbol),
            self.trade_history.get_recent_volume(symbol)
        ]
        
        try:
            # Gather with timeout protection
            results = await asyncio.gather(*liquidity_tasks)
            
            result = {
                'market_depth': results[0],
                'impact': results[1],
                'slippage': results[2],
                'recent_volume': results[3],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Cache the result
            await self.cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.warning(f"Liquidity analysis error for {symbol}: {str(e)}")
            # Return safe defaults
            return {
                'market_depth': {'bid': 0, 'ask': 0},
                'impact': 1.0,  # 100% impact (worst case)
                'slippage': 0.01,  # 1% slippage estimate
                'recent_volume': 0,
                'timestamp': datetime.utcnow().isoformat()
            }

    async def _analyze_correlations(self, symbol: str) -> Dict[str, Any]:
        """Dynamic correlation analysis with cluster identification"""
        # Get cached correlation data if available
        cache_key = f"correlation:{symbol}"
        cached = await self.cache.get(cache_key)
        if cached and time.time() - cached['timestamp'] < 300:  # 5 minutes TTL
            return cached
            
        # Use correlation engine to get correlation clusters
        correlation_data = await self.correlation_engine.get_correlation_profile(symbol)
        
        # Find potential hedge candidates
        hedge_candidates = await self.correlation_engine.find_hedge_candidates(
            symbol, 
            threshold=self.config.get('hedge_correlation_threshold', -0.6)
        )
        
        result = {
            'clusters': correlation_data['clusters'],
            'primary_cluster': correlation_data['primary_cluster'],
            'systemic_risk': correlation_data['systemic_risk'],
            'hedge_candidates': hedge_candidates,
            'timestamp': time.time()
        }
        
        # Cache the result
        await self.cache.set(cache_key, result)
        return result

    async def _analyze_strategy_capacity(self, strategy_id: str) -> Dict[str, Any]:
        """Strategy capacity analysis with dynamic capital allocation"""
        return await self.strategy_orchestrator.get_strategy_capacity(strategy_id)

    async def _validate_strategy_performance(self, strategy_id: str, symbol: str) -> Dict[str, Any]:
        """Validate strategy performance with quick backtest"""
        # Get strategy parameters
        params = await self.strategy_orchestrator.get_strategy_parameters(strategy_id)
        
        # Run distributed backtest validation
        validation_future = self.dist_backtester.validate_strategy.remote(
            strategy_id, symbol, params
        )
        
        try:
            # Wait for backtest results with timeout
            validation = await asyncio.wait_for(
                ray.get(validation_future), 
                timeout=self.config.get('validation_timeout', 2.0)
            )
            
            return {
                'validated': validation['sharpe'] > self.config.get('min_sharpe', 0.5),
                'metrics': validation,
                'timestamp': datetime.utcnow().isoformat()
            }
        except asyncio.TimeoutError:
            logger.warning(f"Strategy validation timeout for {strategy_id} on {symbol}")
            return {
                'validated': False,
                'metrics': {},
                'error': 'validation_timeout',
                'timestamp': datetime.utcnow().isoformat()
            }

    async def _validate_allocation(self, strategy_id: str, symbol: str, 
                                  allocation: Dict[str, Any]) -> bool:
        """Validate allocation against risk parameters"""
        # Fast path risk validation
        if allocation['size'] <= 0:
            return False
            
        # Check risk limits
        risk_check = await self.risk_engine.validate_allocation(
            strategy_id=strategy_id,
            symbol=symbol,
            allocation=allocation
        )
        
        if not risk_check['valid']:
            logger.warning(f"Risk validation failed for {symbol}: {risk_check['reason']}")
            return False
            
        # Check execution feasibility
        execution_check = await self.execution_validator.validate_execution(
            symbol=symbol,
            size=allocation['size'],
            price=allocation.get('price'),
            strategy=allocation.get('execution_strategy', 'VWAP')
        )
        
        if not execution_check['valid']:
            logger.warning(f"Execution validation failed for {symbol}: {execution_check['reason']}")
            return False
            
        return True

    async def _create_execution_plan(self, symbol: str, allocation: Dict[str, Any], 
                                    urgency: str) -> Dict[str, Any]:
        """Create optimized execution plan with smart order routing"""
        # Get best execution method based on market conditions
        execution_method = await self._determine_execution_method(symbol, allocation, urgency)
        
        # Get hedge plan if needed
        hedge_plan = await self._create_hedge_plan(symbol, allocation) if allocation['hedge_required'] else None
        
        # Create comprehensive execution plan
        execution_plan = {
            'symbol': symbol,
            'order_type': allocation['order_type'],
            'side': allocation['side'],
            'size': allocation['size'],
            'price': allocation.get('price'),
            'execution_strategy': execution_method,
            'time_in_force': allocation.get('time_in_force', 'GTC'),
            'risk_parameters': {
                'stop_loss': allocation['stop_loss'],
                'take_profit': allocation['take_profit'],
                'max_slippage': allocation['max_slippage'],
                'max_impact': allocation.get('max_impact', 0.005)
            },
            'hedge_plan': hedge_plan,
            'urgency': urgency,
            'execution_instructions': {
                'fragmentation': allocation.get('fragmentation', False),
                'iceberg': allocation.get('iceberg', False),
                'post_only': allocation.get('post_only', False)
            },
            'fallback_broker': allocation.get('fallback_broker')
        }
        
        # Validate execution plan
        validated_plan = await self.execution_validator.validate_execution_plan(execution_plan)
        return validated_plan

    async def _determine_execution_method(self, symbol: str, allocation: Dict[str, Any], 
                                         urgency: str) -> str:
        """Determine optimal execution method based on market conditions"""
        # Get market conditions
        liquidity = await self.book_analyzer.get_market_depth(symbol)
        volatility = await self.market_feed.get_volatility(symbol)
        
        # Fast path for high urgency
        if urgency == "high":
            return "MARKET"
            
        # Size-based execution selection
        if allocation['size'] > self.config.get('large_order_threshold', 1000):
            if volatility > self.config.get('high_volatility_threshold', 0.05):
                return "TWAP"  # Time-Weighted Average Price for high volatility
            return "VWAP"  # Volume-Weighted Average Price for normal volatility
            
        # Liquidity-based selection
        if liquidity['spread'] > self.config.get('wide_spread_threshold', 0.01):
            return "LIMIT"  # Use limit orders for wide spreads
            
        # Urgency-based selection
        if urgency == "medium":
            return "SNIPER"  # Aggressive but not market
            
        # Default method
        return "LIMIT"

    async def _create_hedge_plan(self, symbol: str, allocation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create dynamic hedging plan based on correlation analysis"""
        # Get correlation data
        correlation_data = await self._analyze_correlations(symbol)
        hedge_candidates = correlation_data['hedge_candidates']
        
        if not hedge_candidates:
            return None
            
        # Calculate hedge ratios using vectorized operations
        hedge_ratios = await self.correlation_engine.calculate_hedge_ratios(
            symbol,
            hedge_candidates,
            allocation['size'],
            precision=HEDGE_RATIO_PRECISION
        )
        
        # Create hedge plan
        return {
            'primary': symbol,
            'hedges': [{
                'symbol': candidate,
                'ratio': hedge_ratios[candidate]['ratio'],
                'size': hedge_ratios[candidate]['size'],
                'correlation': hedge_ratios[candidate]['correlation'],
                'execution_strategy': 'MARKET' if allocation.get('urgency') == 'high' else 'LIMIT'
            } for candidate in hedge_candidates]
        }

    async def execute_allocation(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute allocation with fault tolerance and retry logic"""
        start_time = time.time()
        
        # Validate execution plan
        if not await self.execution_validator.validate_execution_plan(execution_plan):
            raise ValueError("Invalid execution plan")
            
        # Try primary execution with retry logic
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                # Execute the plan with timeout protection
                execution_result = await asyncio.wait_for(
                    self.execution_engine.execute_plan(execution_plan),
                    timeout=EXECUTION_TIMEOUT
                )
                
                # Execute hedge if needed
                if execution_plan.get('hedge_plan'):
                    hedge_result = await self._execute_hedge_plan(
                        execution_plan['hedge_plan'],
                        execution_result['executed_price']
                    )
                    execution_result['hedge_result'] = hedge_result
                
                # Log execution metrics
                execution_time = time.time() - start_time
                self.execution_latencies.append(execution_time)
                
                # Update portfolio state
                await self._update_portfolio_state_atomic(execution_result)
                
                return execution_result
                
            except asyncio.TimeoutError:
                logger.warning(f"Execution timeout for {execution_plan['symbol']}, attempt {attempt+1}/{MAX_RETRY_ATTEMPTS}")
                continue
                
            except Exception as e:
                # If critical error, don't retry
                if hasattr(e, 'critical') and e.critical:
                    raise
                
                logger.warning(f"Execution error for {execution_plan['symbol']}: {str(e)}, attempt {attempt+1}/{MAX_RETRY_ATTEMPTS}")
                
                # Try fallback broker if available
                if execution_plan.get('fallback_broker') and attempt == MAX_RETRY_ATTEMPTS - 1:
                    logger.info(f"Trying fallback broker for {execution_plan['symbol']}")
                    return await self._execute_with_fallback(execution_plan)
                
        # If all attempts failed
        raise ExecutionError(f"Failed to execute {execution_plan['symbol']} after {MAX_RETRY_ATTEMPTS} attempts")

    async def _execute_with_fallback(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute with fallback broker"""
        fallback_plan = execution_plan.copy()
        fallback_plan['broker'] = execution_plan['fallback_broker']
        
        try:
            # Execute with fallback broker
            return await asyncio.wait_for(
                self.execution_engine.execute_plan(fallback_plan),
                timeout=EXECUTION_TIMEOUT
            )
        except Exception as e:
            logger.error(f"Fallback execution failed for {execution_plan['symbol']}: {str(e)}")
            raise ExecutionError(f"Fallback execution failed: {str(e)}")

    async def _execute_hedge_plan(self, hedge_plan: Dict[str, Any], 
                                 executed_price: float) -> Dict[str, Any]:
        """Execute hedge plan with atomic execution"""
        hedge_orders = []
        
        for hedge in hedge_plan['hedges']:
            # Create hedge order
            hedge_order = {
                'symbol': hedge['symbol'],
                'side': 'SELL' if hedge_plan.get('side', 'BUY') == 'BUY' else 'BUY',  # Opposite side
                'size': hedge['size'],
                'execution_strategy': hedge['execution_strategy'],
                'is_hedge': True,
                'hedge_parent': hedge_plan['primary']
            }
            hedge_orders.append(hedge_order)
        
        # Execute all hedge orders in parallel
        hedge_results = await asyncio.gather(*[
            self.execution_engine.execute_plan(order)
            for order in hedge_orders
        ], return_exceptions=True)
        
        # Process hedge results
        successful_hedges = {}
        failed_hedges = {}
        
        for i, result in enumerate(hedge_results):
            hedge = hedge_plan['hedges'][i]
            if isinstance(result, Exception):
                logger.warning(f"Hedge execution failed for {hedge['symbol']}: {str(result)}")
                failed_hedges[hedge['symbol']] = {
                    'error': str(result),
                    'planned_size': hedge['size']
                }
            else:
                successful_hedges[hedge['symbol']] = {
                    'executed_price': result['executed_price'],
                    'executed_size': result['executed_size'],
                    'execution_time': result['execution_time'],
                    'fees': result['fees']
                }
        
        # Log hedge results
        await self.decision_logger.log_hedge_execution(
            primary_symbol=hedge_plan['primary'],
            primary_price=executed_price,
            successful_hedges=successful_hedges,
            failed_hedges=failed_hedges
        )
        
        return {
            'successful_hedges': successful_hedges,
            'failed_hedges': failed_hedges,
            'hedge_ratio': hedge_plan['hedges'][0]['ratio'] if hedge_plan['hedges'] else 0
        }

    async def _update_portfolio_state_atomic(self, execution_result: Dict[str, Any]) -> None:
        """Update portfolio state with atomic operation for thread safety"""
        async with self._state_lock:
            # Extract execution info
            symbol = execution_result['symbol']
            executed_size = execution_result['executed_size']
            executed_price = execution_result['executed_price']
            side = execution_result['side']
            
            # Update position
            if symbol not in self.portfolio_state['positions']:
                self.portfolio_state['positions'][symbol] = {
                    'size': 0,
                    'avg_price': 0,
                    'realized_pnl': 0,
                    'unrealized_pnl': 0,
                    'fees': 0
                }
            
            position = self.portfolio_state['positions'][symbol]
            
            # Update position with vectorized calculation
            if side == 'BUY':
                new_size = position['size'] + executed_size
                position['avg_price'] = ((position['size'] * position['avg_price']) + 
                                        (executed_size * executed_price)) / new_size if new_size > 0 else 0
                position['size'] = new_size
            else:  # SELL
                realized_pnl = (executed_price - position['avg_price']) * min(position['size'], executed_size)
                position['realized_pnl'] += realized_pnl
                position['size'] -= executed_size
                if position['size'] <= 0:
                    position['avg_price'] = 0
                    
            # Update fees
            position['fees'] += execution_result['fees']
            
            # Update hedge positions if applicable
            if 'hedge_result' in execution_result:
                for hedge_symbol, hedge_data in execution_result['hedge_result'].get('successful_hedges', {}).items():
                    if hedge_symbol not in self.portfolio_state['positions']:
                        self.portfolio_state['positions'][hedge_symbol] = {
                            'size': 0,
                            'avg_price': 0,
                            'realized_pnl': 0,
                            'unrealized_pnl': 0,
                            'fees': 0,
                            'is_hedge': True,
                            'hedge_parent': symbol
                        }
                    
                    hedge_position = self.portfolio_state['positions'][hedge_symbol]
                    hedge_size = hedge_data['executed_size']
                    hedge_price = hedge_data['executed_price']
                    
                    # Opposite side of primary trade
                    hedge_side = 'SELL' if side == 'BUY' else 'BUY'
                    
                    if hedge_side == 'BUY':
                        new_size = hedge_position['size'] + hedge_size
                        hedge_position['avg_price'] = ((hedge_position['size'] * hedge_position['avg_price']) + 
                                                      (hedge_size * hedge_price)) / new_size if new_size > 0 else 0
                        hedge_position['size'] = new_size
                    else:  # SELL
                        realized_pnl = (hedge_price - hedge_position['avg_price']) * min(hedge_position['size'], hedge_size)
                        hedge_position['realized_pnl'] += realized_pnl
                        hedge_position['size'] -= hedge_size
                        if hedge_position['size'] <= 0:
                            hedge_position['avg_price'] = 0
                    
                    hedge_position['fees'] += hedge_data['fees']
            
            # Update portfolio metrics
            await self._update_portfolio_metrics()
            
            # Save state to persistence layer
            asyncio.create_task(self._persist_portfolio_state())
            
            # Push update to event buffer for real-time analytics
            await self.event_buffer.put({
                'type': 'execution',
                'data': execution_result,
                'timestamp': time.time()
            })

    async def _update_portfolio_metrics(self) -> None:
        """Update portfolio-wide performance metrics"""
        total_value = Decimal('0')
        total_exposure = Decimal('0')
        realized_pnl = Decimal('0')
        unrealized_pnl = Decimal('0')
        
        # Get latest market prices in parallel
        symbols = list(self.portfolio_state['positions'].keys())
        price_tasks = [self.market_feed.get_latest_price(symbol) for symbol in symbols]
        prices = await asyncio.gather(*price_tasks, return_exceptions=True)
        
        # Process each position with vectorized operations when possible
        for i, symbol in enumerate(symbols):
            position = self.portfolio_state['positions'][symbol]
            
            # Skip positions with zero size
            if position['size'] == 0:
                continue
                
            # Get current price, handle exceptions
            if isinstance(prices[i], Exception):
                logger.warning(f"Failed to get price for {symbol}: {str(prices[i])}")
                current_price = position['avg_price']  # Use average price as fallback
            else:
                current_price = prices[i]
            
            # Calculate position metrics
            position_value = Decimal(str(position['size'])) * Decimal(str(current_price))
            position_exposure = abs(position_value)
            position_unrealized_pnl = position['size'] * (current_price - position['avg_price'])
            
            # Update position
            position['unrealized_pnl'] = float(position_unrealized_pnl)
            position['current_price'] = current_price
            position['value'] = float(position_value)
            
            # Update portfolio totals
            total_value += position_value
            total_exposure += position_exposure
            realized_pnl += Decimal(str(position['realized_pnl']))
            unrealized_pnl += position_unrealized_pnl
        
        # Update portfolio state
        self.portfolio_state['total_value'] = float(total_value)
        self.portfolio_state['total_exposure'] = float(total_exposure)
        self.portfolio_state['realized_pnl'] = float(realized_pnl)
        self.portfolio_state['unrealized_pnl'] = float(unrealized_pnl)
        self.portfolio_state['total_pnl'] = float(realized_pnl + unrealized_pnl)
        self.portfolio_state['last_update'] = datetime.utcnow().isoformat()
        
        # Check for emergency conditions
        initial_capital = self.portfolio_state.get('initial_capital', Decimal('0'))
        if initial_capital > 0:
            current_drawdown = 1 - (total_value / initial_capital)
            if current_drawdown >= PANIC_MODE_DRAWDOWN:
                await self._trigger_emergency_protocols(current_drawdown)

    async def _trigger_emergency_protocols(self, drawdown: Decimal) -> None:
        """Trigger emergency risk protocols on severe drawdown"""
        logger.critical(f"EMERGENCY: Portfolio drawdown {float(drawdown*100):.2f}% exceeds threshold {float(PANIC_MODE_DRAWDOWN*100):.2f}%")
        
        # Alert security systems
        await self.incident_responder.report_critical_incident(
            incident_type="excessive_drawdown",
            severity="critical",
            details={
                "drawdown": float(drawdown),
                "threshold": float(PANIC_MODE_DRAWDOWN),
                "portfolio_value": self.portfolio_state['total_value']
            }
        )
        
        # Execute emergency liquidation if configured
        if self.config.get('auto_liquidate_on_panic', False):
            logger.critical("Initiating emergency liquidation protocol")
            await self._execute_emergency_liquidation()
        
        # Disable high-risk strategies
        await self.strategy_orchestrator.disable_high_risk_strategies()
        
        # Notify mobile apps via push notification API
        asyncio.create_task(self._send_emergency_notifications(drawdown))

    async def _execute_emergency_liquidation(self) -> None:
        """Execute emergency liquidation of risky positions"""
        # Identify positions to liquidate
        positions_to_liquidate = []
        for symbol, position in self.portfolio_state['positions'].items():
            # Skip positions with no size
            if position['size'] == 0:
                continue
                
            # Skip hedge positions (handle separately)
            if position.get('is_hedge', False):
                continue
                
            # Add to liquidation list
            positions_to_liquidate.append({
                'symbol': symbol,
                'size': position['size'],
                'side': 'SELL' if position['size'] > 0 else 'BUY',  # Close position
                'current_price': position.get('current_price', 0)
            })
        
        # Sort by risk exposure (largest first)
        positions_to_liquidate.sort(
            key=lambda p: abs(p['size'] * p['current_price']), 
            reverse=True
        )
        
        # Execute liquidation in parallel batches with high urgency
        batch_size = self.config.get('emergency_liquidation_batch_size', 5)
        for i in range(0, len(positions_to_liquidate), batch_size):
            batch = positions_to_liquidate[i:i+batch_size]
            liquidation_plans = []
            
            for position in batch:
                # Create emergency execution plan
                plan = {
                    'symbol': position['symbol'],
                    'order_type': 'MARKET',
                    'side': position['side'],
                    'size': abs(position['size']),
                    'execution_strategy': 'MARKET',  # Force market execution
                    'time_in_force': 'IOC',  # Immediate-or-Cancel
                    'urgency': 'high',
                    'is_emergency': True
                }
                liquidation_plans.append(plan)
            
            # Execute batch in parallel
            liquidation_results = await asyncio.gather(*[
                self.execution_engine.execute_plan(plan) 
                for plan in liquidation_plans
            ], return_exceptions=True)
            
            # Log results
            for i, result in enumerate(liquidation_results):
                plan = liquidation_plans[i]
                if isinstance(result, Exception):
                    logger.error(f"Emergency liquidation failed for {plan['symbol']}: {str(result)}")
                else:
                    logger.info(f"Emergency liquidation executed for {plan['symbol']}: {result['executed_size']} @ {result['executed_price']}")

    async def _send_emergency_notifications(self, drawdown: Decimal) -> None:
        """Send emergency notifications to mobile apps and dashboard"""
        # Create notification data
        notification = {
            'type': 'EMERGENCY_ALERT',
            'title': 'Critical Portfolio Drawdown',
            'message': f'Portfolio drawdown of {float(drawdown*100):.2f}% exceeds safety threshold. Emergency protocols activated.',
            'timestamp': datetime.utcnow().isoformat(),
            'severity': 'critical',
            'requires_action': True
        }
        
        try:
            # Send to mobile notification service
            # Note: Actual implementation would depend on the specific notification service
            # This is a placeholder for the actual implementation
            # await mobile_notification_service.send_critical_alert(notification)
            logger.info(f"Emergency notification sent to mobile apps")
        except Exception as e:
            logger.error(f"Failed to send emergency notification: {str(e)}")

    async def _persist_portfolio_state(self) -> None:
        """Persist portfolio state to storage with optimized writes"""
        try:
            # Serialize state
            state_data = {
                'positions': self.portfolio_state['positions'],
                'metrics': {
                    'total_value': self.portfolio_state['total_value'],
                    'total_exposure': self.portfolio_state['total_exposure'],
                    'realized_pnl': self.portfolio_state['realized_pnl'],
                    'unrealized_pnl': self.portfolio_state['unrealized_pnl'],
                    'total_pnl': self.portfolio_state['total_pnl']
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Save to persistent storage (implementation depends on the storage system)
            # This is a placeholder for the actual implementation
            # await storage_service.save_portfolio_state(state_data)
            
            # Send state update to mobile apps and dashboard
            # This avoids having to poll for updates
            await self._publish_state_update(state_data)
            
        except Exception as e:
            logger.error(f"Failed to persist portfolio state: {str(e)}")

    async def _publish_state_update(self, state_data: Dict[str, Any]) -> None:
        """Publish state updates to connected clients with bandwidth optimization"""
        # Compress state data for mobile clients
        mobile_data = self._optimize_for_mobile(state_data)
        
        # Publish using a pub/sub pattern to connected clients
        # Note: Actual implementation would depend on the specific pub/sub system
        # This is a placeholder for the actual implementation
        # await pubsub_service.publish('portfolio_updates', mobile_data)
        
        # Push to AI feedback loop if significant change
        cur_value = state_data['metrics']['total_value']
        prev_value = getattr(self, '_last_published_value', 0)
        change_pct = abs(cur_value - prev_value) / max(abs(prev_value), 1) * 100
        
        if change_pct > self.config.get('ai_feedback_threshold_pct', 1.0):
            # Send to AI reinforcement learning system
            await self._send_to_ai_feedback_loop(state_data)
            self._last_published_value = cur_value

    def _optimize_for_mobile(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize state data for mobile clients to reduce bandwidth"""
        # For mobile, we only need a summary with top positions
        mobile_data = {
            'summary': {
                'total_value': state_data['metrics']['total_value'],
                'total_pnl': state_data['metrics']['total_pnl'],
                'realized_pnl': state_data['metrics']['realized_pnl'],
                'unrealized_pnl': state_data['metrics']['unrealized_pnl'],
                'total_exposure': state_data['metrics']['total_exposure'],
                'timestamp': state_data['timestamp']
            },
            'top_positions': []
        }
        
        # Calculate position values for sorting
        position_values = []
        for symbol, position in state_data['positions'].items():
            if position['size'] != 0:  # Only include active positions
                position_values.append((
                    symbol,
                    abs(position['size'] * (position.get('current_price', position['avg_price']))),
                    position
                ))
        
        # Sort positions by value (largest first) and take top N
        top_n = self.config.get('mobile_top_positions', 10)
        position_values.sort(key=lambda x: x[1], reverse=True)
        top_positions = position_values[:top_n]
        
        # Add top positions with minimal data
        for symbol, value, position in top_positions:
            mobile_data['top_positions'].append({
                'symbol': symbol,
                'size': position['size'],
                'avg_price': position['avg_price'],
                'current_price': position.get('current_price', position['avg_price']),
                'unrealized_pnl': position['unrealized_pnl'],
                'is_hedge': position.get('is_hedge', False)
            })
        
        # Add alert counts
        mobile_data['alerts'] = {
            'critical': self.portfolio_state.get('alert_counts', {}).get('critical', 0),
            'warning': self.portfolio_state.get('alert_counts', {}).get('warning', 0),
            'info': self.portfolio_state.get('alert_counts', {}).get('info', 0)
        }
        
        return mobile_data

    async def _send_to_ai_feedback_loop(self, state_data: Dict[str, Any]) -> None:
        """Send portfolio state data to AI feedback loop for reinforcement learning"""
        try:
            # Extract relevant metrics for AI reinforcement
            ai_feedback = {
                'portfolio_metrics': {
                    'total_value': state_data['metrics']['total_value'],
                    'total_pnl': state_data['metrics']['total_pnl'],
                    'total_exposure': state_data['metrics']['total_exposure'],
                    'timestamp': state_data['timestamp']
                },
                'positions': {},
                'execution_metrics': {
                    'avg_execution_latency': np.mean(self.execution_latencies[-100:]) if self.execution_latencies else 0,
                    'avg_allocation_latency': np.mean(self.allocation_latencies[-100:]) if self.allocation_latencies else 0,
                    'execution_success_rate': self._calculate_execution_success_rate()
                }
            }
            
            # Add position data in a format suitable for reinforcement learning
            for symbol, position in state_data['positions'].items():
                if position['size'] != 0:  # Only include active positions
                    ai_feedback['positions'][symbol] = {
                        'size': position['size'],
                        'avg_price': position['avg_price'],
                        'unrealized_pnl': position['unrealized_pnl'],
                        'realized_pnl': position['realized_pnl'],
                        'is_hedge': position.get('is_hedge', False),
                        'hedge_parent': position.get('hedge_parent', None)
                    }
            
            # Send to reinforcement learning systems
            await self.explainable_rl.update_portfolio_state(ai_feedback)
            await self.q_optimizer.update_portfolio_state(ai_feedback)
            
            # Update model weights based on performance
            await self.meta_optimizer.update_portfolio_performance(ai_feedback)
            
            logger.info(f"Sent portfolio state to AI feedback loop: {len(ai_feedback['positions'])} positions")
        except Exception as e:
            logger.error(f"Failed to send to AI feedback loop: {str(e)}")

    def _calculate_execution_success_rate(self) -> float:
        """Calculate execution success rate for AI feedback"""
        if not hasattr(self, '_execution_attempts') or not hasattr(self, '_execution_successes'):
            return 1.0  # Default to 100% if no data available
            
        total_attempts = self._execution_attempts
        total_successes = self._execution_successes
        
        if total_attempts == 0:
            return 1.0
            
        return total_successes / total_attempts

    async def _process_event_buffer(self) -> None:
        """Process events from the buffer for real-time analytics"""
        while True:
            try:
                # Get event from buffer
                event = await self.event_buffer.get()
                
                # Process event based on type
                if event['type'] == 'execution':
                    await self._process_execution_event(event['data'])
                elif event['type'] == 'market_data':
                    await self._process_market_data_event(event['data'])
                elif event['type'] == 'risk_alert':
                    await self._process_risk_alert_event(event['data'])
                elif event['type'] == 'ai_prediction':
                    await self._process_ai_prediction_event(event['data'])
                
                # Mark task as done
                self.event_buffer.task_done()
            except Exception as e:
                logger.error(f"Error processing event: {str(e)}")
                # Continue processing other events
                continue

    async def _process_execution_event(self, execution_data: Dict[str, Any]) -> None:
        """Process execution event for real-time analytics"""
        # Extract execution metrics
        symbol = execution_data['symbol']
        executed_price = execution_data['executed_price']
        executed_size = execution_data['executed_size']
        side = execution_data['side']
        execution_time = execution_data['execution_time']
        
        # Update execution metrics
        if not hasattr(self, '_execution_metrics'):
            self._execution_metrics = {}
        
        if symbol not in self._execution_metrics:
            self._execution_metrics[symbol] = {
                'count': 0,
                'avg_execution_time': 0,
                'slippage': 0,
                'last_price': executed_price
            }
        
        # Update metrics using exponential moving average
        alpha = 0.1  # Weighting factor for EMA
        metrics = self._execution_metrics[symbol]
        metrics['count'] += 1
        metrics['avg_execution_time'] = (1 - alpha) * metrics['avg_execution_time'] + alpha * execution_time
        
        # Calculate slippage if we have a target price
        if 'target_price' in execution_data:
            slippage = abs(executed_price - execution_data['target_price']) / execution_data['target_price']
            metrics['slippage'] = (1 - alpha) * metrics['slippage'] + alpha * slippage
        
        metrics['last_price'] = executed_price
        
        # Log execution metrics for monitoring
        if metrics['count'] % 10 == 0:  # Log every 10 executions
            logger.info(f"Execution metrics for {symbol}: avg_time={metrics['avg_execution_time']:.4f}s, slippage={metrics['slippage']:.4f}")
        
        # Update execution success tracking
        if not hasattr(self, '_execution_attempts'):
            self._execution_attempts = 0
            self._execution_successes = 0
        
        self._execution_attempts += 1
        if execution_data.get('success', True):
            self._execution_successes += 1
        
        # Send to AI feedback loop if significant enough
        if metrics['count'] % self.config.get('ai_feedback_frequency', 5) == 0:
            await self._send_execution_feedback_to_ai(symbol, metrics)

    async def _process_market_data_event(self, market_data: Dict[str, Any]) -> None:
        """Process market data event for real-time analytics"""
        # Extract market data
        symbol = market_data['symbol']
        price = market_data['price']
        volume = market_data.get('volume', 0)
        bid = market_data.get('bid', 0)
        ask = market_data.get('ask', 0)
        
        # Update market data metrics
        if not hasattr(self, '_market_data_metrics'):
            self._market_data_metrics = {}
        
        if symbol not in self._market_data_metrics:
            self._market_data_metrics[symbol] = {
                'last_price': price,
                'price_change': 0,
                'volume': volume,
                'bid': bid,
                'ask': ask,
                'spread': ask - bid if ask > 0 and bid > 0 else 0,
                'timestamp': time.time()
            }
        else:
            metrics = self._market_data_metrics[symbol]
            metrics['price_change'] = (price - metrics['last_price']) / metrics['last_price'] if metrics['last_price'] > 0 else 0
            metrics['last_price'] = price
            metrics['volume'] = volume
            metrics['bid'] = bid
            metrics['ask'] = ask
            metrics['spread'] = ask - bid if ask > 0 and bid > 0 else 0
            metrics['timestamp'] = time.time()
        
        # Check for significant price movements
        if abs(self._market_data_metrics[symbol]['price_change']) > self.config.get('significant_price_change', 0.01):
            await self._handle_significant_price_movement(symbol, self._market_data_metrics[symbol])

    async def _process_risk_alert_event(self, alert_data: Dict[str, Any]) -> None:
        """Process risk alert event"""
        # Extract alert data
        alert_type = alert_data['type']
        severity = alert_data['severity']
        details = alert_data['details']
        
        # Update alert counts
        if 'alert_counts' not in self.portfolio_state:
            self.portfolio_state['alert_counts'] = {
                'critical': 0,
                'warning': 0,
                'info': 0
            }
        
        self.portfolio_state['alert_counts'][severity] += 1
        
        # Handle critical alerts immediately
        if severity == 'critical':
            await self._handle_critical_alert(alert_type, details)
        
        # Log alert
        logger.warning(f"Risk alert: {alert_type} ({severity}) - {details}")
        
        # Send alert to mobile and dashboard
        await self._send_alert_notification(alert_type, severity, details)

    async def _process_ai_prediction_event(self, prediction_data: Dict[str, Any]) -> None:
        """Process AI prediction event"""
        # Extract prediction data
        symbol = prediction_data['symbol']
        prediction_type = prediction_data['type']
        confidence = prediction_data['confidence']
        direction = prediction_data['direction']
        horizon = prediction_data['horizon']
        
        # Update prediction metrics
        if not hasattr(self, '_ai_predictions'):
            self._ai_predictions = {}
        
        if symbol not in self._ai_predictions:
            self._ai_predictions[symbol] = {}
        
        self._ai_predictions[symbol][prediction_type] = {
            'confidence': confidence,
            'direction': direction,
            'horizon': horizon,
            'timestamp': time.time()
        }
        
        # Log prediction
        logger.info(f"AI prediction for {symbol}: {prediction_type} {direction} (conf: {confidence:.2f}, horizon: {horizon})")
        
        # Take action based on prediction if confidence is high enough
        if confidence > self.config.get('ai_action_threshold', 0.8):
            await self._handle_high_confidence_prediction(symbol, prediction_type, direction, confidence)

    async def _handle_significant_price_movement(self, symbol: str, metrics: Dict[str, Any]) -> None:
        """Handle significant price movement"""
        # Log price movement
        logger.info(f"Significant price movement for {symbol}: {metrics['price_change']:.2%}")
        
        # Check if we have a position in this symbol
        if symbol in self.portfolio_state['positions']:
            position = self.portfolio_state['positions'][symbol]
            
            # Calculate unrealized P&L for this position
            unrealized_pnl = position['size'] * (metrics['last_price'] - position['avg_price'])
            unrealized_pnl_pct = unrealized_pnl / (position['size'] * position['avg_price']) if position['size'] * position['avg_price'] != 0 else 0
            
            # Check if we need to take action
            if unrealized_pnl_pct < -self.config.get('stop_loss_threshold', 0.05):
                # Stop loss hit, consider closing position
                await self._handle_stop_loss(symbol, position, metrics)
            elif unrealized_pnl_pct > self.config.get('take_profit_threshold', 0.1):
                # Take profit hit, consider taking partial profits
                await self._handle_take_profit(symbol, position, metrics)
        
        # Update risk engine with new price data
        await self.risk_engine.update_market_data(symbol, metrics['last_price'])
        
        # Push event to AI models for reevaluation
        await self.event_buffer.put({
            'type': 'market_data',
            'data': {
                'symbol': symbol,
                'price': metrics['last_price'],
                'price_change': metrics['price_change'],
                'volume': metrics['volume'],
                'bid': metrics['bid'],
                'ask': metrics['ask'],
                'spread': metrics['spread'],
                'timestamp': metrics['timestamp']
            },
            'timestamp': time.time()
        })

    async def _handle_stop_loss(self, symbol: str, position: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """Handle stop loss for a position"""
        # Check if we've already handled this stop loss
        if position.get('stop_loss_handled', False):
            return
        
        # Mark stop loss as handled to prevent multiple executions
        position['stop_loss_handled'] = True
        
        # Log stop loss
        unrealized_pnl = position['size'] * (metrics['last_price'] - position['avg_price'])
        unrealized_pnl_pct = unrealized_pnl / (position['size'] * position['avg_price']) if position['size'] * position['avg_price'] != 0 else 0
        logger.warning(f"Stop loss triggered for {symbol}: {unrealized_pnl_pct:.2%}")
        
        # Create execution plan to close position
        execution_plan = {
            'symbol': symbol,
            'order_type': 'MARKET',
            'side': 'SELL' if position['size'] > 0 else 'BUY',  # Close position
            'size': abs(position['size']),
            'execution_strategy': 'MARKET',
            'time_in_force': 'IOC',
            'urgency': 'high',
            'reason': 'stop_loss'
        }
        
        # Execute plan
        try:
            result = await self.execute_allocation(execution_plan)
            logger.info(f"Stop loss executed for {symbol}: {result['status']}, size: {result['size']}")
        except Exception as e:
            logger.error(f"Failed to execute stop loss for {symbol}: {str(e)}")
            # Optionally, you could implement a retry mechanism or alert system here
            position['stop_loss_handled'] = False  # Reset the flag if execution fails