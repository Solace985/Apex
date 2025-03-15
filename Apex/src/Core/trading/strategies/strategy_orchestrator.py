# src/Core/trading/strategies/strategy_orchestrator.py

import asyncio
import hashlib
import hmac
import logging
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Lock, RLock
import functools
import traceback
from contextlib import asynccontextmanager

# Core System Imports
from src.Core.trading.execution.broker_manager import BrokerRouter
from src.Core.trading.execution.order_execution import StealthOrderExecutor
from src.Core.trading.execution.market_impact import ImpactPredictor
from src.Core.trading.execution.slippage_calculator import SlippageEstimator
from src.Core.trading.risk.risk_management import AdaptiveRiskController
from src.Core.trading.risk.portfolio_manager import PortfolioBalancer
from src.Core.trading.risk.risk_registry import RiskRegistry
from src.Core.trading.strategies.strategy_selector import AIStrategySelector
from src.Core.trading.strategies.strategy_evaluator import StrategyPerformanceEvaluator
from src.Core.trading.hft.liquidity_manager import LiquidityOptimizer
from src.Core.trading.hft.hft import LatencyOptimizer
from src.Core.trading.security.blacklist import StrategyBlacklist
from src.Core.trading.security.security import SecurityManager
from src.Core.trading.security.order_obfuscator import StealthManager
from src.Core.trading.logging.decision_logger import DecisionLogger
from src.Core.trading.logging.logger import TradingLogger
from src.Core.data.realtime.market_data import UnifiedMarketFeed
from src.Core.data.realtime.websocket_manager import WebsocketManager
from src.Core.data.trade_monitor import TradeMonitor
from src.ai.ensembles.meta_trader import MetaStrategyOptimizer
from src.ai.ensembles.ensemble_voting import EnsembleVoter
from src.ai.reinforcement.maddpg_model import StrategyAdaptationLearner
from src.ai.reinforcement.q_learning.agent import QAgent
from src.ai.analysis.market_regime_classifier import RegimeDetector
from Tests.backtesting.backtest_runner import BacktestEngine
from utils.logging.structured_logger import QuantumLogger
from utils.logging.telegram_alerts import AlertSystem
from utils.helpers.error_handler import CriticalSystemGuard
from utils.analytics.monte_carlo_simulator import AdversarialTester
from metrics.performance_metrics import StrategyBenchmark

# Performance optimization
from functools import lru_cache
from cachetools import TTLCache, cached

class ExecutionPriority:
    """Constants for execution priority levels with clear semantics"""
    CRITICAL = 0  # Risk-mitigation, liquidation prevention strategies
    HIGH = 1      # HFT and time-sensitive strategies
    STANDARD = 2  # Regular trading strategies
    LOW = 3       # Background tasks, non-urgent operations

class QuantumStrategyOrchestrator:
    """
    Institutional-Grade Strategy Orchestration Engine with AI-Driven Execution
    
    Coordinates multiple trading strategies with priority-based execution,
    adaptive risk management, and multi-layered security validation.
    
    Core responsibilities:
    1. Orchestrate strategy execution with priority queuing
    2. Validate strategies against risk parameters
    3. Optimize execution based on market liquidity
    4. Provide feedback to AI systems for continuous improvement
    """
    
    # Class-level constants for optimization and configuration
    MAX_THREADS = 24
    HFT_SLICE_THRESHOLD = 0.05  # % of liquidity that triggers slicing
    MARKET_REFRESH_RATE_MS = 10  # Market data refresh in milliseconds (100Hz)
    EXECUTION_TIMEOUT_MS = 50    # Maximum execution time in milliseconds
    CACHE_SIZE = 10000           # Size of memoization caches
    CACHE_TTL = 30               # Cache lifetime in seconds
    
    def __init__(self, asset_universe: List[str], config: Optional[Dict] = None):
        """
        Initialize the QuantumStrategyOrchestrator with the specified asset universe
        
        Args:
            asset_universe: List of tradable assets
            config: Optional configuration dictionary to override defaults
        """
        self.asset_universe = asset_universe
        self.config = config or {}
        
        # Runtime optimization settings
        self._market_data_cache = TTLCache(maxsize=self.CACHE_SIZE, ttl=self.CACHE_TTL)
        self._signal_cache = TTLCache(maxsize=self.CACHE_SIZE, ttl=5)  # 5 second signal cache
        
        # Initialize execution state
        self._init_runtime_environment()
        self._init_components()
        self._init_execution_state()
        
        # Operational flags
        self.running = False
        self.emergency_mode = False
        self.last_heartbeat = time.time()
        
        # Metrics tracking
        self._latency_metrics = {
            'validation': [],
            'execution': [],
            'feedback': []
        }
        
        # Initialize execution queues with multiple priority levels
        self._initialize_execution_queues()

    def _init_runtime_environment(self):
        """Initialize optimized runtime environment for parallel execution"""
        # Thread pools for CPU-bound tasks
        self._executor = ThreadPoolExecutor(max_workers=self.MAX_THREADS)
        
        # Process pool for heavy computations that benefit from true parallelism
        self._process_pool = ProcessPoolExecutor(max_workers=max(4, self.MAX_THREADS // 2))
        
        # Execution locks for thread safety and synchronized access
        self._priority_lock = RLock()
        self._execution_lock = RLock()
        self._data_lock = RLock()
        
        # Performance settings
        self._event_loop = asyncio.get_event_loop()
        self._event_loop.set_default_executor(self._executor)
        
        # Enable uvloop if available for performance boost
        try:
            import uvloop
            uvloop.install()
        except ImportError:
            pass

    def _init_components(self):
        """Initialize integrated system components with optimal connection paths"""
        # Core Trading Components
        self.market_feed = UnifiedMarketFeed(refresh_rate_ms=self.MARKET_REFRESH_RATE_MS)
        self.broker_router = BrokerRouter()
        self.order_executor = StealthOrderExecutor()
        self.trade_monitor = TradeMonitor()
        
        # Risk Management Components
        self.risk_controller = AdaptiveRiskController()
        self.portfolio_balancer = PortfolioBalancer()
        self.risk_registry = RiskRegistry()
        
        # Strategy Components
        self.strategy_selector = AIStrategySelector()
        self.strategy_evaluator = StrategyPerformanceEvaluator()
        
        # Market Structure Components
        self.liquidity_optimizer = LiquidityOptimizer()
        self.latency_optimizer = LatencyOptimizer()
        self.impact_predictor = ImpactPredictor()
        self.slippage_estimator = SlippageEstimator()
        
        # Security Components
        self.blacklist = StrategyBlacklist()
        self.security_manager = SecurityManager()
        self.stealth_manager = StealthManager()
        
        # AI/ML Components
        self.meta_optimizer = MetaStrategyOptimizer()
        self.ensemble_voter = EnsembleVoter()
        self.adaptation_learner = StrategyAdaptationLearner()
        self.q_agent = QAgent()
        self.regime_detector = RegimeDetector()
        
        # Testing & Validation Components
        self.backtest_engine = BacktestEngine()
        self.adversarial_tester = AdversarialTester()
        
        # Benchmarking Components
        self.benchmark = StrategyBenchmark()
        
        # Logging & Monitoring Components
        self.structured_logger = QuantumLogger("quantum_orchestrator")
        self.decision_logger = DecisionLogger()
        self.trading_logger = TradingLogger()
        self.alert_system = AlertSystem()
        
        # System Safety Components
        self.system_guard = CriticalSystemGuard()
        
        # Initialize security subsystem
        self._init_security()

    def _init_security(self):
        """Initialize cryptographic security components and key management"""
        from src.Core.trading.security.secure_channel import MessageSigner
        self.signer = MessageSigner()
        self.execution_counter = 0
        self.nonce_registry = set()  # Track used nonces to prevent replay attacks

    def _init_execution_state(self):
        """Initialize execution state tracking and metrics collection"""
        self.active_strategies = set()
        self.pending_executions = {}
        self.execution_history = TTLCache(maxsize=10000, ttl=3600)  # 1-hour history
        
        # Strategy performance metrics
        self.strategy_metrics = {}
        
        # Market regimes and current conditions
        self.current_regime = "neutral"
        self.current_volatility = 0.0
        self.last_regime_update = 0.0

    def _initialize_execution_queues(self):
        """Initialize priority-based execution queues with more granularity"""
        self.execution_queues = {
            ExecutionPriority.CRITICAL: asyncio.PriorityQueue(),  # Risk-critical strategies
            ExecutionPriority.HIGH: asyncio.PriorityQueue(),      # High-frequency strategies
            ExecutionPriority.STANDARD: asyncio.PriorityQueue(),  # Regular strategies
            ExecutionPriority.LOW: asyncio.PriorityQueue()        # Background tasks
        }
        
        # Sequence counters for maintaining FIFO within same priority
        self.priority_counters = {p: 0 for p in self.execution_queues.keys()}

    async def start(self):
        """Start the orchestration engine with proper initialization sequence"""
        if self.running:
            return
            
        self.running = True
        
        # Initialize dependent subsystems in order
        await self._initialize_subsystems()
        
        # Start the execution loops in parallel
        execution_tasks = [
            self._event_loop.create_task(self.orchestrate_strategies()),
            self._event_loop.create_task(self._monitor_system_health()),
            self._event_loop.create_task(self._update_market_models()),
        ]
        
        # Report system startup
        self.structured_logger.info(
            "QuantumStrategyOrchestrator started",
            asset_count=len(self.asset_universe),
            config=self.config
        )
        
        return execution_tasks

    async def stop(self):
        """Gracefully stop the orchestration engine with proper shutdown sequence"""
        if not self.running:
            return
            
        self.running = False
        
        # Cancel all pending orders first
        await self.order_executor.cancel_all_pending()
        
        # Shut down executors
        self._executor.shutdown(wait=True)
        self._process_pool.shutdown(wait=True)
        
        self.structured_logger.info("QuantumStrategyOrchestrator stopped gracefully")

    async def _initialize_subsystems(self):
        """Initialize all dependent subsystems in correct order"""
        # Initialize market data first
        await self.market_feed.initialize(self.asset_universe)
        
        # Initialize execution systems
        await asyncio.gather(
            self.broker_router.initialize(),
            self.liquidity_optimizer.initialize(self.asset_universe),
            self.risk_controller.initialize(self.asset_universe),
        )
        
        # Initialize AI models and strategy systems
        await asyncio.gather(
            self.strategy_selector.initialize(),
            self.meta_optimizer.initialize(self.asset_universe),
            self.adaptation_learner.initialize(),
            self.regime_detector.initialize()
        )
        
        # Warm up caches and predictive models
        await self._warm_up_system()

    async def _warm_up_system(self):
        """Pre-load caches and models to minimize latency during live trading"""
        await asyncio.gather(
            self._preload_market_data(),
            self._calibrate_latency_baseline(),
            self._precompute_strategy_metrics()
        )

    async def _preload_market_data(self):
        """Pre-load market data for the entire asset universe"""
        for asset in self.asset_universe:
            await self.market_feed.get_market_data(asset)
            
        self.structured_logger.info(
            "Market data preloaded",
            asset_count=len(self.asset_universe)
        )

    async def _calibrate_latency_baseline(self):
        """Calibrate baseline latency metrics for performance monitoring"""
        latency_samples = []
        for _ in range(10):
            start = time.time()
            await self.market_feed.get_market_data(self.asset_universe[0])
            latency_samples.append((time.time() - start) * 1000)  # Convert to ms
            
        # Store baseline metrics
        self.baseline_latency = {
            'mean': np.mean(latency_samples),
            'median': np.median(latency_samples),
            'p95': np.percentile(latency_samples, 95),
            'min': min(latency_samples),
            'max': max(latency_samples)
        }
        
        self.structured_logger.info(
            "Latency baseline calibrated",
            baseline_latency=self.baseline_latency
        )

    async def _precompute_strategy_metrics(self):
        """Pre-compute strategy performance metrics for quick lookup"""
        # This will be populated as strategies run
        pass

    async def orchestrate_strategies(self):
        """
        Main orchestration loop with priority-based execution
        
        This core loop handles all strategy orchestration, prioritizing execution
        based on strategy importance and time sensitivity.
        """
        while self.running:
            try:
                # Process execution queues in parallel with priority
                await self._process_execution_queues()
                
                # Update system state - this happens after executions to incorporate latest data
                if not self.emergency_mode:
                    await self._update_system_state()
                
                # Adaptive sleep based on market conditions
                sleep_time = await self._calculate_optimal_sleep()
                await asyncio.sleep(sleep_time)
                
                # Update heartbeat
                self.last_heartbeat = time.time()
                
            except Exception as e:
                error_msg = f"Orchestration error: {str(e)}, {traceback.format_exc()}"
                self.structured_logger.error(error_msg)
                await self._activate_failsafe(e)

    async def _calculate_optimal_sleep(self) -> float:
        """
        Calculate optimal sleep time based on market conditions and queue depth
        
        Returns:
            float: Sleep time in seconds (lower for higher volatility)
        """
        base_sleep = self.MARKET_REFRESH_RATE_MS / 1000  # Convert ms to seconds
        
        # If queues are non-empty, process faster
        if any(not q.empty() for q in self.execution_queues.values()):
            return base_sleep / 2
            
        # If in high volatility regime, process faster
        if self.current_volatility > 0.5:
            return base_sleep / 2
            
        return base_sleep

    async def _process_execution_queues(self):
        """
        Process all execution queues in priority order with adaptive parallelism
        
        Processes highest priority queues first, but allows concurrent processing 
        of lower priority queues if system resources permit.
        """
        # Check if any critical executions need processing first
        critical_tasks = []
        
        # Process critical queue with highest priority
        if not self.execution_queues[ExecutionPriority.CRITICAL].empty():
            while not self.execution_queues[ExecutionPriority.CRITICAL].empty():
                _, critical_task = await self.execution_queues[ExecutionPriority.CRITICAL].get()
                critical_tasks.append(critical_task)
                
            # Execute critical tasks immediately and wait for completion
            if critical_tasks:
                await asyncio.gather(*critical_tasks)
        
        # Process remaining queues with controlled concurrency
        all_tasks = []
        
        # Collect tasks from remaining priority queues
        for priority in sorted(self.execution_queues.keys()):
            if priority == ExecutionPriority.CRITICAL:
                continue  # Already processed
                
            # Limit how many tasks we pull from each queue
            task_limit = {
                ExecutionPriority.HIGH: 8,      # High priority: process up to 8 at once
                ExecutionPriority.STANDARD: 4,  # Standard: process up to 4 at once
                ExecutionPriority.LOW: 2        # Low priority: process up to 2 at once
            }.get(priority, 1)
            
            # Pull tasks from queue up to limit
            tasks_from_queue = []
            for _ in range(task_limit):
                if self.execution_queues[priority].empty():
                    break
                _, task = await self.execution_queues[priority].get()
                tasks_from_queue.append(task)
            
            all_tasks.extend(tasks_from_queue)
        
        # Execute all remaining tasks concurrently
        if all_tasks:
            await asyncio.gather(*all_tasks)

    async def submit_strategy(self, strategy: Any, priority: int = ExecutionPriority.STANDARD):
        """
        Submit strategy for execution with specified priority
        
        Args:
            strategy: Strategy object to execute
            priority: Execution priority (0-3, lower is higher priority)
        
        Returns:
            str: Execution ID for tracking
        """
        # Generate unique execution ID
        execution_id = f"exec_{strategy.name}_{int(time.time()*1000)}_{self.execution_counter}"
        self.execution_counter += 1
        
        # Pre-validate strategy before accepting into queue
        if not await self._quick_prevalidate_strategy(strategy):
            self.structured_logger.warning(
                "Strategy pre-validation failed",
                strategy_name=strategy.name,
                execution_id=execution_id
            )
            return None
        
        # Assign to appropriate priority queue with sequence number for stable sorting
        with self._priority_lock:
            sequence = self.priority_counters[priority]
            self.priority_counters[priority] += 1
            
        # Create execution task
        task = self._create_execution_task(strategy, execution_id)
        
        # Track pending execution
        self.pending_executions[execution_id] = {
            'strategy': strategy.name,
            'priority': priority,
            'timestamp': time.time(),
            'status': 'queued'
        }
        
        # Add to execution queue
        await self.execution_queues[priority].put((sequence, task))
        
        self.structured_logger.debug(
            "Strategy submitted to execution queue",
            strategy_name=strategy.name,
            priority=priority,
            execution_id=execution_id
        )
        
        return execution_id

    async def _quick_prevalidate_strategy(self, strategy: Any) -> bool:
        """
        Fast pre-validation check before accepting strategy into queue
        
        This is a lightweight check to quickly reject obviously invalid strategies
        before they enter the execution queue. Full validation happens later.
        
        Args:
            strategy: Strategy to validate
            
        Returns:
            bool: True if strategy passes quick validation
        """
        try:
            # Quick rejection checks (blacklist, circuit breakers)
            if await self.blacklist.is_blacklisted_cached(strategy.name):
                return False
                
            if await self.risk_controller.circuit_breaker_active():
                return False
                
            # Strategy must have required methods
            if not hasattr(strategy, 'generate_signal') or not callable(strategy.generate_signal):
                return False
                
            return True
        except Exception:
            return False

    async def _comprehensive_validate_strategy(self, strategy: Any, execution_id: str) -> bool:
        """
        Comprehensive strategy validation with full security and risk checks
        
        Args:
            strategy: Strategy to validate
            execution_id: Unique execution identifier
            
        Returns:
            bool: True if strategy passes all validation
        """
        validation_start = time.time()
        
        try:
            # Run validations in parallel for speed
            validation_results = await asyncio.gather(
                self._validate_strategy_signature(strategy),
                self.blacklist.is_blacklisted(strategy.name),
                self._run_adversarial_tests(strategy),
                self.risk_controller.preapprove_strategy(strategy),
                return_exceptions=True
            )
            
            # Check if any validation threw an exception
            for result in validation_results:
                if isinstance(result, Exception):
                    self.structured_logger.warning(
                        "Strategy validation error",
                        strategy_name=strategy.name,
                        error=str(result),
                        execution_id=execution_id
                    )
                    return False
            
            # Unpack validation results
            signature_valid, is_blacklisted, adversarial_passed, risk_approved = validation_results
            
            # Update validation latency metrics
            validation_time = (time.time() - validation_start) * 1000  # ms
            self._latency_metrics['validation'].append(validation_time)
            
            # All checks must pass
            if not signature_valid:
                self.structured_logger.warning(
                    "Strategy signature validation failed",
                    strategy_name=strategy.name,
                    execution_id=execution_id
                )
                return False
                
            if is_blacklisted:
                self.structured_logger.warning(
                    "Strategy is blacklisted",
                    strategy_name=strategy.name,
                    execution_id=execution_id
                )
                return False
                
            if not adversarial_passed:
                self.structured_logger.warning(
                    "Strategy failed adversarial testing",
                    strategy_name=strategy.name,
                    execution_id=execution_id
                )
                return False
                
            if not risk_approved:
                self.structured_logger.warning(
                    "Strategy rejected by risk controller",
                    strategy_name=strategy.name,
                    execution_id=execution_id
                )
                return False
            
            # All validations passed
            return True
            
        except Exception as e:
            self.structured_logger.error(
                "Strategy validation error",
                strategy_name=strategy.name,
                error=str(e),
                execution_id=execution_id
            )
            return False

    async def _validate_strategy_signature(self, strategy: Any) -> bool:
        """
        Cryptographically validate strategy integrity with replay protection
        
        Args:
            strategy: Strategy to validate
            
        Returns:
            bool: True if signature is valid and not replayed
        """
        try:
            # Extract strategy parameters and nonce
            strategy_hash = hashlib.sha256(str(strategy.parameters).encode()).hexdigest()
            
            # Check if nonce has been used before (replay protection)
            if hasattr(strategy, 'nonce'):
                if strategy.nonce in self.nonce_registry:
                    return False
                self.nonce_registry.add(strategy.nonce)
                
                # Prevent nonce registry from growing too large
                if len(self.nonce_registry) > 10000:
                    # Keep only most recent 5000 nonces
                    self.nonce_registry = set(list(self.nonce_registry)[-5000:])
            
            # Verify signature
            return hmac.compare_digest(
                strategy.signature,
                self.signer.sign_message(strategy_hash)
            )
        except Exception:
            return False

    async def _run_adversarial_tests(self, strategy: Any) -> bool:
        """
        Run stress tests against the strategy to ensure robustness
        
        Args:
            strategy: Strategy to test
            
        Returns:
            bool: True if strategy passes adversarial testing
        """
        market_conditions = await self.market_feed.current_conditions()
        
        # Run simplified test in normal mode, full test in emergency mode
        if self.emergency_mode:
            test_mode = "complete"
        else:
            test_mode = "simplified"
            
        return await self.adversarial_tester.test_strategy(
            strategy, 
            market_conditions,
            mode=test_mode
        )

    def _create_execution_task(self, strategy: Any, execution_id: str) -> asyncio.Task:
        """
        Create prioritized execution task with full tracing and monitoring
        
        Args:
            strategy: Strategy to execute
            execution_id: Unique execution identifier
            
        Returns:
            asyncio.Task: Execution task
        """
        return asyncio.create_task(
            self._execute_strategy(strategy, execution_id),
            name=f"strategy_exec_{execution_id}"
        )

    async def _execute_strategy(self, strategy: Any, execution_id: str):
        """
        Execute validated trading strategy with full monitoring and feedback
        
        Args:
            strategy: Strategy to execute
            execution_id: Unique execution identifier
        """
        execution_start = time.time()
        self.pending_executions[execution_id]['status'] = 'executing'
        
        try:
            # Full validation (only run expensive validation now)
            if not await self._comprehensive_validate_strategy(strategy, execution_id):
                self.pending_executions[execution_id]['status'] = 'rejected'
                return
                
            # Check execution timeout (prevent stale strategies)
            if time.time() - self.pending_executions[execution_id]['timestamp'] > 5:  # 5 second timeout
                self.structured_logger.warning(
                    "Strategy execution timed out in queue",
                    strategy_name=strategy.name,
                    execution_id=execution_id,
                    queue_time=time.time() - self.pending_executions[execution_id]['timestamp']
                )
                self.pending_executions[execution_id]['status'] = 'timeout'
                return
                
            # Update market data cache
            market_data = await self._get_strategy_market_data(strategy)
            
            # Get current market regime 
            regime = await self._get_current_regime()
            
            # Ensure circuit breakers aren't active
            if await self.risk_controller.circuit_breaker_active():
                self.structured_logger.warning(
                    "Circuit breaker active - halting execution",
                    strategy_name=strategy.name,
                    execution_id=execution_id
                )
                self.pending_executions[execution_id]['status'] = 'circuit_breaker'
                return
                
            # Generate trading signal
            signal_start = time.time()
            signal = await self._generate_signal(strategy, market_data, regime, execution_id)
            signal_time = (time.time() - signal_start) * 1000  # ms
            
            if not signal:
                self.pending_executions[execution_id]['status'] = 'no_signal'
                return
                
            # Log signal generation for AI training
            self.decision_logger.log_decision(
                strategy_name=strategy.name,
                signal=signal,
                market_data=market_data,
                regime=regime,
                execution_id=execution_id,
                latency_ms=signal_time
            )
                
            # Optimize execution
            execution_plan = await self._optimize_execution(signal, execution_id)
            
            # Final risk check before execution
            if not await self.risk_controller.validate_execution(execution_plan):
                self.structured_logger.warning(
                    "Execution blocked by risk controller",
                    strategy_name=strategy.name,
                    execution_id=execution_id,
                    plan=execution_plan
                )
                self.pending_executions[execution_id]['status'] = 'risk_rejected'
                return
                
            # Execute order
            execution_result = await self._execute_order_flow(execution_plan, execution_id)
            
            # Update pending execution status
            self.pending_executions[execution_id]['status'] = 'completed'
            
            # Calculate and log execution latency
            execution_time = (time.time() - execution_start) * 1000  # ms
            self._latency_metrics['execution'].append(execution_time)
            
            # Process execution results
            await self._process_execution_results(execution_plan, execution_result, execution_id)
            
            # Update learning models
            await self._update_learning_models(strategy, execution_plan, execution_result)
            
            # Log execution metrics
            self.structured_logger.info(
                "Strategy execution completed",
                strategy_name=strategy.name,
                execution_id=execution_id,
                execution_time_ms=execution_time,
                signal_time_ms=signal_time
            )
            
        except Exception as e:
            error_msg = f"Strategy execution failed: {str(e)}, {traceback.format_exc()}"
            self.structured_logger.error(
                error_msg,
                strategy_name=strategy.name,
                execution_id=execution_id
            )
            
            # Update execution status
            self.pending_executions[execution_id]['status'] = 'failed'
            self.pending_executions[execution_id]['error'] = str(e)
            
            # Handle execution failure
            await self._handle_execution_failure(strategy, execution_id)

    async def _get_strategy_market_data(self, strategy: Any) -> Dict:
        """
        Get optimized market data for a specific strategy
        
        Returns only the data needed by this strategy type to reduce overhead
        
        Args:
            strategy: Strategy requiring market data
            
        Returns:
            Dict: Tailored market data for this strategy
        """
        # Get strategy data requirements
        required_data = strategy.required_data if hasattr(strategy, 'required_data') else None
        
        # If strategy has specific data requirements, honor them
        if required_data:
            return await self.market_feed.get_custom_data(required_data)
            
        # Otherwise, get standard strategy data
        # Use cache to avoid repeated fetches within short timeframes
        strategy_type = type(strategy).__name__
        cache_key = f"{strategy_type}_{int(time.time()*10)}"  # 100ms resolution
        
        if cache_key in self._market_data_cache:
            return self._market_data_cache[cache_key]
            
        data = await self.market_feed.get_strategy_data()
        self._market_data_cache[cache_key] = data
        return data

    @cached(cache=TTLCache(maxsize=100, ttl=1))  # 1-second cache
    @cached(cache=TTLCache(maxsize=100, ttl=1))  # 1-second cache
    async def _get_current_regime(self) -> str:
        """
        Get current market regime with efficient caching
        
        Returns:
            str: Current market regime label
        """
        # Only update regime at most once per second
        current_time = time.time()
        if current_time - self.last_regime_update > 1.0:
            self.current_regime = await self.regime_detector.detect_regime()
            self.current_volatility = await self.market_feed.get_volatility_metrics()
            self.last_regime_update = current_time
            
        return self.current_regime
        
    async def _generate_signal(self, strategy: Any, market_data: Dict, regime: str, 
                               execution_id: str) -> Optional[Dict]:
        """
        Generate trading signal from strategy with optimized execution path
        
        Args:
            strategy: Strategy to execute
            market_data: Current market data
            regime: Current market regime
            execution_id: Unique execution identifier
            
        Returns:
            Optional[Dict]: Trading signal or None if no signal
        """
        # Signal generation timing
        start_time = time.time()
        
        try:
            # Check signal cache first for identical strategies
            cache_key = f"{strategy.name}_{hash(frozenset(strategy.parameters.items()))}"
            if cache_key in self._signal_cache:
                return self._signal_cache[cache_key]
            
            # Generate signal with timeout protection
            signal_future = asyncio.ensure_future(
                strategy.generate_signal(market_data, regime)
            )
            
            # Apply execution timeout - vital for HFT applications
            try:
                signal = await asyncio.wait_for(
                    signal_future, 
                    timeout=self.EXECUTION_TIMEOUT_MS / 1000
                )
            except asyncio.TimeoutError:
                self.structured_logger.warning(
                    "Strategy signal generation timed out",
                    strategy_name=strategy.name,
                    execution_id=execution_id,
                    timeout_ms=self.EXECUTION_TIMEOUT_MS
                )
                return None
                
            # No trade signal
            if not signal or not signal.get('action'):
                return None
                
            # Cache valid signals briefly
            self._signal_cache[cache_key] = signal
            
            # Log signal generation performance
            signal_time = (time.time() - start_time) * 1000  # Convert to ms
            self.structured_logger.debug(
                "Signal generated",
                strategy_name=strategy.name,
                signal_time_ms=signal_time,
                execution_id=execution_id
            )
            
            return signal
            
        except Exception as e:
            self.structured_logger.error(
                f"Signal generation error: {str(e)}",
                strategy_name=strategy.name,
                execution_id=execution_id
            )
            return None

    async def _optimize_execution(self, signal: Dict, execution_id: str) -> Dict:
        """
        Optimize trade execution based on market microstructure and liquidity
        
        Args:
            signal: Trading signal from strategy
            execution_id: Unique execution identifier
            
        Returns:
            Dict: Optimized execution plan
        """
        # Initialize execution plan with signal data
        execution_plan = {
            'signal': signal,
            'execution_id': execution_id,
            'timestamp': time.time(),
            'market_impact': {},
            'slippage_estimate': {},
            'execution_slices': [],
            'stealth_mode': False
        }
        
        # Get market depth and liquidity data
        ticker = signal.get('ticker')
        size = signal.get('size', 0)
        
        # Early exit for invalid signals
        if not ticker or size <= 0:
            return execution_plan
        
        # Get liquidity profile for target asset
        liquidity_profile = await self.liquidity_optimizer.get_liquidity_profile(ticker)
        
        # Calculate market impact
        market_impact = await self.impact_predictor.predict_impact(
            ticker, size, signal.get('action'), liquidity_profile
        )
        execution_plan['market_impact'] = market_impact
        
        # Estimate slippage
        slippage = await self.slippage_estimator.estimate_slippage(
            ticker, size, signal.get('action'), liquidity_profile
        )
        execution_plan['slippage_estimate'] = slippage
        
        # Determine if order needs to be sliced based on size relative to liquidity
        liquidity_ratio = size / liquidity_profile.get('available_liquidity', float('inf'))
        
        if liquidity_ratio > self.HFT_SLICE_THRESHOLD:
            # Order is large relative to available liquidity, slice it
            execution_plan['execution_slices'] = await self._create_execution_slices(
                signal, liquidity_profile, market_impact
            )
            execution_plan['stealth_mode'] = True
        else:
            # Order can be executed at once
            execution_plan['execution_slices'] = [{
                'size': size,
                'price': signal.get('price'),
                'timing': 'immediate'
            }]
            
        return execution_plan
        
    async def _create_execution_slices(self, signal: Dict, 
                                       liquidity_profile: Dict, 
                                       market_impact: Dict) -> List[Dict]:
        """
        Create optimal execution slices to minimize market impact
        
        Args:
            signal: Trading signal
            liquidity_profile: Current liquidity data
            market_impact: Predicted market impact
            
        Returns:
            List[Dict]: List of execution slices with size, price and timing
        """
        # Get optimal execution from stealth executor
        return await self.stealth_manager.create_execution_plan(
            signal.get('ticker'),
            signal.get('size'),
            signal.get('action'),
            liquidity_profile,
            market_impact
        )

    async def _execute_order_flow(self, execution_plan: Dict, execution_id: str) -> Dict:
        """
        Execute optimized order flow with stealth execution and latency tracking
        
        Args:
            execution_plan: Optimized execution plan
            execution_id: Unique execution identifier
            
        Returns:
            Dict: Execution results
        """
        # Initialize execution results
        execution_results = {
            'execution_id': execution_id,
            'timestamp': time.time(),
            'slices_executed': 0,
            'total_slices': len(execution_plan['execution_slices']),
            'actual_slippage': 0.0,
            'actual_impact': 0.0,
            'execution_latency_ms': 0.0,
            'status': 'pending',
            'slice_results': []
        }
        
        # Execute each slice
        start_time = time.time()
        
        try:
            if execution_plan['stealth_mode']:
                # Execute with stealth mode for large orders
                slice_results = await self.order_executor.execute_sliced_order(
                    execution_plan['signal'].get('ticker'),
                    execution_plan['signal'].get('action'),
                    execution_plan['execution_slices'],
                    execution_id
                )
            else:
                # Execute simple order
                slice_results = [await self.order_executor.execute_order(
                    execution_plan['signal'].get('ticker'),
                    execution_plan['signal'].get('action'),
                    execution_plan['execution_slices'][0]['size'],
                    execution_plan['signal'].get('price'),
                    execution_id
                )]
            
            # Calculate execution metrics
            execution_results['slices_executed'] = len(slice_results)
            execution_results['slice_results'] = slice_results
            execution_results['status'] = 'completed'
            execution_results['execution_latency_ms'] = (time.time() - start_time) * 1000
            
            # Calculate actual slippage and impact
            if slice_results:
                execution_results['actual_slippage'] = np.mean([
                    slice.get('actual_slippage', 0.0) for slice in slice_results
                ])
                execution_results['actual_impact'] = np.mean([
                    slice.get('market_impact', 0.0) for slice in slice_results
                ])
            
            return execution_results
            
        except Exception as e:
            execution_results['status'] = 'failed'
            execution_results['error'] = str(e)
            execution_results['execution_latency_ms'] = (time.time() - start_time) * 1000
            
            self.structured_logger.error(
                f"Order execution failed: {str(e)}",
                execution_id=execution_id,
                ticker=execution_plan['signal'].get('ticker')
            )
            
            return execution_results

    async def _process_execution_results(self, execution_plan: Dict, 
                                         execution_results: Dict, 
                                         execution_id: str) -> None:
        """
        Process execution results and update system state
        
        Args:
            execution_plan: The original execution plan
            execution_results: Results from order execution
            execution_id: Unique execution identifier
        """
        # Update execution history
        self.execution_history[execution_id] = {
            'plan': execution_plan,
            'results': execution_results,
            'timestamp': time.time()
        }
        
        # Update strategy metrics
        strategy_name = self.pending_executions[execution_id]['strategy']
        if strategy_name not in self.strategy_metrics:
            self.strategy_metrics[strategy_name] = {
                'executions': 0,
                'successful': 0,
                'failed': 0,
                'avg_slippage': 0.0,
                'avg_impact': 0.0,
                'avg_latency': 0.0
            }
        
        metrics = self.strategy_metrics[strategy_name]
        metrics['executions'] += 1
        
        if execution_results['status'] == 'completed':
            metrics['successful'] += 1
        else:
            metrics['failed'] += 1
            
        # Update running averages
        n = metrics['executions']
        metrics['avg_slippage'] = (metrics['avg_slippage'] * (n-1) + execution_results['actual_slippage']) / n
        metrics['avg_impact'] = (metrics['avg_impact'] * (n-1) + execution_results['actual_impact']) / n
        metrics['avg_latency'] = (metrics['avg_latency'] * (n-1) + execution_results['execution_latency_ms']) / n
        
        # Log execution results
        if execution_results['status'] == 'completed':
            self.structured_logger.info(
                "Order execution completed",
                execution_id=execution_id,
                slices_executed=execution_results['slices_executed'],
                latency_ms=execution_results['execution_latency_ms'],
                slippage=execution_results['actual_slippage']
            )
        else:
            self.structured_logger.warning(
                "Order execution failed",
                execution_id=execution_id,
                error=execution_results.get('error', 'Unknown error')
            )
            
        # Update portfolio state
        await self.portfolio_balancer.update_positions(execution_results)
        
        # Send feedback to the risk controller
        await self.risk_controller.process_execution_feedback(
            execution_plan, execution_results
        )

    async def _update_learning_models(self, strategy: Any, execution_plan: Dict, 
                                      execution_results: Dict) -> None:
        """
        Update AI models with execution feedback for continuous learning
        
        Args:
            strategy: Strategy that was executed
            execution_plan: Original execution plan
            execution_results: Results from order execution
        """
        # Create feedback data packet
        feedback_data = {
            'strategy': strategy.name,
            'parameters': strategy.parameters,
            'signal': execution_plan['signal'],
            'market_regime': self.current_regime,
            'volatility': self.current_volatility,
            'predicted_impact': execution_plan['market_impact'],
            'predicted_slippage': execution_plan['slippage_estimate'],
            'actual_impact': execution_results['actual_impact'],
            'actual_slippage': execution_results['actual_slippage'],
            'execution_latency': execution_results['execution_latency_ms'],
            'success': execution_results['status'] == 'completed'
        }
        
        # Update learning models asynchronously
        feedback_start = time.time()
        
        # Start feedback tasks concurrently
        feedback_tasks = [
            self.meta_optimizer.update_model(feedback_data),
            self.adaptation_learner.process_feedback(feedback_data),
            self.strategy_selector.update_strategy_performance(strategy.name, feedback_data),
            self.q_agent.update_q_values(feedback_data)
        ]
        
        # Wait for all feedback tasks to complete
        await asyncio.gather(*feedback_tasks)
        
        # Update feedback latency metrics
        feedback_time = (time.time() - feedback_start) * 1000  # ms
        self._latency_metrics['feedback'].append(feedback_time)
        
        self.structured_logger.debug(
            "Learning models updated",
            strategy_name=strategy.name,
            feedback_time_ms=feedback_time
        )

    async def _handle_execution_failure(self, strategy: Any, execution_id: str) -> None:
        """
        Handle execution failures with appropriate recovery actions
        
        Args:
            strategy: Strategy that failed
            execution_id: Unique execution identifier
        """
        # Get failure details
        failure_info = self.pending_executions[execution_id]
        
        # Log failure
        self.structured_logger.error(
            "Strategy execution failed",
            strategy_name=strategy.name,
            execution_id=execution_id,
            error=failure_info.get('error', 'Unknown error')
        )
        
        # Update strategy blacklist score
        await self.blacklist.increment_failure_score(strategy.name)
        
        # Alert if critical strategy
        if failure_info.get('priority') == ExecutionPriority.CRITICAL:
            await self.alert_system.send_alert(
                f"Critical strategy execution failed: {strategy.name}",
                level="high"
            )
            
        # Check for circuit breaker conditions
        consecutive_failures = await self.blacklist.get_consecutive_failures(strategy.name)
        if consecutive_failures >= 3:
            self.structured_logger.warning(
                "Strategy blacklisted due to consecutive failures",
                strategy_name=strategy.name,
                consecutive_failures=consecutive_failures
            )

    async def _update_system_state(self) -> None:
        """
        Update global system state and adapt to changing market conditions
        """
        # Update market regime if needed
        regime = await self._get_current_regime()
        
        # Update volatility metrics
        volatility_metrics = await self.market_feed.get_volatility_metrics()
        self.current_volatility = volatility_metrics.get('current_volatility', 0.0)
        
        # Check market circuit breakers
        if self.current_volatility > self.config.get('circuit_breaker_threshold', 0.8):
            await self._activate_circuit_breaker("High volatility detected")
            
        # Update risk parameters based on current regime
        await self.risk_controller.update_risk_parameters(regime, self.current_volatility)
        
        # Update strategy selector with latest market conditions
        await self.strategy_selector.update_market_conditions(
            regime, self.current_volatility
        )
        
        # Clean up old execution records
        self._cleanup_old_records()

    async def _activate_circuit_breaker(self, reason: str) -> None:
        """
        Activate system-wide circuit breaker with coordinated risk mitigation
        
        Args:
            reason: Reason for circuit breaker activation
        """
        if self.emergency_mode:
            return  # Already in emergency mode
            
        self.emergency_mode = True
        circuit_breaker_id = f"cb_{int(time.time())}"
        
        self.structured_logger.warning(
            "Circuit breaker activated",
            reason=reason,
            circuit_breaker_id=circuit_breaker_id,
            timestamp=datetime.now().isoformat()
        )
        
        # Cancel all non-critical pending orders
        await self.order_executor.cancel_all_non_critical()
        
        # Notify risk system
        await self.risk_controller.circuit_breaker_activated(reason)
        
        # Send high-priority alert
        await self.alert_system.send_alert(
            f"CIRCUIT BREAKER ACTIVATED: {reason}",
            level="critical"
        )
        
        # Schedule circuit breaker review
        self._event_loop.create_task(
            self._circuit_breaker_review(circuit_breaker_id)
        )

    async def _circuit_breaker_review(self, circuit_breaker_id: str) -> None:
        """
        Periodically review circuit breaker conditions for potential deactivation
        
        Args:
            circuit_breaker_id: Unique circuit breaker identifier
        """
        review_count = 0
        
        while self.emergency_mode and self.running:
            review_count += 1
            
            # Wait before review (exponential backoff with cap)
            wait_time = min(30, 5 * (2 ** min(review_count - 1, 3)))
            await asyncio.sleep(wait_time)
            
            # Get latest market conditions
            volatility = await self.market_feed.get_volatility_metrics()
            current_volatility = volatility.get('current_volatility', 1.0)
            
            # Check if conditions have normalized
            if current_volatility < self.config.get('circuit_breaker_reset_threshold', 0.4):
                await self._deactivate_circuit_breaker(circuit_breaker_id)
                break
                
            # Log continued circuit breaker
            self.structured_logger.info(
                "Circuit breaker review",
                circuit_breaker_id=circuit_breaker_id,
                review_count=review_count,
                current_volatility=current_volatility,
                emergency_mode=self.emergency_mode
            )

    async def _deactivate_circuit_breaker(self, circuit_breaker_id: str) -> None:
        """
        Deactivate circuit breaker and resume normal operations gradually
        
        Args:
            circuit_breaker_id: Circuit breaker identifier to deactivate
        """
        if not self.emergency_mode:
            return
            
        self.emergency_mode = False
        
        self.structured_logger.info(
            "Circuit breaker deactivated",
            circuit_breaker_id=circuit_breaker_id,
            timestamp=datetime.now().isoformat()
        )
        
        # Notify risk system
        await self.risk_controller.circuit_breaker_deactivated()
        
        # Send alert about deactivation
        await self.alert_system.send_alert(
            "Circuit breaker deactivated - resuming normal operations",
            level="medium"
        )
        
        # Resume operations with caution - start with low risk strategies first
        await self._gradual_system_restart()

    async def _gradual_system_restart(self) -> None:
        """
        Gradually restart system operations after emergency mode
        """
        # Step 1: Reset execution queues
        for priority_queue in self.execution_queues.values():
            while not priority_queue.empty():
                try:
                    await priority_queue.get()
                except Exception:
                    pass
        
        # Step 2: Update market data and models
        await self._warm_up_system()
        
        # Step 3: Rebalance portfolio if needed
        await self.portfolio_balancer.rebalance_after_circuit_breaker()
        
        # Step 4: Reset priority counters
        self.priority_counters = {p: 0 for p in self.execution_queues.keys()}
        
        self.structured_logger.info("System restart complete after circuit breaker")

    async def _activate_failsafe(self, exception: Exception) -> None:
        """
        Activate system failsafe mode for critical errors
        
        Args:
            exception: Exception that triggered failsafe
        """
        error_msg = f"CRITICAL ERROR: {str(exception)}"
        tb_str = traceback.format_exc()
        
        # Log critical error
        self.structured_logger.critical(
            error_msg,
            traceback=tb_str,
            timestamp=datetime.now().isoformat()
        )
        
        # Activate circuit breaker
        await self._activate_circuit_breaker(f"Failsafe: {str(exception)}")
        
        # Send critical alert
        await self.alert_system.send_alert(
            f"SYSTEM FAILSAFE ACTIVATED: {error_msg}",
            level="critical"
        )
        
        # Perform emergency position management if needed
        if self.config.get('failsafe_liquidation', False):
            await self._emergency_position_management()

    async def _emergency_position_management(self) -> None:
        """
        Perform emergency position management during failsafe activation
        """
        # Get current positions
        positions = await self.portfolio_balancer.get_current_positions()
        
        if not positions:
            return
            
        # Check if emergency liquidation is required
        if self.config.get('emergency_liquidation_threshold', 0.0) > 0:
            # Calculate emergency risk metrics
            risk_ratio = await self.risk_controller.calculate_emergency_risk_ratio()
            
            if risk_ratio >= self.config['emergency_liquidation_threshold']:
                # Initiate emergency liquidation
                self.structured_logger.critical(
                    "INITIATING EMERGENCY POSITION LIQUIDATION",
                    risk_ratio=risk_ratio,
                    threshold=self.config['emergency_liquidation_threshold']
                )
                
                await self.portfolio_balancer.execute_emergency_liquidation()
                return
        
        # Otherwise just hedge positions
        await self.portfolio_balancer.execute_emergency_hedging()

    async def _monitor_system_health(self) -> None:
        """
        Monitor system health metrics and performance
        """
        while self.running:
            try:
                # Check heartbeat
                heartbeat_age = time.time() - self.last_heartbeat
                if heartbeat_age > 5.0:  # No heartbeat for 5 seconds
                    self.structured_logger.warning(
                        "System heartbeat delayed",
                        delay_seconds=heartbeat_age
                    )
                
                # Monitor execution latency
                if self._latency_metrics['execution']:
                    avg_execution = np.mean(self._latency_metrics['execution'][-100:])
                    if avg_execution > self.EXECUTION_TIMEOUT_MS * 0.8:
                        self.structured_logger.warning(
                            "Execution latency approaching timeout",
                            avg_latency_ms=avg_execution,
                            timeout_ms=self.EXECUTION_TIMEOUT_MS
                        )
                        
                # Monitor memory usage
                memory_usage = self.system_guard.get_memory_usage()
                if memory_usage > self.config.get('memory_warning_threshold', 80):
                    self.structured_logger.warning(
                        "High memory usage detected",
                        memory_percent=memory_usage
                    )
                    # Trigger memory optimization
                    self._cleanup_old_records(aggressive=True)
                
                # Wait before next health check
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.structured_logger.error(
                    f"Health monitoring error: {str(e)}",
                    traceback=traceback.format_exc()
                )
                await asyncio.sleep(5.0)  # Back off on error

    def _cleanup_old_records(self, aggressive: bool = False) -> None:
        """
        Clean up old execution records to manage memory usage
        
        Args:
            aggressive: If True, perform more aggressive cleanup
        """
        # Standard cleanup happens automatically with TTLCache
        
        # Perform additional cleanup when requested
        if aggressive:
            # Clear signal cache completely
            self._signal_cache.clear()
            
            # Clear market data cache
            self._market_data_cache.clear()
            
            # Trim latency metrics
            for key in self._latency_metrics:
                if len(self._latency_metrics[key]) > 100:
                    self._latency_metrics[key] = self._latency_metrics[key][-100:]
            
            # Run garbage collection
            import gc
            gc.collect()

    async def _update_market_models(self) -> None:
        """
        Periodically update market models with latest data
        """
        while self.running:
            try:
                # Only update in non-emergency mode
                if not self.emergency_mode:
                    # Update regime detector with latest market data
                    full_market_data = await self.market_feed.get_full_market_data()
                    
                    # Start model updates concurrently
                    await asyncio.gather(
                        self.regime_detector.update_model(full_market_data),
                        self.meta_optimizer.update_market_data(full_market_data),
                        self.impact_predictor.update_model(full_market_data),
                        self.slippage_estimator.calibrate(full_market_data)
                    )
                
                # Model update frequency (adaptive based on market conditions)
                update_interval = 60.0  # Base: 1 minute
                
                # Update more frequently in high-volatility regimes
                if self.current_volatility > 0.6:
                    update_interval = 30.0  # High volatility: 30 seconds
                elif self.current_regime in ["trending", "volatile"]:
                    update_interval = 45.0  # Active markets: 45 seconds
                
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                self.structured_logger.error(
                    f"Market model update error: {str(e)}",
                    traceback=traceback.format_exc()
                )
                await asyncio.sleep(60.0)  # Back off on error

    # Mobile app and web dashboard API methods
    async def get_system_status(self) -> Dict:
        """
        Get current system status for API/Dashboard
        
        Returns:
            Dict: Current system status metrics
        """
        # Calculate execution statistics
        execution_stats = {}
        if self._latency_metrics['execution']:
            recent_execs = self._latency_metrics['execution'][-100:]
            execution_stats = {
                'avg_latency_ms': float(np.mean(recent_execs)),
                'p95_latency_ms': float(np.percentile(recent_execs, 95)),
                'min_latency_ms': float(np.min(recent_execs)),
                'max_latency_ms': float(np.max(recent_execs))
            }
            
        # Calculate strategy performance metrics
        top_strategies = []
        for strategy_name, metrics in sorted(
            self.strategy_metrics.items(), 
            key=lambda x: x[1]['executions'], 
            reverse=True
        )[:5]:
            if metrics['executions'] > 0:
                success_rate = (metrics['successful'] / metrics['executions']) * 100
                top_strategies.append({
                    'name': strategy_name,
                    'executions': metrics['executions'],
                    'success_rate': round(success_rate, 2),
                    'avg_slippage': round(metrics['avg_slippage'], 4),
                    'avg_latency_ms': round(metrics['avg_latency'], 2)
                })
        
        # Build system status response
        return {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'emergency' if self.emergency_mode else 'normal',
            'market_regime': self.current_regime,
            'volatility': round(self.current_volatility, 4),
            'active_strategies': len(self.active_strategies),
            'pending_executions': len(self.pending_executions),
            'execution_metrics': execution_stats,
            'top_strategies': top_strategies,
            'memory_usage': self.system_guard.get_memory_usage(),
            'cpu_usage': self.system_guard.get_cpu_usage()
        }

    async def get_execution_history(self, limit: int = 20) -> List[Dict]:
        """
        Get recent execution history for API/Dashboard
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List[Dict]: Recent execution history
        """
        # Get recent executions sorted by timestamp (newest first)
        recent_executions = []
        
        for exec_id, execution in sorted(
            self.execution_history.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )[:limit]:
            # Create simplified view for dashboard
            execution_plan = execution['plan']
            execution_results = execution['results']
            
            signal = execution_plan['signal']
            
            recent_executions.append({
                'execution_id': exec_id,
                'timestamp': datetime.fromtimestamp(execution['timestamp']).isoformat(),
                'strategy': self.pending_executions.get(exec_id, {}).get('strategy', 'unknown'),
                'ticker': signal.get('ticker', 'unknown'),
                'action': signal.get('action', 'unknown'),
                'size': signal.get('size', 0),
                'slices': execution_results.get('slices_executed', 0),
                'actual_slippage': execution_results.get('actual_slippage', 0),
                'execution_latency_ms': execution_results.get('execution_latency_ms', 0),
                'status': execution_results.get('status', 'unknown')
            })
            
        return recent_executions

    # Mobile-specific API methods for lightweight access
    async def get_lightweight_status(self) -> Dict:
        """
        Get lightweight system status for mobile app
        
        Returns:
            Dict: Minimal status data optimized for mobile bandwidth
        """
        # Only essential metrics for mobile view
        return {
            'status': 'emergency' if self.emergency_mode else 'normal',
            'regime': self.current_regime,
            'vol': round(self.current_volatility, 2),
            'active': len(self.active_strategies),
            'pending': len(self.pending_executions),
            'timestamp': int(time.time())
        }

    # System integration helpers for institutional deployment
    async def sync_with_meta_trader(self) -> None:
        """
        Synchronize strategy weights and selection with meta_trader.py
        """
        # Get optimal strategy weights from meta trader
        strategy_weights = await self.meta_optimizer.get_optimal_strategy_weights()
        
        # Update strategy selector with latest weights
        await self.strategy_selector.update_strategy_weights(strategy_weights)
        
        # Share execution performance metrics back to meta trader
        await self.meta_optimizer.update_execution_metrics(self.strategy_metrics)
        
        self.structured_logger.debug(
            "Synchronized with meta trader",
            strategy_count=len(strategy_weights)
        )

    async def register_strategy(self, strategy: Any) -> bool:
        """
        Register a new strategy with the orchestration engine
        
        Args:
            strategy: Strategy to register
            
        Returns:
            bool: True if registration was successful, False otherwise
        """
        try:
            # Validate the strategy object
            if not hasattr(strategy, 'name') or not hasattr(strategy, 'execute'):
                self.structured_logger.error("Invalid strategy object provided for registration.")
                return False
            
            # Register the strategy
            self.active_strategies[strategy.name] = strategy
            self.structured_logger.info(f"Strategy registered: {strategy.name}")
            return True
        except Exception as e:
            self.structured_logger.error(f"Failed to register strategy: {str(e)}")
            return False