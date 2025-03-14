# src/Core/trading/execution/broker_manager.py

import asyncio
import time
import hashlib
import hmac
import os
import json
import uuid
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import defaultdict
import uvloop
import aiohttp
from cryptography.fernet import Fernet
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Apex Core Imports
from Apex.utils.logging.structured_logger import StructuredLogger
from Apex.src.Core.trading.execution.broker_registry import QuantumBrokerRegistry
from Apex.src.Core.trading.risk.risk_engine import RiskEngine
from Apex.src.Core.trading.hft.liquidity_manager import LiquidityOracle
from Apex.src.Core.data.trade_history import TradeHistory
from Apex.src.ai.ensembles.meta_trader import MetaTrader
from Apex.utils.helpers.security import QuantumVault, generate_hmac
from Apex.Config.config_loader import load_config
from Apex.src.Core.trading.execution.order_execution import QuantumExecutionEngine
from Apex.src.Core.data.trade_monitor import TradeMonitor
from Apex.src.Core.trading.execution.broker_factory import BrokerFactory
from Apex.src.Core.trading.execution.broker_api import BrokerAPI
from Apex.src.Core.trading.execution.market_impact import MarketImpactCalculator
from Apex.src.Core.trading.execution.conflict_resolver import ConflictResolver
from Apex.src.Core.data.realtime.websocket_handler import WebSocketHandler
from Apex.src.Core.trading.security.security import SecurityManager

# Configure UVLoop for enhanced async performance
uvloop.install()

# Global thread pool for CPU-bound tasks
CPU_POOL = ThreadPoolExecutor(max_workers=os.cpu_count())

logger = StructuredLogger("QuantumBrokerManager")

class LatencyTracker:
    """Track and analyze broker execution latency metrics"""
    
    def __init__(self, decay_factor: float = 0.95, history_size: int = 100):
        self.latencies = defaultdict(lambda: np.zeros(history_size))
        self.counters = defaultdict(int)
        self.decay_factor = decay_factor
        self.history_size = history_size
        
    def record(self, broker_id: str, latency_ns: int) -> None:
        """Record a latency measurement with exponential decay weighting"""
        idx = self.counters[broker_id] % self.history_size
        self.latencies[broker_id][idx] = latency_ns
        self.counters[broker_id] += 1
        
    def get_percentile(self, broker_id: str, percentile: float = 90) -> float:
        """Get specified percentile latency for a broker (default: 90th percentile)"""
        if self.counters[broker_id] == 0:
            return float('inf')
            
        count = min(self.counters[broker_id], self.history_size)
        return np.percentile(self.latencies[broker_id][:count], percentile)
        
    def get_mean(self, broker_id: str) -> float:
        """Get mean latency for a broker"""
        if self.counters[broker_id] == 0:
            return float('inf')
            
        count = min(self.counters[broker_id], self.history_size)
        return np.mean(self.latencies[broker_id][:count])
    
    def get_recent(self, broker_id: str) -> float:
        """Get most recent latency"""
        if self.counters[broker_id] == 0:
            return float('inf')
            
        idx = (self.counters[broker_id] - 1) % self.history_size
        return self.latencies[broker_id][idx]
    
    def get_rankings(self) -> List[Tuple[str, float]]:
        """Get brokers ranked by latency (fastest first)"""
        rankings = []
        for broker_id in self.latencies:
            if self.counters[broker_id] > 0:
                rankings.append((broker_id, self.get_percentile(broker_id)))
        
        return sorted(rankings, key=lambda x: x[1])


class BrokerReliabilityMetrics:
    """Track broker reliability metrics including success rates, failures, and operational status"""
    
    def __init__(self, window_size: int = 100):
        self.success_counts = defaultdict(int)
        self.failure_counts = defaultdict(int)
        self.window_size = window_size
        self.recent_results = defaultdict(lambda: [])
        self.last_failure_time = defaultdict(float)
        self.consecutive_failures = defaultdict(int)
        
    def record_success(self, broker_id: str) -> None:
        """Record a successful execution"""
        self.success_counts[broker_id] += 1
        self.recent_results[broker_id].append(1)
        if len(self.recent_results[broker_id]) > self.window_size:
            self.recent_results[broker_id].pop(0)
        self.consecutive_failures[broker_id] = 0
        
    def record_failure(self, broker_id: str) -> None:
        """Record a failed execution"""
        self.failure_counts[broker_id] += 1
        self.recent_results[broker_id].append(0)
        if len(self.recent_results[broker_id]) > self.window_size:
            self.recent_results[broker_id].pop(0)
        self.last_failure_time[broker_id] = time.time()
        self.consecutive_failures[broker_id] += 1
        
    def get_success_rate(self, broker_id: str) -> float:
        """Get recent success rate for a broker"""
        results = self.recent_results[broker_id]
        if not results:
            return 0.0
        return sum(results) / len(results)
    
    def get_failure_rate(self, broker_id: str) -> float:
        """Get recent failure rate for a broker"""
        return 1.0 - self.get_success_rate(broker_id)
    
    def is_broker_reliable(self, broker_id: str, threshold: float = 0.95) -> bool:
        """Determine if broker is reliable based on recent performance"""
        if not self.recent_results[broker_id]:
            return True  # No data yet, assume reliable
        
        # Check if we have sufficient data
        if len(self.recent_results[broker_id]) < min(10, self.window_size // 10):
            return True
            
        return self.get_success_rate(broker_id) >= threshold
    
    def time_since_last_failure(self, broker_id: str) -> float:
        """Get time in seconds since last failure"""
        if broker_id not in self.last_failure_time:
            return float('inf')
        return time.time() - self.last_failure_time[broker_id]
    
    def get_consecutive_failures(self, broker_id: str) -> int:
        """Get number of consecutive failures"""
        return self.consecutive_failures[broker_id]


class ExecutionCache:
    """Thread-safe LRU cache for execution results to avoid redundant operations"""
    
    def __init__(self, maxsize: int = 1000):
        self.cache = {}
        self.maxsize = maxsize
        self.access_times = {}
        self._lock = asyncio.Lock()
        
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached execution result if available"""
        async with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
            
    async def set(self, key: str, value: Dict[str, Any]) -> None:
        """Cache execution result with LRU eviction"""
        async with self._lock:
            if len(self.cache) >= self.maxsize:
                # Evict least recently used item
                oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
                
            self.cache[key] = value
            self.access_times[key] = time.time()


class QuantumBrokerManager:
    """
    Institutional-Grade Broker Orchestration System with advanced HFT capabilities
    - AI-driven dynamic broker routing with quantum-secure execution
    - Multi-region distributed broker management with nanosecond failover
    - Edge-computing execution pipeline with real-time adaption
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Initialize here to avoid race conditions
        self._initialized = True
        self._initialization_task = None
        self._last_health_check = {}
        self._broker_locks = defaultdict(asyncio.Lock)
        
        # System Configuration
        self.config = load_config()
        self.vault = QuantumVault()
        self.registry = QuantumBrokerRegistry()
        
        # Performance Tracking Systems
        self.latency_tracker = LatencyTracker(
            decay_factor=self.config.get('latency_decay_factor', 0.95),
            history_size=self.config.get('latency_history_size', 100)
        )
        self.reliability_metrics = BrokerReliabilityMetrics(
            window_size=self.config.get('reliability_window_size', 100)
        )
        self.execution_cache = ExecutionCache(
            maxsize=self.config.get('execution_cache_size', 1000)
        )
        
        # Security Systems
        self.security = SecurityManager()
        self.session_nonce = hashlib.sha3_256(os.urandom(32)).hexdigest()
        
        # System Integration Components
        self.broker_factory = BrokerFactory()
        self.risk_engine = RiskEngine()
        self.liquidity_oracle = LiquidityOracle()
        self.trade_history = TradeHistory()
        self.meta_trader = MetaTrader()
        self.execution_engine = QuantumExecutionEngine()
        self.trade_monitor = TradeMonitor()
        self.market_impact = MarketImpactCalculator()
        self.conflict_resolver = ConflictResolver()
        self.websocket_handler = WebSocketHandler()
        
        # Execution Optimization
        self.broker_api = BrokerAPI()
        self.primary_brokers = set()
        self.secondary_brokers = set()
        self.tertiary_brokers = set()
        
        # Edge Computing Support
        self.edge_regions = self.config.get('edge_regions', ['us-east', 'us-west', 'eu', 'asia'])
        self.region_brokers = {region: set() for region in self.edge_regions}
        
        # Rate Limiting & Throttling
        self.rate_limiter = defaultdict(lambda: {
            'last_reset': time.time(),
            'requests': 0,
            'max_requests': self.config.get('broker_max_requests_per_second', 100)
        })
        
        # Instrument-Specific Broker Performance
        self.instrument_broker_performance = defaultdict(lambda: defaultdict(lambda: {
            'success_rate': 0.95,
            'latency': float('inf'),
            'slippage': 0.0,
            'volume_capacity': 0.0
        }))
        
        # Async Connection Pool
        self.websocket_connections = {}
        self.connection_status = defaultdict(lambda: {'connected': False, 'last_attempt': 0})
        
        # Recovery State Management
        self.circuit_breakers = defaultdict(lambda: {
            'failures': 0,
            'last_failure': 0,
            'recovery_threshold': self.config.get('circuit_breaker_threshold', 3),
            'recovery_timeout': self.config.get('circuit_breaker_timeout', 60), # seconds
            'is_open': False
        })
        
        # AI Components - Load dynamically to avoid startup issues
        self.ai_components = {}
        
        # Request tracking for debugging
        self.pending_requests = set()
        self.completed_requests = []

    async def initialize(self):
        """Initialize the quantum broker orchestration system with all dependencies"""
        if self._initialization_task is not None:
            # Wait for existing initialization task if called multiple times
            await self._initialization_task
            return
            
        # Create initialization task
        self._initialization_task = asyncio.create_task(self._initialize_internal())
        await self._initialization_task
    
    async def _initialize_internal(self):
        """Internal initialization logic with parallelized setup"""
        logger.info("Initializing QuantumBrokerManager")
        
        # Load AI components in parallel
        ai_load_task = asyncio.create_task(self._load_ai_components())
        
        # Run initialization tasks in parallel
        init_tasks = [
            self.registry.refresh(),
            self.security.initialize(),
            self.risk_engine.initialize(),
            self.execution_engine.initialize(),
            self.trade_monitor.initialize(),
            self._establish_websocket_connections(),
            self._warm_execution_cache(),
            self._load_broker_performance_metrics()
        ]
        
        # Wait for all initialization tasks
        await asyncio.gather(*init_tasks)
        
        # Wait for AI components to load
        await ai_load_task
        
        # Set up broker tiers based on performance and reliability
        await self._configure_broker_tiers()
        
        logger.info("QuantumBrokerManager initialization complete")
    
    async def _load_ai_components(self):
        """Load AI components asynchronously to avoid startup delays"""
        try:
            self.ai_components['execution_optimizer'] = await self.meta_trader.load_component('execution_optimizer')
            self.ai_components['failover_advisor'] = await self.meta_trader.load_component('failover_advisor')
            self.ai_components['latency_predictor'] = await self.meta_trader.load_component('latency_predictor')
            self.ai_components['slippage_estimator'] = await self.meta_trader.load_component('slippage_estimator')
            self.ai_components['broker_selector'] = await self.meta_trader.load_component('broker_selector')
        except Exception as e:
            logger.error("Failed to load AI components", error=str(e))
            # Fall back to rule-based selection
            self.ai_components['execution_optimizer'] = None
            self.ai_components['failover_advisor'] = None
    
    async def _warm_execution_cache(self):
        """Pre-warm execution cache with commonly used data"""
        common_assets = self.config.get('common_assets', ['BTC-USD', 'ETH-USD', 'SPY', 'QQQ'])
        for asset in common_assets:
            cache_key = f"liquidity_{asset}"
            liquidity_data = await self.liquidity_oracle.get_liquidity(asset)
            await self.execution_cache.set(cache_key, liquidity_data)
    
    async def _load_broker_performance_metrics(self):
        """Load historical broker performance metrics from trade history"""
        try:
            # Get historical execution data for the last 24 hours
            history_data = await self.trade_history.get_execution_history(
                hours=24,
                include_metrics=True
            )
            
            # Process broker performance metrics
            for record in history_data:
                broker_id = record.get('broker_id')
                if not broker_id:
                    continue
                
                # Record latency metrics
                if 'execution_ns' in record:
                    self.latency_tracker.record(broker_id, record['execution_ns'])
                
                # Record reliability metrics
                if record.get('status') == 'success':
                    self.reliability_metrics.record_success(broker_id)
                else:
                    self.reliability_metrics.record_failure(broker_id)
                
                # Record instrument-specific performance
                instrument = record.get('symbol')
                if instrument:
                    metrics = self.instrument_broker_performance[instrument][broker_id]
                    # Update success rate using exponential moving average
                    success = 1.0 if record.get('status') == 'success' else 0.0
                    metrics['success_rate'] = 0.95 * metrics['success_rate'] + 0.05 * success
                    
                    # Update latency metrics
                    if 'execution_ns' in record:
                        if metrics['latency'] == float('inf'):
                            metrics['latency'] = record['execution_ns']
                        else:
                            metrics['latency'] = 0.9 * metrics['latency'] + 0.1 * record['execution_ns']
                    
                    # Update slippage metrics
                    if 'slippage' in record:
                        metrics['slippage'] = 0.9 * metrics['slippage'] + 0.1 * record['slippage']
                        
                    # Update volume capacity
                    if 'volume' in record:
                        metrics['volume_capacity'] = max(metrics['volume_capacity'], record['volume'])
                    
            logger.info("Loaded broker performance metrics", 
                        broker_count=len(self.latency_tracker.latencies))
                        
        except Exception as e:
            logger.error("Failed to load broker performance metrics", error=str(e))
    
    async def _configure_broker_tiers(self):
        """Configure broker tiers based on performance metrics"""
        try:
            # Get all available brokers
            all_brokers = await self.registry.get_all_brokers()
            
            # Reset broker tiers
            self.primary_brokers = set()
            self.secondary_brokers = set()
            self.tertiary_brokers = set()
            
            # Configure region-specific brokers
            for region in self.edge_regions:
                self.region_brokers[region] = set()
            
            # Rank brokers by performance
            for broker in all_brokers:
                # Skip brokers with insufficient data
                if self.latency_tracker.counters.get(broker.id, 0) < 10:
                    self.secondary_brokers.add(broker.id)
                    continue
                
                # Calculate broker score based on latency and reliability
                latency_score = 1.0 / (1.0 + self.latency_tracker.get_percentile(broker.id) / 1e6)
                reliability_score = self.reliability_metrics.get_success_rate(broker.id)
                
                # Combined score with reliability weighted more heavily
                combined_score = 0.4 * latency_score + 0.6 * reliability_score
                
                # Assign broker to tier based on score
                if combined_score >= 0.9:
                    self.primary_brokers.add(broker.id)
                elif combined_score >= 0.7:
                    self.secondary_brokers.add(broker.id)
                else:
                    self.tertiary_brokers.add(broker.id)
                
                # Assign broker to region
                if hasattr(broker, 'region') and broker.region in self.edge_regions:
                    self.region_brokers[broker.region].add(broker.id)
            
            # Ensure at least one broker in each tier
            if not self.primary_brokers and self.secondary_brokers:
                # Promote best secondary broker to primary
                best_secondary = min(self.secondary_brokers, 
                                     key=lambda b: self.latency_tracker.get_percentile(b))
                self.primary_brokers.add(best_secondary)
                self.secondary_brokers.remove(best_secondary)
            
            if not self.secondary_brokers and self.tertiary_brokers:
                # Promote best tertiary broker to secondary
                best_tertiary = min(self.tertiary_brokers, 
                                    key=lambda b: self.latency_tracker.get_percentile(b))
                self.secondary_brokers.add(best_tertiary)
                self.tertiary_brokers.remove(best_tertiary)
            
            logger.info("Configured broker tiers", 
                        primary_count=len(self.primary_brokers),
                        secondary_count=len(self.secondary_brokers),
                        tertiary_count=len(self.tertiary_brokers))
                        
        except Exception as e:
            logger.error("Failed to configure broker tiers", error=str(e))
    
    async def _establish_websocket_connections(self):
        """Establish persistent WebSocket connections to brokers for low-latency execution"""
        try:
            # Get primary brokers that support WebSocket
            brokers = await self.registry.get_brokers_by_feature('websocket')
            
            # Establish connections in parallel
            connection_tasks = []
            for broker in brokers:
                if broker.websocket_endpoint:
                    task = asyncio.create_task(
                        self._connect_websocket(broker.id, broker.websocket_endpoint)
                    )
                    connection_tasks.append(task)
            
            # Wait for all connections (with timeout)
            await asyncio.gather(*connection_tasks, return_exceptions=True)
            
            # Log connection status
            connected = sum(1 for status in self.connection_status.values() if status['connected'])
            logger.info("Established WebSocket connections", 
                        connected=connected, 
                        total=len(brokers))
                        
        except Exception as e:
            logger.error("Failed to establish WebSocket connections", error=str(e))
    
    async def _connect_websocket(self, broker_id, endpoint):
        """Establish a WebSocket connection to a broker"""
        try:
            # Set last attempt time
            self.connection_status[broker_id]['last_attempt'] = time.time()
            
            # Create authentication headers
            auth_headers = await self._create_auth_headers(broker_id)
            
            # Connect to WebSocket
            connection = await self.websocket_handler.connect(
                endpoint,
                headers=auth_headers,
                timeout=self.config.get('websocket_timeout', 5)
            )
            
            # Store connection
            self.websocket_connections[broker_id] = connection
            self.connection_status[broker_id]['connected'] = True
            
            # Set up message handler
            asyncio.create_task(self._handle_websocket_messages(broker_id, connection))
            
            logger.info("Connected to broker WebSocket", broker_id=broker_id)
            
        except Exception as e:
            logger.warning("Failed to connect to broker WebSocket", 
                          broker_id=broker_id, 
                          error=str(e))
            self.connection_status[broker_id]['connected'] = False
    
    async def _handle_websocket_messages(self, broker_id, connection):
        """Handle incoming WebSocket messages from a broker"""
        try:
            async for message in connection:
                # Parse message
                try:
                    data = json.loads(message)
                    
                    # Handle different message types
                    if 'type' in data:
                        if data['type'] == 'execution':
                            # Process execution confirmation
                            await self._process_execution_confirmation(broker_id, data)
                        elif data['type'] == 'heartbeat':
                            # Update broker health
                            await self._update_broker_health(broker_id, data)
                        elif data['type'] == 'error':
                            # Handle error message
                            await self._handle_broker_error(broker_id, data)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from broker WebSocket", 
                                  broker_id=broker_id)
                    
        except Exception as e:
            logger.error("WebSocket connection error", 
                        broker_id=broker_id, 
                        error=str(e))
            
            # Mark connection as disconnected
            self.connection_status[broker_id]['connected'] = False
            
            # Clean up connection
            if broker_id in self.websocket_connections:
                del self.websocket_connections[broker_id]
                
            # Reconnect after delay
            await asyncio.sleep(self.config.get('reconnect_delay', 5))
            asyncio.create_task(
                self._connect_websocket(
                    broker_id, 
                    (await self.registry.get_broker(broker_id)).websocket_endpoint
                )
            )
    
    async def _update_broker_health(self, broker_id, data):
        """Update broker health status based on WebSocket heartbeat"""
        self._last_health_check[broker_id] = time.time()
        
        # Extract health metrics
        health_metrics = data.get('metrics', {})
        
        # Update registry with health information
        await self.registry.update_broker_health(
            broker_id,
            is_healthy=True,
            metrics=health_metrics
        )
    
    async def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an order through the optimal broker with automatic failover
        and distributed execution capabilities
        """
        if not self._initialized or self._initialization_task is None:
            await self.initialize()
        
        # Generate unique order ID if not provided
        if 'id' not in order:
            order['id'] = str(uuid.uuid4())
        
        # Record start time for latency tracking
        start_time = time.time_ns()
        
        try:
            # Phase 1: Pre-execution validation
            validation_result = await self._validate_order(order)
            if not validation_result['valid']:
                return {
                    "status": "rejected",
                    "reason": validation_result['reason'],
                    "details": validation_result.get('details', {}),
                    "order_id": order['id'],
                    "timestamp": time.time_ns()
                }
            
            # Phase 2: Execution strategy calculation
            strategy = await self._calculate_execution_strategy(order)
            
            # Phase 3: Dynamic broker selection
            broker = await self._select_optimal_broker(order, strategy)
            if not broker:
                return {
                    "status": "rejected",
                    "reason": "no_broker_available",
                    "order_id": order['id'],
                    "timestamp": time.time_ns()
                }
            
            # Phase 4: Atomic execution with failover
            execution_result = await self._execute_with_failover(order, broker, strategy)
            
            # Phase 5: Post-execution verification and recording
            if execution_result['status'] == 'success':
                verification_result = await self._verify_execution(order, execution_result, broker)
                if not verification_result['valid']:
                    # Handle verification failure
                    logger.warning("Execution verification failed", 
                                  order_id=order['id'],
                                  broker_id=broker.id,
                                  reason=verification_result['reason'])
                    
                    # Trigger audit if verification fails
                    await self._trigger_execution_audit(order, execution_result, broker)
                    
                    # Still return the result but flag the verification issue
                    execution_result['verification_failed'] = True
                    execution_result['verification_reason'] = verification_result['reason']
            
            # Calculate and add execution latency
            execution_latency = time.time_ns() - start_time
            execution_result['execution_latency_ns'] = execution_latency
            
            # Record execution metrics
            await self._record_execution_metrics(order, execution_result, broker, execution_latency)
            
            return execution_result
            
        except Exception as e:
            end_time = time.time_ns()
            logger.error("Order execution failed", 
                        order_id=order.get('id', 'unknown'),
                        error=str(e),
                        execution_time_ns=end_time - start_time)
            
            return {
                "status": "error",
                "reason": str(e),
                "order_id": order.get('id', 'unknown'),
                "timestamp": time.time_ns()
            }
    
    async def _validate_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive order validation with multi-layer checks
        """
        try:
            # Check order structure
            required_fields = ['symbol', 'side', 'quantity', 'type']
            missing_fields = [field for field in required_fields if field not in order]
            if missing_fields:
                return {
                    'valid': False,
                    'reason': 'missing_fields',
                    'details': {'missing': missing_fields}
                }
            
            # Run validations in parallel
            validation_tasks = [
                self.risk_engine.validate_order(order),
                self.liquidity_oracle.verify_liquidity(order['symbol'], order['quantity']),
                self._check_security_constraints(order),
                self._check_regulatory_compliance(order)
            ]
            
            validation_results = await asyncio.gather(*validation_tasks)
            
            # Check if any validation failed
            for i, result in enumerate(validation_results):
                if not result.get('valid', False):
                    validation_type = ['risk', 'liquidity', 'security', 'regulatory'][i]
                    return {
                        'valid': False,
                        'reason': f'{validation_type}_validation_failed',
                        'details': result
                    }
            
            # Calculate potential market impact
            market_impact = await self.market_impact.calculate(
                order['symbol'], 
                order['side'], 
                order['quantity']
            )
            
            # Check if market impact is too high
            if market_impact > self.config.get('max_market_impact', 0.05):
                return {
                    'valid': False,
                    'reason': 'market_impact_too_high',
                    'details': {'impact': market_impact}
                }
            
            return {'valid': True}
            
        except Exception as e:
            logger.error("Order validation error", error=str(e))
            return {
                'valid': False,
                'reason': 'validation_error',
                'details': {'error': str(e)}
            }
    
    async def _check_rate_limits(self, order: Dict[str, Any]) -> bool:
        """Check if the order exceeds rate limits for the instrument or account"""
        # Get instrument symbol
        symbol = order.get('symbol')
        
        # Check instrument-specific rate limits
        rate_limit = self.rate_limiter[symbol]
        current_time = time.time()
        
        # Reset counter if we're in a new time window
        if current_time - rate_limit['last_reset'] > 1.0:  # 1 second window
            rate_limit['requests'] = 0
            rate_limit['last_reset'] = current_time
        
        # Check if we've exceeded the rate limit
        if rate_limit['requests'] >= rate_limit['max_requests']:
            return False
        
        # Increment request counter
        rate_limit['requests'] += 1
        return True
    async def _check_regulatory_compliance(self, order: Dict[str, Any]) -> Dict[str, bool]:
        """Check order against regulatory constraints and compliance rules"""
        # Get key order details
        symbol = order.get('symbol')
        quantity = order.get('quantity', 0)
        side = order.get('side')
        
        # Quick vectorized check using NumPy for performance
        try:
            # Get relevant compliance rules
            compliance_rules = await CPU_POOL.submit(
                self.risk_engine.get_compliance_rules,
                symbol
            )
            
            # Check position limits
            current_position = await self.trade_monitor.get_position(symbol)
            if side == 'buy':
                new_position = current_position + quantity
            else:
                new_position = current_position - quantity
                
            # Check if new position exceeds limits
            max_position = compliance_rules.get('max_position', float('inf'))
            if abs(new_position) > max_position:
                return {'valid': False, 'reason': 'position_limit_exceeded'}
                
            # Check for restricted trading periods
            current_hour = time.localtime().tm_hour
            restricted_hours = compliance_rules.get('restricted_hours', [])
            if current_hour in restricted_hours:
                return {'valid': False, 'reason': 'restricted_trading_period'}
                
            # All checks passed
            return {'valid': True}
            
        except Exception as e:
            logger.error("Regulatory compliance check failed", error=str(e))
            # Fail-safe: reject if we can't verify compliance
            return {'valid': False, 'reason': 'compliance_check_error'}

    async def _calculate_execution_strategy(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal execution strategy for the order"""
        # Get order details
        symbol = order.get('symbol')
        quantity = order.get('quantity', 0)
        side = order.get('side')
        order_type = order.get('type')
        
        # Check if we have AI execution optimizer
        if self.ai_components.get('execution_optimizer'):
            # Use AI for intelligent execution strategy
            market_data = await self.liquidity_oracle.get_liquidity(symbol)
            
            # Fast-path for time-sensitive execution
            if order.get('time_in_force') == 'IOC' or quantity < 100:
                return {
                    'strategy': 'single_broker',
                    'time_slice': 1,
                    'aggressive_factor': 0.8
                }
            
            # Use CPU pool for intensive calculations
            strategy = await CPU_POOL.submit(
                self.ai_components['execution_optimizer'].optimize,
                symbol, side, quantity, market_data
            )
            
            # Add caching for similar orders
            cache_key = f"strategy_{symbol}_{side}_{quantity}"
            await self.execution_cache.set(cache_key, strategy)
            
            return strategy
        else:
            # Fallback to rule-based execution strategy
            if quantity > self.config.get('large_order_threshold', 1000):
                # TWAP strategy for large orders
                return {
                    'strategy': 'time_sliced',
                    'slices': min(5, int(quantity / 100)),
                    'interval_seconds': 10
                }
            else:
                # Simple execution for smaller orders
                return {
                    'strategy': 'single_broker',
                    'time_slice': 1,
                    'aggressive_factor': 0.5
                }

    async def _select_optimal_broker(self, order: Dict[str, Any], 
                                    strategy: Dict[str, Any]) -> Optional[Any]:
        """Select the optimal broker based on order characteristics and execution strategy"""
        try:
            symbol = order.get('symbol')
            quantity = order.get('quantity', 0)
            
            # Fast path: check cache for recent broker selection
            cache_key = f"broker_{symbol}_{quantity}"
            cached_broker = await self.execution_cache.get(cache_key)
            if cached_broker and time.time() - cached_broker.get('timestamp', 0) < 5:
                broker_id = cached_broker.get('broker_id')
                return await self.registry.get_broker(broker_id)
            
            # Get candidate brokers
            candidate_brokers = []
            
            # Primary search: instrument-specific performance
            if symbol in self.instrument_broker_performance:
                # Sort brokers by success rate for this instrument
                perf_data = self.instrument_broker_performance[symbol]
                broker_scores = []
                
                for broker_id, metrics in perf_data.items():
                    # Skip brokers with circuit breakers open
                    if self.circuit_breakers[broker_id]['is_open']:
                        continue
                        
                    # Calculate combined score (weighted by importance)
                    success_weight = 0.4
                    latency_weight = 0.3
                    slippage_weight = 0.2
                    capacity_weight = 0.1
                    
                    # Normalize latency (lower is better)
                    norm_latency = 1.0 / (1.0 + metrics['latency'] / 1e6)
                    
                    # Calculate combined score
                    score = (
                        success_weight * metrics['success_rate'] +
                        latency_weight * norm_latency +
                        slippage_weight * (1.0 - min(1.0, metrics['slippage'])) +
                        capacity_weight * min(1.0, quantity / max(1.0, metrics['volume_capacity']))
                    )
                    
                    broker_scores.append((broker_id, score))
                
                # Sort by score (highest first)
                broker_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Take top 3 brokers
                top_brokers = [bs[0] for bs in broker_scores[:3]]
                for broker_id in top_brokers:
                    broker = await self.registry.get_broker(broker_id)
                    if broker:
                        candidate_brokers.append(broker)
            
            # Fallback if no instrument-specific brokers found
            if not candidate_brokers:
                # Use primary brokers first
                for broker_id in self.primary_brokers:
                    # Skip brokers with circuit breakers open
                    if self.circuit_breakers[broker_id]['is_open']:
                        continue
                    broker = await self.registry.get_broker(broker_id)
                    if broker:
                        candidate_brokers.append(broker)
            
            # Further fallback to secondary brokers if needed
            if not candidate_brokers:
                for broker_id in self.secondary_brokers:
                    # Skip brokers with circuit breakers open
                    if self.circuit_breakers[broker_id]['is_open']:
                        continue
                    broker = await self.registry.get_broker(broker_id)
                    if broker:
                        candidate_brokers.append(broker)
            
            # Final fallback to any available broker
            if not candidate_brokers:
                all_brokers = await self.registry.get_all_brokers()
                for broker in all_brokers:
                    # Skip brokers with circuit breakers open
                    if self.circuit_breakers[broker.id]['is_open']:
                        continue
                    candidate_brokers.append(broker)
            
            # If we have candidates, pick the best one
            if candidate_brokers:
                # If AI broker selector is available, use it
                if self.ai_components.get('broker_selector'):
                    # Prepare broker data for AI selection
                    broker_data = []
                    for broker in candidate_brokers:
                        broker_data.append({
                            'id': broker.id,
                            'latency': self.latency_tracker.get_percentile(broker.id),
                            'success_rate': self.reliability_metrics.get_success_rate(broker.id),
                            'features': broker.features
                        })
                    
                    # Get AI selection
                    selected_broker_id = await CPU_POOL.submit(
                        self.ai_components['broker_selector'].select,
                        broker_data, order, strategy
                    )
                    
                    # Find the selected broker
                    for broker in candidate_brokers:
                        if broker.id == selected_broker_id:
                            # Cache the selection
                            await self.execution_cache.set(cache_key, {
                                'broker_id': broker.id,
                                'timestamp': time.time()
                            })
                            return broker
                
                # Fallback: return the first candidate
                broker = candidate_brokers[0]
                await self.execution_cache.set(cache_key, {
                    'broker_id': broker.id,
                    'timestamp': time.time()
                })
                return broker
            
            # No brokers available
            return None
            
        except Exception as e:
            logger.error("Broker selection error", error=str(e))
            return None

    async def _execute_with_failover(self, order: Dict[str, Any], 
                                broker: Any, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute order with automatic failover to backup brokers if needed"""
        try:
            # Track order in pending requests
            request_id = str(uuid.uuid4())
            self.pending_requests.add(request_id)
            
            # Create execution context
            execution_context = {
                'order': order,
                'broker': broker,
                'strategy': strategy,
                'attempt': 1,
                'start_time': time.time_ns(),
                'failover_attempts': 0,
                'max_attempts': self.config.get('max_execution_attempts', 3)
            }
            
            # Execute order with primary broker
            result = await self._execute_single(execution_context)
            
            # Check if execution succeeded
            if result['status'] == 'success':
                # Mark broker as reliable
                self.reliability_metrics.record_success(broker.id)
                
                # Reset circuit breaker if it was partially open
                if self.circuit_breakers[broker.id]['failures'] > 0:
                    self.circuit_breakers[broker.id]['failures'] = 0
                    
                # Update latency metrics
                execution_time = time.time_ns() - execution_context['start_time']
                self.latency_tracker.record(broker.id, execution_time)
                
                # Complete order tracking
                self.pending_requests.remove(request_id)
                self.completed_requests.append({
                    'id': request_id,
                    'order_id': order['id'],
                    'broker_id': broker.id,
                    'timestamp': time.time_ns(),
                    'execution_time_ns': execution_time
                })
                
                return result
            
            # Execution failed, try failover
            logger.warning("Execution failed, attempting failover", 
                        order_id=order['id'],
                        broker_id=broker.id,
                        reason=result.get('reason', 'unknown'))
            
            # Record failure
            self.reliability_metrics.record_failure(broker.id)
            
            # Update circuit breaker
            circuit = self.circuit_breakers[broker.id]
            circuit['failures'] += 1
            circuit['last_failure'] = time.time()
            
            # Open circuit breaker if too many failures
            if circuit['failures'] >= circuit['recovery_threshold']:
                circuit['is_open'] = True
                logger.warning("Circuit breaker opened for broker", 
                            broker_id=broker.id,
                            failures=circuit['failures'])
                
                # Schedule circuit breaker reset
                asyncio.create_task(self._reset_circuit_breaker(broker.id))
            
            # Try failover if we haven't exceeded max attempts
            if execution_context['attempt'] < execution_context['max_attempts']:
                # Get failover broker
                failover_broker = await self._get_failover_broker(broker, order)
                if failover_broker:
                    # Update execution context
                    execution_context['broker'] = failover_broker
                    execution_context['attempt'] += 1
                    execution_context['failover_attempts'] += 1
                    
                    # Execute with failover broker
                    logger.info("Attempting failover execution", 
                            order_id=order['id'],
                            original_broker=broker.id,
                            failover_broker=failover_broker.id)
                    
                    return await self._execute_with_failover(order, failover_broker, strategy)
            
            # All attempts failed
            self.pending_requests.remove(request_id)
            return {
                'status': 'failed',
                'reason': 'all_attempts_failed',
                'order_id': order['id'],
                'attempts': execution_context['attempt'],
                'timestamp': time.time_ns()
            }
            
        except Exception as e:
            logger.error("Execution failover error", 
                        order_id=order.get('id', 'unknown'),
                        error=str(e))
            
            if request_id in self.pending_requests:
                self.pending_requests.remove(request_id)
                
            return {
                'status': 'error',
                'reason': str(e),
                'order_id': order.get('id', 'unknown'),
                'timestamp': time.time_ns()
            }

    async def _execute_single(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute order with a single broker"""
        order = context['order']
        broker = context['broker']
        strategy = context['strategy']
        
        try:
            # Acquire broker lock to prevent concurrent operations on same broker
            async with self._broker_locks[broker.id]:
                # Check if we can use WebSocket for faster execution
                if broker.id in self.websocket_connections and self.connection_status[broker.id]['connected']:
                    # Create WebSocket execution message
                    execution_message = {
                        'type': 'execution',
                        'order': order,
                        'timestamp': time.time_ns(),
                        'nonce': str(uuid.uuid4())
                    }
                    
                    # Add authentication
                    signature = await self._create_order_signature(broker.id, execution_message)
                    execution_message['signature'] = signature
                    
                    # Send order via WebSocket
                    ws = self.websocket_connections[broker.id]
                    await ws.send(json.dumps(execution_message))
                    
                    # Wait for response with timeout
                    timeout = self.config.get('execution_timeout_ms', 1000) / 1000.0
                    response = await asyncio.wait_for(
                        self._wait_for_execution_response(broker.id, order['id']),
                        timeout=timeout
                    )
                    
                    return response
                else:
                    # Fallback to REST API
                    logger.debug("Using REST API for execution", 
                                broker_id=broker.id, 
                                order_id=order['id'])
                    
                    # Use the broker API for execution
                    return await self.broker_api.execute_order(
                        broker_id=broker.id,
                        order=order,
                        strategy=strategy
                    )
        
        except asyncio.TimeoutError:
            # Execution timed out
            logger.warning("Execution timed out", 
                        broker_id=broker.id, 
                        order_id=order['id'])
            return {
                'status': 'failed',
                'reason': 'timeout',
                'order_id': order['id'],
                'broker_id': broker.id,
                'timestamp': time.time_ns()
            }
            
        except Exception as e:
            # Execution failed with exception
            logger.error("Execution error", 
                        broker_id=broker.id, 
                        order_id=order['id'],
                        error=str(e))
            return {
                'status': 'failed',
                'reason': str(e),
                'order_id': order['id'],
                'broker_id': broker.id,
                'timestamp': time.time_ns()
            }

    async def _wait_for_execution_response(self, broker_id: str, order_id: str) -> Dict[str, Any]:
        """Wait for execution response from WebSocket"""
        # Create response queue
        response_queue = asyncio.Queue()
        
        # Define message handler
        async def _message_handler(message):
            try:
                data = json.loads(message)
                if (data.get('type') == 'execution_response' and 
                    data.get('order_id') == order_id):
                    await response_queue.put(data)
            except Exception:
                pass
        
        # Register handler with WebSocket
        handler_id = await self.websocket_handler.add_message_handler(
            broker_id, _message_handler
        )
        
        try:
            # Wait for response
            response = await response_queue.get()
            return response
        finally:
            # Clean up handler
            await self.websocket_handler.remove_message_handler(broker_id, handler_id)

    async def _get_failover_broker(self, failed_broker: Any, order: Dict[str, Any]) -> Optional[Any]:
        """Get the best failover broker for an order when the primary broker fails"""
        try:
            # Get all available brokers excluding the failed one
            all_brokers = await self.registry.get_all_brokers()
            candidates = [b for b in all_brokers if b.id != failed_broker.id and 
                        not self.circuit_breakers[b.id]['is_open']]
            
            if not candidates:
                return None
                
            # If failover advisor AI is available, use it
            if self.ai_components.get('failover_advisor'):
                # Prepare broker data
                broker_data = []
                for broker in candidates:
                    broker_data.append({
                        'id': broker.id,
                        'features': broker.features,
                        'latency': self.latency_tracker.get_percentile(broker.id),
                        'success_rate': self.reliability_metrics.get_success_rate(broker.id)
                    })
                    
                # Get AI recommendation
                failover_id = await CPU_POOL.submit(
                    self.ai_components['failover_advisor'].recommend,
                    broker_data, order, failed_broker.id
                )
                
                # Find recommended broker
                for broker in candidates:
                    if broker.id == failover_id:
                        return broker
            
            # Fallback: sort by reliability and latency
            candidates.sort(key=lambda b: (
                -self.reliability_metrics.get_success_rate(b.id),
                self.latency_tracker.get_percentile(b.id)
            ))
            
            return candidates[0] if candidates else None
            
        except Exception as e:
            logger.error("Failover broker selection error", error=str(e))
            
            # Last resort: return any broker that's not the failed one
            try:
                all_brokers = await self.registry.get_all_brokers()
                for broker in all_brokers:
                    if broker.id != failed_broker.id and not self.circuit_breakers[broker.id]['is_open']:
                        return broker
            except:
                pass
                
            return None

    async def _reset_circuit_breaker(self, broker_id: str):
        """Reset circuit breaker after timeout period"""
        # Get circuit breaker
        circuit = self.circuit_breakers[broker_id]
        
        # Wait for recovery timeout
        await asyncio.sleep(circuit['recovery_timeout'])
        
        # Reset circuit breaker
        circuit['is_open'] = False
        circuit['failures'] = 0
        
        logger.info("Circuit breaker reset for broker", broker_id=broker_id)

    async def _verify_execution(self, order: Dict[str, Any], 
                            result: Dict[str, Any], broker: Any) -> Dict[str, bool]:
        """Verify execution result matches order parameters"""
        try:
            # Check basic verification
            if result.get('status') != 'success':
                return {'valid': False, 'reason': 'execution_failed'}
                
            # Check executed quantity
            executed_quantity = result.get('executed_quantity', 0)
            if executed_quantity <= 0:
                return {'valid': False, 'reason': 'zero_executed_quantity'}
                
            # Check if executed quantity matches order
            order_quantity = order.get('quantity', 0)
            if not order.get('allow_partial', False) and executed_quantity < order_quantity:
                return {
                    'valid': False, 
                    'reason': 'incomplete_execution',
                    'details': {
                        'ordered': order_quantity,
                        'executed': executed_quantity
                    }
                }
                
            # Check execution price for limit orders
            if order.get('type') == 'limit':
                limit_price = order.get('price', 0)
                execution_price = result.get('price', 0)
                
                # Validate price based on side
                if order.get('side') == 'buy' and execution_price > limit_price:
                    return {
                        'valid': False,
                        'reason': 'price_worse_than_limit',
                        'details': {
                            'limit': limit_price,
                            'execution': execution_price
                        }
                    }
                elif order.get('side') == 'sell' and execution_price < limit_price:
                    return {
                        'valid': False,
                        'reason': 'price_worse_than_limit',
                        'details': {
                            'limit': limit_price,
                            'execution': execution_price
                        }
                    }
                    
            # Check execution timestamp
            if 'timestamp' in result:
                current_time = time.time_ns()
                execution_time = result['timestamp']
                time_diff = (current_time - execution_time) / 1e9  # in seconds
                
                # If execution timestamp is too far in the past or future
                if abs(time_diff) > self.config.get('max_execution_time_diff', 60):
                    return {
                        'valid': False,
                        'reason': 'suspicious_timestamp',
                        'details': {
                            'time_diff_seconds': time_diff
                        }
                    }
                    
            return {'valid': True}
            
        except Exception as e:
            logger.error("Execution verification error", error=str(e))
            return {'valid': False, 'reason': 'verification_error'}

    async def _trigger_execution_audit(self, order: Dict[str, Any], 
                                    result: Dict[str, Any], broker: Any):
        """Trigger audit for suspicious execution results"""
        try:
            # Record audit event
            audit_data = {
                'order': order,
                'result': result,
                'broker_id': broker.id,
                'timestamp': time.time_ns(),
                'reason': result.get('verification_reason', 'unknown')
            }
            
            # Add to audit log
            await self.trade_history.add_audit_event(audit_data)
            
            # Notify risk engine
            await self.risk_engine.notify_execution_audit(audit_data)
            
            logger.warning("Triggered execution audit", 
                        order_id=order['id'],
                        broker_id=broker.id,
                        reason=audit_data['reason'])
                        
        except Exception as e:
            logger.error("Failed to trigger execution audit", error=str(e))

    async def _record_execution_metrics(self, order: Dict[str, Any], 
                                    result: Dict[str, Any], broker: Any,
                                    latency_ns: int):
        """Record execution metrics for performance tracking"""
        try:
            # Create metrics record
            metrics = {
                'order_id': order['id'],
                'broker_id': broker.id,
                'symbol': order.get('symbol'),
                'side': order.get('side'),
                'quantity': order.get('quantity'),
                'executed_quantity': result.get('executed_quantity'),
                'type': order.get('type'),
                'latency_ns': latency_ns,
                'timestamp': time.time_ns(),
                'status': result.get('status')
            }
            
            # Add price information if available
            if 'price' in result:
                metrics['execution_price'] = result['price']
            if 'price' in order:
                metrics['order_price'] = order['price']
                
            # Calculate slippage if possible
            if 'price' in order and 'price' in result:
                base_price = order['price']
                exec_price = result['price']
                side_factor = 1 if order.get('side') == 'sell' else -1
                metrics['slippage'] = side_factor * (exec_price - base_price) / base_price
            
            # Add to execution history
            await self.trade_history.add_execution_metrics(metrics)
            
            # Update broker performance tracking
            if result.get('status') == 'success':
                self.reliability_metrics.record_success(broker.id)
                
                # Update instrument-specific metrics
                symbol = order.get('symbol')
                if symbol:
                    perf = self.instrument_broker_performance[symbol][broker.id]
                    
                    # Update success rate with exponential decay
                    perf['success_rate'] = 0.95 * perf['success_rate'] + 0.05 * 1.0
                    
                    # Update latency with exponential decay
                    if perf['latency'] == float('inf'):
                        perf['latency'] = latency_ns
                    else:
                        perf['latency'] = 0.9 * perf['latency'] + 0.1 * latency_ns
                    
                    # Update slippage if available
                    if 'slippage' in metrics:
                        perf['slippage'] = 0.9 * perf['slippage'] + 0.1 * abs(metrics['slippage'])
                    
                    # Update volume capacity
                    if 'executed_quantity' in result:
                        perf['volume_capacity'] = max(
                            perf['volume_capacity'], 
                            result['executed_quantity']
                        )
            else:
                self.reliability_metrics.record_failure(broker.id)
            
        except Exception as e:
            logger.error("Failed to record execution metrics", error=str(e))

    async def _create_auth_headers(self, broker_id: str) -> Dict[str, str]:
        """Create authentication headers for broker API requests"""
        try:
            # Get broker credentials
            credentials = await self.vault.get_broker_credentials(broker_id)
            
            # Create authentication headers
            timestamp = str(int(time.time() * 1000))
            nonce = hashlib.sha256(os.urandom(16)).hexdigest()
            
            # Create signature payload
            payload = f"{timestamp}{nonce}{broker_id}"
            
            # Create HMAC signature
            signature = hmac.new(
                credentials['api_secret'].encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Return headers
            return {
                'X-API-Key': credentials['api_key'],
                'X-Timestamp': timestamp,
                'X-Nonce': nonce,
                'X-Signature': signature
            }
            
        except Exception as e:
            logger.error("Failed to create auth headers", broker_id=broker_id, error=str(e))
            return {}

    async def _create_order_signature(self, broker_id: str, data: Dict[str, Any]) -> str:
        """Create cryptographic signature for order data"""
        try:
            # Get broker credentials
            credentials = await self.vault.get_broker_credentials(broker_id)
            
            # Convert data to canonical string representation
            ordered_data = json.dumps(data, sort_keys=True)
            
            # Create HMAC signature
            signature = hmac.new(
                credentials['api_secret'].encode(),
                ordered_data.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            logger.error("Failed to create order signature", broker_id=broker_id, error=str(e))
            return ""

    async def _process_execution_confirmation(self, broker_id: str, data: Dict[str, Any]):
        """Process execution confirmation from WebSocket"""
        try:
            # Verify the confirmation
            order_id = data.get('order_id')
            if not order_id:
                return
                
            # Get original order if available
            # (WebSocket callbacks might be for orders we didn't track directly)
            original_order = None
            for req in self.completed_requests:
                if req.get('order_id') == order_id:
                    original_order = req
                    break
            
            # Update trade history
            await self.trade_history.update_execution_status(
                order_id=order_id,
                broker_id=broker_id,
                status=data.get('status', 'unknown'),
                details=data
            )
            
            # Notify trade monitor
            await self.trade_monitor.process_execution_confirmation(
                broker_id=broker_id,
                data=data
            )
            
            logger.debug("Processed execution confirmation", 
                        broker_id=broker_id,
                        order_id=order_id,
                        status=data.get('status'))
                        
        except Exception as e:
            logger.error("Failed to process execution confirmation", error=str(e))

    async def _handle_broker_error(self, broker_id: str, data: Dict[str, Any]):
        """Handle error message from broker WebSocket"""
        try:
            error_code = data.get('code', 'unknown')
            error_msg = data.get('message', 'Unknown error')
            order_id = data.get('order_id')
            
            # Log the error
            logger.warning("Broker error received", 
                        broker_id=broker_id,
                        error_code=error_code,
                        error_message=error_msg,
                        order_id=order_id)
            
            # Update circuit breaker if error is severe
            severe_error_codes = self.config.get('severe_error_codes', ['AUTH_FAILED', 'RATE_LIMIT'])
            if error_code in severe_error_codes:
                circuit = self.circuit_breakers[broker_id]
                circuit['failures'] += 1
                circuit['last_failure'] = time.time()
                
                # Open circuit breaker if too many failures
                if circuit['failures'] >= circuit['recovery_threshold']:
                    circuit['is_open'] = True
                    logger.warning("Circuit breaker opened due to error", 
                                broker_id=broker_id,
                                error_code=error_code)
                    
                    # Schedule circuit breaker reset
                    asyncio.create_task(self._reset_circuit_breaker(broker_id))
            
            # Update reliability metrics
            self.reliability_metrics.record_failure(broker_id)
            
            # If the error is related to a specific order, update its status
            if order_id:
                await self.trade_history.update_execution_status(
                    order_id=order_id,
                    status="ERROR",
                    error_code=error_code,
                    error_message=error_msg,
                    timestamp=time.time(),
                    # Add additional broker-specific error details
                    broker_details=data.get('broker_details', {})
                )
                
                # Record execution metrics for AI optimization
                execution_metrics = {
                    'broker_id': broker_id,
                    'order_id': order_id,
                    'error_type': error_code,
                    'timestamp': time.time(),
                    'latency': data.get('latency'),
                    'market_conditions': data.get('market_conditions', {}),
                    'severity': 'high' if error_code in severe_error_codes else 'medium'
                }
                
                # Store execution metrics for ML-based broker optimization
                await self.metrics_store.record_execution_metrics(execution_metrics)
                
                # Trigger failover if needed for critical orders
                order_info = await self.trade_history.get_order_details(order_id) if order_id else None
                if order_info and order_info.get('priority') == 'high':
                    logger.info("Initiating broker failover for high-priority order", 
                                order_id=order_id, 
                                original_broker=broker_id)
                    await self._initiate_broker_failover(order_id, broker_id, error_code)
                
                # Update AI broker selection model with negative reinforcement
                await self.broker_optimizer.update_model(
                    broker_id=broker_id,
                    execution_result='error',
                    error_type=error_code,
                    market_context=data.get('market_context', {})
                )
                
                # Verify if this error requires regulatory reporting
                if error_code in self.config.get('reportable_errors', []):
                    await self.compliance_manager.log_reportable_incident(
                        broker_id=broker_id,
                        order_id=order_id,
                        error_code=error_code,
                        error_message=error_msg,
                        timestamp=time.time()
                    )
                
                # Update risk limits if error indicates potential risk breach
                if error_code in self.config.get('risk_related_errors', []):
                    await self.risk_manager.adjust_broker_risk_limits(
                        broker_id=broker_id,
                        adjustment_factor=0.8,  # Reduce risk limits by 20%
                        reason=f"Error {error_code}: {error_msg}"
                    )
                
                # Create cryptographic hash of the error for verification
                error_hash = self._create_verification_hash(data)
                await self.trade_history.store_execution_verification(
                    order_id=order_id,
                    verification_hash=error_hash,
                    verification_type="error_confirmation"
                )
                
                # Return error information to caller
                return {
                    'success': False,
                    'error_code': error_code,
                    'error_message': error_msg,
                    'broker_id': broker_id,
                    'order_id': order_id,
                    'timestamp': time.time(),
                    'verification_hash': error_hash
                }
        except Exception as e:
            logger.error("Failed to handle broker error", error=str(e), broker_id=broker_id)
            
    def __del__(self):
        """Clean up resources when the broker manager is destroyed."""
        logger.info("Shutting down BrokerManager and releasing resources")
        # Close any open connections or resources
        if hasattr(self, 'metrics_store') and self.metrics_store:
            asyncio.create_task(self.metrics_store.close())
        
        # Ensure all pending operations are completed
        for task in asyncio.all_tasks():
            if not task.done() and task.get_name().startswith('broker_'):
                logger.warning(f"Cancelling pending broker task: {task.get_name()}")
                task.cancel()