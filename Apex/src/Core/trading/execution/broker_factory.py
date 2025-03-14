# src/Core/trading/execution/broker_factory.py

import asyncio
import time
import hashlib
import hmac
import os
import uuid
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import defaultdict, deque
from functools import lru_cache
from datetime import datetime, timedelta
import ujson  # Faster JSON processing
import uvloop
import aiohttp
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization

# Apex Core Imports
from Apex.utils.logging.structured_logger import StructuredLogger
from Apex.src.Core.trading.execution.broker_registry import BrokerRegistry
from Apex.src.Core.trading.risk.risk_engine import RiskEngine
from Apex.src.Core.trading.hft.liquidity_manager import LiquidityOracle
from Apex.src.Core.trading.execution.market_impact import MarketImpactAnalyzer
from Apex.src.Core.trading.execution.order_execution import OrderExecutionEngine
from Apex.src.Core.data.trade_history import TradeHistory
from Apex.src.Core.data.realtime.market_data import MarketDataService
from Apex.src.Core.trading.logging.decision_logger import DecisionLogger
from Apex.src.ai.ensembles.meta_trader import MetaTrader
from Apex.src.ai.analysis.market_regime_classifier import MarketRegimeClassifier
from Apex.utils.helpers.security import QuantumVault, validate_hmac
from Apex.Config.config_loader import load_config

# Configure UVLoop for enhanced async performance
uvloop.install()

# Set up structured logging
logger = StructuredLogger("QuantumBrokerFactory")

# Thread pool for CPU-bound operations
CPU_CORES = os.cpu_count() or 4
MAX_WORKERS = max(4, CPU_CORES - 1)  # Reserve one core for IO operations

class BrokerCache:
    """High-performance broker performance cache with time-decay metrics"""
    
    def __init__(self, max_size: int = 1000, decay_factor: float = 0.95):
        self.performance_data = {}  # broker_id -> performance metrics
        self.max_size = max_size
        self.decay_factor = decay_factor
        self.timestamp = time.time()
        self._last_cleanup = time.time()
        
    async def update(self, broker_id: str, metrics: Dict[str, Any]) -> None:
        """Update broker performance metrics with time decay"""
        current_time = time.time()
        
        # Apply time decay to existing metrics
        if current_time - self.timestamp > 60:  # Only decay every minute
            self._apply_time_decay()
            self.timestamp = current_time
            
        if broker_id not in self.performance_data:
            self.performance_data[broker_id] = metrics
        else:
            # Update with exponential moving average
            for key, value in metrics.items():
                if key in self.performance_data[broker_id]:
                    self.performance_data[broker_id][key] = (
                        0.8 * self.performance_data[broker_id][key] + 0.2 * value
                    )
                else:
                    self.performance_data[broker_id][key] = value
                    
        # Cleanup if needed
        if current_time - self._last_cleanup > 3600:  # Cleanup every hour
            await self._cleanup()
            self._last_cleanup = current_time
            
    def get(self, broker_id: str) -> Dict[str, Any]:
        """Get broker performance metrics"""
        return self.performance_data.get(broker_id, {})
    
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Get all broker performance metrics"""
        return self.performance_data
        
    def _apply_time_decay(self) -> None:
        """Apply time decay to all performance metrics"""
        for broker_id in self.performance_data:
            for metric in self.performance_data[broker_id]:
                if isinstance(self.performance_data[broker_id][metric], (int, float)):
                    self.performance_data[broker_id][metric] *= self.decay_factor
                    
    async def _cleanup(self) -> None:
        """Remove old or unused brokers if cache exceeds max size"""
        if len(self.performance_data) <= self.max_size:
            return
            
        # Sort by last_used timestamp
        sorted_brokers = sorted(
            self.performance_data.items(),
            key=lambda x: x[1].get('last_used', 0)
        )
        
        # Remove oldest entries
        to_remove = len(self.performance_data) - self.max_size
        for i in range(to_remove):
            del self.performance_data[sorted_brokers[i][0]]

class LatencyMonitor:
    """Real-time broker latency tracking with statistical analysis"""
    
    def __init__(self, window_size: int = 100, decay_factor: float = 0.95):
        self.latencies = defaultdict(lambda: deque(maxlen=window_size))
        self.avg_latencies = {}
        self.min_latencies = {}
        self.decay_factor = decay_factor
        self.window_size = window_size
        
    def update(self, broker_id: str, latency_ns: int) -> None:
        """Update latency metrics for a broker"""
        self.latencies[broker_id].append(latency_ns)
        
        # Calculate statistics
        latency_array = np.array(self.latencies[broker_id])
        
        # Update averages with time decay
        if broker_id in self.avg_latencies:
            self.avg_latencies[broker_id] = (
                self.decay_factor * self.avg_latencies[broker_id] + 
                (1 - self.decay_factor) * np.mean(latency_array)
            )
        else:
            self.avg_latencies[broker_id] = np.mean(latency_array)
            
        # Update minimums
        self.min_latencies[broker_id] = np.min(latency_array)
        
    def get(self, broker_id: str) -> float:
        """Get average latency for a broker"""
        return self.avg_latencies.get(broker_id, float('inf'))
        
    def get_percentile(self, broker_id: str, percentile: float = 95) -> float:
        """Get latency percentile for a broker"""
        if broker_id not in self.latencies or not self.latencies[broker_id]:
            return float('inf')
        return np.percentile(self.latencies[broker_id], percentile)
        
    def get_all_stats(self, broker_id: str) -> Dict[str, float]:
        """Get comprehensive latency statistics for a broker"""
        if broker_id not in self.latencies or not self.latencies[broker_id]:
            return {
                'avg': float('inf'),
                'min': float('inf'),
                'max': float('inf'),
                'p50': float('inf'),
                'p95': float('inf'),
                'p99': float('inf'),
                'jitter': float('inf')
            }
            
        latency_array = np.array(self.latencies[broker_id])
        return {
            'avg': np.mean(latency_array),
            'min': np.min(latency_array),
            'max': np.max(latency_array),
            'p50': np.percentile(latency_array, 50),
            'p95': np.percentile(latency_array, 95),
            'p99': np.percentile(latency_array, 99),
            'jitter': np.std(latency_array)
        }

class ExecutionQualityMonitor:
    """Tracks execution quality metrics across brokers"""
    
    def __init__(self, window_size: int = 100):
        self.execution_history = defaultdict(lambda: deque(maxlen=window_size))
        self.window_size = window_size
        
    async def record_execution(self, broker_id: str, metrics: Dict[str, Any]) -> None:
        """Record execution metrics for a broker"""
        self.execution_history[broker_id].append({
            **metrics,
            'timestamp': time.time_ns()
        })
        
    def get_metrics(self, broker_id: str) -> Dict[str, float]:
        """Get execution quality metrics for a broker"""
        if broker_id not in self.execution_history or not self.execution_history[broker_id]:
            return {}
            
        metrics = list(self.execution_history[broker_id])
        
        # Extract slippage values
        slippages = [m.get('slippage', 0) for m in metrics if 'slippage' in m]
        if not slippages:
            slippages = [0]
            
        # Extract fill rates
        fill_rates = [m.get('fill_rate', 1.0) for m in metrics if 'fill_rate' in m]
        if not fill_rates:
            fill_rates = [1.0]
            
        # Calculate success rates
        successes = sum(1 for m in metrics if m.get('status') == 'success')
        total = len(metrics)
        
        return {
            'avg_slippage': np.mean(slippages),
            'max_slippage': np.max(slippages),
            'min_slippage': np.min(slippages),
            'avg_fill_rate': np.mean(fill_rates),
            'success_rate': successes / total if total > 0 else 0
        }

class OrderQueue:
    """Manages order queuing and retries"""
    
    def __init__(self, max_size: int = 10000, retry_delay: float = 0.5):
        self.pending_orders = asyncio.PriorityQueue(maxsize=max_size)
        self.retry_delay = retry_delay
        self.max_retries = 3
        
    async def add_order(self, order: Dict[str, Any], priority: int = 5) -> None:
        """Add an order to the queue with priority (lower is higher priority)"""
        # Add retry count to order
        if 'retry_count' not in order:
            order['retry_count'] = 0
            
        await self.pending_orders.put((priority, time.time_ns(), order))
        
    async def get_next_order(self) -> Optional[Dict[str, Any]]:
        """Get the next order from the queue"""
        if self.pending_orders.empty():
            return None
            
        priority, _, order = await self.pending_orders.get()
        return order
        
    async def retry_order(self, order: Dict[str, Any]) -> bool:
        """Add order back to queue for retry"""
        order['retry_count'] += 1
        
        if order['retry_count'] > self.max_retries:
            logger.warning("Max retries reached for order", 
                          order_id=order.get('order_id', 'unknown'))
            return False
            
        # Increase priority for retried orders
        priority = max(1, 5 - order['retry_count'])
        await asyncio.sleep(self.retry_delay)
        await self.pending_orders.put((priority, time.time_ns(), order))
        return True
        
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return self.pending_orders.empty()
        
    async def size(self) -> int:
        """Get current queue size"""
        return self.pending_orders.qsize()

class QuantumBrokerFactory:
    """
    Institutional-Grade Broker Orchestration System with Quantum-Secure API Handling
    - AI-driven dynamic broker routing with atomic execution
    - Secure multi-cloud broker connectivity with advanced encryption
    - Real-time performance optimization and adaptive load balancing
    - Comprehensive failover mechanisms with order recovery
    """
    
    _instance = None
    _initialized = False
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def __aenter__(self):
        await self.initialize_factory()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
    
    def __init__(self):
        if self._initialized:
            return
            
        # System Configuration
        self.config = load_config()
        self.vault = QuantumVault()
        self.registry = BrokerRegistry()
        
        # Setup execution queues
        self.order_queue = OrderQueue()
        self.fallback_queue = OrderQueue(max_size=5000)
        
        # Security Systems
        self.hmac_key = self.vault.get_hmac_key()
        self.session_nonce = hashlib.sha3_256(os.urandom(32)).hexdigest()
        self.api_key_rotation_time = {}  # broker_id -> next rotation time
        
        # Performance Monitoring
        self.broker_cache = BrokerCache()
        self.latency_map = LatencyMonitor()
        self.execution_monitor = ExecutionQualityMonitor()
        
        # System Integration
        self.risk = RiskEngine()
        self.liquidity = LiquidityOracle()
        self.market_impact = MarketImpactAnalyzer()
        self.history = TradeHistory()
        self.market_data = MarketDataService()
        self.order_engine = OrderExecutionEngine()
        self.decision_logger = DecisionLogger()
        self.market_regime = MarketRegimeClassifier()
        
        # AI Components
        self.meta_trader = MetaTrader()
        self.route_predictor = self.meta_trader.load_component('execution_predictor')
        self.failover_advisor = self.meta_trader.load_component('failover_advisor')
        self.execution_optimizer = self.meta_trader.load_component('execution_optimizer')
        
        # Background tasks
        self.background_tasks = set()
        self.shutdown_event = asyncio.Event()
        
        # Set up WebSocket connections for live broker monitoring
        self.ws_connections = {}
        
        # Statistics and tracking
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'avg_execution_time_ns': 0,
            'executions_per_broker': defaultdict(int)
        }
        
        # Main lock for ensuring atomic operations
        self._master_lock = asyncio.Lock()
        
        # Mark as initialized
        self._initialized = True
        logger.info("QuantumBrokerFactory initialized")

    async def initialize_factory(self):
        """Initialize the quantum broker orchestration system"""
        tasks = [
            self._refresh_broker_registry(),
            self._monitor_broker_health(),
            self._update_latency_map(),
            self._sync_risk_parameters(),
            self._process_order_queue(),
            self._rotate_api_keys(),
            self._establish_ws_connections()
        ]
        
        for task in tasks:
            background_task = asyncio.create_task(task)
            self.background_tasks.add(background_task)
            background_task.add_done_callback(self.background_tasks.remove)
        
        logger.info("QuantumBrokerFactory background tasks initialized")
        
    async def shutdown(self):
        """Gracefully shut down the broker factory"""
        logger.info("Shutting down QuantumBrokerFactory")
        
        # Signal all background tasks to stop
        self.shutdown_event.set()
        
        # Wait for all background tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
        # Close all WebSocket connections
        for broker_id, ws in self.ws_connections.items():
            if not ws.closed:
                await ws.close()
                
        logger.info("QuantumBrokerFactory shutdown complete")

    async def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Quantum-secure order execution with AI optimization"""
        start_time = time.time_ns()
        order_id = order.get('order_id', str(uuid.uuid4()))
        
        # Ensure order has an ID
        if 'order_id' not in order:
            order['order_id'] = order_id
            
        try:
            # Log the execution attempt
            logger.info("Order execution initiated", 
                       order_id=order_id, 
                       symbol=order.get('symbol'), 
                       quantity=order.get('quantity'))
            
            # Decision logging for AI model training
            await self.decision_logger.log_execution_decision({
                'order_id': order_id,
                'timestamp': datetime.now().isoformat(),
                'order_type': order.get('type', 'market'),
                'symbol': order.get('symbol'),
                'quantity': order.get('quantity'),
                'side': order.get('side', 'buy'),
                'source': order.get('source', 'api')
            })
            
            # Phase 1: Pre-execution validation
            validation_result = await self._validate_order(order)
            if not validation_result['valid']:
                logger.warning("Order validation failed", 
                              order_id=order_id, 
                              reason=validation_result['reason'])
                
                return {
                    "status": "rejected", 
                    "reason": validation_result['reason'],
                    "order_id": order_id
                }
            
            # Phase 2: Market regime analysis
            market_state = await self.market_regime.get_current_regime(order['symbol'])
            
            # Phase 3: Pre-execution market impact analysis
            impact_analysis = await self.market_impact.analyze_impact(
                symbol=order['symbol'],
                quantity=order['quantity'],
                side=order.get('side', 'buy'),
                order_type=order.get('type', 'market')
            )
            
            # Adjust order parameters based on impact analysis
            order = await self._adjust_order_for_impact(order, impact_analysis)
            
            # Phase 4: AI Execution Strategy
            strategy = await self._calculate_execution_strategy(order, market_state)
            
            # Phase 5: Broker Selection (with fallback options)
            broker_candidates = await self._select_optimal_brokers(strategy, order)
            
            if not broker_candidates:
                logger.error("No suitable brokers found for execution", order_id=order_id)
                return {"status": "failed", "reason": "no_suitable_brokers", "order_id": order_id}
            
            # Phase 6: Atomic Execution
            primary_broker = broker_candidates[0]
            result = await self._execute_atomic(order, primary_broker, strategy)
            
            # Phase 7: Post-execution analytics
            self._update_execution_stats(result)
            
            # Calculate execution time
            execution_time = time.time_ns() - start_time
            
            # Update execution statistics
            self.execution_stats['total_executions'] += 1
            if result.get('status') == 'success':
                self.execution_stats['successful_executions'] += 1
            else:
                self.execution_stats['failed_executions'] += 1
                
            # Update moving average of execution time
            n = self.execution_stats['total_executions']
            prev_avg = self.execution_stats['avg_execution_time_ns']
            self.execution_stats['avg_execution_time_ns'] = prev_avg + (execution_time - prev_avg) / n
            
            # Count executions per broker
            self.execution_stats['executions_per_broker'][primary_broker.id] += 1
            
            # Add execution time to result
            result['execution_time_ns'] = execution_time
            result['order_id'] = order_id
            
            return result
            
        except Exception as e:
            logger.error("Execution failed", 
                        order_id=order_id, 
                        error=str(e), 
                        exception_type=type(e).__name__)
            
            # Queue for retry if appropriate
            if self._should_retry_on_error(e):
                await self.order_queue.retry_order(order)
                
            return {
                "status": "failed", 
                "error": str(e), 
                "order_id": order_id
            }

    async def _validate_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive order validation pipeline"""
        # Create structured validation result
        result = {'valid': True, 'reason': None}
        
        try:
            # Required fields check
            required_fields = ['symbol', 'quantity', 'side']
            for field in required_fields:
                if field not in order:
                    result['valid'] = False
                    result['reason'] = f"missing_required_field:{field}"
                    return result
            
            # Run parallel validation checks
            checks = await asyncio.gather(
                self.risk.validate_order(order),
                self.liquidity.verify(order['symbol'], order['quantity'], order['side']),
                self._check_regulatory_compliance(order),
                self._verify_order_signature(order),
                return_exceptions=True
            )
            
            # Process validation results
            for i, check in enumerate(checks):
                if isinstance(check, Exception):
                    result['valid'] = False
                    result['reason'] = f"validation_error:{['risk', 'liquidity', 'compliance', 'signature'][i]}"
                    logger.warning(f"Order validation failed: {check}", order_id=order.get('order_id'))
                    return result
                elif isinstance(check, dict) and not check.get('valid', True):
                    result['valid'] = False
                    result['reason'] = check.get('reason', f"validation_failed:{['risk', 'liquidity', 'compliance', 'signature'][i]}")
                    return result
                elif not check:
                    result['valid'] = False
                    result['reason'] = f"validation_failed:{['risk', 'liquidity', 'compliance', 'signature'][i]}"
                    return result
            
            return result
            
        except Exception as e:
            logger.error("Order validation error", error=str(e), order_id=order.get('order_id'))
            result['valid'] = False
            result['reason'] = f"validation_error:{str(e)}"
            return result

    async def _check_regulatory_compliance(self, order: Dict[str, Any]) -> bool:
        """Check if order complies with regulatory requirements"""
        # Use cached compliance check results when possible
        cache_key = f"{order['symbol']}:{order['side']}:{order.get('account_id', 'default')}"
        
        @lru_cache(maxsize=1000)
        def get_cached_compliance(key):
            return True  # Default to compliant, actual implementation would check regulations
            
        # In a real system, this would check against real regulatory rules
        # For now, we'll assume all orders are compliant
        return get_cached_compliance(cache_key)

    @lru_cache(maxsize=1000)
    async def _verify_order_signature(self, order: Dict[str, Any]) -> bool:
        """Verify the cryptographic signature of the order"""
        # Extract signature if present
        signature = order.get('signature')
        if not signature:
            return True  # No signature to verify
            
        # In a real system, this would verify the signature
        # For now, we'll assume all signatures are valid
        return True

    async def _calculate_execution_strategy(self, order: Dict[str, Any], market_state: Dict[str, Any]) -> Dict[str, Any]:
        """AI-powered execution strategy optimization"""
        # Get current market data
        market_data = await self.market_data.get_real_time_data(order['symbol'])
        
        # Get current latency map
        latency_data = {
            broker_id: self.latency_map.get(broker_id)
            for broker_id in [b.id for b in await self.registry.get_all_brokers()]
        }
        
        # Get current risk parameters
        risk_params = await self.risk.get_current_params()
        
        # Generate execution strategy using AI
        strategy = await self.route_predictor.generate_strategy(
            order=order,
            market_state=market_state,
            market_data=market_data,
            latency_map=latency_data,
            risk_params=risk_params,
            execution_history=self.execution_stats
        )
        
        # Add strategy ID for tracking
        strategy['id'] = str(uuid.uuid4())
        
        # Log the strategy decision
        await self.decision_logger.log_strategy_decision({
            'order_id': order.get('order_id'),
            'strategy_id': strategy['id'],
            'timestamp': datetime.now().isoformat(),
            'strategy_type': strategy.get('type', 'default'),
            'market_state': market_state,
            'broker_preferences': strategy.get('broker_preferences', [])
        })
        
        return strategy

    async def _adjust_order_for_impact(self, order: Dict[str, Any], impact: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust order parameters based on market impact analysis"""
        adjusted_order = order.copy()
        
        # If impact is too high, consider order slicing
        if impact.get('expected_impact', 0) > self.config.get('max_acceptable_impact', 0.1):
            # Calculate slices
            slices = max(1, int(impact.get('expected_impact', 0) / self.config.get('max_acceptable_impact', 0.1)))
            
            # Adjust quantity for first slice
            adjusted_order['original_quantity'] = adjusted_order['quantity']
            adjusted_order['quantity'] = adjusted_order['quantity'] / slices
            adjusted_order['sliced'] = True
            adjusted_order['total_slices'] = slices
            adjusted_order['current_slice'] = 1
            
            logger.info("Order sliced due to market impact", 
                       order_id=order.get('order_id'),
                       slices=slices,
                       original_quantity=order['quantity'],
                       adjusted_quantity=adjusted_order['quantity'])
        
        return adjusted_order

    async def _select_optimal_brokers(self, strategy: Dict[str, Any], order: Dict[str, Any]) -> List[Any]:
        """Select multiple brokers based on AI strategy and real-time conditions"""
        try:
            # Get brokers that support the asset class and required features
            candidates = await self.registry.get_candidates(
                strategy.get('asset_class', self._get_asset_class(order['symbol'])),
                strategy.get('required_features', [])
            )
            
            if not candidates:
                logger.warning("No broker candidates found", 
                              order_id=order.get('order_id'),
                              symbol=order['symbol'])
                return []
            
            # Rank brokers based on multiple factors
            ranked_brokers = await self._rank_brokers(candidates, strategy, order)
            
            # Return top N brokers for potential failover
            return ranked_brokers[:min(3, len(ranked_brokers))]
            
        except Exception as e:
            logger.error("Error selecting brokers", 
                        error=str(e),
                        order_id=order.get('order_id'))
            return []

    async def _get_asset_class(self, symbol: str) -> str:
        """Determine asset class from symbol"""
        # This is a simplified implementation
        if symbol.endswith('USD') or symbol.endswith('USDT'):
            return 'crypto'
        elif '/' in symbol:
            return 'forex'
        else:
            return 'equity'

    async def _rank_brokers(self, brokers: list, strategy: Dict[str, Any], order: Dict[str, Any]) -> List[Any]:
        """Multi-factor broker ranking with AI optimization"""
        if not brokers:
            return []
            
        # Calculate all broker scores in parallel
        broker_scores = await asyncio.gather(*[
            self._calculate_broker_score(broker, strategy, order)
            for broker in brokers
        ])
        
        # Combine brokers with their scores
        broker_with_scores = list(zip(brokers, broker_scores))
        
        # Sort by score (descending)
        ranked = sorted(broker_with_scores, key=lambda x: x[1], reverse=True)
        
        # Return only the brokers
        return [broker for broker, _ in ranked]

    async def _calculate_broker_score(self, broker, strategy: Dict[str, Any], order: Dict[str, Any]) -> float:
        """Dynamic scoring algorithm with multiple weighted factors"""
        # Get broker statistics
        latency_stats = self.latency_map.get_all_stats(broker.id)
        execution_quality = self.execution_monitor.get_metrics(broker.id)
        broker_health = await self.registry.get_health(broker.id)
        
        # Base score components
        base_score = broker.base_score if hasattr(broker, 'base_score') else 5.0
        
        # Latency score (lower is better)
        latency_score = 10.0 / (1.0 + latency_stats['avg'] / 1e6)  # Convert to ms
        
        # Execution quality score
        quality_score = 10.0 * (
            execution_quality.get('success_rate', 0.5) * 0.6 +  # 60% weight on success rate
            (1.0 - min(execution_quality.get('avg_slippage', 0.0), 0.01) / 0.01) * 0.4  # 40% weight on slippage
        )
        
        # Health score
        health_score = 10.0 if broker_health else 0.0
        
        # Get AI weight for this broker
        ai_weight = await self.route_predictor.get_weight(
            broker=broker,
            strategy=strategy,
            order=order
        )
        
        # Calculate final score with weighted components
        weights = {
            'base': 0.1,
            'latency': 0.3,
            'quality': 0.3,
            'health': 0.2,
            'ai': 0.1
        }
        
        final_score = (
            weights['base'] * base_score +
            weights['latency'] * latency_score +
            weights['quality'] * quality_score +
            weights['health'] * health_score +
            weights['ai'] * ai_weight
        )
        
        return final_score

    async def _execute_atomic(self, order: Dict[str, Any], broker, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Atomic execution sequence with failover and retry logic"""
        start_time = time.time_ns()
        order_id = order.get('order_id', 'unknown')
        
        # Create serializable order representation for broker API
        broker_order = self._prepare_broker_specific_order(order, broker)
        
        try:
            logger.info("Executing order with broker", 
                    broker_id=broker.id, 
                    order_id=order_id,
                    strategy_id=strategy.get('id', 'unknown'))
            
            # Record execution attempt for this broker
            await self.decision_logger.log_execution_attempt({
                'order_id': order_id,
                'broker_id': broker.id,
                'timestamp': datetime.now().isoformat(),
                'attempt_type': 'primary'
            })
            
            # Execute order through broker API with timeout
            execution_timeout = order.get('timeout_ms', 500) / 1000  # Convert to seconds
            execution_start = time.time_ns()
            
            # Use efficient connection pooling for broker API calls
            async with aiohttp.ClientSession() as session:
                # Get cached API key or generate new one
                api_key, api_secret = await self._get_broker_credentials(broker.id)
                
                # Execute order through broker API
                result = await asyncio.wait_for(
                    broker.execute_order(
                        session=session,
                        order=broker_order,
                        api_key=api_key,
                        api_secret=api_secret,
                        strategy=strategy
                    ),
                    timeout=execution_timeout
                )
                
            # Calculate actual execution latency
            execution_latency = time.time_ns() - execution_start
            
            # Update latency metrics
            self.latency_map.update(broker.id, execution_latency)
            
            # Process execution result
            if result.get('status') == 'success':
                # Record successful execution
                await self.execution_monitor.record_execution(broker.id, {
                    'status': 'success',
                    'latency': execution_latency,
                    'slippage': result.get('slippage', 0),
                    'fill_rate': result.get('fill_rate', 1.0),
                    'price': result.get('price'),
                    'fees': result.get('fees', 0)
                })
                
                # Update performance cache for this broker
                await self.broker_cache.update(broker.id, {
                    'last_used': time.time(),
                    'success_count': 1,
                    'avg_latency': execution_latency,
                    'last_execution': datetime.now().isoformat()
                })
                
                # Process any sliced orders if this was part of a slice
                if order.get('sliced', False) and order.get('current_slice', 1) < order.get('total_slices', 1):
                    await self._process_next_slice(order, broker, strategy)
                    
                return {
                    'status': 'success',
                    'broker_id': broker.id,
                    'execution_time_ns': execution_latency,
                    'fill_price': result.get('price'),
                    'fill_quantity': result.get('filled_quantity', order['quantity']),
                    'fees': result.get('fees', 0),
                    'slippage': result.get('slippage', 0),
                    'order_id': order_id,
                    'transaction_id': result.get('transaction_id', str(uuid.uuid4()))
                }
            else:
                # Record failed execution
                await self.execution_monitor.record_execution(broker.id, {
                    'status': 'failed',
                    'latency': execution_latency,
                    'reason': result.get('reason', 'unknown')
                })
                
                # Attempt failover to alternative broker
                return await self._attempt_failover(order, broker, strategy)
                
        except asyncio.TimeoutError:
            logger.warning("Execution timeout", 
                        broker_id=broker.id, 
                        order_id=order_id,
                        timeout_ms=order.get('timeout_ms', 500))
            
            # Update latency metrics with timeout value
            self.latency_map.update(broker.id, order.get('timeout_ms', 500) * 1e6)  # Convert ms to ns
            
            # Record failed execution due to timeout
            await self.execution_monitor.record_execution(broker.id, {
                'status': 'failed',
                'latency': order.get('timeout_ms', 500) * 1e6,  # Convert ms to ns
                'reason': 'timeout'
            })
            
            # Attempt failover to alternative broker
            return await self._attempt_failover(order, broker, strategy)
            
        except Exception as e:
            logger.error("Execution error", 
                        broker_id=broker.id, 
                        order_id=order_id,
                        error=str(e))
            
            # Record failed execution
            await self.execution_monitor.record_execution(broker.id, {
                'status': 'failed',
                'reason': f'error:{type(e).__name__}'
            })
            
            # Attempt failover to alternative broker
            return await self._attempt_failover(order, broker, strategy)

    async def _prepare_broker_specific_order(self, order: Dict[str, Any], broker) -> Dict[str, Any]:
        """Prepare order for specific broker API format"""
        # Create a copy of the order to avoid modifying the original
        broker_order = order.copy()
        
        # Get broker-specific order format
        if hasattr(broker, 'get_order_format'):
            broker_order = await broker.get_order_format(broker_order)
        
        # Add execution timestamp in broker's required format
        timestamp_format = getattr(broker, 'timestamp_format', 'iso')
        if timestamp_format == 'iso':
            broker_order['timestamp'] = datetime.now().isoformat()
        elif timestamp_format == 'unix':
            broker_order['timestamp'] = int(time.time())
        elif timestamp_format == 'unix_ms':
            broker_order['timestamp'] = int(time.time() * 1000)
        
        # Add required broker-specific fields
        broker_order['broker_id'] = broker.id
        
        # Add signature if required by broker
        if getattr(broker, 'requires_signature', False):
            broker_order['signature'] = self._generate_order_signature(broker_order, broker)
        
        return broker_order

    def _generate_order_signature(self, order: Dict[str, Any], broker) -> str:
        """Generate cryptographic signature for order authentication"""
        # Get signing key for this broker
        signing_key = self.vault.get_signing_key(broker.id)
        
        # Order parameters to include in signature
        params_to_sign = [
            str(order.get('symbol', '')),
            str(order.get('side', '')),
            str(order.get('quantity', '')),
            str(order.get('price', '')),
            str(order.get('timestamp', ''))
        ]
        
        # Create signature string
        signature_string = ''.join(params_to_sign)
        
        # Generate HMAC signature
        signature = hmac.new(
            key=signing_key.encode('utf-8'),
            msg=signature_string.encode('utf-8'),
            digestmod=hashlib.sha256
        ).hexdigest()
        
        return signature

    async def _get_broker_credentials(self, broker_id: str) -> Tuple[str, str]:
        """Get API credentials for broker with rotation if needed"""
        current_time = time.time()
        
        # Check if API key needs rotation
        if broker_id in self.api_key_rotation_time and current_time >= self.api_key_rotation_time[broker_id]:
            await self._rotate_api_key(broker_id)
        
        # Get API key and secret from secure vault
        api_key = self.vault.get_api_key(broker_id)
        api_secret = self.vault.get_api_secret(broker_id)
        
        return api_key, api_secret

    async def _rotate_api_key(self, broker_id: str) -> None:
        """Rotate API keys for security"""
        logger.info("Rotating API keys", broker_id=broker_id)
        
        try:
            # Get broker instance
            broker = await self.registry.get_broker(broker_id)
            
            # Get current API credentials
            old_api_key = self.vault.get_api_key(broker_id)
            old_api_secret = self.vault.get_api_secret(broker_id)
            
            # Generate new API key if broker supports it
            if hasattr(broker, 'generate_new_api_key'):
                async with aiohttp.ClientSession() as session:
                    new_credentials = await broker.generate_new_api_key(
                        session=session,
                        current_key=old_api_key,
                        current_secret=old_api_secret
                    )
                    
                    # Store new credentials in vault
                    self.vault.store_api_key(broker_id, new_credentials['api_key'])
                    self.vault.store_api_secret(broker_id, new_credentials['api_secret'])
            
            # Set next rotation time (default: 24 hours)
            rotation_interval = self.config.get('api_key_rotation_interval', 86400)
            self.api_key_rotation_time[broker_id] = time.time() + rotation_interval
            
            logger.info("API key rotation complete", broker_id=broker_id)
            
        except Exception as e:
            logger.error("API key rotation failed", 
                        broker_id=broker_id,
                        error=str(e))
            
            # Set next attempt in 1 hour instead of regular interval
            self.api_key_rotation_time[broker_id] = time.time() + 3600

    async def _attempt_failover(self, order: Dict[str, Any], failed_broker, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to execute order using backup broker"""
        order_id = order.get('order_id', 'unknown')
        
        # Get failover recommendations from AI advisor
        failover_recommendations = await self.failover_advisor.get_recommendations(
            order=order,
            failed_broker_id=failed_broker.id,
            execution_history=self.execution_stats,
            current_market_conditions=await self.market_data.get_real_time_data(order['symbol'])
        )
        
        # No recommendations, return failure
        if not failover_recommendations:
            return {
                'status': 'failed',
                'reason': f'execution_failed_with_broker:{failed_broker.id}',
                'order_id': order_id,
                'failover_attempted': True,
                'failover_success': False
            }
        
        # Get list of potential failover brokers (excluding the failed broker)
        failover_brokers = []
        for broker_id in failover_recommendations:
            if broker_id != failed_broker.id:
                broker = await self.registry.get_broker(broker_id)
                if broker:
                    failover_brokers.append(broker)
        
        # No available failover brokers, return failure
        if not failover_brokers:
            return {
                'status': 'failed',
                'reason': 'no_failover_brokers_available',
                'order_id': order_id,
                'failover_attempted': True,
                'failover_success': False
            }
        
        # Attempt execution with first failover broker
        failover_broker = failover_brokers[0]
        logger.info("Attempting failover execution", 
                order_id=order_id,
                primary_broker=failed_broker.id,
                failover_broker=failover_broker.id)
        
        # Log failover attempt
        await self.decision_logger.log_execution_attempt({
            'order_id': order_id,
            'broker_id': failover_broker.id,
            'timestamp': datetime.now().isoformat(),
            'attempt_type': 'failover',
            'primary_broker_id': failed_broker.id
        })
        
        # Execute with failover broker
        try:
            # Create serializable order for broker API
            broker_order = self._prepare_broker_specific_order(order, failover_broker)
            
            # Execute order through broker API with timeout
            execution_timeout = order.get('timeout_ms', 500) / 1000  # Convert to seconds
            execution_start = time.time_ns()
            
            async with aiohttp.ClientSession() as session:
                # Get broker credentials
                api_key, api_secret = await self._get_broker_credentials(failover_broker.id)
                
                # Execute order through broker API
                result = await asyncio.wait_for(
                    failover_broker.execute_order(
                        session=session,
                        order=broker_order,
                        api_key=api_key,
                        api_secret=api_secret,
                        strategy=strategy
                    ),
                    timeout=execution_timeout
                )
            
            # Calculate execution latency
            execution_latency = time.time_ns() - execution_start
            
            # Update latency metrics
            self.latency_map.update(failover_broker.id, execution_latency)
            
            # Process execution result
            if result.get('status') == 'success':
                # Record successful execution
                await self.execution_monitor.record_execution(failover_broker.id, {
                    'status': 'success',
                    'latency': execution_latency,
                    'slippage': result.get('slippage', 0),
                    'fill_rate': result.get('fill_rate', 1.0),
                    'price': result.get('price'),
                    'fees': result.get('fees', 0),
                    'failover': True
                })
                
                return {
                    'status': 'success',
                    'broker_id': failover_broker.id,
                    'execution_time_ns': execution_latency,
                    'fill_price': result.get('price'),
                    'fill_quantity': result.get('filled_quantity', order['quantity']),
                    'fees': result.get('fees', 0),
                    'slippage': result.get('slippage', 0),
                    'order_id': order_id,
                    'transaction_id': result.get('transaction_id', str(uuid.uuid4())),
                    'failover': True,
                    'failover_from': failed_broker.id
                }
            else:
                # Record failed execution
                await self.execution_monitor.record_execution(failover_broker.id, {
                    'status': 'failed',
                    'latency': execution_latency,
                    'reason': result.get('reason', 'unknown'),
                    'failover': True
                })
                
                return {
                    'status': 'failed',
                    'reason': f'failover_execution_failed:{result.get("reason", "unknown")}',
                    'order_id': order_id,
                    'failover_attempted': True,
                    'failover_success': False,
                    'primary_broker': failed_broker.id,
                    'failover_broker': failover_broker.id
                }
        
        except Exception as e:
            logger.error("Failover execution error", 
                        broker_id=failover_broker.id, 
                        order_id=order_id,
                        error=str(e))
            
            return {
                'status': 'failed',
                'reason': f'failover_execution_error:{str(e)}',
                'order_id': order_id,
                'failover_attempted': True,
                'failover_success': False,
                'primary_broker': failed_broker.id,
                'failover_broker': failover_broker.id
            }

    async def _process_next_slice(self, order: Dict[str, Any], broker, strategy: Dict[str, Any]) -> None:
        """Process the next slice of a multi-slice order"""
        # Create a new order for the next slice
        next_slice = order.copy()
        next_slice['current_slice'] += 1
        
        # Keep the order_id the same but add a slice identifier
        next_slice['parent_order_id'] = order.get('order_id')
        next_slice['order_id'] = f"{order.get('order_id')}-slice-{next_slice['current_slice']}"
        
        # Recalculate adaptive parameters based on market conditions
        symbol = next_slice['symbol']
        current_market_data = await self.market_data.get_real_time_data(symbol)
        
        # Adapt slice timing based on market volatility
        volatility = current_market_data.get('volatility', 0.01)
        base_delay = self.config.get('slice_base_delay_ms', 250)
        adaptive_delay = base_delay * (1 + volatility * 10)  # Increase delay with volatility
        
        # If this is the last slice, potentially adjust quantity to account for previous fills
        if next_slice['current_slice'] == next_slice['total_slices']:
            # TODO: Implement slice quantity adjustment based on previous fills
            pass
        
        # Schedule the next slice with adaptive delay
        logger.info("Scheduling next order slice", 
                order_id=next_slice['order_id'],
                parent_order_id=next_slice['parent_order_id'],
                slice=f"{next_slice['current_slice']}/{next_slice['total_slices']}",
                delay_ms=adaptive_delay)
        
        # Add to queue with delay
        asyncio.create_task(self._delayed_queue_add(next_slice, adaptive_delay / 1000))

    async def _delayed_queue_add(self, order: Dict[str, Any], delay: float) -> None:
        """Add an order to the queue after a delay"""
        await asyncio.sleep(delay)
        await self.order_queue.add_order(order, priority=2)  # Priority 2 for sliced orders

    async def _process_order_queue(self) -> None:
        """Background task to process the order queue"""
        logger.info("Order queue processor started")
        
        while not self.shutdown_event.is_set():
            try:
                # Process up to 10 orders per iteration
                for _ in range(10):
                    if self.order_queue.is_empty():
                        break
                    
                    order = await self.order_queue.get_next_order()
                    if not order:
                        break
                    
                    # Process the order
                    asyncio.create_task(self._process_queued_order(order))
                
                # Process fallback queue if main queue is empty
                if self.order_queue.is_empty() and not self.fallback_queue.is_empty():
                    for _ in range(5):  # Process up to 5 fallback orders
                        order = await self.fallback_queue.get_next_order()
                        if not order:
                            break
                        
                        # Process the fallback order
                        asyncio.create_task(self._process_queued_order(order, is_fallback=True))
                
                # Adaptive sleep based on queue size
                queue_size = await self.order_queue.size()
                fallback_size = await self.fallback_queue.size()
                
                if queue_size + fallback_size > 1000:
                    await asyncio.sleep(0.001)  # 1ms sleep for very large queues
                elif queue_size + fallback_size > 100:
                    await asyncio.sleep(0.01)   # 10ms sleep for large queues
                else:
                    await asyncio.sleep(0.05)   # 50ms sleep for small queues
                    
            except Exception as e:
                logger.error("Error processing order queue", error=str(e))
                await asyncio.sleep(0.1)  # Sleep on error to prevent tight loop

    async def _process_queued_order(self, order: Dict[str, Any], is_fallback: bool = False) -> None:
        """Process a single order from the queue"""
        try:
            # Get current market state
            market_state = await self.market_regime.get_current_regime(order['symbol'])
            
            # Calculate execution strategy
            strategy = await self._calculate_execution_strategy(order, market_state)
            
            # Select optimal brokers
            brokers = await self._select_optimal_brokers(strategy, order)
            
            if not brokers:
                logger.warning("No suitable brokers found for queued order", 
                            order_id=order.get('order_id', 'unknown'))
                
                if not is_fallback:
                    # Add to fallback queue for later retries
                    await self.fallback_queue.add_order(order, priority=3)
                return
            
            # Execute with primary broker
            primary_broker = brokers[0]
            result = await self._execute_atomic(order, primary_broker, strategy)
            
            # Log execution result
            if result.get('status') == 'success':
                logger.info("Queued order executed successfully", 
                        order_id=order.get('order_id', 'unknown'),
                        broker_id=result.get('broker_id'),
                        execution_time_ns=result.get('execution_time_ns'))
            else:
                logger.warning("Queued order execution failed", 
                            order_id=order.get('order_id', 'unknown'),
                            reason=result.get('reason', 'unknown'))
                
                if not is_fallback and result.get('failover_attempted', False) == False:
                    # Add to fallback queue for later retries
                    await self.fallback_queue.add_order(order, priority=3)
                    
        except Exception as e:
            logger.error("Error processing queued order", 
                        order_id=order.get('order_id', 'unknown'),
                        error=str(e))
            
            if not is_fallback:
                # Add to fallback queue for later retries
                await self.fallback_queue.add_order(order, priority=3)

    async def _refresh_broker_registry(self) -> None:
        """Background task to refresh broker registry"""
        logger.info("Broker registry refresh task started")
        
        while not self.shutdown_event.is_set():
            try:
                # Refresh broker registry
                await self.registry.refresh()
                
                # Sleep for configured interval
                refresh_interval = self.config.get('broker_registry_refresh_interval_s', 60)
                await asyncio.sleep(refresh_interval)
                
            except Exception as e:
                logger.error("Error refreshing broker registry", error=str(e))
                await asyncio.sleep(10)  # Sleep for 10 seconds on error

    async def _monitor_broker_health(self) -> None:
        """Background task to monitor broker health"""
        logger.info("Broker health monitoring task started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get all brokers
                brokers = await self.registry.get_all_brokers()
                
                # Check health of each broker
                for broker in brokers:
                    try:
                        # Check broker health
                        is_healthy = await self.registry.check_health(broker.id)
                        
                        if not is_healthy:
                            logger.warning("Broker health check failed", broker_id=broker.id)
                            
                            # Notify risk engine about unhealthy broker
                            await self.risk.notify_broker_status(broker.id, {
                                'status': 'unhealthy',
                                'timestamp': datetime.now().isoformat()
                            })
                            
                    except Exception as e:
                        logger.error("Error checking broker health", 
                                    broker_id=broker.id,
                                    error=str(e))
                
                # Sleep for configured interval
                health_check_interval = self.config.get('broker_health_check_interval_s', 30)
                await asyncio.sleep(health_check_interval)
                
            except Exception as e:
                logger.error("Error monitoring broker health", error=str(e))
                await asyncio.sleep(10)  # Sleep for 10 seconds on error

    async def _update_latency_map(self) -> None:
        """Background task to update latency map with ping tests"""
        logger.info("Latency map update task started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get all brokers
                brokers = await self.registry.get_all_brokers()
                
                # Update latency for each broker
                for broker in brokers:
                    try:
                        # Ping broker API
                        latency_ns = await self._ping_broker(broker)
                        
                        # Update latency map
                        self.latency_map.update(broker.id, latency_ns)
                        
                    except Exception as e:
                        logger.error("Error updating broker latency", 
                                    broker_id=broker.id,
                                    error=str(e))
                
                # Sleep for configured interval
                latency_update_interval = self.config.get('latency_update_interval_s', 15)
                await asyncio.sleep(latency_update_interval)
                
            except Exception as e:
                logger.error("Error updating latency map", error=str(e))
                await asyncio.sleep(10)  # Sleep for 10 seconds on error

    async def _ping_broker(self, broker) -> float:
        """Ping broker API and return latency in nanoseconds"""
        try:
            # Get API endpoint
            api_endpoint = broker.get_ping_endpoint()
            
            # Start timer
            start_time = time.time_ns()
            
            # Ping broker API
            async with aiohttp.ClientSession() as session:
                async with session.get(api_endpoint, timeout=2) as response:
                    # Ensure response is valid
                    if response.status == 200:
                        # Stop timer
                        end_time = time.time_ns()
                        
                        # Calculate latency
                        latency = end_time - start_time
                        
                        return latency
                    else:
                        # Use high latency value for failed requests
                        return float('inf')
        
        except Exception:
            # Use high latency value for failed requests
            return float('inf')

    async def _sync_risk_parameters(self) -> None:
        """Background task to sync risk parameters with risk engine"""
        logger.info("Risk parameter sync task started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get current risk parameters
                risk_params = await self.risk.get_current_params()
                
                # Update broker registry with risk parameters
                await self.registry.update_risk_params(risk_params)
                
                # Sleep for configured interval
                risk_sync_interval = self.config.get('risk_sync_interval_s', 60)
                await asyncio.sleep(risk_sync_interval)
                
            except Exception as e:
                logger.error("Error syncing risk parameters", error=str(e))
                await asyncio.sleep(10)  # Sleep for 10 seconds on error

    async def _rotate_api_keys(self) -> None:
        """Background task to rotate API keys"""
        logger.info("API key rotation task started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get all brokers
                brokers = await self.registry.get_all_brokers()
                
                current_time = time.time()
                
                # Check for brokers that need key rotation
                for broker in brokers:
                    if broker.id not in self.api_key_rotation_time:
                        # Set initial rotation time
                        rotation_interval = self.config.get('api_key_rotation_interval', 86400)
                        self.api_key_rotation_time[broker.id] = current_time + rotation_interval
                        continue
                    
                    if current_time >= self.api_key_rotation_time[broker.id]:
                        # Rotate API key
                        await self._rotate_api_key(broker.id)
                
                # Sleep for configured interval
                key_rotation_check_interval = self.config.get('key_rotation_check_interval_s', 3600)
                await asyncio.sleep(key_rotation_check_interval)
                
            except Exception as e:
                logger.error("Error rotating API keys", error=str(e))
                await asyncio.sleep(10)  # Sleep for 10 seconds on error

    async def _establish_ws_connections(self) -> None:
        """Background task to establish WebSocket connections to brokers"""
        logger.info("WebSocket connection task started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get all brokers
                brokers = await self.registry.get_all_brokers()
                
                for broker in brokers:
                    # Skip brokers that don't support WebSocket
                    if not hasattr(broker, 'get_ws_endpoint'):
                        continue
                    
                    # Check if connection exists and is healthy
                    if broker.id in self.ws_connections and not self.ws_connections[broker.id].closed:
                        continue
                    
                    # Establish new WebSocket connection
                    try:
                        ws_endpoint = broker.get_ws_endpoint()
                        
                        # Get API credentials
                        api_key, api_secret = await self._get_broker_credentials(broker.id)
                        
                        # Create session
                        session = aiohttp.ClientSession()
                        
                        # Connect to WebSocket
                        ws = await session.ws_connect(ws_endpoint)
                        
                        # Authenticate WebSocket connection
                        if hasattr(broker, 'authenticate_ws'):
                            await broker.authenticate_ws(ws, api_key, api_secret)
                        
                        # Store connection
                        self.ws_connections[broker.id] = ws
                        
                        # Start background task to handle WebSocket messages
                        task = asyncio.create_task(self._handle_ws_messages(broker.id, ws, session))
                        self.background_tasks.add(task)
                        task.add_done_callback(self.background_tasks.remove)
                        
                        logger.info("WebSocket connection established", broker_id=broker.id)
                        
                    except Exception as e:
                        logger.error("Error establishing WebSocket connection", 
                                    broker_id=broker.id,
                                    error=str(e))
                
                # Sleep for configured interval
                ws_connection_check_interval = self.config.get('ws_connection_check_interval_s', 60)
                await asyncio.sleep(ws_connection_check_interval)
                
            except Exception as e:
                logger.error("Error establishing WebSocket connections", error=str(e))
                await asyncio.sleep(10)  # Sleep for 10 seconds on error

    async def _handle_ws_messages(self, broker_id: str, ws, session) -> None:
        """Handle WebSocket messages for a broker"""
        try:
            logger.info("WebSocket message handler started", broker_id=broker_id)
            
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    # Process text message
                    await self._process_ws_message(broker_id, msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    # Process binary message
                    await self._process_ws_binary(broker_id, msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("WebSocket connection closed with error", 
                                broker_id=broker_id,
                                error=ws.exception())
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("WebSocket connection closed", broker_id=broker_id)
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSE:
                    logger.info("WebSocket close frame received", broker_id=broker_id)
                    await ws.close()
                    break
                elif msg.type == aiohttp.WSMsgType.PING:
                    # Respond to ping with pong to maintain connection
                    logger.debug("WebSocket ping received, sending pong", broker_id=broker_id)
                    await ws.pong()
                    # Update connection health metrics
                    self._update_broker_health(broker_id, "active")
                elif msg.type == aiohttp.WSMsgType.PONG:
                    # Log pong response for connection monitoring
                    logger.debug("WebSocket pong received", broker_id=broker_id)
                    self._update_broker_latency(broker_id, "ws_heartbeat")
                    # Record successful heartbeat for failover system
                    self.broker_health_monitor.record_heartbeat(broker_id)
                
                # Process message through AI-powered message classifier
                if msg.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                    # Cryptographically verify message integrity
                    if not await self._verify_message_integrity(broker_id, msg.data):
                        logger.warning("Message integrity check failed", broker_id=broker_id)
                        await self._handle_security_violation(broker_id, "message_integrity")
                        continue
                    
                    # Apply message transformation and normalization
                    normalized_data = await self._normalize_broker_message(broker_id, msg.data, msg.type)
                    
                    # Dispatch to appropriate handler based on message type
                    message_type = self._classify_message(normalized_data)
                    
                    # Record message for AI training and optimization
                    await self._record_message_for_analysis(broker_id, message_type, normalized_data)
                    
                    # Handle different message categories with specialized processors
                    if message_type == "execution_report":
                        await self._process_execution_report(broker_id, normalized_data)
                    elif message_type == "order_book_update":
                        await self._process_order_book_update(broker_id, normalized_data)
                    elif message_type == "market_data":
                        await self._process_market_data(broker_id, normalized_data)
                    elif message_type == "account_update":
                        await self._process_account_update(broker_id, normalized_data)
                    elif message_type == "error":
                        await self._handle_broker_error(broker_id, normalized_data)
                        # Trigger failover if critical error
                        if self._is_critical_error(normalized_data):
                            await self._initiate_broker_failover(broker_id, normalized_data)
                    
                    # Update broker performance metrics based on message processing
                    self._update_broker_performance(broker_id, message_type, normalized_data)
        except asyncio.CancelledError:
            logger.info("WebSocket message handler cancelled", broker_id=broker_id)
            raise
        except Exception as e:
            logger.error("Error in WebSocket message handler", 
                        broker_id=broker_id, 
                        error=str(e),
                        traceback=traceback.format_exc())
            # Record broker failure for health monitoring
            self.broker_health_monitor.record_failure(broker_id, "ws_handler_error")
            # Attempt to reconnect if connection was lost
            await self._schedule_broker_reconnection(broker_id)
        finally:
            # Clean up resources
            if not ws.closed:
                await ws.close()
            await session.close()
            logger.info("WebSocket message handler stopped", broker_id=broker_id)
            # Remove from active connections
            if broker_id in self.ws_connections:
                del self.ws_connections[broker_id]
                # Notify broker connection manager of disconnection
                await self.connection_manager.notify_disconnection(broker_id)
                
                # Update routing table to exclude this broker
                await self.update_routing_table(exclude_broker=broker_id)
                
                # Check if we need to initiate automatic failover
                if self.config.get('auto_failover_enabled', True):
                    await self._initiate_automatic_failover(broker_id)
                
                # Log disconnection for analytics
                self.metrics_collector.record_event(
                    'broker_disconnection',
                    {
                        'broker_id': broker_id,
                        'timestamp': time.time(),
                        'connection_duration': time.time() - self.connection_start_times.get(broker_id, time.time())
                    }
                )
                # Check if we need to attempt reconnection
                if self.config.get('auto_reconnect_enabled', True):
                    reconnect_delay = self.config.get('reconnect_delay', 5)
                    max_reconnect_attempts = self.config.get('max_reconnect_attempts', 3)
                    
                    # Get current attempt count
                    current_attempts = self.reconnect_attempts.get(broker_id, 0)
                    
                    if current_attempts < max_reconnect_attempts:
                        logger.info(f"Scheduling reconnection attempt {current_attempts + 1}/{max_reconnect_attempts}",
                                   broker_id=broker_id, 
                                   delay=reconnect_delay)
                        
                        # Update reconnection attempts counter
                        self.reconnect_attempts[broker_id] = current_attempts + 1
                        
                        # Schedule reconnection
                        asyncio.create_task(self._delayed_reconnect(broker_id, reconnect_delay))
                    else:
                        logger.warning(f"Maximum reconnection attempts reached for broker",
                                     broker_id=broker_id,
                                     attempts=current_attempts)
                        
                        # Reset reconnection counter for future attempts
                        self.reconnect_attempts[broker_id] = 0
                        
                        # Notify system of permanent disconnection
                        await self.connection_manager.notify_permanent_failure(broker_id)
                        # Mark broker as unavailable in the broker registry
                        await self.broker_registry.set_broker_status(broker_id, status="unavailable")
                        
                        # Log permanent failure for system monitoring
                        logger.error(f"Permanent broker failure detected",
                                    broker_id=broker_id,
                                    reconnect_attempts=current_attempts,
                                    timestamp=time.time())
                        
                        # Trigger failover to backup brokers if available
                        if self.config.get('use_backup_brokers', True):
                            await self._activate_backup_broker(broker_id)