import os
import time
import uuid
import numpy as np
import hashlib
import asyncio
import threading
import msgpack
import zstandard as zstd
from collections import deque
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Apex Core Imports - maintain integration with existing system
from Apex.utils.logging.structured_logger import StructuredLogger
from Apex.src.Core.trading.execution.order_execution import OrderExecutionManager
from Apex.src.Core.trading.risk.risk_management import RiskEngine
from Apex.src.ai.ensembles.meta_trader import MetaTrader
from Apex.src.ai.analysis.market_regime_classifier import MarketRegimeClassifier
from Apex.utils.analytics.market_impact import MarketImpactCalculator
from Apex.src.Core.data.insider_monitor import InstitutionalActivityDetector
from Apex.src.Core.trading.hft.liquidity_manager import LiquidityOracle
from Apex.src.Core.trading.strategies.strategy_evaluator import StrategyEvaluator

class TradeMonitor:
    """
    Institutional-Grade Trade Surveillance & Execution Optimizer
    Focused on high-frequency, low-latency trade monitoring with nanosecond precision
    """
    
    _instance = None
    _lock = threading.Lock()
    _strategy_blacklist: Set[str] = set()
    _MAX_CACHE_SIZE = 100000
    _SURVEILLANCE_INTERVAL_MS = 1  # 1ms for ultra-high-frequency surveillance
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # Initialize core system integrations
        self.logger = StructuredLogger("TradeMonitor")
        self.execution = OrderExecutionManager()
        self.risk = RiskEngine()
        self.meta = MetaTrader()
        self.regime = MarketRegimeClassifier()
        self.impact = MarketImpactCalculator()
        self.institutions = InstitutionalActivityDetector()
        self.liquidity = LiquidityOracle()
        self.strategy_evaluator = StrategyEvaluator()
        
        # Optimized data structures for trade tracking
        self.active_trades: Dict[str, Dict] = {}
        self.trade_cache = deque(maxlen=self._MAX_CACHE_SIZE)
        self.venue_performance = {}
        
        # Pre-allocation of memory for venue metrics (avoids dynamic resizing)
        self._prealloc_venue_metrics()
        
        # Compression for efficient data serialization
        self.compressor = zstd.ZstdCompressor(level=3)  # Balance between speed and compression
        
        # Initialize thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 4))
        
        # Start surveillance systems in non-blocking mode
        self._start_surveillance_loop()
        
    def _prealloc_venue_metrics(self):
        """Pre-allocate memory for venue metrics to avoid dynamic allocation during trading"""
        self.venue_metrics = {
            'latency': np.zeros((100,), dtype=np.float32),
            'slippage': np.zeros((100,), dtype=np.float32),
            'fill_rates': np.zeros((100,), dtype=np.float32),
            'impact_scores': np.zeros((100,), dtype=np.float32),
            'venue_map': {},  # Maps venue names to array indices
            'counter': 0
        }
        
    def _start_surveillance_loop(self):
        """Start surveillance systems with optimized threading"""
        self.monitor_task = asyncio.create_task(self._surveillance_loop())
        
        # Analysis thread for non-critical path operations
        self.analysis_thread = threading.Thread(
            target=self._realtime_analysis, 
            daemon=True
        )
        self.analysis_thread.start()

    def track_trade(self, trade: Dict[str, Any]) -> str:
        """
        Register a new trade for surveillance (entry point from execution system)
        Returns a secure trade ID for subsequent tracking
        """
        # Generate quantum-resistant trade ID
        trade_id = self._generate_secure_id()
        
        # Optimized trade context capture - only store what's needed
        self.active_trades[trade_id] = {
            'meta': {
                'id': trade_id,
                'strategy': trade['strategy'],
                'model_version': self.meta.active_version,
                'timestamp_ns': time.time_ns(),
                'expiry_ns': time.time_ns() + 60_000_000_000  # 1-minute TTL for stale trades
            },
            'execution': {
                'expected_price': trade['price'],
                'allowed_slippage': self.risk.get_slippage_limit(trade['asset']),
                'venue': trade.get('venue', 'auto'),
                'order_type': trade['type'],
                'size': trade['size'],
                'asset': trade['asset']
            },
            'context': {
                'liquidity_snapshot': self.liquidity.order_book_snapshot(trade['asset']),
                'institutional_presence': self.institutions.detect_activity(trade['asset']),
                'regime': self.regime.current_regime,
                'liquidity_score': self.liquidity.current_state(trade['asset'])
            },
            'risk': {
                'max_size': self.risk.get_position_limit(trade['asset']),
                'current_exposure': self.risk.get_current_exposure(trade['asset']),
                'leverage_limit': self.risk.get_leverage_limit(trade['strategy'])
            }
        }
        
        # Concurrent validation to avoid blocking the main execution path
        self.executor.submit(self._validate_strategy_compliance, trade_id, trade)
        self.executor.submit(self._adapt_to_market_conditions, trade_id, trade)
        
        self.logger.info("Trade registered for monitoring", trade_id=trade_id)
        return trade_id

    def _generate_secure_id(self) -> str:
        """Generate a quantum-resistant unique trade identifier"""
        # SHA3-256 is quantum-resistant and provides stronger security than CRC32
        unique_bytes = f"{uuid.uuid4().hex}:{time.time_ns()}".encode()
        return hashlib.sha3_256(unique_bytes).hexdigest()[:16]

    def _validate_strategy_compliance(self, trade_id: str, trade: Dict):
        """Ensures strategy alignment with current market conditions"""
        if trade['strategy'] in self._strategy_blacklist:
            self.logger.warning("Attempted execution with blacklisted strategy", 
                               strategy=trade['strategy'], 
                               trade_id=trade_id)
            self.executor.submit(self.execution.cancel_order, trade_id)
            return
            
        if not self.regime.is_strategy_allowed(trade['strategy']):
            self.logger.critical("Strategy-regime mismatch detected", 
                                strategy=trade['strategy'])
            self._strategy_blacklist.add(trade['strategy'])
            self.executor.submit(self.execution.cancel_order, trade_id)

    def _adapt_to_market_conditions(self, trade_id: str, trade: Dict):
        """Adapts execution based on institutional presence and liquidity"""
        if trade_id not in self.active_trades:
            return
            
        trade_data = self.active_trades[trade_id]
        asset = trade_data['execution']['asset']
        
        # Detect institutional activity and adapt execution method
        if self.institutions.detect_activity(asset):
            # If institutions are active, adjust execution to avoid front-running
            self.execution.adjust_params(
                asset=asset,
                trade_id=trade_id,
                params={
                    'execution_style': 'iceberg',
                    'venue_preference': 'dark_pool',
                    'time_in_force': 'day'
                }
            )
            
        # Adapt to liquidity conditions
        if trade_data['context']['liquidity_score'] < 0.3:  # Low liquidity
            # In low liquidity, use time-weighted execution to minimize impact
            self.execution.adjust_params(
                asset=asset,
                trade_id=trade_id,
                params={
                    'execution_style': 'twap',
                    'duration_minutes': 30
                }
            )

    async def update_trade(self, trade_id: str, execution_result: Dict):
        """
        Update trade with execution results and perform post-execution analysis
        This is called when an order is filled or partially filled
        """
        if trade_id not in self.active_trades:
            self.logger.error("Unknown trade update received", trade_id=trade_id)
            return
        
        trade = self.active_trades[trade_id]
        
        # Record execution metrics with nanosecond precision
        trade['execution']['actual_price'] = execution_result['price']
        trade['execution']['execution_time_ns'] = time.time_ns()
        trade['execution']['latency_ns'] = trade['execution']['execution_time_ns'] - trade['meta']['timestamp_ns']
        trade['execution']['latency_ms'] = trade['execution']['latency_ns'] / 1_000_000  # Convert to ms for human readability
        
        # Calculate execution quality metrics
        trade['metrics'] = self._calculate_execution_metrics(trade, execution_result)
        
        # Perform parallel system updates to avoid blocking the main thread
        await asyncio.gather(
            self._feed_ai_systems(trade),
            self._update_risk_parameters(trade),
            self._update_venue_metrics(trade)
        )
        
        # Archive trade data
        self._archive_trade(trade)
        
        # If fully executed, remove from active trades
        if execution_result.get('status') == 'filled':
            del self.active_trades[trade_id]

    def _calculate_execution_metrics(self, trade: Dict, execution_result: Dict) -> Dict:
        """
        Fast vectorized calculation of execution quality metrics
        Using pre-computed values where possible to reduce latency
        """
        expected_price = trade['execution']['expected_price']
        actual_price = execution_result['price']
        asset = trade['execution']['asset']
        size = trade['execution']['size']
        
        # Calculate metrics in a vectorized way
        metrics = {
            'slippage_bps': round(10000 * (actual_price - expected_price) / expected_price, 2),
            'market_impact': self.impact.measure_impact(asset, size),
            'execution_quality': min(1.0, max(0.0, 1.0 - abs(actual_price - expected_price) / expected_price * 100)),
            'liquidity_score': trade['context']['liquidity_score'],
            'venue_latency': trade['execution']['latency_ms']
        }
        
        # Add frontrunning detection score (pre-computed by AI model)
        metrics['frontrunning_detected'] = self.meta.detect_frontrunning(
            asset, 
            trade['context']['liquidity_snapshot'],
            actual_price
        )
        
        return metrics

    async def _feed_ai_systems(self, trade: Dict):
        """Feed execution data to AI subsystems for continuous improvement"""
        # All system updates run in parallel to minimize blocking
        update_tasks = [
            self.meta.process_execution_outcome(trade),
            self.regime.update_regime_model(trade),
            self.institutions.update_activity_model(trade),
            self.liquidity.update_profile(trade),
            self.strategy_evaluator.update_strategy_performance(
                trade['meta']['strategy'], 
                trade['metrics']
            )
        ]
        await asyncio.gather(*update_tasks)

    async def _update_risk_parameters(self, trade: Dict):
        """Dynamic risk recalibration based on execution outcomes"""
        asset = trade['execution']['asset']
        strategy = trade['meta']['strategy']
        
        # Update risk parameters based on execution quality
        if trade['metrics']['slippage_bps'] > 15:  # High slippage
            await self.risk.adjust_slippage_limits(
                asset=asset,
                new_limit=trade['execution']['allowed_slippage'] * 0.9  # Tighten limits
            )
        
        if trade['metrics']['frontrunning_detected']:
            # If frontrunning detected, reduce position sizes
            await self.risk.adjust_position_limits(
                asset=asset,
                strategy=strategy,
                limit_multiplier=0.8  # Reduce size by 20%
            )

    def _update_venue_metrics(self, trade: Dict):
        """Update venue performance metrics for future venue selection"""
        venue = trade['execution']['venue']
        
        # Get or create venue index in pre-allocated arrays
        if venue not in self.venue_metrics['venue_map']:
            idx = self.venue_metrics['counter'] % 100  # Circular buffer
            self.venue_metrics['venue_map'][venue] = idx
            self.venue_metrics['counter'] += 1
        else:
            idx = self.venue_metrics['venue_map'][venue]
        
        # Update venue metrics in pre-allocated arrays
        self.venue_metrics['latency'][idx] = trade['execution']['latency_ms']
        self.venue_metrics['slippage'][idx] = abs(trade['metrics']['slippage_bps'])
        self.venue_metrics['fill_rates'][idx] = 1.0  # Assume filled for now
        self.venue_metrics['impact_scores'][idx] = trade['metrics']['market_impact']
        
        # Simple venue performance tracking (separate from numpy arrays)
        if venue not in self.venue_performance:
            self.venue_performance[venue] = {
                'trades': 0,
                'avg_latency': 0,
                'avg_slippage': 0,
                'quality_score': 0
            }
            
        perf = self.venue_performance[venue]
        count = perf['trades']
        
        # Update rolling averages
        perf['avg_latency'] = (perf['avg_latency'] * count + trade['execution']['latency_ms']) / (count + 1)
        perf['avg_slippage'] = (perf['avg_slippage'] * count + trade['metrics']['slippage_bps']) / (count + 1)
        perf['quality_score'] = (perf['quality_score'] * count + trade['metrics']['execution_quality']) / (count + 1)
        perf['trades'] = count + 1

    def _archive_trade(self, trade: Dict):
        """Archive trade data for backtesting and strategy optimization"""
        # Compress trade data before archiving
        compressed_data = self.compressor.compress(msgpack.packb(trade))
        
        # Store in cache for immediate access
        self.trade_cache.append(compressed_data)
        
        # Submit async storage task (non-blocking)
        self.executor.submit(self._persist_trade_data, trade)

    def _persist_trade_data(self, trade: Dict):
        """Background task to persist trade data (non-blocking)"""
        # This would connect to a storage system like RocksDB or append to a Parquet file
        # Implementation depends on the specific storage system used
        pass

    async def _surveillance_loop(self):
        """Ultra-low latency surveillance loop for real-time trade monitoring"""
        while True:
            try:
                current_time_ns = time.time_ns()
                
                # Execute surveillance checks in parallel
                await asyncio.gather(
                    self._check_trade_expiry(current_time_ns),
                    self._enforce_risk_limits(),
                    self._detect_execution_anomalies()
                )
                
                # Ultra-low latency loop - 1ms precision
                await asyncio.sleep(self._SURVEILLANCE_INTERVAL_MS / 1000)
                
            except Exception as e:
                self.logger.error("Surveillance loop error", error=str(e))
                await asyncio.sleep(0.1)  # Brief pause on error

    async def _check_trade_expiry(self, current_time_ns: int):
        """Identifies and cancels stale trades based on TTL"""
        # Vectorized check for expired trades (faster than iterating)
        expired_ids = [
            trade_id for trade_id, trade in self.active_trades.items()
            if trade['meta']['expiry_ns'] < current_time_ns
        ]
        
        # Cancel expired trades in parallel
        if expired_ids:
            cancel_tasks = [self.execution.cancel_order(trade_id) for trade_id in expired_ids]
            await asyncio.gather(*cancel_tasks)
            
            # Log once for all expired trades rather than individual logs
            self.logger.warning(f"Cancelled {len(expired_ids)} expired trades")

    async def _enforce_risk_limits(self):
        """Real-time risk limit enforcement across all active trades"""
        # Skip if no active trades
        if not self.active_trades:
            return
            
        # Group trades by asset for more efficient risk processing
        trades_by_asset = {}
        for trade_id, trade in self.active_trades.items():
            asset = trade['execution']['asset']
            if asset not in trades_by_asset:
                trades_by_asset[asset] = []
            trades_by_asset[asset].append((trade_id, trade))
        
        # Process each asset group in parallel
        enforcement_tasks = []
        for asset, trades in trades_by_asset.items():
            enforcement_tasks.append(self._enforce_asset_risk(asset, trades))
        
        await asyncio.gather(*enforcement_tasks)

    async def _enforce_asset_risk(self, asset: str, trades: List[Tuple[str, Dict]]):
        """Enforce risk limits for a specific asset group"""
        # Get current exposure from risk engine
        current_exposure = self.risk.get_current_exposure(asset)
        max_exposure = self.risk.get_max_exposure(asset)
        
        # If approaching max exposure, cancel the lowest priority trades
        if current_exposure > max_exposure * 0.9:  # Within 90% of limit
            # Sort trades by strategy priority (higher priority = keep)
            sorted_trades = sorted(
                trades,
                key=lambda x: self.strategy_evaluator.get_strategy_priority(x[1]['meta']['strategy']),
                reverse=True  # Highest priority first
            )
            
            # Cancel lowest priority trades until under the threshold
            for trade_id, trade in sorted_trades[int(len(sorted_trades)/2):]:  # Cancel bottom half
                await self.execution.cancel_order(trade_id)
                current_exposure -= trade['execution']['size']
                if current_exposure <= max_exposure * 0.7:  # Cancel until 70% of max
                    break

    async def _detect_execution_anomalies(self):
        """Detect patterns of market manipulation or execution issues"""
        if len(self.trade_cache) < 10:  # Need sufficient data
            return
            
        # This would call into specialized anomaly detection models
        # Implementation would depend on the specific detection algorithm
        pass

    def _realtime_analysis(self):
        """Background thread for non-time-critical analysis"""
        while True:
            try:
                # Optimize execution venue selection
                self._optimize_venue_selection()
                
                # Update strategy performance metrics
                self._update_strategy_performance()
                
                # Sleep between analysis cycles (less time-critical)
                time.sleep(0.1)  # 100ms cycle time for non-critical analysis
            except Exception as e:
                self.logger.error("Analysis thread error", error=str(e))
                time.sleep(1)  # Brief pause on error

    def _optimize_venue_selection(self):
        """AI-driven venue optimization for future executions"""
        if not self.venue_performance:
            return
        
        # Calculate venue scores (combining latency, slippage, impact)
        venue_scores = {}
        for venue, metrics in self.venue_performance.items():
            if metrics['trades'] < 5:  # Skip venues with insufficient data
                continue
                
            # Lower values are better for latency and slippage
            latency_score = max(0, 1 - metrics['avg_latency'] / 1000)  # Normalize to 0-1
            slippage_score = max(0, 1 - abs(metrics['avg_slippage']) / 50)  # Normalize to 0-1
            
            # Combined score (quality is already 0-1)
            venue_scores[venue] = (
                0.3 * latency_score + 
                0.4 * slippage_score + 
                0.3 * metrics['quality_score']
            )
        
        if venue_scores:
            # Set best venue as primary
            best_venue = max(venue_scores, key=venue_scores.get)
            self.execution.set_primary_venue(best_venue)

    def _update_strategy_performance(self):
        """Update AI-driven strategy performance metrics"""
        # Delegated to StrategyEvaluator for separation of concerns
        pass

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for dashboard and reporting"""
        # Return compressed summary for mobile app
        return {
            'venue_performance': self.venue_performance,
            'active_trades': len(self.active_trades),
            'total_monitored': len(self.trade_cache)
        }

    def get_execution_heatmap(self) -> Dict[str, Any]:
        """Get detailed execution heatmap for web dashboard"""
        # Delegate to StrategyEvaluator for detailed metrics
        return self.strategy_evaluator.get_strategy_heatmap()

# Initialize as a singleton
trade_monitor = TradeMonitor()