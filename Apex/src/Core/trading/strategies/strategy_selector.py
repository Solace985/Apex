# src/Core/trading/strategies/strategy_selector.py

import asyncio
import hashlib
import hmac
import time
import numpy as np
import os
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from functools import lru_cache
import logging
from dataclasses import dataclass

# Core System Imports
from src.Core.data.realtime.market_data import UnifiedMarketFeed
from src.Core.data.realtime.websocket_handler import WebSocketHandler
from src.Core.trading.risk.risk_management import AdaptiveRiskController
from src.Core.trading.hft.liquidity_manager import LiquidityOptimizer
from src.Core.trading.security.security import SecurityManager
from src.Core.trading.logging.decision_logger import ExecutionAuditLogger
from src.Core.trading.execution.order_execution import OrderExecutionEngine
from src.Core.trading.execution.market_impact import MarketImpactAnalyzer
from src.Core.trading.strategies.strategy_orchestrator import StrategyOrchestrator
from src.Core.trading.strategies.strategy_evaluator import StrategyEvaluator

# AI System Imports
from src.ai.ensembles.meta_trader import MetaStrategyOptimizer
from src.ai.ensembles.ensemble_voting import EnsembleVotingSystem
from src.ai.reinforcement.reinforcement_learning import ReinforcementLearningEngine
from src.ai.analysis.market_regime_classifier import MarketRegimeClassifier
from src.ai.analysis.institutional_clusters import InstitutionalClusterAnalyzer
from src.ai.analysis.correlation_engine import CorrelationEngine

# Utility Imports
from utils.logging.structured_logger import StructuredLogger
from utils.helpers.error_handler import ErrorHandler
from utils.analytics.monte_carlo_simulator import MonteCarloSimulator
from utils.helpers.stealth_api import StealthAPI
from metrics.performance_metrics import PerformanceMetrics

# Cache decorator for strategy scoring
def cached_strategy_score(ttl_seconds=1):
    """Time-bounded LRU cache for strategy scoring with TTL"""
    def decorator(func):
        cache = {}
        lock = RLock()
        
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            with lock:
                current_time = time.time()
                if key in cache:
                    result, timestamp = cache[key]
                    if current_time - timestamp < ttl_seconds:
                        return result
                result = func(*args, **kwargs)
                cache[key] = (result, current_time)
                # Cleanup expired entries
                expired_keys = [k for k, (_, t) in cache.items() if current_time - t >= ttl_seconds]
                for k in expired_keys:
                    del cache[k]
                return result
        return wrapper
    return decorator

@dataclass
class StrategyPerformanceMetrics:
    """Container for strategy performance metrics"""
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_return: float = 0.0
    volatility: float = 0.0
    recent_success: bool = False
    latency_ms: float = 0.0
    market_impact: float = 0.0

class QuantumStrategySelector:
    """Advanced AI-powered strategy selection engine with HFT optimization"""
    
    # Class-level constants
    MAX_CONCURRENT_EVALUATIONS = 16
    STRATEGY_EVAL_TIMEOUT_MS = 50
    CIRCUIT_BREAKER_THRESHOLD = 3
    BLACKLIST_DURATION_SECONDS = 300
    SCORE_CACHE_TTL_SECONDS = 1
    REGIME_CACHE_TTL_SECONDS = 5
    
    def __init__(self, asset_universe: List[str], config: Dict[str, Any] = None):
        """Initialize the strategy selector with the given asset universe"""
        self.asset_universe = asset_universe
        self.config = config or {}
        
        # Thread and concurrency management
        self._executor = ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT_EVALUATIONS)
        self._strategy_lock = RLock()
        self._selector_lock = RLock()
        
        # Initialize internal state
        self._init_components()
        self._init_state()
        
        # Initialize the shared memory for strategy performance
        self._init_shared_memory()
        
        # Start background tasks
        self._start_background_tasks()
        
        # Log initialization
        self.logger.info("QuantumStrategySelector initialized", extra={"assets": len(self.asset_universe)})

    def _init_components(self):
        """Initialize integrated system components"""
        # Core market data components
        self.market_feed = UnifiedMarketFeed()
        self.websocket_handler = WebSocketHandler()
        
        # Risk and execution components
        self.risk_controller = AdaptiveRiskController()
        self.liquidity_optimizer = LiquidityOptimizer()
        self.security_manager = SecurityManager()
        self.execution_engine = OrderExecutionEngine()
        self.market_impact = MarketImpactAnalyzer()
        
        # Strategy components
        self.strategy_orchestrator = StrategyOrchestrator()
        self.strategy_evaluator = StrategyEvaluator()
        
        # AI components
        self.meta_optimizer = MetaStrategyOptimizer()
        self.ensemble_voting = EnsembleVotingSystem()
        self.rl_engine = ReinforcementLearningEngine()
        self.regime_classifier = MarketRegimeClassifier()
        self.institutional_analyzer = InstitutionalClusterAnalyzer()
        self.correlation_engine = CorrelationEngine()
        
        # Utilities
        self.logger = StructuredLogger("quantum_selector")
        self.error_handler = ErrorHandler()
        self.monte_carlo = MonteCarloSimulator()
        self.stealth_api = StealthAPI()
        self.audit_logger = ExecutionAuditLogger()
        self.performance_metrics = PerformanceMetrics()

    def _init_state(self):
        """Initialize internal state variables"""
        # Strategy registry stores available strategies per asset
        self.strategy_registry = {asset: set() for asset in self.asset_universe}
        
        # Strategy performance metrics
        self.strategy_performance = {}
        
        # Strategy blacklist for temporarily disabling problematic strategies
        self.strategy_blacklist = {}
        
        # Circuit breaker state
        self.circuit_breakers = {asset: False for asset in self.asset_universe}
        
        # Performance history for reinforcement learning
        self.performance_history = []
        
        # Failure counters for circuit breaker activation
        self.consecutive_failures = {asset: 0 for asset in self.asset_universe}
        
        # Current market regime cache
        self.regime_cache = {}
        
        # Strategy evaluation prioritization queues
        self.evaluation_queues = {
            'high_frequency': asyncio.Queue(),
            'medium_frequency': asyncio.Queue(),
            'low_frequency': asyncio.Queue()
        }

    def _init_shared_memory(self):
        """Initialize shared memory for inter-process communication"""
        # Strategy performance shared memory
        self.strategy_scores = {}
        
        # Market state shared memory
        self.market_states = {}
        
        # Execution feedback shared memory
        self.execution_feedback = {}

    def _start_background_tasks(self):
        """Start background monitoring and optimization tasks"""
        # These will run in separate threads to avoid blocking the main execution
        self._executor.submit(self._monitor_strategy_performance)
        self._executor.submit(self._update_ai_models)
        self._executor.submit(self._cleanup_blacklist)

    async def select_optimal_strategy(self, asset: str, 
                                    market_state: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Select the optimal trading strategy for the given asset
        
        Args:
            asset: The asset to select a strategy for
            market_state: Optional pre-fetched market state to avoid redundant API calls
            
        Returns:
            Dict containing strategy name and parameters, or None if no strategy is suitable
        """
        start_time = time.time()
        
        try:
            # Check if circuit breaker is active for this asset
            if self._is_circuit_breaker_active(asset):
                self.logger.warning(f"Circuit breaker active for {asset}", extra={"asset": asset})
                return None
            
            # Get or fetch market state
            state = market_state or await self._fetch_market_state(asset)
            if not state:
                self.logger.warning(f"No market state available for {asset}", extra={"asset": asset})
                return None
            
            # Get current market regime
            regime = await self._get_current_regime(asset, state)
            
            # Get valid strategies for this asset and regime
            valid_strategies = await self._get_valid_strategies(asset, regime)
            if not valid_strategies:
                self.logger.info(f"No valid strategies found for {asset}", extra={"asset": asset})
                return None
            
            # Evaluate strategies in parallel
            tasks = [
                self._evaluate_strategy(strategy, asset, state, regime)
                for strategy in valid_strategies
            ]
            
            # Wait for all evaluations with timeout
            results = await asyncio.gather(*tasks)
            
            # Filter out None results
            results = [r for r in results if r is not None]
            if not results:
                self.logger.warning(f"All strategy evaluations failed for {asset}", extra={"asset": asset})
                return None
            
            # Select the best strategy
            best_strategy = max(results, key=lambda x: x['score'])
            
            # Update shared memory with selection
            self._update_selection_memory(asset, best_strategy)
            
            # Log selection
            await self._log_strategy_selection(asset, best_strategy)
            
            # Prepare result
            result = {
                'name': best_strategy['name'],
                'parameters': best_strategy['parameters'],
                'score': best_strategy['score'],
                'regime': regime,
                'selection_time_ms': (time.time() - start_time) * 1000
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error selecting strategy for {asset}: {str(e)}", 
                              extra={"asset": asset, "error": str(e)})
            self.error_handler.handle_error(e)
            return None

    async def _fetch_market_state(self, asset: str) -> Dict[str, Any]:
        """Fetch current market state for the given asset"""
        try:
            # Try to get from cache first
            if asset in self.market_states and time.time() - self.market_states[asset]['timestamp'] < 1:
                return self.market_states[asset]['data']
            
            # Fetch from market feed
            state = await self.market_feed.get_market_state(asset)
            
            # Cache the result
            self.market_states[asset] = {
                'data': state,
                'timestamp': time.time()
            }
            
            return state
        except Exception as e:
            self.logger.error(f"Error fetching market state: {str(e)}", extra={"asset": asset})
            return None

    @lru_cache(maxsize=128)
    async def _get_current_regime(self, asset: str, state: Dict[str, Any] = None) -> str:
        """Get current market regime for the given asset"""
        try:
            # Check cache first
            cache_key = f"{asset}_{int(time.time() / self.REGIME_CACHE_TTL_SECONDS)}"
            if cache_key in self.regime_cache:
                return self.regime_cache[cache_key]
            
            # Fetch market state if not provided
            if not state:
                state = await self._fetch_market_state(asset)
                if not state:
                    return "unknown"
            
            # Determine regime using AI classifier
            regime = await self.regime_classifier.classify_regime(asset, state)
            
            # Cache the result
            self.regime_cache[cache_key] = regime
            
            return regime
        except Exception as e:
            self.logger.error(f"Error determining market regime: {str(e)}", extra={"asset": asset})
            return "unknown"

    async def _get_valid_strategies(self, asset: str, regime: str) -> List[str]:
        """Get valid strategies for the given asset and regime"""
        try:
            with self._strategy_lock:
                # Get all strategies for this asset
                all_strategies = self.strategy_registry.get(asset, set())
                
                # Filter strategies
                valid_strategies = [
                    s for s in all_strategies
                    if await self._is_strategy_valid(s, asset, regime)
                ]
                
                return valid_strategies
        except Exception as e:
            self.logger.error(f"Error getting valid strategies: {str(e)}", extra={"asset": asset})
            return []

    async def _is_strategy_valid(self, strategy: str, asset: str, regime: str) -> bool:
        """Check if a strategy is valid for the current market conditions"""
        # Skip blacklisted strategies
        if self._is_strategy_blacklisted(strategy, asset):
            return False
        
        # Check if strategy is suitable for current regime
        if not await self._is_regime_compatible(strategy, regime):
            return False
        
        # Check if strategy has sufficient liquidity
        if not await self._has_sufficient_liquidity(strategy, asset):
            return False
        
        # Check if strategy meets risk constraints
        if not await self._meets_risk_constraints(strategy, asset):
            return False
        
        return True

    def _is_strategy_blacklisted(self, strategy: str, asset: str) -> bool:
        """Check if a strategy is blacklisted for the given asset"""
        blacklist_key = f"{asset}_{strategy}"
        if blacklist_key in self.strategy_blacklist:
            blacklist_time = self.strategy_blacklist[blacklist_key]
            if time.time() - blacklist_time < self.BLACKLIST_DURATION_SECONDS:
                return True
            else:
                # Remove from blacklist if duration has expired
                with self._strategy_lock:
                    self.strategy_blacklist.pop(blacklist_key, None)
        return False

    async def _is_regime_compatible(self, strategy: str, regime: str) -> bool:
        """Check if a strategy is compatible with the current market regime"""
        try:
            # Get strategy metadata
            metadata = await self.meta_optimizer.get_strategy_metadata(strategy)
            if not metadata:
                return False
            
            # Check if strategy is compatible with current regime
            compatible_regimes = metadata.get('compatible_regimes', [])
            return regime in compatible_regimes or 'all' in compatible_regimes
        except Exception as e:
            self.logger.error(f"Error checking regime compatibility: {str(e)}", 
                             extra={"strategy": strategy, "regime": regime})
            return False

    async def _has_sufficient_liquidity(self, strategy: str, asset: str) -> bool:
        """Check if an asset has sufficient liquidity for a strategy"""
        try:
            # Get strategy liquidity requirements
            metadata = await self.meta_optimizer.get_strategy_metadata(strategy)
            if not metadata:
                return False
            
            min_liquidity = metadata.get('min_liquidity', 0)
            # Get current liquidity for the asset
            liquidity = await self.liquidity_optimizer.get_asset_liquidity(asset)
            if not liquidity:
                return False
            
            # Check if liquidity is sufficient
            return liquidity['available'] >= min_liquidity
        except Exception as e:
            self.logger.error(f"Error checking liquidity: {str(e)}", 
                             extra={"strategy": strategy, "asset": asset})
            return False

    async def _meets_risk_constraints(self, strategy: str, asset: str) -> bool:
        """Check if a strategy meets risk constraints for the given asset"""
        try:
            # Get strategy risk profile
            metadata = await self.meta_optimizer.get_strategy_metadata(strategy)
            if not metadata:
                return False
            
            # Get current portfolio risk exposure
            risk_exposure = await self.risk_controller.get_current_exposure(asset)
            if risk_exposure is None:
                return True  # Default to allowing if risk data is unavailable
            
            # Check if adding this strategy would exceed risk limits
            strategy_risk = metadata.get('risk_profile', {})
            max_exposure = strategy_risk.get('max_exposure', 1.0)
            
            return risk_exposure + max_exposure <= 1.0
        except Exception as e:
            self.logger.error(f"Error checking risk constraints: {str(e)}", 
                             extra={"strategy": strategy, "asset": asset})
            return False

    @cached_strategy_score(ttl_seconds=SCORE_CACHE_TTL_SECONDS)
    async def _evaluate_strategy(self, strategy: str, asset: str, 
                               market_state: Dict[str, Any], regime: str) -> Optional[Dict[str, Any]]:
        """
        Evaluate a strategy for the given asset and market state
        
        Uses vectorized calculations and caching for performance optimization
        """
        try:
            # Start timer for latency measurement
            start_time = time.time()
            
            # Get strategy parameters
            parameters = await self.meta_optimizer.get_strategy_parameters(strategy, asset, regime)
            if not parameters:
                return None
            
            # Calculate AI-based score using ensemble voting
            ai_scores = await self._calculate_ai_scores(strategy, asset, market_state, regime)
            
            # Calculate reinforcement learning adjustment
            rl_adjustment = await self._calculate_rl_adjustment(strategy, asset)
            
            # Calculate institutional cluster alignment
            institutional_alignment = await self._calculate_institutional_alignment(strategy, asset)
            
            # Calculate liquidity score
            liquidity_score = await self._calculate_liquidity_score(strategy, asset)
            
            # Calculate risk-adjusted score
            risk_score = await self._calculate_risk_score(strategy, asset)
            
            # Calculate market impact score
            impact_score = await self._calculate_impact_score(strategy, asset)
            
            # Calculate performance history score
            performance_score = self._calculate_performance_score(strategy, asset)
            
            # Combine all scores using weighted average
            weights = np.array([0.25, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1])
            scores = np.array([
                ai_scores['combined'], 
                rl_adjustment, 
                institutional_alignment,
                liquidity_score,
                risk_score,
                impact_score,
                performance_score
            ])
            
            # Use vectorized calculation for performance
            combined_score = np.sum(weights * scores)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Prepare result
            result = {
                'name': strategy,
                'parameters': parameters,
                'score': float(combined_score),  # Convert from numpy to Python float
                'component_scores': {
                    'ai_score': ai_scores,
                    'rl_adjustment': float(rl_adjustment),
                    'institutional_alignment': float(institutional_alignment),
                    'liquidity_score': float(liquidity_score),
                    'risk_score': float(risk_score),
                    'impact_score': float(impact_score),
                    'performance_score': float(performance_score)
                },
                'latency_ms': latency_ms
            }
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Strategy evaluation timed out: {strategy}", 
                              extra={"strategy": strategy, "asset": asset})
            return None
        except Exception as e:
            self.logger.error(f"Error evaluating strategy: {str(e)}", 
                             extra={"strategy": strategy, "asset": asset})
            return None

    async def _calculate_ai_scores(self, strategy: str, asset: str, 
                                 market_state: Dict[str, Any], regime: str) -> Dict[str, float]:
        """Calculate AI-based scores using multiple models"""
        try:
            # Get scores from multiple AI models in parallel
            results = await asyncio.gather(
                self.meta_optimizer.get_strategy_score(strategy, asset, regime),
                self.regime_classifier.get_regime_compatibility(strategy, regime),
                self.correlation_engine.get_correlation_score(strategy, asset),
                return_exceptions=True
            )
            
            # Filter out exceptions
            scores = [r for r in results if not isinstance(r, Exception)]
            if not scores:
                return {'combined': 0.5, 'models': {}}
            
            # Use vectorized calculation for mean
            combined = float(np.mean(scores))
            
            return {
                'combined': combined,
                'models': {
                    'meta_optimizer': scores[0] if len(scores) > 0 else 0,
                    'regime_classifier': scores[1] if len(scores) > 1 else 0,
                    'correlation_engine': scores[2] if len(scores) > 2 else 0
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating AI scores: {str(e)}", 
                             extra={"strategy": strategy, "asset": asset})
            return {'combined': 0.5, 'models': {}}

    async def _calculate_rl_adjustment(self, strategy: str, asset: str) -> float:
        """Calculate reinforcement learning adjustment for strategy score"""
        try:
            # Get RL adjustment from reinforcement learning engine
            adjustment = await self.rl_engine.get_strategy_adjustment(
                strategy=strategy,
                asset=asset,
                timeframe=self.config.get('rl_timeframe', '1d')
            )
            
            return adjustment
        except Exception as e:
            self.logger.error(f"Error calculating RL adjustment: {str(e)}", 
                             extra={"strategy": strategy, "asset": asset})
            return 0.0

    async def _calculate_institutional_alignment(self, strategy: str, asset: str) -> float:
        """Calculate alignment with institutional investor patterns"""
        try:
            # Get institutional alignment score
            alignment = await self.institutional_analyzer.get_alignment_score(
                strategy=strategy,
                asset=asset
            )
            
            return alignment
        except Exception as e:
            self.logger.error(f"Error calculating institutional alignment: {str(e)}", 
                             extra={"strategy": strategy, "asset": asset})
            return 0.5

    async def _calculate_liquidity_score(self, strategy: str, asset: str) -> float:
        """Calculate liquidity score for strategy execution"""
        try:
            # Get strategy liquidity requirements
            metadata = await self.meta_optimizer.get_strategy_metadata(strategy)
            if not metadata:
                return 0.5
            
            min_liquidity = metadata.get('min_liquidity', 0)
            
            # Get current liquidity
            liquidity = await self.liquidity_optimizer.get_asset_liquidity(asset)
            if not liquidity:
                return 0.5
            
            # Calculate liquidity score as ratio of available to required
            if min_liquidity <= 0:
                return 1.0
            
            score = min(1.0, liquidity['available'] / min_liquidity)
            return float(score)
        except Exception as e:
            self.logger.error(f"Error calculating liquidity score: {str(e)}", 
                             extra={"strategy": strategy, "asset": asset})
            return 0.5

    async def _calculate_risk_score(self, strategy: str, asset: str) -> float:
        """Calculate risk-adjusted score for strategy"""
        try:
            # Get strategy risk profile
            metadata = await self.meta_optimizer.get_strategy_metadata(strategy)
            if not metadata:
                return 0.5
            
            # Get risk metrics from risk controller
            risk_metrics = await self.risk_controller.get_strategy_risk_metrics(strategy, asset)
            if not risk_metrics:
                return 0.5
            
            # Calculate risk score
            risk_score = 1.0 - risk_metrics.get('normalized_risk', 0.5)
            return float(risk_score)
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {str(e)}", 
                             extra={"strategy": strategy, "asset": asset})
            return 0.5

    async def _calculate_impact_score(self, strategy: str, asset: str) -> float:
        """Calculate market impact score for strategy execution"""
        try:
            # Get expected trade size
            metadata = await self.meta_optimizer.get_strategy_metadata(strategy)
            if not metadata:
                return 0.5
            
            expected_size = metadata.get('expected_size', 0)
            
            # Get expected market impact
            impact = await self.market_impact.estimate_impact(asset, expected_size)
            if impact is None:
                return 0.5
            
            # Calculate impact score (lower impact is better)
            impact_score = 1.0 - min(1.0, impact)
            return float(impact_score)
        except Exception as e:
            self.logger.error(f"Error calculating impact score: {str(e)}", 
                             extra={"strategy": strategy, "asset": asset})
            return 0.5

    def _calculate_performance_score(self, strategy: str, asset: str) -> float:
        """Calculate performance history score for strategy"""
        try:
            # Get strategy performance metrics
            performance_key = f"{asset}_{strategy}"
            if performance_key not in self.strategy_performance:
                return 0.5
            
            metrics = self.strategy_performance[performance_key]
            
            # Calculate combined performance score
            weights = np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
            
            # Normalize metrics and clip to [0,1]
            win_rate = min(1.0, metrics.win_rate)
            profit_factor = min(1.0, metrics.profit_factor / 3.0)  # Normalize to [0,1]
            sharpe = min(1.0, max(0.0, metrics.sharpe_ratio / 3.0))  # Normalize to [0,1]
            drawdown_factor = 1.0 - min(1.0, metrics.max_drawdown)
            recent_success = 1.0 if metrics.recent_success else 0.0
            latency_factor = 1.0 - min(1.0, metrics.latency_ms / 100.0)  # Lower is better
            
            # Vectorized calculation of performance score
            scores = np.array([win_rate, profit_factor, sharpe, drawdown_factor, recent_success, latency_factor])
            performance_score = float(np.sum(weights * scores))
            
            return performance_score
            
        except Exception as e:
            self.logger.error(f"Error calculating performance score: {str(e)}", 
                             extra={"strategy": strategy, "asset": asset})
            return 0.5

    def _is_circuit_breaker_active(self, asset: str) -> bool:
        """Check if circuit breaker is active for the given asset"""
        with self._strategy_lock:
            return self.circuit_breakers.get(asset, False)

    def _update_selection_memory(self, asset: str, strategy: Dict[str, Any]):
        """Update shared memory with strategy selection"""
        with self._strategy_lock:
            self.strategy_scores[asset] = {
                'strategy': strategy['name'],
                'score': strategy['score'],
                'timestamp': time.time()
            }

    async def _log_strategy_selection(self, asset: str, strategy: Dict[str, Any]):
        """Log strategy selection to audit log and structured logger"""
        try:
            # Log to structured logger
            self.logger.info(
                f"Selected strategy {strategy['name']} for {asset}",
                extra={
                    'asset': asset,
                    'strategy': strategy['name'],
                    'score': strategy['score'],
                    'latency_ms': strategy['latency_ms']
                }
            )
            
            # Log to audit logger for compliance
            await self.audit_logger.log_strategy_selection({
                'asset': asset,
                'strategy': strategy['name'],
                'timestamp': datetime.utcnow().isoformat(),
                'score': strategy['score'],
                'component_scores': strategy['component_scores']
            })
        except Exception as e:
            self.logger.error(f"Error logging strategy selection: {str(e)}", 
                             extra={"asset": asset, "strategy": strategy['name']})

    async def update_strategy_registry(self, updates: Dict[str, List[str]]):
        """
        Update the strategy registry with new strategies
        
        Args:
            updates: Dictionary of asset -> list of strategy names
        """
        try:
            with self._strategy_lock:
                for asset, strategies in updates.items():
                    if asset in self.strategy_registry:
                        # Validate strategies first
                        valid_strategies = set()
                        for strategy in strategies:
                            if await self._validate_strategy(strategy):
                                valid_strategies.add(strategy)
                        
                        # Update registry
                        self.strategy_registry[asset] = valid_strategies
                    else:
                        self.logger.warning(f"Unknown asset in update: {asset}")
                        
            self.logger.info("Strategy registry updated", 
                           extra={"assets": len(updates), "total_strategies": sum(len(s) for s in updates.values())})
            
        except Exception as e:
            self.logger.error(f"Error updating strategy registry: {str(e)}")
            self.error_handler.handle_error(e)

    async def _validate_strategy(self, strategy: str) -> bool:
        """Validate a strategy for security and integrity"""
        try:
            # Get strategy metadata for validation
            metadata = await self.meta_optimizer.get_strategy_metadata(strategy)
            if not metadata:
                return False
            
            # Verify cryptographic signature if available
            if 'signature' in metadata:
                strategy_hash = hashlib.sha256(str(metadata.get('parameters', {})).encode()).hexdigest()
                signature_valid = hmac.compare_digest(
                    metadata['signature'],
                    self.security_manager.sign_message(strategy_hash)
                )
                if not signature_valid:
                    self.logger.warning(f"Invalid signature for strategy: {strategy}")
                    return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error validating strategy: {str(e)}", extra={"strategy": strategy})
            return False

    async def record_strategy_performance(self, asset: str, strategy: str, 
                                        result: Dict[str, Any]):
        """
        Record strategy execution performance for feedback loop
        
        Args:
            asset: The asset traded
            strategy: The strategy used
            result: Dictionary containing performance metrics
        """
        try:
            performance_key = f"{asset}_{strategy}"
            
            # Create or update performance metrics
            metrics = StrategyPerformanceMetrics(
                win_rate=result.get('win_rate', 0.0),
                profit_factor=result.get('profit_factor', 0.0),
                sharpe_ratio=result.get('sharpe_ratio', 0.0),
                max_drawdown=result.get('max_drawdown', 0.0),
                avg_return=result.get('avg_return', 0.0),
                volatility=result.get('volatility', 0.0),
                recent_success=result.get('profit', 0.0) > 0,
                latency_ms=result.get('execution_latency_ms', 0.0),
                market_impact=result.get('market_impact', 0.0)
            )
            
            # Update performance metrics in shared memory
            with self._strategy_lock:
                self.strategy_performance[performance_key] = metrics
                
                # Update performance history for reinforcement learning
                self.performance_history.append({
                    'asset': asset,
                    'strategy': strategy,
                    'timestamp': time.time(),
                    'metrics': {k: v for k, v in vars(metrics).items()}
                })
                
                # Keep history limited to avoid memory bloat
                if len(self.performance_history) > self.config.get('max_history_size', 1000):
                    self.performance_history = self.performance_history[-self.config.get('max_history_size', 1000):]
                
                # Check for consecutive failures
                if not metrics.recent_success:
                    self.consecutive_failures[asset] = self.consecutive_failures.get(asset, 0) + 1
                    
                    # Check if circuit breaker should be activated
                    if self.consecutive_failures[asset] >= self.CIRCUIT_BREAKER_THRESHOLD:
                        self._activate_circuit_breaker(asset)
                        
                        # Blacklist the strategy temporarily
                        self._blacklist_strategy(strategy, asset)
                else:
                    # Reset failure counter on success
                    self.consecutive_failures[asset] = 0
            
            # Update reinforcement learning engine
            await self.rl_engine.update_strategy_performance(
                strategy=strategy,
                asset=asset,
                result=result
            )
            
            # Log performance
            self.logger.info(
                f"Recorded performance for {strategy} on {asset}",
                extra={
                    'asset': asset,
                    'strategy': strategy,
                    'win_rate': metrics.win_rate,
                    'profit_factor': metrics.profit_factor,
                    'sharpe_ratio': metrics.sharpe_ratio
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error recording strategy performance: {str(e)}", 
                             extra={"asset": asset, "strategy": strategy})
            self.error_handler.handle_error(e)

    def _activate_circuit_breaker(self, asset: str):
        """Activate circuit breaker for an asset to temporarily halt trading"""
        with self._strategy_lock:
            self.circuit_breakers[asset] = True
            
            # Schedule circuit breaker deactivation
            deactivation_time = time.time() + self.config.get('circuit_breaker_duration_seconds', 300)
            self._executor.submit(self._deactivate_circuit_breaker, asset, deactivation_time)
            
            self.logger.warning(f"Circuit breaker activated for {asset}", extra={"asset": asset})
            
            # Notify risk management system
            asyncio.create_task(self.risk_controller.notify_circuit_breaker(asset, True))

    def _deactivate_circuit_breaker(self, asset: str, deactivation_time: float):
        """Deactivate circuit breaker after specified time"""
        # Sleep until deactivation time
        sleep_time = max(0, deactivation_time - time.time())
        time.sleep(sleep_time)
        
        with self._strategy_lock:
            self.circuit_breakers[asset] = False
            self.consecutive_failures[asset] = 0
            
            self.logger.info(f"Circuit breaker deactivated for {asset}", extra={"asset": asset})
            
            # Notify risk management system
            asyncio.create_task(self.risk_controller.notify_circuit_breaker(asset, False))

    def _blacklist_strategy(self, strategy: str, asset: str):
        """Blacklist a strategy for an asset temporarily"""
        blacklist_key = f"{asset}_{strategy}"
        with self._strategy_lock:
            self.strategy_blacklist[blacklist_key] = time.time()
            
            self.logger.warning(
                f"Strategy {strategy} blacklisted for {asset}",
                extra={"asset": asset, "strategy": strategy}
            )

    def _monitor_strategy_performance(self):
        """Background task to monitor strategy performance"""
        try:
            while True:
                # Sleep interval
                time.sleep(self.config.get('performance_monitoring_interval_seconds', 60))
                
                # Snapshot current strategies
                with self._strategy_lock:
                    performance_data = self.strategy_performance.copy()
                
                # Analyze performance data
                for key, metrics in performance_data.items():
                    try:
                        asset, strategy = key.split('_', 1)
                        
                        # Check for underperforming strategies
                        if (metrics.win_rate < self.config.get('min_win_rate', 0.4) or
                            metrics.sharpe_ratio < self.config.get('min_sharpe_ratio', 0.5)):
                            
                            # Only if we have significant data
                            if metrics.win_rate > 0:  # Non-zero means we have data
                                self._blacklist_strategy(strategy, asset)
                                
                                self.logger.warning(
                                    f"Underperforming strategy detected: {strategy} on {asset}",
                                    extra={
                                        'asset': asset,
                                        'strategy': strategy,
                                        'win_rate': metrics.win_rate,
                                        'sharpe_ratio': metrics.sharpe_ratio
                                    }
                                )
                    except Exception as e:
                        self.logger.error(f"Error analyzing performance data: {str(e)}", 
                                        extra={"key": key})
        except Exception as e:
            self.logger.error(f"Performance monitoring thread error: {str(e)}")
            # Restart the monitoring thread
            self._executor.submit(self._monitor_strategy_performance)

    def _update_ai_models(self):
        """Background task to update AI models with latest performance data"""
        try:
            while True:
                # Sleep interval
                time.sleep(self.config.get('ai_update_interval_seconds', 300))
                
                # Snapshot current performance history
                with self._strategy_lock:
                    history = self.performance_history.copy()
                
                if not history:
                    continue
                
                # Update AI models asynchronously
                asyncio.run(self._update_models_async(history))
                
        except Exception as e:
            self.logger.error(f"AI model update thread error: {str(e)}")
            # Restart the update thread
            self._executor.submit(self._update_ai_models)

    async def _update_models_async(self, history: List[Dict[str, Any]]):
        """Update AI models with latest performance data"""
        try:
            # Prepare update tasks
            update_tasks = [
                self.meta_optimizer.update_model(history),
                self.rl_engine.batch_update(history),
                self.regime_classifier.update_model(history),
                self.institutional_analyzer.update_model(history)
            ]
            
            # Execute updates in parallel
            results = await asyncio.gather(*update_tasks, return_exceptions=True)
            
            # Log results
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            self.logger.info(
                f"Updated AI models with {len(history)} data points",
                extra={"success_count": success_count, "total": len(update_tasks)}
            )
            
            # Log any errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error updating AI model {i}: {str(result)}")
            
        except Exception as e:
            self.logger.error(f"Error in async model update: {str(e)}")

    def _cleanup_blacklist(self):
        """Background task to clean up expired blacklist entries"""
        try:
            while True:
                # Sleep interval
                time.sleep(self.config.get('blacklist_cleanup_interval_seconds', 60))
                
                # Get current time
                current_time = time.time()
                
                # Clean up expired entries
                with self._strategy_lock:
                    expired_keys = [
                        key for key, timestamp in self.strategy_blacklist.items()
                        if current_time - timestamp >= self.BLACKLIST_DURATION_SECONDS
                    ]
                    
                    # Remove expired entries
                    for key in expired_keys:
                        self.strategy_blacklist.pop(key, None)
                    
                    if expired_keys:
                        self.logger.info(f"Cleaned up {len(expired_keys)} expired blacklist entries")
                
        except Exception as e:
            self.logger.error(f"Blacklist cleanup thread error: {str(e)}")
            # Restart the cleanup thread
            self._executor.submit(self._cleanup_blacklist)

    async def get_strategy_metrics(self, asset: str = None) -> Dict[str, Any]:
        """
        Get performance metrics for all strategies or a specific asset
        
        Args:
            asset: Optional asset to filter metrics
            
        Returns:
            Dictionary of strategy metrics
        """
        try:
            with self._strategy_lock:
                if asset:
                    # Filter metrics for specific asset
                    asset_metrics = {
                        key.split('_', 1)[1]: vars(metrics)
                        for key, metrics in self.strategy_performance.items()
                        if key.startswith(f"{asset}_")
                    }
                    return {
                        'asset': asset,
                        'strategy_count': len(asset_metrics),
                        'strategies': asset_metrics
                    }
                else:
                    # Aggregate metrics for all assets
                    assets = {}
                    for key, metrics in self.strategy_performance.items():
                        asset, strategy = key.split('_', 1)
                        if asset not in assets:
                            assets[asset] = {
                                'strategy_count': 0,
                                'strategies': {}
                            }
                        assets[asset]['strategy_count'] += 1
                        assets[asset]['strategies'][strategy] = vars(metrics)
                    
                    return {
                        'asset_count': len(assets),
                        'total_strategy_count': sum(data['strategy_count'] for data in assets.values()),
                        'assets': assets
                    }
        except Exception as e:
            self.logger.error(f"Error getting strategy metrics: {str(e)}", 
                             extra={"asset": asset})
            return {}

    async def get_best_strategy(self, asset: str) -> Optional[str]:
        """
        Get the best performing strategy for the given asset
        
        Args:
            asset: The asset to get the best strategy for
            
        Returns:
            Name of the best strategy, or None if no strategies available
        """
        try:
            # Get current market state
            state = await self._fetch_market_state(asset)
            if not state:
                return None
            
            # Get current market regime
            regime = await self._get_current_regime(asset, state)
            
            # Get valid strategies
            valid_strategies = await self._get_valid_strategies(asset, regime)
            if not valid_strategies:
                return None
            
            # Evaluate all strategies
            tasks = [
                self._evaluate_strategy(strategy, asset, state, regime)
                for strategy in valid_strategies
            ]
            
            # Wait for all evaluations
            results = await asyncio.gather(*tasks)
            
            # Filter out None results
            results = [r for r in results if r is not None]
            if not results:
                return None
            
            # Select the best strategy
            best_strategy = max(results, key=lambda x: x['score'])
            
            return best_strategy['name']
        except Exception as e:
            self.logger.error(f"Error getting best strategy: {str(e)}", 
                             extra={"asset": asset})
            return None

    async def simulate_strategy_performance(self, strategy: str, asset: str, 
                                           timeframe: str = '1d', 
                                           periods: int = 30) -> Dict[str, Any]:
        """
        Simulate strategy performance using Monte Carlo simulation
        
        Args:
            strategy: The strategy to simulate
            asset: The asset to simulate on
            timeframe: The timeframe to simulate (e.g., '1d', '1h')
            periods: Number of periods to simulate
            
        Returns:
            Dictionary of simulation results
        """
        try:
            # Get strategy parameters
            parameters = await self.meta_optimizer.get_strategy_parameters(strategy, asset)
            if not parameters:
                return {'success': False, 'error': 'Strategy parameters not found'}
            
            # Get historical data
            historical_data = await self.market_feed.get_historical_data(
                asset=asset,
                timeframe=timeframe,
                periods=periods * 2  # Get more data for better simulation
            )
            if not historical_data or len(historical_data) < periods:
                return {'success': False, 'error': 'Insufficient historical data'}
            
            # Run Monte Carlo simulation
            simulation_results = await self.monte_carlo.simulate_strategy(
                strategy=strategy,
                parameters=parameters,
                historical_data=historical_data,
                periods=periods,
                iterations=self.config.get('monte_carlo_iterations', 1000)
            )
            
            # Add risk metrics
            risk_metrics = await self.risk_controller.calculate_risk_metrics(
                asset=asset,
                strategy=strategy,
                simulation_results=simulation_results
            )
            
            # Combine results
            combined_results = {
                'success': True,
                'strategy': strategy,
                'asset': asset,
                'timeframe': timeframe,
                'periods': periods,
                'simulation': simulation_results,
                'risk': risk_metrics
            }
            
            return combined_results
        except Exception as e:
            self.logger.error(f"Error simulating strategy performance: {str(e)}", 
                             extra={"strategy": strategy, "asset": asset})
            return {'success': False, 'error': str(e)}

    async def optimize_strategy_parameters(self, strategy: str, asset: str) -> Dict[str, Any]:
        """
        Optimize strategy parameters using AI and historical data
        
        Args:
            strategy: The strategy to optimize
            asset: The asset to optimize for
            
        Returns:
            Dictionary of optimized parameters
        """
        try:
            # Get current parameters
            current_parameters = await self.meta_optimizer.get_strategy_parameters(strategy, asset)
            if not current_parameters:
                return {'success': False, 'error': 'Strategy parameters not found'}
            
            # Get current market regime
            market_state = await self._fetch_market_state(asset)
            if not market_state:
                return {'success': False, 'error': 'Could not fetch market state'}
                
            regime = await self._get_current_regime(asset, market_state)
            
            # Get historical data for optimization
            historical_data = await self.market_feed.get_historical_data(
                asset=asset,
                timeframe=self.config.get('optimization_timeframe', '1d'),
                periods=self.config.get('optimization_periods', 60)
            )
            
            if not historical_data or len(historical_data) < self.config.get('min_optimization_periods', 30):
                return {'success': False, 'error': 'Insufficient historical data for optimization'}
            
            # Use meta-optimizer to optimize parameters
            optimization_result = await self.meta_optimizer.optimize_parameters(
                strategy=strategy,
                asset=asset,
                regime=regime,
                historical_data=historical_data,
                current_parameters=current_parameters,
                constraints=await self.risk_controller.get_optimization_constraints(asset)
            )
            
            if not optimization_result.get('success', False):
                return {'success': False, 'error': optimization_result.get('error', 'Optimization failed')}
            
            # Validate optimized parameters with risk controller
            risk_validation = await self.risk_controller.validate_strategy_parameters(
                strategy=strategy,
                asset=asset,
                parameters=optimization_result['parameters']
            )
            
            if not risk_validation.get('valid', False):
                return {'success': False, 'error': risk_validation.get('reason', 'Risk validation failed')}
            
            # Test optimized parameters with Monte Carlo simulation
            simulation_result = await self.simulate_strategy_performance(
                strategy=strategy,
                asset=asset,
                timeframe=self.config.get('optimization_timeframe', '1d'),
                periods=self.config.get('optimization_test_periods', 30)
            )
            
            # Log optimization results
            self.logger.info(
                f"Optimized parameters for {strategy} on {asset}",
                extra={
                    'asset': asset,
                    'strategy': strategy,
                    'improvement': optimization_result.get('improvement', 0),
                    'parameter_count': len(optimization_result['parameters'])
                }
            )
            
            # Return combined results
            return {
                'success': True,
                'strategy': strategy,
                'asset': asset,
                'original_parameters': current_parameters,
                'optimized_parameters': optimization_result['parameters'],
                'improvement': optimization_result.get('improvement', 0),
                'simulation': simulation_result.get('simulation', {}),
                'risk_metrics': simulation_result.get('risk', {})
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing strategy parameters: {str(e)}", 
                             extra={"strategy": strategy, "asset": asset})
            return {'success': False, 'error': str(e)}

    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """
        Get real-time performance metrics for the strategy selector
        
        Returns:
            Dictionary of real-time metrics
        """
        try:
            # Get basic metrics
            with self._strategy_lock:
                active_strategies = sum(len(strategies) for strategies in self.strategy_registry.values())
                blacklisted_count = len(self.strategy_blacklist)
                circuit_breaker_count = sum(1 for active in self.circuit_breakers.values() if active)
            
            # Calculate performance metrics
            win_rates = []
            profit_factors = []
            sharpe_ratios = []
            
            for metrics in self.strategy_performance.values():
                if metrics.win_rate > 0:  # Only include strategies with data
                    win_rates.append(metrics.win_rate)
                    profit_factors.append(metrics.profit_factor)
                    sharpe_ratios.append(metrics.sharpe_ratio)
            
            # Use vectorized calculations for performance
            avg_win_rate = float(np.mean(win_rates)) if win_rates else 0
            avg_profit_factor = float(np.mean(profit_factors)) if profit_factors else 0
            avg_sharpe = float(np.mean(sharpe_ratios)) if sharpe_ratios else 0
            
            # Get system status
            system_status = await self._get_system_status()
            
            return {
                'timestamp': time.time(),
                'active_strategies': active_strategies,
                'blacklisted_strategies': blacklisted_count,
                'circuit_breakers_active': circuit_breaker_count,
                'assets_count': len(self.asset_universe),
                'performance': {
                    'average_win_rate': avg_win_rate,
                    'average_profit_factor': avg_profit_factor,
                    'average_sharpe_ratio': avg_sharpe,
                    'strategies_with_data': len(win_rates)
                },
                'system_status': system_status
            }
            
        except Exception as e:
            self.logger.error(f"Error getting real-time metrics: {str(e)}")
            return {'error': str(e)}

    async def _get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        try:
            # Get component statuses in parallel
            tasks = [
                self.market_feed.get_status(),
                self.risk_controller.get_status(),
                self.liquidity_optimizer.get_status(),
                self.meta_optimizer.get_status(),
                self.rl_engine.get_status()
            ]
            
            # Wait for all statuses
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            statuses = {}
            for i, component in enumerate(['market_feed', 'risk_controller', 'liquidity_optimizer', 
                                          'meta_optimizer', 'rl_engine']):
                if isinstance(results[i], Exception):
                    statuses[component] = {'status': 'error', 'error': str(results[i])}
                else:
                    statuses[component] = results[i]
            
            return statuses
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    async def export_strategy_data(self, format: str = 'json') -> Dict[str, Any]:
        """
        Export strategy data for external analysis
        
        Args:
            format: Export format ('json' or 'csv')
            
        Returns:
            Dictionary containing export data or file path
        """
        try:
            # Prepare export data
            export_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'asset_universe': self.asset_universe,
                'strategy_registry': {asset: list(strategies) for asset, strategies in self.strategy_registry.items()},
                'performance_metrics': {
                    key: vars(metrics) for key, metrics in self.strategy_performance.items()
                },
                'performance_history': self.performance_history[-self.config.get('export_history_limit', 100):]
            }
            
            # Export based on format
            if format.lower() == 'json':
                export_path = f"exports/strategy_data_{int(time.time())}.json"
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(export_path), exist_ok=True)
                
                # Write to file
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                return {'success': True, 'format': 'json', 'path': export_path}
                
            elif format.lower() == 'csv':
                # Convert to CSV format
                export_path = f"exports/strategy_data_{int(time.time())}"
                os.makedirs(export_path, exist_ok=True)
                
                # Write multiple CSV files for different data types
                
                # Performance metrics
                metrics_data = []
                for key, metrics in self.strategy_performance.items():
                    asset, strategy = key.split('_', 1)
                    metrics_dict = vars(metrics)
                    metrics_dict['asset'] = asset
                    metrics_dict['strategy'] = strategy
                    metrics_data.append(metrics_dict)
                
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.to_csv(f"{export_path}/performance_metrics.csv", index=False)
                
                # Performance history
                history_df = pd.DataFrame(self.performance_history)
                history_df.to_csv(f"{export_path}/performance_history.csv", index=False)
                
                # Strategy registry
                registry_data = []
                for asset, strategies in self.strategy_registry.items():
                    for strategy in strategies:
                        registry_data.append({'asset': asset, 'strategy': strategy})
                
                registry_df = pd.DataFrame(registry_data)
                registry_df.to_csv(f"{export_path}/strategy_registry.csv", index=False)
                
                return {'success': True, 'format': 'csv', 'path': export_path}
                
            else:
                return {'success': False, 'error': f"Unsupported export format: {format}"}
            
        except Exception as e:
            self.logger.error(f"Error exporting strategy data: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def import_strategy_data(self, import_path: str) -> Dict[str, Any]:
        """
        Import strategy data from external source
        
        Args:
            import_path: Path to import data from
            
        Returns:
            Dictionary containing import results
        """
        try:
            if not os.path.exists(import_path):
                return {'success': False, 'error': f"Import path does not exist: {import_path}"}
            
            # Determine import format from extension
            if import_path.endswith('.json'):
                # Import from JSON
                with open(import_path, 'r') as f:
                    import_data = json.load(f)
                
                # Validate data structure
                required_keys = ['asset_universe', 'strategy_registry', 'performance_metrics']
                for key in required_keys:
                    if key not in import_data:
                        return {'success': False, 'error': f"Missing required key in import data: {key}"}
                
                # Update asset universe if needed
                if set(import_data['asset_universe']) != set(self.asset_universe):
                    self.logger.warning("Asset universe in import data differs from current universe")
                
                # Update strategy registry
                with self._strategy_lock:
                    # Merge with existing registry
                    for asset, strategies in import_data['strategy_registry'].items():
                        if asset in self.strategy_registry:
                            self.strategy_registry[asset].update(strategies)
                        else:
                            self.strategy_registry[asset] = set(strategies)
                    
                    # Import performance metrics
                    for key, metrics_dict in import_data['performance_metrics'].items():
                        self.strategy_performance[key] = StrategyPerformanceMetrics(**metrics_dict)
                    
                    # Import performance history if available
                    if 'performance_history' in import_data:
                        # Add only new entries to avoid duplicates
                        existing_timestamps = {entry['timestamp'] for entry in self.performance_history}
                        new_history = [
                            entry for entry in import_data['performance_history']
                            if entry['timestamp'] not in existing_timestamps
                        ]
                        
                        self.performance_history.extend(new_history)
                        
                        # Keep history limited to avoid memory bloat
                        if len(self.performance_history) > self.config.get('max_history_size', 1000):
                            self.performance_history = self.performance_history[-self.config.get('max_history_size', 1000):]
                
                return {
                    'success': True,
                    'format': 'json',
                    'assets_imported': len(import_data['strategy_registry']),
                    'strategies_imported': sum(len(strategies) for strategies in import_data['strategy_registry'].values()),
                    'metrics_imported': len(import_data['performance_metrics']),
                    'history_imported': len(new_history) if 'performance_history' in import_data else 0
                }
                
            elif os.path.isdir(import_path) and os.path.exists(f"{import_path}/performance_metrics.csv"):
                # Import from CSV directory
                # Read performance metrics
                metrics_df = pd.read_csv(f"{import_path}/performance_metrics.csv")
                
                # Read strategy registry if available
                registry_data = None
                if os.path.exists(f"{import_path}/strategy_registry.csv"):
                    registry_df = pd.read_csv(f"{import_path}/strategy_registry.csv")
                    registry_data = registry_df.to_dict('records')
                
                # Read performance history if available
                history_data = None
                if os.path.exists(f"{import_path}/performance_history.csv"):
                    history_df = pd.read_csv(f"{import_path}/performance_history.csv")
                    history_data = history_df.to_dict('records')
                
                # Update data
                with self._strategy_lock:
                    # Update strategy registry
                    if registry_data:
                        for record in registry_data:
                            asset = record['asset']
                            strategy = record['strategy']
                            
                            if asset in self.strategy_registry:
                                self.strategy_registry[asset].add(strategy)
                            else:
                                self.strategy_registry[asset] = {strategy}
                    
                    # Update performance metrics
                    metrics_imported = 0
                    for _, row in metrics_df.iterrows():
                        asset = row['asset']
                        strategy = row['strategy']
                        key = f"{asset}_{strategy}"
                        
                        # Create metrics object
                        metrics_dict = row.to_dict()
                        # Remove asset and strategy from dict since they're not part of the metrics
                        metrics_dict.pop('asset', None)
                        metrics_dict.pop('strategy', None)
                        
                        self.strategy_performance[key] = StrategyPerformanceMetrics(**metrics_dict)
                        metrics_imported += 1
                    
                    # Update performance history
                    history_imported = 0
                    if history_data:
                        existing_timestamps = {entry['timestamp'] for entry in self.performance_history}
                        new_history = [
                            entry for entry in history_data
                            if entry['timestamp'] not in existing_timestamps
                        ]
                        
                        self.performance_history.extend(new_history)
                        history_imported = len(new_history)
                        
                        # Keep history limited
                        if len(self.performance_history) > self.config.get('max_performance_history_size'):
                            excess_count = len(self.performance_history) - self.config.get('max_performance_history_size')
                            self.performance_history = self.performance_history[excess_count:]

                return {
                    'success': True,
                    'format': 'csv',
                    'assets_imported': len(registry_data) if registry_data else 0,
                    'metrics_imported': metrics_imported,
                    'history_imported': history_imported
                }
                
            else:
                return {'success': False, 'error': f"Unsupported import format or missing files in directory: {import_path}"}
            
        except Exception as e:
            self.logger.error(f"Error importing strategy data: {str(e)}")
            return {'success': False, 'error': str(e)}
