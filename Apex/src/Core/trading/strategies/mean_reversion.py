import numpy as np
import logging
import asyncio
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor

# Core System Imports
from src.Core.trading.execution.order_execution import StealthOrderExecutor
from src.Core.trading.risk.risk_management import DynamicRiskAssessor
from src.Core.trading.hft.liquidity_manager import LiquidityOptimizer, DarkPoolRouter
from src.Core.data.realtime.market_data import MultiFeedDataStream
from src.Core.data.order_book_analyzer import OrderBookMicrostructureAnalyzer
from src.ai.analysis.market_regime_classifier import MarketRegimeClassifier
from src.ai.ensembles.meta_trader import MeanReversionEnsemble
from src.Core.trading.logging.decision_logger import ExecutionAuditLogger
from utils.logging.structured_logger import StructuredLogger
from utils.helpers.error_handler import CriticalErrorHandler, ExecutionErrorHandler
from utils.analytics.monte_carlo_simulator import WorstCaseSimulator
from utils.helpers.stealth_api import OrderCamouflage, LatencyRandomizer
from src.ai.analysis.institutional_clusters import MarketMakerDetector
from src.ai.reinforcement.q_learning.agent import RLTradingAgent
from src.Core.trading.strategies.strategy_selector import StrategySelector
from src.Core.trading.execution.slippage_calculator import SlippageAnalyzer
from src.Core.trading.execution.market_impact import MarketImpactEstimator

class QuantumMeanReversionEngine:
    """
    AI-driven mean reversion strategy with institutional-grade execution optimization
    
    This class serves as the core mean reversion strategy implementation for Apex,
    providing signal generation, risk assessment, and execution optimization for
    mean reversion trading across multiple asset classes.
    
    Attributes:
        asset_universe (list): List of assets to monitor for mean reversion opportunities
        data_stream (MultiFeedDataStream): Real-time market data provider
        order_executor (StealthOrderExecutor): Low-latency trade execution engine
        risk_assessor (DynamicRiskAssessor): Real-time risk management system
        liquidity_optimizer (LiquidityOptimizer): Order sizing and execution optimization
        dark_pool_router (DarkPoolRouter): Alternative venue execution router
        regime_classifier (MarketRegimeClassifier): Market condition analyzer
        ensemble_model (MeanReversionEnsemble): AI signal generation ensemble
        max_position_size_pct (float): Maximum position size as percentage of capital
        max_concurrent_assets (int): Maximum number of assets to trade simultaneously
    """
    
    # Static configuration
    MAX_BATCH_SIZE = 128
    SIGNAL_UPDATE_INTERVAL_MS = 250
    MARKET_REGIME_CACHE_SECONDS = 60
    LIQUIDITY_CACHE_SECONDS = 5
    
    def __init__(self, asset_universe: list, config: Dict = None):
        """
        Initialize the mean reversion engine with the specified asset universe
        
        Args:
            asset_universe: List of assets to monitor for mean reversion opportunities
            config: Optional configuration dictionary to override defaults
        """
        self.asset_universe = asset_universe
        self.config = config or {}
        
        # Initialize components and state
        self._init_components()
        self._init_state()
        self._init_caches()
        
        # Set configuration parameters
        self.max_position_size_pct = self.config.get('max_position_size_pct', 0.05)
        self.max_concurrent_assets = self.config.get('max_concurrent_assets', 15)
        self.signal_threshold = self.config.get('signal_threshold', 0.65)
        self.mobile_mode = self.config.get('mobile_mode', False)
        
        # Register with strategy selector
        self._register_with_strategy_selector()
        
        # Initialize metrics
        self.metrics = {
            'signals_generated': 0,
            'trades_executed': 0,
            'avg_execution_latency_ms': 0,
            'win_rate': 0.0,
            'avg_slippage_bps': 0.0
        }
        
        # Start background tasks
        self._start_background_tasks()

    def _init_components(self):
        """Initialize integrated system components"""
        # Core Modules
        self.data_stream = MultiFeedDataStream()
        self.order_executor = StealthOrderExecutor()
        self.risk_assessor = DynamicRiskAssessor()
        self.liquidity_optimizer = LiquidityOptimizer()
        self.dark_pool_router = DarkPoolRouter()
        self.regime_classifier = MarketRegimeClassifier()
        self.audit_logger = ExecutionAuditLogger()
        self.market_maker_detector = MarketMakerDetector()
        self.slippage_analyzer = SlippageAnalyzer()
        self.impact_estimator = MarketImpactEstimator()
        
        # AI/ML Components
        self.ensemble_model = MeanReversionEnsemble()
        self.reinforcement_learner = self._init_reinforcement_learner()
        
        # Utilities
        self.logger = StructuredLogger("quantum_mean_reversion")
        self.error_handler = CriticalErrorHandler()
        self.execution_error_handler = ExecutionErrorHandler()
        self.stealth = OrderCamouflage()
        self.latency_randomizer = LatencyRandomizer()
        
        # Thread Pool
        self._executor = ThreadPoolExecutor(max_workers=min(32, len(self.asset_universe)))

    def _init_reinforcement_learner(self):
        """Initialize RL component for strategy self-improvement"""
        if self.mobile_mode:
            self.logger.info("Mobile mode enabled: Using lightweight RL model")
            return RLTradingAgent(strategy_type='mean_reversion', lightweight=True)
        return RLTradingAgent(strategy_type='mean_reversion')

    def _init_state(self):
        """Initialize strategy state variables"""
        self.active_signals = {}
        self.pending_executions = {}
        self.active_trades = {}
        self.signal_history = {}
        self.market_regimes = {}
        self.trade_performance = {}
        self.engine_state = {"status": "initialized", "last_update": time.time()}

    def _init_caches(self):
        """Initialize in-memory caches for performance optimization"""
        self.regime_cache = {}
        self.liquidity_cache = {}
        self.market_maker_cache = {}
        self.order_book_cache = {}

    def _register_with_strategy_selector(self):
        """Register this strategy with the strategy selection system"""
        try:
            strategy_selector = StrategySelector()
            strategy_selector.register_strategy(
                name="quantum_mean_reversion",
                strategy=self,
                description="AI-driven mean reversion strategy with stealth execution",
                category="statistical_arbitrage",
                risk_profile="medium"
            )
        except Exception as e:
            self.logger.warning(f"Failed to register with strategy selector: {str(e)}")

    def _start_background_tasks(self):
        """Start background tasks for data processing and cache updates"""
        asyncio.create_task(self._update_market_regime_cache())
        asyncio.create_task(self._update_liquidity_cache())
        asyncio.create_task(self._update_model_periodically())

    async def _update_market_regime_cache(self):
        """Background task to update market regime cache"""
        while True:
            try:
                self.regime_cache = {
                    asset: await self.regime_classifier.get_current_regime(asset)
                    for asset in self.asset_universe
                }
                self.engine_state["last_regime_update"] = time.time()
            except Exception as e:
                self.logger.error(f"Error updating market regime cache: {str(e)}")
            
            await asyncio.sleep(self.MARKET_REGIME_CACHE_SECONDS)

    async def _update_liquidity_cache(self):
        """Background task to update liquidity cache"""
        while True:
            try:
                self.liquidity_cache = {
                    asset: await self.liquidity_optimizer.get_liquidity_snapshot(asset)
                    for asset in self.asset_universe
                }
                self.engine_state["last_liquidity_update"] = time.time()
            except Exception as e:
                self.logger.error(f"Error updating liquidity cache: {str(e)}")
            
            await asyncio.sleep(self.LIQUIDITY_CACHE_SECONDS)

    async def _update_model_periodically(self):
        """Background task to update AI models periodically"""
        while True:
            try:
                if self.signal_history and len(self.trade_performance) > 10:
                    await self.ensemble_model.update_model(self.signal_history)
                    await self.reinforcement_learner.process_batch_update()
                    self.engine_state["last_model_update"] = time.time()
            except Exception as e:
                self.logger.error(f"Error updating models: {str(e)}")
            
            # Update less frequently in mobile mode
            await asyncio.sleep(3600 if self.mobile_mode else 900)

    async def compute_signals(self) -> Dict[str, Dict]:
        """
        Compute mean reversion signals across asset universe
        
        Returns:
            Dict[str, Dict]: Dictionary of assets with valid trade signals
        """
        start_time = time.time()
        try:
            # Get market data for all assets
            market_data = await self.data_stream.get_multi_asset_feed(self.asset_universe)
            
            # Validate data integrity
            validated_data = self._validate_and_preprocess(market_data)
            
            # Process assets in parallel
            batch_size = self.MAX_BATCH_SIZE
            signal_batches = []
            
            # Process assets in batches for better resource management
            for i in range(0, len(validated_data), batch_size):
                batch = {k: validated_data[k] for k in list(validated_data.keys())[i:i+batch_size]}
                batch_signals = await self._process_asset_batch(batch)
                signal_batches.extend(batch_signals)
            
            # Combine signals and filter by strength
            signals = {s['asset']: s for s in signal_batches if s and s['strength'] > self.signal_threshold}
            
            # Limit number of concurrent trades
            signals = self._limit_concurrent_trades(signals)
            
            # Update metrics
            self.metrics['signals_generated'] += len(signals)
            
            # Log signal generation performance
            elapsed_ms = (time.time() - start_time) * 1000
            self.logger.debug(f"Signal computation complete in {elapsed_ms:.2f}ms for {len(signals)} valid signals")
            
            return signals
        
        except Exception as e:
            self.error_handler.log_and_alert(e, "Error in compute_signals")
            return {}

    async def _process_asset_batch(self, batch_data: Dict) -> List[Dict]:
        """Process a batch of assets in parallel"""
        # Create tasks for each asset
        tasks = [
            self._process_asset(asset, data)
            for asset, data in batch_data.items()
        ]
        
        # Run tasks concurrently
        return await asyncio.gather(*tasks)

    async def _process_asset(self, asset: str, data: Dict) -> Optional[Dict]:
        """
        Process mean reversion logic for single asset
        
        Args:
            asset: Asset symbol
            data: Market data for the asset
            
        Returns:
            Optional[Dict]: Signal dictionary if valid, None otherwise
        """
        try:
            # Apply jitter to prevent pattern detection
            if not self.mobile_mode:
                await self.latency_randomizer.apply_jitter()
            
            # Skip if asset is already in active trades
            if asset in self.active_trades:
                return None
            
            # Check if strategy is applicable in current market regime
            if not await self._is_strategy_applicable(asset, data):
                return None
            
            # Generate signal using ensemble model
            signal_strength = await self.ensemble_model.predict(data)
            
            # Validate liquidity for execution
            if not await self._validate_liquidity(asset, signal_strength):
                return None
            
            # Assess risk profile
            risk_profile = await self._assess_risk(asset, data, signal_strength)
            if not risk_profile['approved']:
                return None
            
            # Build final signal package
            signal = self._build_signal_package(asset, data, signal_strength, risk_profile)
            
            # Store in signal history
            self.signal_history[asset] = signal
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Asset {asset} processing failed: {str(e)}")
            return None

    async def _is_strategy_applicable(self, asset: str, data: Dict) -> bool:
        """
        Check if market regime is suitable for mean reversion strategy
        
        Args:
            asset: Asset symbol
            data: Market data for the asset
            
        Returns:
            bool: True if strategy is applicable, False otherwise
        """
        # Get market regime from cache if available
        if asset in self.regime_cache:
            regime = self.regime_cache[asset]
        else:
            regime = await self.regime_classifier.get_current_regime(asset)
        
        # Skip if regime is unfavorable
        unsuitable_regimes = ['strong_trend', 'hyper_volatile', 'news_driven']
        if regime in unsuitable_regimes:
            self.logger.debug(f"Skipping {asset} - unfavorable regime: {regime}")
            return False
        
        # Check for market maker activity
        if await self._detect_market_maker_presence(asset):
            self.logger.debug(f"Skipping {asset} - market maker dominance")
            return False
        
        return True

    async def _detect_market_maker_presence(self, asset: str) -> bool:
        """
        Check for institutional market maker patterns
        
        Args:
            asset: Asset symbol
            
        Returns:
            bool: True if market makers detected, False otherwise
        """
        # Check cache first
        cache_key = f"{asset}_mm"
        if cache_key in self.market_maker_cache:
            cache_time, result = self.market_maker_cache[cache_key]
            if time.time() - cache_time < 60:  # Cache for 1 minute
                return result
        
        # Detect market maker presence
        result = await self.market_maker_detector.detect(asset)
        
        # Update cache
        self.market_maker_cache[cache_key] = (time.time(), result)
        
        return result

    async def _validate_liquidity(self, asset: str, signal_strength: float) -> bool:
        """
        Validate if there's sufficient liquidity for safe execution
        
        Args:
            asset: Asset symbol
            signal_strength: Signal strength value
            
        Returns:
            bool: True if liquidity is sufficient, False otherwise
        """
        # Get liquidity profile from cache if available
        if asset in self.liquidity_cache:
            liquidity_profile = self.liquidity_cache[asset]
        else:
            liquidity_profile = await self.liquidity_optimizer.get_liquidity_snapshot(asset)
        
        # Calculate required liquidity based on signal strength
        min_liquidity = self._calculate_required_liquidity(signal_strength)
        
        # Check if liquidity is sufficient
        if liquidity_profile['effective'] < min_liquidity:
            self.logger.debug(f"Insufficient liquidity for {asset}: {liquidity_profile['effective']} < {min_liquidity}")
            return False
        
        return True

    def _calculate_required_liquidity(self, signal_strength: float) -> float:
        """
        Calculate required liquidity based on signal strength
        
        Args:
            signal_strength: Signal strength value
            
        Returns:
            float: Minimum required liquidity
        """
        # Base liquidity requirement
        base = 100000  # Minimum liquidity threshold
        
        # Scale requirement based on signal strength (stronger signals require more liquidity)
        return base * (1 + abs(signal_strength) ** 2)

    async def _assess_risk(self, asset: str, data: Dict, signal_strength: float) -> Dict:
        """
        Perform comprehensive risk assessment
        
        Args:
            asset: Asset symbol
            data: Market data for the asset
            signal_strength: Signal strength value
            
        Returns:
            Dict: Risk profile with approval status and trade parameters
        """
        # Use lightweight risk assessment in mobile mode
        if self.mobile_mode:
            return await self._assess_risk_lightweight(asset, data, signal_strength)
        
        # Perform Monte Carlo simulation for worst-case scenario
        worst_case = await WorstCaseSimulator().simulate(
            data, 
            iterations=1000, 
            confidence_level=0.99
        )
        
        # Calculate position size based on risk tolerance
        position_size = self._calculate_position_size(data, worst_case)
        
        # Estimate market impact
        impact = await self.impact_estimator.estimate_impact(asset, position_size)
        
        # Get fresh liquidity profile
        liquidity_profile = await self.liquidity_optimizer.get_liquidity_snapshot(asset)
        
        # Complete risk assessment
        return await self.risk_assessor.evaluate_mean_reversion_trade({
            'asset': asset,
            'signal_strength': signal_strength,
            'worst_case_loss': worst_case['value_at_risk'],
            'position_size': position_size,
            'liquidity_profile': liquidity_profile,
            'market_impact': impact,
            'volatility': data.get('volatility', 0),
            'spread': data.get('spread', 0)
        })

    async def _assess_risk_lightweight(self, asset: str, data: Dict, signal_strength: float) -> Dict:
        """
        Lightweight risk assessment for mobile devices
        
        Args:
            asset: Asset symbol
            data: Market data for the asset
            signal_strength: Signal strength value
            
        Returns:
            Dict: Risk profile with approval status and trade parameters
        """
        # Use historical volatility as proxy for risk
        volatility = data.get('volatility', 0.02)
        
        # Calculate simplified position size
        position_size = 100000 * min(1.0, signal_strength * 2) * (1 - volatility * 5)
        position_size = max(0, position_size)  # Prevent negative position sizes
        
        # Calculate stop loss and take profit levels
        stop_loss = data['current_price'] * (1 - volatility * 2)
        take_profit = data['current_price'] * (1 + volatility * 3)
        
        return {
            'approved': True,
            'approved_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

    def _calculate_position_size(self, data: Dict, worst_case: Dict) -> float:
        """
        Calculate optimal position size using dynamic volatility adjustment
        
        Args:
            data: Market data for the asset
            worst_case: Worst-case scenario simulation results
            
        Returns:
            float: Calculated position size
        """
        # Risk-adjusted position sizing
        max_risk = 0.02  # Maximum 2% risk per trade
        
        # Calculate risk-adjusted size
        risk_adjusted_size = (max_risk / worst_case['value_at_risk']) * 1000000
        
        # Limit position size based on liquidity
        liquidity_limit = data['available_liquidity'] * 0.1
        
        # Apply additional constraints
        max_position = 1000000 * self.max_position_size_pct
        
        # Return the minimum of the constraints
        return min(risk_adjusted_size, liquidity_limit, max_position)

    def _build_signal_package(self, asset: str, data: Dict, 
                            signal_strength: float, risk_profile: Dict) -> Dict:
        """
        Build final trade signal package
        
        Args:
            asset: Asset symbol
            data: Market data for the asset
            signal_strength: Signal strength value
            risk_profile: Risk assessment results
            
        Returns:
            Dict: Complete signal package with execution parameters
        """
        # Determine trade direction
        direction = 'BUY' if signal_strength > 0 else 'SELL'
        
        # Calculate execution parameters
        urgency = self._determine_execution_urgency(signal_strength)
        stealth_level = self.stealth.calculate_camouflage_level(risk_profile['approved_size'])
        dark_pool_allocation = self._calculate_dark_pool_allocation(asset)
        
        # Assemble signal package
        signal = {
            'asset': asset,
            'signal': direction,
            'strength': abs(signal_strength),
            'entry_price': data['current_price'],
            'position_size': risk_profile['approved_size'],
            'stop_loss': risk_profile['stop_loss'],
            'take_profit': risk_profile['take_profit'],
            'timestamp': time.time(),
            'execution_parameters': {
                'urgency': urgency,
                'stealth_level': stealth_level,
                'dark_pool_allocation': dark_pool_allocation,
                'order_type': 'LIMIT' if urgency == 'normal' else 'MARKET',
                'time_in_force': 'GTC',
                'reduce_only': False
            },
            'risk_metrics': {
                'expected_return': (risk_profile['take_profit'] - data['current_price']) / data['current_price'],
                'stop_loss_pct': (data['current_price'] - risk_profile['stop_loss']) / data['current_price'],
                'risk_reward_ratio': abs((risk_profile['take_profit'] - data['current_price']) / 
                                        (data['current_price'] - risk_profile['stop_loss']))
            }
        }
        
        return signal

    def _determine_execution_urgency(self, signal_strength: float) -> str:
        """
        Classify execution urgency based on signal strength
        
        Args:
            signal_strength: Signal strength value
            
        Returns:
            str: Urgency classification
        """
        if abs(signal_strength) > 0.8:
            return 'immediate'
        elif abs(signal_strength) > 0.5:
            return 'high_priority'
        else:
            return 'normal'

    def _calculate_dark_pool_allocation(self, asset: str) -> float:
        """
        Determine optimal dark pool allocation
        
        Args:
            asset: Asset symbol
            
        Returns:
            float: Dark pool allocation percentage (0-1)
        """
        # Base allocation
        base_allocation = 0.3  # 30% to dark pools by default
        
        # Adjust based on liquidity profile
        liquidity_quality = self.liquidity_optimizer.get_liquidity_quality(asset)
        
        # Increase dark pool allocation for less transparent markets
        transparency_adjustment = (1 - liquidity_quality.get('transparency', 0.5)) * 0.4
        
        # Adjust based on asset type
        asset_type = self._get_asset_type(asset)
        if asset_type == 'large_cap':
            type_adjustment = 0.1
        elif asset_type == 'mid_cap':
            type_adjustment = 0.2
        else:  # small_cap
            type_adjustment = 0.0  # Small caps often have limited dark pool liquidity
        
        # Calculate final allocation
        allocation = base_allocation + transparency_adjustment + type_adjustment
        
        # Ensure allocation is within bounds
        return max(0.0, min(0.8, allocation))

    def _get_asset_type(self, asset: str) -> str:
        """
        Determine asset type based on market cap
        
        Args:
            asset: Asset symbol
            
        Returns:
            str: Asset type classification
        """
        # Get market cap from data stream
        market_cap = self.data_stream.get_asset_info(asset).get('market_cap', 0)
        
        # Classify asset
        if market_cap > 10000000000:  # $10B+
            return 'large_cap'
        elif market_cap > 2000000000:  # $2B+
            return 'mid_cap'
        else:
            return 'small_cap'

    def _limit_concurrent_trades(self, signals: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Limit the number of concurrent trades
        
        Args:
            signals: Dictionary of signal packages
            
        Returns:
            Dict: Filtered signals
        """
        # Return all signals if under limit
        if len(signals) <= self.max_concurrent_assets:
            return signals
        
        # Sort signals by strength in descending order
        sorted_signals = sorted(
            signals.items(), 
            key=lambda x: x[1]['strength'], 
            reverse=True
        )
        
        # Take top signals up to max_concurrent_assets
        top_signals = dict(sorted_signals[:self.max_concurrent_assets])
        
        # Log the filtering
        filtered_count = len(signals) - len(top_signals)
        self.logger.info(f"Limited concurrent trades: filtered {filtered_count} signals")
        
        return top_signals

    async def execute_strategy(self, signals: Dict[str, Dict]):
        """
        Execute mean reversion strategy across generated signals
        
        Args:
            signals: Dictionary of signal packages
        """
        start_time = time.time()
        execution_tasks = []
        
        # Skip if no signals
        if not signals:
            return
        
        # Create execution tasks for each signal
        for asset, signal in signals.items():
            # Skip if position size is too small
            if signal['position_size'] <= 0:
                continue
            
            # Create execution task
            execution_tasks.append(
                self._execute_single_trade(asset, signal)
            )
        
        # Execute all trades concurrently
        await asyncio.gather(*execution_tasks)
        
        # Update AI models
        await self._update_learning_model(signals)
        
        # Calculate and log execution time
        execution_time_ms = (time.time() - start_time) * 1000
        self.metrics['avg_execution_latency_ms'] = (
            (self.metrics['avg_execution_latency_ms'] * self.metrics['trades_executed'] + execution_time_ms) / 
            (self.metrics['trades_executed'] + len(execution_tasks))
        )
        self.metrics['trades_executed'] += len(execution_tasks)
        
        self.logger.info(f"Executed {len(execution_tasks)} trades in {execution_time_ms:.2f}ms")

    async def _execute_single_trade(self, asset: str, signal: Dict):
        """
        Execute a single trade with stealth optimization
        
        Args:
            asset: Asset symbol
            signal: Signal package
        """
        try:
            # Apply order camouflage
            camouflaged_order = self.stealth.camouflage_order(signal)
            
            # Split order between lit and dark pools
            lit_order, dark_order = self._split_order(camouflaged_order)
            
            # Execute lit portion
            lit_result = None
            if lit_order['position_size'] > 0:
                lit_result = await self.order_executor.execute_order(lit_order)
                
            # Execute dark pool portion
            dark_result = None
            if dark_order['position_size'] > 0:
                dark_result = await self.dark_pool_router.execute_dark_pool_order(dark_order)
                
            # Track active trade
            self.active_trades[asset] = {
                'entry_time': time.time(),
                'signal': signal,
                'lit_execution': lit_result,
                'dark_execution': dark_result
            }
            
            # Log execution
            await self._log_trade_execution(asset, signal, lit_order, dark_order)
            
        except Exception as e:
            self.execution_error_handler.handle_execution_error(e, asset)

    def _split_order(self, order: Dict) -> Tuple[Dict, Dict]:
        """
        Split order between lit and dark venues
        
        Args:
            order: Order details
            
        Returns:
            Tuple[Dict, Dict]: Lit and dark portions of the order
        """
        # Get dark pool allocation
        dark_pool_allocation = order['execution_parameters']['dark_pool_allocation']
        
        # Calculate size for each venue
        dark_size = order['position_size'] * dark_pool_allocation
        lit_size = order['position_size'] - dark_size
        
        # Create venue-specific orders
        lit_order = {**order, 'position_size': lit_size, 'venue': 'lit'}
        dark_order = {**order, 'position_size': dark_size, 'venue': 'dark'}
        
        return lit_order, dark_order

    async def _log_trade_execution(self, asset: str, signal: Dict, 
                                 lit_order: Dict, dark_order: Dict):
        """
        Log trade details to multiple systems
        
        Args:
            asset: Asset symbol
            signal: Signal package
            lit_order: Lit market order details
            dark_order: Dark pool order details
        """
        # Calculate execution metrics
        slippage = await self._calculate_execution_slippage(asset)
        impact_cost = await self.liquidity_optimizer.calculate_impact_cost(asset)
        
        # Log to audit system
        self.audit_logger.log_trade({
            'asset': asset,
            'signal_strength': signal['strength'],
            'execution_details': {
                'lit': lit_order,
                'dark': dark_order
            },
            'performance_metrics': {
                'slippage': slippage,
                'impact_cost': impact_cost
            }
        })
        
        # Update reinforcement learning model
        self.reinforcement_learner.record_decision(signal)
        
        # Update metrics
        self.metrics['avg_slippage_bps'] = (
            (self.metrics['avg_slippage_bps'] * (self.metrics['trades_executed'] - 1) + slippage * 10000) / 
            self.metrics['trades_executed']
        )

    async def _update_learning_model(self, signals: Dict):
        """
        Update AI models with latest execution results
        
        Args:
            signals: Dictionary of signal packages
        """
        if not signals:
            return
            
        # Only update models if we have enough signals
        if len(signals) < 2:
            return
            
        # Extract signal features for model updates
        features = []
        for asset, signal in signals.items():
            features.append({
                'asset': asset,
                'signal_strength': signal['strength'],
                'entry_price': signal['entry_price'],
                'regime': self.regime_cache.get(asset, 'unknown'),
                'direction': 1 if signal['signal'] == 'BUY' else -1,
                'execution_time': time.time()
            })
            
        # Update reinforcement learning model
        if len(features) >= 5:
            await self.reinforcement_learner.batch_update(features)
            
        # Update ensemble model weights
        await self.ensemble_model.update_weights(features)
            
        # Mark models as updated
        self.engine_state["last_model_update"] = time.time()
        
    async def _validate_and_preprocess(self, market_data: Dict) -> Dict:
        """
        Validate and preprocess market data for signal generation
        
        Args:
            market_data: Raw market data from data stream
            
        Returns:
            Dict: Validated and preprocessed data
        """
        validated_data = {}
        
        for asset, data in market_data.items():
            # Skip assets with missing data
            if not data or 'current_price' not in data:
                self.logger.warning(f"Missing data for {asset}")
                continue
                
            # Check for data integrity
            if not self._validate_data_integrity(data):
                self.logger.warning(f"Data integrity check failed for {asset}")
                continue
                
            # Preprocess data
            try:
                # Add volatility if not present
                if 'volatility' not in data:
                    data['volatility'] = self._calculate_volatility(data)
                    
                # Add available liquidity if not present
                if 'available_liquidity' not in data:
                    data['available_liquidity'] = self.liquidity_optimizer.get_available_liquidity(asset)
                    
                # Add spread if not present
                if 'spread' not in data:
                    data['spread'] = data.get('ask', 0) - data.get('bid', 0)
                    
                # Add to validated data
                validated_data[asset] = data
                    
            except Exception as e:
                self.logger.error(f"Error preprocessing data for {asset}: {str(e)}")
                
        return validated_data

    def _validate_data_integrity(self, data: Dict) -> bool:
        """
        Validate market data integrity
        
        Args:
            data: Market data for an asset
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        # Check for required fields
        required_fields = ['current_price', 'timestamp']
        for field in required_fields:
            if field not in data:
                return False
                
        # Check if timestamp is recent
        if time.time() - data['timestamp'] > 5:  # Older than 5 seconds
            return False
            
        # Validate price
        if data['current_price'] <= 0:
            return False
            
        # Check for unrealistic price changes
        if 'previous_price' in data:
            pct_change = abs(data['current_price'] - data['previous_price']) / data['previous_price']
            if pct_change > 0.2:  # 20% price change
                return False
                
        return True

    def _calculate_volatility(self, data: Dict) -> float:
        """
        Calculate volatility from price data
        
        Args:
            data: Market data for an asset
            
        Returns:
            float: Volatility value
        """
        if 'price_history' in data and len(data['price_history']) > 1:
            prices = np.array(data['price_history'])
            returns = np.diff(prices) / prices[:-1]
            return np.std(returns)
        return 0.02  # Default volatility if no history available

    async def monitor_active_trades(self):
        """
        Monitor and manage active trades
        """
        # Skip if no active trades
        if not self.active_trades:
            return
            
        # Get current market data for active trades
        assets = list(self.active_trades.keys())
        current_data = await self.data_stream.get_multi_asset_feed(assets)
        
        # Process each active trade
        for asset, trade in list(self.active_trades.items()):
            # Skip if asset data is missing
            if asset not in current_data:
                continue
                
            # Check if trade needs to be closed
            if await self._should_close_trade(asset, trade, current_data[asset]):
                await self._close_trade(asset, trade, current_data[asset])
                
            # Update trade statistics
            await self._update_trade_stats(asset, trade, current_data[asset])

    async def _should_close_trade(self, asset: str, trade: Dict, current_data: Dict) -> bool:
        """
        Determine if a trade should be closed
        
        Args:
            asset: Asset symbol
            trade: Trade information
            current_data: Current market data
            
        Returns:
            bool: True if trade should be closed, False otherwise
        """
        # Get signal details
        signal = trade['signal']
        entry_price = signal['entry_price']
        current_price = current_data['current_price']
        
        # Check stop loss
        if signal['signal'] == 'BUY' and current_price <= signal['stop_loss']:
            self.logger.info(f"Closing {asset} trade - Stop loss triggered")
            return True
            
        if signal['signal'] == 'SELL' and current_price >= signal['stop_loss']:
            self.logger.info(f"Closing {asset} trade - Stop loss triggered")
            return True
            
        # Check take profit
        if signal['signal'] == 'BUY' and current_price >= signal['take_profit']:
            self.logger.info(f"Closing {asset} trade - Take profit reached")
            return True
            
        if signal['signal'] == 'SELL' and current_price <= signal['take_profit']:
            self.logger.info(f"Closing {asset} trade - Take profit reached")
            return True
            
        # Check for mean reversion completion
        if self._is_mean_reversion_complete(asset, trade, current_data):
            self.logger.info(f"Closing {asset} trade - Mean reversion complete")
            return True
            
        # Check for regime change
        if self._is_regime_unfavorable(asset):
            self.logger.info(f"Closing {asset} trade - Market regime unfavorable")
            return True
            
        # Check for time-based exit
        if time.time() - trade['entry_time'] > 86400:  # 24 hours
            self.logger.info(f"Closing {asset} trade - Time-based exit")
            return True
            
        return False

    def _is_mean_reversion_complete(self, asset: str, trade: Dict, current_data: Dict) -> bool:
        """
        Check if mean reversion is complete
        
        Args:
            asset: Asset symbol
            trade: Trade information
            current_data: Current market data
            
        Returns:
            bool: True if mean reversion is complete, False otherwise
        """
        # Get signal and price data
        signal = trade['signal']
        entry_price = signal['entry_price']
        current_price = current_data['current_price']
        
        # Calculate reversion percentage
        if 'price_mean' in current_data:
            mean_price = current_data['price_mean']
            
            # Calculate distance from mean
            entry_distance = abs(entry_price - mean_price) / mean_price
            current_distance = abs(current_price - mean_price) / mean_price
            
            # If current price is closer to mean than entry price
            if current_distance < entry_distance * 0.3:  # 70% reversion
                return True
        
        return False

    def _is_regime_unfavorable(self, asset: str) -> bool:
        """
        Check if market regime is unfavorable for mean reversion
        
        Args:
            asset: Asset symbol
            
        Returns:
            bool: True if regime is unfavorable, False otherwise
        """
        # Get current regime from cache
        regime = self.regime_cache.get(asset)
        if not regime:
            return False
            
        # Check if regime is unfavorable
        unfavorable_regimes = ['strong_trend', 'hyper_volatile', 'news_driven']
        return regime in unfavorable_regimes

    async def _close_trade(self, asset: str, trade: Dict, current_data: Dict):
        """
        Close a trade
        
        Args:
            asset: Asset symbol
            trade: Trade information
            current_data: Current market data
        """
        # Get signal details
        signal = trade['signal']
        entry_price = signal['entry_price']
        current_price = current_data['current_price']
        position_size = signal['position_size']
        
        # Calculate profit/loss
        if signal['signal'] == 'BUY':
            pnl = (current_price - entry_price) * position_size
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # SELL
            pnl = (entry_price - current_price) * position_size
            pnl_pct = (entry_price - current_price) / entry_price
            
        # Create close order
        close_order = {
            'asset': asset,
            'signal': 'SELL' if signal['signal'] == 'BUY' else 'BUY',
            'strength': signal['strength'],
            'position_size': position_size,
            'entry_price': current_price,
            'timestamp': time.time(),
            'execution_parameters': {
                'urgency': 'immediate',
                'stealth_level': 0,
                'dark_pool_allocation': 0,
                'order_type': 'MARKET',
                'time_in_force': 'IOC',
                'reduce_only': True
            }
        }
        
        # Execute close order
        lit_order, dark_order = self._split_order(close_order)
        
        # Close lit portion
        if lit_order['position_size'] > 0:
            await self.order_executor.execute_order(lit_order)
            
        # Close dark pool portion
        if dark_order['position_size'] > 0:
            await self.dark_pool_router.execute_dark_pool_order(dark_order)
            
        # Record trade performance
        self.trade_performance[asset] = {
            'entry_time': trade['entry_time'],
            'exit_time': time.time(),
            'entry_price': entry_price,
            'exit_price': current_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'position_size': position_size,
            'win': pnl > 0,
            'signal_strength': signal['strength']
        }
        
        # Update win rate metric
        self.metrics['win_rate'] = (
            (self.metrics['win_rate'] * (self.metrics['trades_executed'] - 1) + (1 if pnl > 0 else 0)) / 
            self.metrics['trades_executed']
        )
        
        # Remove from active trades
        del self.active_trades[asset]
        
        # Log trade closure
        self.logger.info(f"Closed {asset} trade - PnL: {pnl:.2f} ({pnl_pct:.2%})")

    async def _update_trade_stats(self, asset: str, trade: Dict, current_data: Dict):
        """
        Update trade statistics
        
        Args:
            asset: Asset symbol
            trade: Trade information
            current_data: Current market data
        """
        # Get signal details
        signal = trade['signal']
        entry_price = signal['entry_price']
        current_price = current_data['current_price']
        
        # Calculate unrealized profit/loss
        if signal['signal'] == 'BUY':
            unrealized_pnl = (current_price - entry_price) * signal['position_size']
            unrealized_pnl_pct = (current_price - entry_price) / entry_price
        else:  # SELL
            unrealized_pnl = (entry_price - current_price) * signal['position_size']
            unrealized_pnl_pct = (entry_price - current_price) / entry_price
            
        # Update trade information
        trade['current_price'] = current_price
        trade['unrealized_pnl'] = unrealized_pnl
        trade['unrealized_pnl_pct'] = unrealized_pnl_pct
        trade['duration'] = time.time() - trade['entry_time']

    async def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics for the mean reversion strategy
        
        Returns:
            Dict: Performance metrics
        """
        # Calculate performance metrics
        metrics = {**self.metrics}
        
        # Calculate additional metrics
        if self.trade_performance:
            # Calculate PnL metrics
            pnl_values = [trade['pnl'] for trade in self.trade_performance.values()]
            pnl_pct_values = [trade['pnl_pct'] for trade in self.trade_performance.values()]
            
            # Calculate win rate
            win_count = sum(1 for trade in self.trade_performance.values() if trade['win'])
            win_rate = win_count / len(self.trade_performance) if self.trade_performance else 0
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            if pnl_pct_values:
                mean_return = np.mean(pnl_pct_values)
                std_return = np.std(pnl_pct_values) if len(pnl_pct_values) > 1 else 1
                sharpe = mean_return / std_return if std_return > 0 else 0
            else:
                sharpe = 0
                
            # Add metrics
            metrics['total_pnl'] = sum(pnl_values)
            metrics['avg_pnl_pct'] = np.mean(pnl_pct_values) if pnl_pct_values else 0
            metrics['win_rate'] = win_rate
            metrics['sharpe_ratio'] = sharpe
            metrics['trade_count'] = len(self.trade_performance)
            
        return metrics

    async def export_performance_data(self) -> Dict:
        """
        Export performance data for visualization
        
        Returns:
            Dict: Performance data
        """
        # Get performance metrics
        metrics = await self.get_performance_metrics()
        
        # Get trade history
        trade_history = list(self.trade_performance.values())
        
        # Get current active trades
        active_trades = []
        for asset, trade in self.active_trades.items():
            active_trades.append({
                'asset': asset,
                'entry_time': trade['entry_time'],
                'entry_price': trade['signal']['entry_price'],
                'current_price': trade.get('current_price', trade['signal']['entry_price']),
                'unrealized_pnl': trade.get('unrealized_pnl', 0),
                'unrealized_pnl_pct': trade.get('unrealized_pnl_pct', 0),
                'signal_strength': trade['signal']['strength'],
                'duration': time.time() - trade['entry_time']
            })
        
        # Prepare performance data
        performance_data = {
            'metrics': metrics,
            'trade_history': trade_history,
            'active_trades': active_trades,
            'asset_universe': self.asset_universe,
            'engine_state': self.engine_state,
            'market_regimes': self.regime_cache
        }
        
        return performance_data

    async def generate_mobile_dashboard_data(self) -> Dict:
        """
        Generate optimized data for mobile dashboard
        
        Returns:
            Dict: Mobile-optimized dashboard data
        """
        # Get basic metrics
        metrics = await self.get_performance_metrics()
        
        # Calculate summary statistics
        total_trades = metrics.get('trades_executed', 0)
        win_rate = metrics.get('win_rate', 0) * 100
        
        # Get active trades (limited for mobile)
        active_trades = []
        for asset, trade in list(self.active_trades.items())[:5]:  # Limit to 5 trades for mobile
            active_trades.append({
                'asset': asset,
                'entry_time': trade['entry_time'],
                'unrealized_pnl_pct': trade.get('unrealized_pnl_pct', 0) * 100,
                'signal': trade['signal']['signal']
            })
        
        # Create mobile-optimized data package
        mobile_data = {
            'summary': {
                'total_trades': total_trades,
                'win_rate': f"{win_rate:.1f}%",
                'total_pnl': f"{metrics.get('total_pnl', 0):.2f}",
                'active_count': len(self.active_trades)
            },
            'active_trades': active_trades,
            'recent_closed': list(self.trade_performance.values())[-5:] if self.trade_performance else [],
            'last_update': time.time()
        }
        
        return mobile_data

    async def run_cycle(self):
        """
        Run a complete strategy cycle
        """
        try:
            # Update market regimes if cache is stale
            if (time.time() - self.engine_state.get("last_regime_update", 0) > 
                    self.MARKET_REGIME_CACHE_SECONDS):
                await self._update_market_regime_cache()
            
            # Update liquidity cache if stale
            if (time.time() - self.engine_state.get("last_liquidity_update", 0) > 
                    self.LIQUIDITY_CACHE_SECONDS):
                await self._update_liquidity_cache()
            
            # Monitor active trades
            await self.monitor_active_trades()
            
            # Generate new signals
            signals = await self.compute_signals()
            
            # Execute strategy if signals present
            if signals:
                await self.execute_strategy(signals)
            
            # Update engine state
            self.engine_state["status"] = "running"
            self.engine_state["last_cycle"] = time.time()
            
        except Exception as e:
            self.error_handler.log_and_alert(e, "Error in run_cycle")
            self.engine_state["status"] = "error"
            self.engine_state["last_error"] = str(e)
            self.engine_state["last_error_time"] = time.time()

    async def generate_institutional_report(self) -> Dict:
        """
        Generate comprehensive institutional-grade performance report
        
        Returns:
            Dict: Institutional report data
        """
        # Get base performance data
        performance_data = await self.export_performance_data()
        metrics = performance_data['metrics']
        
        # Calculate advanced metrics
        if self.trade_performance:
            trades = list(self.trade_performance.values())
            
            # Extract PnL values
            pnl_values = [trade['pnl'] for trade in trades]
            pnl_pct_values = [trade['pnl_pct'] for trade in trades]
            
            # Calculate drawdown
            cumulative_returns = np.cumsum(pnl_pct_values)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (peak - cumulative_returns)
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            # Calculate Sortino ratio (downside deviation only)
            negative_returns = [r for r in pnl_pct_values if r < 0]
            downside_deviation = np.std(negative_returns) if negative_returns else 1
            mean_return = np.mean(pnl_pct_values) if pnl_pct_values else 0
            sortino = mean_return / downside_deviation if downside_deviation > 0 else 0
            
            # Calculate Calmar ratio
            annualized_return = mean_return * 252  # Assuming daily returns and 252 trading days
            calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0
            
            # Add to metrics
            advanced_metrics = {
                'max_drawdown': max_drawdown,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'avg_win_size': np.mean([t['pnl'] for t in trades if t['win']]) if any(t['win'] for t in trades) else 0,
                'avg_loss_size': np.mean([t['pnl'] for t in trades if not t['win']]) if any(not t['win'] for t in trades) else 0,
                'profit_factor': (sum(t['pnl'] for t in trades if t['win']) / 
                                abs(sum(t['pnl'] for t in trades if not t['win']))) 
                                if sum(t['pnl'] for t in trades if not t['win']) != 0 else float('inf'),
                'expectancy': np.mean(pnl_values) if pnl_values else 0,
                'recovery_factor': (sum(pnl_values) / max_drawdown) if max_drawdown > 0 else float('inf')
            }
            metrics.update(advanced_metrics)
        
        # Generate trade analytics by signal strength
        signal_strength_buckets = {
            'very_strong': [0.8, 1.0],
            'strong': [0.65, 0.8],
            'moderate': [0.5, 0.65]
        }
        
        signal_analytics = {}
        for bucket, [min_val, max_val] in signal_strength_buckets.items():
            bucket_trades = [t for t in self.trade_performance.values() 
                            if min_val <= t['signal_strength'] < max_val]
            
            if bucket_trades:
                win_rate = sum(1 for t in bucket_trades if t['win']) / len(bucket_trades)
                avg_pnl = np.mean([t['pnl'] for t in bucket_trades])
                signal_analytics[bucket] = {
                    'count': len(bucket_trades),
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl
                }
        
        # Generate regime analytics
        regime_analytics = {}
        for regime in set(self.regime_cache.values()):
            regime_assets = [asset for asset, r in self.regime_cache.items() if r == regime]
            regime_trades = [t for asset, t in self.trade_performance.items() if asset in regime_assets]
            
            if regime_trades:
                win_rate = sum(1 for t in regime_trades if t['win']) / len(regime_trades)
                avg_pnl = np.mean([t['pnl'] for t in regime_trades])
                regime_analytics[regime] = {
                    'count': len(regime_trades),
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl
                }
        
        # Create institutional report
        institutional_report = {
            'summary_metrics': metrics,
            'signal_strength_analytics': signal_analytics,
            'regime_analytics': regime_analytics,
            'execution_quality': {
                'avg_slippage_bps': metrics.get('avg_slippage_bps', 0),
                'avg_execution_latency_ms': metrics.get('avg_execution_latency_ms', 0)
            },
            'model_performance': {
                'signal_accuracy': metrics.get('win_rate', 0),
                'prediction_confidence': np.mean([t['signal_strength'] for t in self.trade_performance.values()]) 
                                    if self.trade_performance else 0
            },
            'risk_metrics': {
                'max_drawdown': metrics.get('max_drawdown', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'sortino_ratio': metrics.get('sortino_ratio', 0),
                'calmar_ratio': metrics.get('calmar_ratio', 0)
            }
        }
        
        return institutional_report

    async def adapt_to_market_conditions(self):
        """
        Adapt strategy parameters to current market conditions
        """
        # Skip if not enough trade history
        if len(self.trade_performance) < 10:
            return
        
        # Calculate overall performance by regime
        regime_performance = {}
        for asset, trade in self.trade_performance.items():
            regime = self.regime_cache.get(asset, 'unknown')
            if regime not in regime_performance:
                regime_performance[regime] = {'pnl': 0, 'count': 0, 'win': 0}
            
            regime_performance[regime]['pnl'] += trade['pnl']
            regime_performance[regime]['count'] += 1
            regime_performance[regime]['win'] += 1 if trade['win'] else 0
        
        # Calculate win rate by regime
        for regime, perf in regime_performance.items():
            if perf['count'] > 0:
                perf['win_rate'] = perf['win'] / perf['count']
            else:
                perf['win_rate'] = 0
        
        # Adapt signal threshold based on regime performance
        for regime, perf in regime_performance.items():
            if perf['count'] >= 5:  # Only adapt if we have enough data
                if perf['win_rate'] < 0.4:  # Poor performance
                    # Increase signal threshold for this regime
                    self.regime_thresholds[regime] = min(0.85, self.regime_thresholds.get(regime, 0.65) + 0.05)
                    self.logger.info(f"Increased signal threshold for {regime} to {self.regime_thresholds[regime]}")
                elif perf['win_rate'] > 0.7:  # Good performance
                    # Decrease signal threshold for this regime
                    self.regime_thresholds[regime] = max(0.5, self.regime_thresholds.get(regime, 0.65) - 0.03)
                    self.logger.info(f"Decreased signal threshold for {regime} to {self.regime_thresholds[regime]}")
        
        # Adapt position sizing based on performance
        overall_win_rate = sum(1 for t in self.trade_performance.values() if t['win']) / len(self.trade_performance)
        if overall_win_rate > 0.6:
            # Increase max position size
            self.max_position_size_pct = min(0.08, self.max_position_size_pct * 1.05)
            self.logger.info(f"Increased max position size to {self.max_position_size_pct}")
        elif overall_win_rate < 0.4:
            # Decrease max position size
            self.max_position_size_pct = max(0.02, self.max_position_size_pct * 0.9)
            self.logger.info(f"Decreased max position size to {self.max_position_size_pct}")

        # Update engine state
        self.engine_state["last_adaptation"] = time.time()
        self.engine_state["adaptations"] = self.engine_state.get("adaptations", 0) + 1

    async def perform_health_check(self) -> Dict:
        """
        Perform system health check
        
        Returns:
            Dict: Health check results
        """
        health_status = {
            'status': 'healthy',
            'components': {},
            'errors': [],
            'warnings': []
        }
        
        # Check data feed
        try:
            data_feed_status = await self.data_stream.check_connection()
            health_status['components']['data_feed'] = {
                'status': 'online' if data_feed_status['connected'] else 'offline',
                'latency_ms': data_feed_status.get('latency_ms', 0)
            }
            
            if not data_feed_status['connected']:
                health_status['status'] = 'degraded'
                health_status['errors'].append('Data feed connection failed')
        except Exception as e:
            health_status['components']['data_feed'] = {'status': 'error'}
            health_status['status'] = 'degraded'
            health_status['errors'].append(f'Data feed check error: {str(e)}')
        
        # Check order execution system
        try:
            execution_status = await self.order_executor.check_connection()
            health_status['components']['order_execution'] = {
                'status': 'online' if execution_status['connected'] else 'offline',
                'latency_ms': execution_status.get('latency_ms', 0)
            }
            
            if not execution_status['connected']:
                health_status['status'] = 'degraded'
                health_status['errors'].append('Order execution system connection failed')
        except Exception as e:
            health_status['components']['order_execution'] = {'status': 'error'}
            health_status['status'] = 'degraded'
            health_status['errors'].append(f'Order execution check error: {str(e)}')
        
        # Check AI model status
        try:
            ensemble_status = self.ensemble_model.get_status()
            health_status['components']['ensemble_model'] = {
                'status': ensemble_status['status'],
                'version': ensemble_status.get('version', 'unknown'),
                'last_update': ensemble_status.get('last_update', 0)
            }
            
            if ensemble_status['status'] != 'ready':
                health_status['status'] = 'degraded'
                health_status['warnings'].append('Ensemble model not ready')
        except Exception as e:
            health_status['components']['ensemble_model'] = {'status': 'error'}
            health_status['status'] = 'degraded'
            health_status['errors'].append(f'Ensemble model check error: {str(e)}')
        
        # Check for stale data
        current_time = time.time()
        last_data_time = self.data_stream.get_last_update_time()
        data_age = current_time - last_data_time
        
        if data_age > self.MAX_DATA_AGE:
            health_status['status'] = 'degraded'
            health_status['warnings'].append(f'Data feed stale: Last update {data_age:.1f}s ago')
            
        # Check for stale model predictions
        last_prediction_time = self.ensemble_model.get_last_prediction_time() 
        prediction_age = current_time - last_prediction_time
        
        if prediction_age > self.MAX_PREDICTION_AGE:
            health_status['status'] = 'degraded'
            health_status['warnings'].append(f'Model predictions stale: Last update {prediction_age:.1f}s ago')
            # Check for regime detection service
            try:
                from .regime_detection import RegimeDetectionStrategy
                regime_detector = RegimeDetectionStrategy()
                market_data = self.data_stream.get_latest_market_data()
                current_regime = regime_detector.detect_regime(market_data)
                
                health_status['components']['regime_detection'] = {
                    'status': 'online',
                    'current_regime': current_regime,
                    'confidence': regime_detector.get_confidence() if hasattr(regime_detector, 'get_confidence') else 0.85
                }
                
                if current_regime == "NEUTRAL":
                    health_status['warnings'].append('Market regime is neutral - mean reversion signals may be unreliable')
            except Exception as e:
                health_status['components']['regime_detection'] = {'status': 'error'}
                health_status['warnings'].append(f'Regime detection error: {str(e)}')
            
            # Check liquidity conditions
            try:
                liquidity_metrics = self.liquidity_manager.get_current_metrics()
                health_status['components']['liquidity'] = {
                    'status': 'normal' if liquidity_metrics['status'] == 'sufficient' else 'warning',
                    'spread_factor': liquidity_metrics.get('spread_factor', 1.0),
                    'market_depth': liquidity_metrics.get('depth_score', 0.0)
                }
                
                if liquidity_metrics['status'] != 'sufficient':
                    health_status['warnings'].append('Low liquidity conditions detected - execution may be impacted')
            except Exception as e:
                health_status['components']['liquidity'] = {'status': 'unknown'}
                health_status['warnings'].append(f'Liquidity check error: {str(e)}')
            
            # Check risk management system
            try:
                risk_status = self.risk_manager.get_system_status()
                health_status['components']['risk_management'] = {
                    'status': risk_status['status'],
                    'current_exposure': risk_status.get('current_exposure', 0.0),
                    'max_drawdown': risk_status.get('max_drawdown', 0.0)
                }
                
                if risk_status['status'] == 'at_limit':
                    health_status['status'] = 'degraded'
                    health_status['warnings'].append('Risk limits reached - new positions restricted')
            except Exception as e:
                health_status['components']['risk_management'] = {'status': 'error'}
                health_status['errors'].append(f'Risk management check error: {str(e)}')
            
            # Check GPU acceleration status for AI models
            try:
                gpu_status = self.ensemble_model.check_gpu_availability()
                health_status['components']['gpu_acceleration'] = {
                    'status': 'available' if gpu_status['available'] else 'unavailable',
                    'device': gpu_status.get('device', 'cpu'),
                    'memory_available': gpu_status.get('memory_available_mb', 0)
                }
                
                if not gpu_status['available'] and self.config.get('require_gpu', False):
                    health_status['status'] = 'degraded'
                    health_status['warnings'].append('GPU acceleration unavailable - performance will be degraded')
            except Exception as e:
                health_status['components']['gpu_acceleration'] = {'status': 'error'}
                health_status['warnings'].append(f'GPU acceleration check error: {str(e)}')
            
            # Check multi-threading status
            try:
                threading_status = self.thread_pool.get_status()
                health_status['components']['multi_threading'] = {
                    'status': 'optimal' if threading_status['active_threads'] < threading_status['max_threads'] * 0.8 else 'near_capacity',
                    'active_threads': threading_status['active_threads'],
                    'max_threads': threading_status['max_threads'],
                    'queue_depth': threading_status.get('queue_depth', 0)
                }
                
                if threading_status['active_threads'] > threading_status['max_threads'] * 0.9:
                    health_status['warnings'].append('Thread pool near capacity - performance may be degraded')
            except Exception as e:
                health_status['components']['multi_threading'] = {'status': 'unknown'}
                health_status['warnings'].append(f'Thread pool check error: {str(e)}')
            
            # Check data feed integrity
            try:
                data_integrity = self.data_validator.verify_feed_integrity()
                health_status['components']['data_integrity'] = {
                    'status': data_integrity['status'],
                    'checksum_valid': data_integrity.get('checksum_valid', False),
                    'feed_latency': data_integrity.get('feed_latency_ms', 0)
                }
                
                if data_integrity['status'] != 'valid':
                    health_status['status'] = 'degraded'
                    health_status['errors'].append('Data feed integrity compromised - trading suspended')
            except Exception as e:
                health_status['components']['data_integrity'] = {'status': 'unknown'}
                health_status['warnings'].append(f'Data integrity check error: {str(e)}')
            
            # Check anti-front-running protection
            try:
                protection_status = self.execution_protector.get_status()
                health_status['components']['front_running_protection'] = {
                    'status': protection_status['status'],
                    'detection_active': protection_status.get('detection_active', False),
                    'suspicious_activity': protection_status.get('suspicious_activity_level', 0.0)
                }
                
                if protection_status.get('suspicious_activity_level', 0.0) > 0.7:
                    health_status['warnings'].append('High suspicious activity detected - enhanced protection active')
            except Exception as e:
                health_status['components']['front_running_protection'] = {'status': 'unknown'}
                health_status['warnings'].append(f'Front-running protection check error: {str(e)}')
            
            # Final health determination
            if len(health_status['errors']) > 0:
                health_status['status'] = 'degraded'
            elif len(health_status['warnings']) > 3:
                health_status['status'] = 'degraded'
            
            return health_status
    
    async def generate_mean_reversion_signals(self, market_data, timeframe='1h', lookback_period=20):
        """
        Generates mean reversion trading signals based on statistical deviations from mean.
        
        Args:
            market_data (dict): Market data containing price, volume, etc.
            timeframe (str): Timeframe for analysis (e.g., '1h', '4h', '1d')
            lookback_period (int): Period for calculating mean and standard deviation
            
        Returns:
            dict: Signal information with strength, direction, and confidence
        """
        # Initialize signal result
        signal = {
            'timestamp': time.time(),
            'symbol': market_data.get('symbol', 'unknown'),
            'timeframe': timeframe,
            'signal_type': 'mean_reversion',
            'direction': 'neutral',
            'strength': 0.0,
            'confidence': 0.0,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'indicators': {},
            'metadata': {}
        }
        
        # Extract price data
        prices = np.array(market_data.get('price', []))
        if len(prices) < lookback_period * 2:
            signal['metadata']['error'] = 'Insufficient price data'
            return signal
            
        # Calculate basic statistical indicators
        rolling_mean = np.mean(prices[-lookback_period:])
        rolling_std = np.std(prices[-lookback_period:])
        current_price = prices[-1]
        
        # Calculate z-score (how many standard deviations from mean)
        z_score = (current_price - rolling_mean) / rolling_std if rolling_std > 0 else 0
        
        # Calculate Bollinger Bands
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
        
        # Calculate RSI
        deltas = np.diff(prices[-lookback_period-1:])
        seed = deltas[:lookback_period+1]
        up = seed[seed >= 0].sum() / lookback_period
        down = -seed[seed < 0].sum() / lookback_period
        rs = up / down if down != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        # Store indicator values
        signal['indicators']['z_score'] = z_score
        signal['indicators']['bollinger_bands'] = {
            'upper': upper_band,
            'middle': rolling_mean,
            'lower': lower_band
        }
        signal['indicators']['rsi'] = rsi
        signal['indicators']['price_to_mean_ratio'] = current_price / rolling_mean
        
        # Check for regime using the regime detection strategy
        try:
            from .regime_detection import RegimeDetectionStrategy
            regime_detector = RegimeDetectionStrategy()
            current_regime = regime_detector.detect_regime(market_data)
            signal['metadata']['market_regime'] = current_regime
            
            # Adjust signal based on market regime
            if current_regime == "TRENDING":
                # Reduce mean reversion signal strength in trending markets
                signal['metadata']['regime_adjustment'] = 'reduced_strength_due_to_trend'
                regime_factor = 0.5  # Reduce signal strength by 50% in trending markets
            elif current_regime == "RANGING":
                # Enhance mean reversion signal strength in ranging markets
                signal['metadata']['regime_adjustment'] = 'enhanced_strength_due_to_range'
                regime_factor = 1.5  # Increase signal strength by 50% in ranging markets
            else:  # NEUTRAL
                regime_factor = 1.0  # No adjustment
                
        except Exception as e:
            self.logger.warning(f"Regime detection failed: {str(e)}")
            regime_factor = 1.0  # Default to no adjustment
            signal['metadata']['regime_detection_error'] = str(e)
        
        # Generate signal based on z-score and other indicators
        if z_score < -2.0 and rsi < 30:
            # Oversold condition - potential buy signal
            signal['direction'] = 'buy'
            signal['strength'] = min(abs(z_score) / 3.0, 1.0) * regime_factor
            signal['entry_price'] = current_price
            signal['stop_loss'] = current_price * 0.98  # 2% stop loss
            signal['take_profit'] = rolling_mean  # Target the mean for take profit
            
        elif z_score > 2.0 and rsi > 70:
            # Overbought condition - potential sell signal
            signal['direction'] = 'sell'
            signal['strength'] = min(abs(z_score) / 3.0, 1.0) * regime_factor
            signal['entry_price'] = current_price
            signal['stop_loss'] = current_price * 1.02  # 2% stop loss
            signal['take_profit'] = rolling_mean  # Target the mean for take profit
            
        # Calculate confidence based on multiple factors
        # 1. Strength of z-score deviation
        z_score_confidence = min(abs(z_score) / 4.0, 1.0)
        
        # 2. RSI extremity
        rsi_confidence = 0.0
        if rsi < 30:
            rsi_confidence = (30 - rsi) / 30.0
        elif rsi > 70:
            rsi_confidence = (rsi - 70) / 30.0
            
        # 3. Volume confirmation
        volume = np.array(market_data.get('volume', []))
        if len(volume) >= lookback_period:
            avg_volume = np.mean(volume[-lookback_period:-1])
            current_volume = volume[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            volume_confidence = min(volume_ratio / 2.0, 1.0)
        else:
            volume_confidence = 0.5  # Neutral if not enough volume data
            
        # 4. Price momentum
        momentum = prices[-1] - prices[-5] if len(prices) >= 5 else 0
        momentum_direction = np.sign(momentum)
        signal_direction = 1 if signal['direction'] == 'buy' else -1 if signal['direction'] == 'sell' else 0
        
        # Higher confidence if momentum is opposite to the mean reversion direction
        # (e.g., falling price for buy signal indicates stronger reversion potential)
        momentum_confidence = 0.7 if momentum_direction * signal_direction < 0 else 0.3
        
        # Combine confidence factors with weights
        signal['confidence'] = (
            0.4 * z_score_confidence +
            0.3 * rsi_confidence +
            0.2 * volume_confidence +
            0.1 * momentum_confidence
        ) * regime_factor
        
        # Add additional metadata
        signal['metadata']['lookback_period'] = lookback_period
        signal['metadata']['current_price'] = current_price
        signal['metadata']['rolling_mean'] = rolling_mean
        signal['metadata']['rolling_std'] = rolling_std
        signal['metadata']['regime_factor'] = regime_factor
        
        # Apply AI enhancement if available
        try:
            ai_enhancement = await self.ensemble_model.enhance_signal(signal, market_data)
            if ai_enhancement:
                signal['confidence'] = ai_enhancement.get('confidence', signal['confidence'])
                signal['strength'] = ai_enhancement.get('strength', signal['strength'])
                signal['metadata']['ai_enhanced'] = True
                signal['metadata']['ai_model_version'] = ai_enhancement.get('model_version', 'unknown')
        except Exception as e:
            self.logger.warning(f"AI signal enhancement failed: {str(e)}")
            signal['metadata']['ai_enhancement_error'] = str(e)
        
        return signal
    
    async def validate_signal(self, signal, market_data):
        """
        Validates a mean reversion signal against additional market conditions
        to filter out false positives and low-quality signals.
        
        Args:
            signal (dict): The generated trading signal
            market_data (dict): Current market data
            
        Returns:
            dict: Validated signal with additional validation metadata
        """
        # Clone the signal to avoid modifying the original
        validated_signal = signal.copy()
        validated_signal['metadata'] = signal.get('metadata', {}).copy()
        validated_signal['validation'] = {
            'passed': True,
            'checks': [],
            'score': 0.0
        }
        
        # Skip validation for neutral signals
        if signal['direction'] == 'neutral':
            validated_signal['validation']['passed'] = False
            validated_signal['validation']['checks'].append({
                'name': 'direction_check',
                'passed': False,
                'message': 'Neutral signals do not require validation'
            })
            return validated_signal
        
        # 1. Check signal strength threshold
        strength_check = {
            'name': 'strength_threshold',
            'passed': signal['strength'] >= self.config.get('min_signal_strength', 0.5),
            'message': f"Signal strength {signal['strength']:.2f} vs threshold {self.config.get('min_signal_strength', 0.5)}"
        }
        validated_signal['validation']['checks'].append(strength_check)
        
        # 2. Check signal confidence threshold
        confidence_check = {
            'name': 'confidence_threshold',
            'passed': signal['confidence'] >= self.config.get('min_signal_confidence', 0.6),
            'message': f"Signal confidence {signal['confidence']:.2f} vs threshold {self.config.get('min_signal_confidence', 0.6)}"
        }
        validated_signal['validation']['checks'].append(confidence_check)
        
        # 3. Check market regime compatibility
        regime = signal['metadata'].get('market_regime', 'UNKNOWN')
        is_buy = signal['direction'] == 'buy'
        is_sell = signal['direction'] == 'sell'
        
        regime_compatible = True
        regime_message = f"Market regime {regime} is compatible with mean reversion"
        
        if regime == "TRENDING":
            # In strongly trending markets, be more cautious with mean reversion
            trend_direction = market_data.get('trend_direction', 0)
            
            # If buying against a strong downtrend or selling against a strong uptrend,
            # this could be dangerous for mean reversion
            if (is_buy and trend_direction < -0.7) or (is_sell and trend_direction > 0.7):
                regime_compatible = False
                regime_message = f"Mean reversion {signal['direction']} signal against strong {regime} market"
        
        regime_check = {
            'name': 'regime_compatibility',
            'passed': regime_compatible,
            'message': regime_message
        }
        validated_signal['validation']['checks'].append(regime_check)
        
        # 4. Check for liquidity conditions
        try:
            liquidity_metrics = self.liquidity_manager.get_current_metrics()
            liquidity_sufficient = liquidity_metrics['status'] == 'sufficient'
            spread_acceptable = liquidity_metrics.get('spread_factor', 1.0) <= self.config.get('max_spread_factor', 1.5)
            
            liquidity_check = {
                'name': 'liquidity_check',
                'passed': liquidity_sufficient and spread_acceptable,
                'message': f"Liquidity: {liquidity_metrics['status']}, Spread factor: {liquidity_metrics.get('spread_factor', 1.0):.2f}"
            }
            validated_signal['validation']['checks'].append(liquidity_check)
        except Exception as e:
            liquidity_check = {
                'name': 'liquidity_check',
                'passed': False,
                'message': f"Liquidity check error: {str(e)}"
            }
            validated_signal['validation']['checks'].append(liquidity_check)
        
        # 5. Check for news events that might disrupt mean reversion
        try:
            news_impact = self.news_analyzer.get_current_impact()
            news_check = {
                'name': 'news_impact',
                'passed': news_impact['impact_level'] < self.config.get('max_news_impact', 0.7),
                'message': f"News impact: {news_impact['impact_level']:.2f}, Recent news: {news_impact.get('recent_headlines', 'None')}"
            }
            validated_signal['validation']['checks'].append(news_check)
        except Exception as e:
            news_check = {
                'name': 'news_impact',
                'passed': True,  # Default to passing if news analyzer is unavailable
                'message': f"News impact check error: {str(e)}"
            }
            validated_signal['validation']['checks'].append(news_check)
        
        # 6. Check for technical pattern conflicts
        try:
            patterns = self.pattern_detector.detect_patterns(market_data)
            conflicting_patterns = []
            
            # Define which patterns conflict with mean reversion signals
            if is_buy:
                conflicting_patterns = [p for p in patterns if p['type'] in ['head_and_shoulders', 'double_top', 'descending_triangle']]
            elif is_sell:
                conflicting_patterns = [p for p in patterns if p['type'] in ['inverse_head_and_shoulders', 'double_bottom', 'ascending_triangle']]
            
            pattern_check = {
                'name': 'pattern_conflict',
                'passed': len(conflicting_patterns) == 0,
                'message': f"Conflicting patterns: {[p['type'] for p in conflicting_patterns]}" if conflicting_patterns else "No conflicting patterns"
            }
            validated_signal['validation']['checks'].append(pattern_check)
        except Exception as e:
            pattern_check = {
                'name': 'pattern_conflict',
                'passed': True,  # Default to passing if pattern detector is unavailable
                'message': f"Pattern detection error: {str(e)}"
            }
            validated_signal['validation']['checks'].append(pattern_check)
        
        # 7. Risk management check
        try:
            risk_assessment = await self.risk_manager.assess_trade_risk(signal)
            risk_check = {
                'name': 'risk_assessment',
                'passed': risk_assessment['risk_level'] <= self.config.get('max_risk_level', 0.7),
                'message': f"Risk level: {risk_assessment['risk_level']:.2f}, Max drawdown: {risk_assessment.get('potential_drawdown', 0):.2f}%"
            }
            validated_signal['validation']['checks'].append(risk_check)
        except Exception as e:
            risk_check = {
                'name': 'risk_assessment',
                'passed': False,
                'message': f"Risk assessment error: {str(e)}"
            }
            validated_signal['validation']['checks'].append(risk_check)
        
        # Calculate overall validation score
        passed_checks = [check for check in validated_signal['validation']['checks'] if check['passed']]
        validation_score = len(passed_checks) / len(validated_signal['validation']['checks'])
        validated_signal['validation']['score'] = validation_score
        
        # Determine if signal passes validation
        # Critical checks that must pass: strength, confidence, regime, risk
        critical_checks = ['strength_threshold', 'confidence_threshold', 'regime_compatibility', 'risk_assessment']
        critical_passed = all(
            check['passed'] for check in validated_signal['validation']['checks'] 
            if check['name'] in critical_checks
        )
        
        # Signal passes if all critical checks pass and overall score is above threshold
        min_validation_score = self.config.get('min_validation_score', 0.7)
        validated_signal['validation']['passed'] = critical_passed and validation_score >= min_validation_score
        
        # Add validation metadata
        validated_signal['metadata']['validation_timestamp'] = time.time()
        validated_signal['metadata']['validation_score'] = validation_score
        validated_signal['metadata']['critical_checks_passed'] = critical_passed
        
        return validated_signal
    
    async def optimize_entry_exit(self, signal, market_data):
        """
        Optimizes entry and exit points for a validated mean reversion signal
        based on order book analysis and historical price patterns.
        
        Args:
            signal (dict): Validated trading signal
            market_data (dict): Current market data including order book
            
        Returns:
            dict: Signal with optimized entry/exit points
        """
        # Clone the signal to avoid modifying the original
        optimized_signal = signal.copy()
        optimized_signal['metadata'] = signal.get('metadata', {}).copy()
        optimized_signal['execution'] = {
            'optimized': True,
            'original_entry': signal.get('entry_price'),
            'original_stop_loss': signal.get('stop_loss'),
            'original_take_profit': signal.get('take_profit'),
            'entry_method': 'market',  # default, may change to limit
            'time_validity': 'day',    # default, may change based on optimization
            'execution_urgency': 'normal'  # default, may change based on signal strength
        }
        
        # Skip optimization for signals that didn't pass validation
        if not signal.get('validation', {}).get('passed', False):
            optimized_signal['execution']['optimized'] = False
            optimized_signal['execution']['skip_reason'] = 'Signal did not pass validation'
            return optimized_signal
        
        # Get order book data
        order_book = market_data.get('order_book', {})
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        current_price = market_data.get('price', signal.get('entry_price'))
        if not current_price:
            optimized_signal['execution']['optimized'] = False
            optimized_signal['execution']['skip_reason'] = 'Missing current price data'
            return optimized_signal
        
        # Analyze order book for liquidity clusters
        try:
            liquidity_analysis = self.order_book_analyzer.analyze_liquidity_clusters(order_book)
            
            # For buy signals, look for support levels below current price
            if signal['direction'] == 'buy':
                support_levels = [level for level in liquidity_analysis['support_levels'] if level < current_price]
                
                if support_levels:
                    # Find the strongest support level
                    strongest_support = max(support_levels, key=lambda x: liquidity_analysis['levels'][x]['strength'])
                    
                    # If strong support is found, place limit order slightly above it
                    if liquidity_analysis['levels'][strongest_support]['strength'] > 0.6:
                        optimized_entry = strongest_support * 1.001  # Slightly above support
                        optimized_signal['entry_price'] = optimized_entry
                        optimized_signal['execution']['entry_method'] = 'limit'
                        optimized_signal['execution']['entry_rationale'] = f"Limit order above strong support at {strongest_support}"
                
            # For sell signals, look for resistance levels above current price
            elif signal['direction'] == 'sell':
                resistance_levels = [level for level in liquidity_analysis['resistance_levels'] if level > current_price]
                
                if resistance_levels:
                    # Find the strongest resistance level
                    strongest_resistance = min(resistance_levels, key=lambda x: liquidity_analysis['levels'][x]['strength'])
                    
                    # If strong resistance is found, place limit order slightly below it
                    if liquidity_analysis['levels'][strongest_resistance]['strength'] > 0.6:
                        optimized_entry = strongest_resistance * 0.999  # Slightly below resistance
                        optimized_signal['entry_price'] = optimized_entry
                        optimized_signal['execution']['entry_method'] = 'limit'
                        optimized_signal['execution']['entry_rationale'] = f"Limit order below strong resistance at {strongest_resistance}"
        
        except Exception as e:
            self.logger.warning(f"Order book analysis failed: {str(e)}")
            optimized_signal['execution']['order_book_analysis_error'] = str(e)
        
        # Optimize stop loss based on volatility
        try:
            # Calculate Average True Range (ATR) for dynamic stop loss
            highs = market_data.get('high', [])
            lows = market_data.get('low', [])
            closes = market_data.get('price', [])
            
            if len(highs) > 14 and len(lows) > 14 and len(closes) > 14:
                # Calculate ATR
                tr_values = []
                for i in range(1, 14):
                    tr = max(
                        highs[-i] - lows[-i],
                        abs(highs[-i] - closes[-i-1]),
                        abs(lows[-i] - closes[-i-1])
                    )
                    tr_values.append(tr)
                
                atr = sum(tr_values) / len(tr_values)
                
                # Set stop loss based on ATR multiplier
                atr_multiplier = self.config.get('stop_loss_atr_multiplier', 2.0)
                
                if signal['direction'] == 'buy':
                    optimized_stop = current_price - (atr * atr_multiplier)
                    # Ensure stop loss is not too far from entry (max percentage loss)
                    max_stop_distance = current_price * (1 - self.config.get('max_stop_loss_percent', 0.03))
                    optimized_stop = max(optimized_stop, max_stop_distance)
                else:  # sell
                    optimized_stop = current_price + (atr * atr_multiplier)
                    # Ensure stop loss is not too far from entry (max percentage loss)
                    max_stop_distance = current_price * (1 + self.config.get('max_stop_loss_percent', 0.03))
                    optimized_stop = min(optimized_stop, max_stop_distance)
                
                optimized_signal['stop_loss'] = optimized_stop
                optimized_signal['execution']['stop_loss_method'] = 'atr_based'
                optimized_signal['execution']['atr_value'] = atr
            
        except Exception as e:
            self.logger.warning(f"Stop loss optimization failed: {str(e)}")
            optimized_signal['execution']['stop_loss_optimization_error'] = str(e)
        
        # Optimize take profit based on mean reversion target
        try:
            # For mean reversion, target the mean (or a percentage of the distance to the mean)
            rolling_mean = signal['metadata'].get('rolling_mean')
            
            if rolling_mean:
                # Calculate distance to mean
                distance_to_mean = abs(current_price - rolling_mean)
                
                # Take profit at X% of the distance to mean
                mean_reversion_factor = self.config.get('mean_reversion_target_factor', 0.8)
                
                if signal['direction'] == 'buy':
                    # For buy signals, take profit should be higher than entry
                    if rolling_mean > current_price:
                        # Mean is above current price, target the mean
                        optimized_take_profit = current_price + (distance_to_mean * mean_reversion_factor)
                    else:
                        # Mean is below current price (unusual for buy signal), use default take profit
                        optimized_take_profit = current_price * (1 + self.config.get('default_take_profit_percent', 0.02))
                else:  # sell
                    # For sell signals, take profit should be lower than entry
                    if rolling_mean < current_price:
                        # Mean is below current price, target the mean
                        optimized_take_profit = current_price - (distance_to_mean * mean_reversion_factor)
                    else:
                        # Mean is above current price (unusual for sell signal), use default take profit
                        optimized_take_profit = current_price * (1 - self.config.get('default_take_profit_percent', 0.02))
                
                optimized_signal['take_profit'] = optimized_take_profit
                optimized_signal['execution']['take_profit_method'] = 'mean_reversion_target'
                optimized_signal['execution']['mean_reversion_factor'] = mean_reversion_factor
            
        except Exception as e:
            self.logger.warning(f"Take profit optimization failed: {str(e)}")
            optimized_signal['execution']['take_profit_optimization_error'] = str(e)
        
        # Set execution urgency based on signal strength and market conditions
        signal_strength = signal.get('strength', 0.5)
        if signal_strength > 0.8:
            # High signal strength - use aggressive execution
            optimized_signal['execution']['urgency'] = 'high'
            optimized_signal['execution']['order_type'] = 'market'
            optimized_signal['execution']['time_in_force'] = 'gtc'  # Good 'til canceled
        elif signal_strength > 0.6:
            # Medium signal strength - use standard execution
            optimized_signal['execution']['urgency'] = 'medium'
            optimized_signal['execution']['order_type'] = 'limit'
            optimized_signal['execution']['limit_price_buffer'] = 0.001  # 0.1% buffer
            optimized_signal['execution']['time_in_force'] = 'day'
        else:
            # Low signal strength - use conservative execution
            optimized_signal['execution']['urgency'] = 'low'
            optimized_signal['execution']['order_type'] = 'limit'
            optimized_signal['execution']['limit_price_buffer'] = 0.002  # 0.2% buffer
            optimized_signal['execution']['time_in_force'] = 'day'
            
        # Check market regime if available
        try:
            from .regime_detection import RegimeDetectionStrategy
            regime_detector = RegimeDetectionStrategy()
            market_data = self.data_stream.get_latest_market_data()
            current_regime = regime_detector.detect_regime(market_data)
            
            # Adjust execution based on market regime
            if current_regime == "RANGING":
                # Ranging market is ideal for mean reversion - maintain or increase urgency
                pass  # Keep current urgency settings
            elif current_regime == "TRENDING":
                # Trending market is risky for mean reversion - reduce urgency
                if optimized_signal['execution']['urgency'] == 'high':
                    optimized_signal['execution']['urgency'] = 'medium'
                elif optimized_signal['execution']['urgency'] == 'medium':
                    optimized_signal['execution']['urgency'] = 'low'
                
                # Add warning to signal metadata
                optimized_signal['metadata']['warnings'] = optimized_signal['metadata'].get('warnings', [])
                optimized_signal['metadata']['warnings'].append("Trending market detected - mean reversion risk increased")
        except Exception as e:
            self.logger.debug(f"Market regime detection skipped: {str(e)}")
            
        return optimized_signal
    
    async def get_health_status(self):
        """
        Returns the health status of the mean reversion strategy and its components.
        
        Returns:
            dict: Health status information including component statuses and any warnings/errors
        """
        health_status = {
            'status': 'healthy',  # Default status
            'timestamp': time.time(),
            'strategy': 'mean_reversion',
            'version': self.VERSION,
            'components': {},
            'warnings': [],
            'errors': []
        }
        
        return health_status