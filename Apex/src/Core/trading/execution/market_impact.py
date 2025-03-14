# Core/trading/execution/market_impact.py
"""
Advanced Market Impact Analysis Engine for Apex Trading System

This module specializes in analyzing and predicting the market impact of trade orders,
ensuring optimal execution quality with minimal slippage and price movement.
It's designed to work as a fully integrated component of the Apex trading system,
maintaining strict separation of concerns while delivering ultra-low latency performance.

Key responsibilities:
- Pre-trade market impact analysis and prediction
- Order size optimization based on liquidity conditions
- Real-time market condition assessment for execution routing
- Anti-frontrunning and trade camouflage techniques
"""

import numpy as np
import asyncio
import hashlib
import time
import pickle
from typing import Dict, Any, List, Tuple, Optional, Union
from decimal import Decimal
import concurrent.futures
import multiprocessing as mp
from functools import lru_cache
import warnings
from contextlib import asynccontextmanager

# Core System Imports
from Core.data.realtime.market_data import MarketDataStreamManager, OrderBookState
from Core.data.realtime.websocket_handler import WebSocketConnectionPool
from Core.data.order_book_analyzer import OrderBookAnalyzer, LiquidityProfile
from Core.trading.risk.risk_management import RiskManager, RiskEvaluationResult
from Core.trading.hft.liquidity_manager import LiquidityOptimizer
from Core.trading.execution.order_execution import OrderSplitter, ExecutionQualityMonitor
from Core.trading.execution.broker_api import BrokerSelector
from Core.trading.execution.conflict_resolver import OrderConflictResolver
from Core.trading.logging.decision_logger import DecisionLogger

# AI System Imports
from ai.ensembles.meta_trader import MetaTrader
from ai.forecasting.spread_forecaster import SpreadImpactPredictor
from ai.analysis.market_regime_classifier import MarketRegimeClassifier
from ai.analysis.institutional_clusters import InstitutionalOrderDetector

# System Utilities
from utils.helpers.error_handler import log_error, critical_error, log_anomaly
from utils.logging.structured_logger import StructuredLogger
from utils.analytics.monte_carlo_simulator import MarketImpactSimulator
from utils.helpers.stealth_api import OrderCamouflage
from utils.logging.telegram_alerts import AlertManager


class MarketImpactEngine:
    """
    High-performance market impact analysis system with nanosecond-level optimization
    
    This class analyzes the potential market impact of trades before execution, 
    enabling optimal execution strategies and minimizing trading costs.
    """
    # Shared memory resources to avoid redundant computations
    _order_book_cache = {}
    _last_analysis_results = {}
    _market_regime_cache = {}
    
    def __init__(
        self, 
        risk_manager: RiskManager,
        liquidity_optimizer: LiquidityOptimizer,
        meta_trader: MetaTrader,
        order_book_analyzer: OrderBookAnalyzer,
        spread_predictor: SpreadImpactPredictor,
        market_data_manager: MarketDataStreamManager,
        decision_logger: DecisionLogger,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the MarketImpactEngine with required dependencies
        
        Args:
            risk_manager: System risk management component
            liquidity_optimizer: Liquidity analysis and optimization component
            meta_trader: AI model ensemble coordinator
            order_book_analyzer: Order book analysis component
            spread_predictor: Spread movement prediction component
            market_data_manager: Real-time market data manager
            decision_logger: Execution decision logging system
            config: Optional configuration parameters
        """
        # Core dependencies - strict integration with other Apex components
        self.risk_manager = risk_manager
        self.liquidity_optimizer = liquidity_optimizer
        self.meta_trader = meta_trader
        self.order_book_analyzer = order_book_analyzer
        self.spread_predictor = spread_predictor
        self.market_data_manager = market_data_manager
        self.decision_logger = decision_logger
        
        # Set up monitoring and alerts
        self.alert_manager = AlertManager()
        self.logger = StructuredLogger("market_impact")
        
        # Performance optimization components
        self._setup_performance_optimizations()
        
        # Security and anti-frontrunning components
        self.order_camouflage = OrderCamouflage()
        
        # Configuration parameters
        self.config = config or self._default_config()
        
        # Load pre-trained models for impact prediction
        self._load_impact_models()
        
        # Initialize last execution timestamp for performance tracking
        self.last_execution_time = time.time()
        
        # Set up multiprocessing resources
        self._setup_multiprocessing()
        
        # Register for market data updates
        self._register_market_data_handlers()
        
        self.logger.info("MarketImpactEngine initialized successfully")

    def _setup_performance_optimizations(self):
        """Set up performance optimization components for high-frequency trading"""
        # Thread pool for parallel computations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, mp.cpu_count() * 4),
            thread_name_prefix="impact_worker"
        )
        
        # Process pool for CPU-intensive calculations
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=max(2, mp.cpu_count() - 1)
        )
        
        # Shared memory for inter-process communication
        self.shared_memory = {}
        
        # WebSocket connection pool for real-time data
        self.websocket_pool = WebSocketConnectionPool(max_connections=100)

    def _setup_multiprocessing(self):
        """Initialize multiprocessing resources for parallel computation"""
        # Create shared memory manager
        self.manager = mp.Manager()
        # Shared dictionaries for inter-process communication
        self.shared_order_book = self.manager.dict()
        self.shared_market_states = self.manager.dict()
        # Event flags for synchronization
        self.calculation_ready = mp.Event()
        self.model_updated = mp.Event()

    def _register_market_data_handlers(self):
        """Register handlers for real-time market data updates"""
        # Subscribe to order book updates
        self.market_data_manager.register_order_book_handler(
            self._handle_order_book_update
        )
        # Subscribe to trade updates
        self.market_data_manager.register_trade_handler(
            self._handle_trade_update
        )
        # Subscribe to market regime changes
        self.market_data_manager.register_regime_handler(
            self._handle_regime_change
        )

    def _load_impact_models(self):
        """Load pre-trained market impact prediction models"""
        try:
            # Primary models for different market regimes
            self.impact_models = {
                'normal': self._load_model('normal_market_impact'),
                'volatile': self._load_model('volatile_market_impact'),
                'crisis': self._load_model('crisis_market_impact'),
                'thin_liquidity': self._load_model('thin_liquidity_impact')
            }
            
            # Fallback model for emergency situations
            self.fallback_model = self._load_model('fallback_impact')
            
            # Institutional order detection model
            self.institutional_detector = InstitutionalOrderDetector()
            
            # Market regime classifier
            self.regime_classifier = MarketRegimeClassifier()
            
            # Impact simulation engine
            self.impact_simulator = MarketImpactSimulator()
            
        except Exception as e:
            self.logger.error(f"Failed to load impact models: {e}")
            # Use minimal fallback models that don't require external files
            self._setup_fallback_models()

    def _load_model(self, model_name: str) -> Any:
        """Load a pre-trained model from storage
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model object
        """
        try:
            with open(f"models/market_impact/{model_name}.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Model {model_name} not found, using default")
            return self._create_default_model(model_name)

    def _create_default_model(self, model_type: str) -> Any:
        """Create a simple default model when pre-trained models are unavailable
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Simple model object
        """
        # Simple linear model as fallback
        return {
            'type': 'linear_fallback',
            'params': {
                'size_impact_factor': 0.15,
                'volatility_multiplier': 2.0,
                'min_impact': 0.0001,
            }
        }

    def _setup_fallback_models(self):
        """Set up minimal fallback models when loading fails"""
        # Simple models based on standard market impact formulas
        self.impact_models = {
            'normal': self._create_default_model('normal'),
            'volatile': self._create_default_model('volatile'),
            'crisis': self._create_default_model('crisis'),
            'thin_liquidity': self._create_default_model('thin_liquidity')
        }
        self.fallback_model = self._create_default_model('fallback')

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration settings
        
        Returns:
            Dictionary of default configuration parameters
        """
        return {
            'parallelization': {
                'enabled': True,
                'max_workers': mp.cpu_count(),
                'chunk_size': 1000
            },
            'caching': {
                'enabled': True,
                'ttl_seconds': 1.0,  # Ultra-short TTL for HFT
                'max_items': 10000
            },
            'security': {
                'verify_data_integrity': True,
                'anti_frontrunning': True,
                'stealth_routing': True
            },
            'performance': {
                'use_shared_memory': True,
                'vectorize_calculations': True,
                'precompute_common_scenarios': True
            },
            'fallback': {
                'use_historical_data': True,
                'max_historical_age_seconds': 300,
                'conservative_adjustment': 1.5  # Increase estimated impact by 50% when using fallback
            },
            'logging': {
                'log_level': 'INFO',
                'detailed_metrics': True,
                'performance_tracking': True
            }
        }

    @asynccontextmanager
    async def _performance_tracking(self, operation: str):
        """Context manager for tracking operation performance
        
        Args:
            operation: Name of the operation being tracked
        """
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.logger.debug(f"Operation '{operation}' completed in {execution_time:.2f}ms")
            # Log slow operations
            if execution_time > 50:  # More than 50ms is slow for HFT
                self.logger.warning(f"Slow operation detected: '{operation}' took {execution_time:.2f}ms")

    async def analyze_market_impact(self, trade_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the market impact of a proposed trade and provide execution recommendations
        
        This is the primary public method that other Apex components call to assess
        how a trade may affect the market before execution.
        
        Args:
            trade_plan: Dictionary containing trade details including:
                - asset: Symbol of the asset to trade
                - side: 'buy' or 'sell'
                - size: Quantity to trade
                - price: Target price (optional)
                - urgency: Trade urgency level (optional)
                - max_impact: Maximum acceptable impact (optional)
        
        Returns:
            Dictionary with market impact analysis including:
                - expected_impact: Estimated price impact
                - slippage_estimate: Expected execution slippage
                - optimal_size: Recommended order size
                - execution_strategy: Recommended execution approach
                - risk_assessment: Risk evaluation results
        """
        async with self._performance_tracking("analyze_market_impact"):
            try:
                # Apply security measures first
                if self.config['security']['anti_frontrunning']:
                    trade_plan = await self._apply_anti_frontrunning_measures(trade_plan)
                
                # Validate the trade plan
                self._validate_trade_plan(trade_plan)
                
                # Collect all necessary market data in parallel
                market_data = await self._gather_market_data(trade_plan['asset'])
                
                # Determine current market regime
                market_regime = self._determine_market_regime(market_data, trade_plan['asset'])
                
                # Select appropriate impact model based on market regime
                impact_model = self.impact_models.get(market_regime, self.impact_models['normal'])
                
                # Calculate expected market impact using the selected model
                impact_analysis = await self._calculate_market_impact(
                    trade_plan, 
                    market_data,
                    impact_model
                )
                
                # Get risk assessment from risk manager
                risk_assessment = await self._get_risk_assessment(trade_plan, impact_analysis)
                
                # Optimize order execution parameters
                execution_params = self._optimize_execution_parameters(
                    trade_plan,
                    impact_analysis,
                    risk_assessment,
                    market_data
                )
                
                # Perform Monte Carlo simulation for uncertainty analysis
                uncertainty_analysis = await self._run_impact_simulation(
                    trade_plan,
                    impact_analysis,
                    market_data
                )
                
                # Combine all results
                final_analysis = self._prepare_impact_analysis_result(
                    trade_plan,
                    impact_analysis,
                    risk_assessment,
                    execution_params,
                    uncertainty_analysis,
                    market_regime
                )
                
                # Log the decision for audit and analysis
                self.decision_logger.log_market_impact_analysis(
                    trade_plan, 
                    final_analysis,
                    market_data
                )
                
                # Cache the result for potential reuse
                self._cache_analysis_result(trade_plan['asset'], final_analysis)
                
                # Update execution metrics
                self._update_execution_metrics(final_analysis)
                
                return final_analysis
                
            except Exception as e:
                self.logger.error(f"Market impact analysis failed: {e}", exc_info=True)
                await self._handle_analysis_failure(trade_plan, e)
                # Return conservative fallback analysis
                return self._fallback_impact_analysis(trade_plan)

    def _validate_trade_plan(self, trade_plan: Dict[str, Any]) -> None:
        """
        Validate the trade plan for required fields and correct formats
        
        Args:
            trade_plan: Trade plan dictionary to validate
        
        Raises:
            ValueError: If the trade plan is invalid
        """
        required_fields = ['asset', 'side', 'size']
        for field in required_fields:
            if field not in trade_plan:
                raise ValueError(f"Trade plan missing required field: {field}")
        
        # Validate side
        if trade_plan['side'] not in ['buy', 'sell']:
            raise ValueError(f"Invalid trade side: {trade_plan['side']}")
        
        # Validate size
        if not isinstance(trade_plan['size'], (int, float)) or trade_plan['size'] <= 0:
            raise ValueError(f"Invalid trade size: {trade_plan['size']}")

    async def _gather_market_data(self, asset: str) -> Dict[str, Any]:
        """
        Gather all required market data for impact analysis in parallel
        
        Args:
            asset: Asset symbol
        
        Returns:
            Dictionary containing all necessary market data
        """
        # Define all the data gathering tasks to run in parallel
        tasks = [
            self._get_order_book_snapshot(asset),
            self._get_recent_trades(asset),
            self._get_liquidity_profile(asset),
            self._get_volatility_metrics(asset),
            self._get_institutional_activity(asset)
        ]
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks)
        
        # Combine results into a single market data dictionary
        market_data = {
            'order_book': results[0],
            'recent_trades': results[1],
            'liquidity_profile': results[2],
            'volatility_metrics': results[3],
            'institutional_activity': results[4],
            'timestamp': time.time()
        }
        
        return market_data

    async def _get_order_book_snapshot(self, asset: str) -> Dict[str, Any]:
        """
        Get the latest order book snapshot for an asset
        
        Args:
            asset: Asset symbol
        
        Returns:
            Order book data dictionary
        """
        # Check if we have a recent cached order book
        cached = self._order_book_cache.get(asset)
        if cached and time.time() - cached['timestamp'] < self.config['caching']['ttl_seconds']:
            return cached['data']
        
        # Get fresh order book data
        try:
            order_book = await self.market_data_manager.get_order_book(asset)
            
            # Cache the result
            self._order_book_cache[asset] = {
                'data': order_book,
                'timestamp': time.time()
            }
            
            return order_book
        except Exception as e:
            self.logger.warning(f"Failed to get order book for {asset}: {e}")
            # Return empty order book or last known
            return cached['data'] if cached else OrderBookState.empty_book()

    async def _get_recent_trades(self, asset: str) -> List[Dict[str, Any]]:
        """
        Get recent trades for an asset
        
        Args:
            asset: Asset symbol
        
        Returns:
            List of recent trades
        """
        try:
            # Get trades from last 5 minutes
            return await self.market_data_manager.get_recent_trades(
                asset, 
                lookback_seconds=300
            )
        except Exception as e:
            self.logger.warning(f"Failed to get recent trades for {asset}: {e}")
            return []

    async def _get_liquidity_profile(self, asset: str) -> LiquidityProfile:
        """
        Get liquidity profile for an asset
        
        Args:
            asset: Asset symbol
        
        Returns:
            Liquidity profile object
        """
        try:
            return await self.liquidity_optimizer.get_liquidity_profile(asset)
        except Exception as e:
            self.logger.warning(f"Failed to get liquidity profile for {asset}: {e}")
            return LiquidityProfile.default()

    async def _get_volatility_metrics(self, asset: str) -> Dict[str, float]:
        """
        Get volatility metrics for an asset
        
        Args:
            asset: Asset symbol
        
        Returns:
            Dictionary of volatility metrics
        """
        try:
            # Run this in a separate thread as it's CPU-intensive
            return await asyncio.to_thread(
                self.order_book_analyzer.calculate_volatility_metrics,
                asset
            )
        except Exception as e:
            self.logger.warning(f"Failed to get volatility metrics for {asset}: {e}")
            return {
                'realized_volatility': 0.02,  # Default 2% volatility
                'implied_volatility': 0.025,
                'volatility_ratio': 1.25
            }

    async def _get_institutional_activity(self, asset: str) -> Dict[str, Any]:
        """
        Get institutional trading activity for an asset
        
        Args:
            asset: Asset symbol
        
        Returns:
            Dictionary of institutional activity metrics
        """
        try:
            # This is potentially CPU-intensive so run in process pool
            return await asyncio.to_thread(
                self.institutional_detector.detect_institutional_activity,
                asset
            )
        except Exception as e:
            self.logger.warning(f"Failed to get institutional activity for {asset}: {e}")
            return {
                'presence_likelihood': 0.0,
                'direction': 'neutral',
                'confidence': 0.0
            }

    def _determine_market_regime(self, market_data: Dict[str, Any], asset: str) -> str:
        """
        Determine the current market regime for an asset
        
        Args:
            market_data: Market data dictionary
            asset: Asset symbol
        
        Returns:
            Market regime identifier string
        """
        # Check if we have a cached regime that's recent
        cached = self._market_regime_cache.get(asset)
        if cached and time.time() - cached['timestamp'] < 60:  # 1 minute TTL
            return cached['regime']
        
        try:
            # Calculate market regime
            volatility = market_data['volatility_metrics']['realized_volatility']
            liquidity = market_data['liquidity_profile'].depth_factor
            spread = market_data['order_book']['spread_pct'] if 'spread_pct' in market_data['order_book'] else 0.001
            
            # Simple threshold-based regime classification
            if volatility > 0.04:  # 4% volatility
                regime = 'volatile'
            elif liquidity < 0.3:  # Low liquidity
                regime = 'thin_liquidity'
            elif volatility > 0.08 or liquidity < 0.1 or spread > 0.005:  # Extreme conditions
                regime = 'crisis'
            else:
                regime = 'normal'
            
            # Cache the result
            self._market_regime_cache[asset] = {
                'regime': regime,
                'timestamp': time.time()
            }
            
            return regime
        except Exception as e:
            self.logger.warning(f"Failed to determine market regime for {asset}: {e}")
            return 'normal'  # Default to normal regime

    async def _calculate_market_impact(
        self, 
        trade_plan: Dict[str, Any], 
        market_data: Dict[str, Any],
        impact_model: Any
    ) -> Dict[str, Any]:
        """
        Calculate expected market impact using the selected model
        
        Args:
            trade_plan: Trade plan dictionary
            market_data: Market data dictionary
            impact_model: Model to use for impact calculation
        
        Returns:
            Dictionary of impact analysis results
        """
        asset = trade_plan['asset']
        size = trade_plan['size']
        side = trade_plan['side']
        
        try:
            # Extract necessary data for calculation
            order_book = market_data['order_book']
            liquidity = market_data['liquidity_profile']
            volatility = market_data['volatility_metrics']['realized_volatility']
            
            # Normalize trade size relative to average daily volume
            if hasattr(liquidity, 'adv') and liquidity.adv > 0:
                relative_size = size / liquidity.adv
            else:
                relative_size = size / 100000  # Default assumption
            
            # Calculate expected market impact percentage
            if impact_model['type'] == 'linear_fallback':
                # Simple square-root model for fallback
                params = impact_model['params']
                impact_pct = params['size_impact_factor'] * np.sqrt(relative_size) * (1 + params['volatility_multiplier'] * volatility)
                impact_pct = max(impact_pct, params['min_impact'])
            else:
                # Use the pre-trained model
                features = self._prepare_impact_features(trade_plan, market_data)
                impact_pct = await asyncio.to_thread(self._predict_impact, impact_model, features)
            
            # Calculate expected slippage
            slippage = self._calculate_expected_slippage(
                side,
                size,
                order_book,
                volatility
            )
            
            # Calculate optimal execution size to minimize impact
            optimal_size = self._calculate_optimal_size(
                size,
                liquidity,
                impact_pct
            )
            
            # Calculate expected price after impact
            current_price = order_book['mid_price']
            impact_direction = 1 if side == 'buy' else -1
            expected_price = current_price * (1 + impact_direction * impact_pct)
            
            return {
                'impact_pct': impact_pct,
                'impact_absolute': impact_pct * current_price,
                'expected_price': expected_price,
                'slippage_pct': slippage['percent'],
                'slippage_absolute': slippage['absolute'],
                'optimal_size': optimal_size,
                'relative_size': relative_size,
                'current_price': current_price
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate market impact: {e}", exc_info=True)
            # Return conservative default impact estimate
            return self._default_impact_estimate(trade_plan)

    def _prepare_impact_features(
        self, 
        trade_plan: Dict[str, Any], 
        market_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        Prepare features for market impact prediction models
        
        Args:
            trade_plan: Trade plan dictionary
            market_data: Market data dictionary
            
        Returns:
            NumPy array of features
        """
        # Extract relevant data
        order_book = market_data['order_book']
        liquidity = market_data['liquidity_profile']
        volatility = market_data['volatility_metrics']['realized_volatility']
        
        # Basic features
        features = [
            float(trade_plan['size']),  # Order size
            1.0 if trade_plan['side'] == 'buy' else -1.0,  # Order direction
            order_book['bid_ask_spread'],  # Current spread
            liquidity.depth_factor,  # Market depth
            volatility,  # Current volatility
            liquidity.turnover_velocity,  # Trading velocity
            market_data['institutional_activity']['presence_likelihood']  # Institutional presence
        ]
        
        # Additional order book features
        features.extend([
            order_book['bid_depth'],  # Depth on bid side
            order_book['ask_depth'],  # Depth on ask side
            order_book['bid_ask_imbalance']  # Order book imbalance
        ])
        
        return np.array(features).reshape(1, -1)  # Reshape for model input

    def _predict_impact(self, model: Any, features: np.ndarray) -> float:
        """
        Predict market impact using a trained model
        
        Args:
            model: Trained model object
            features: Feature array
            
        Returns:
            Predicted impact percentage
        """
        try:
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Ensure prediction is reasonable (between 0.0001% and 10%)
            return max(0.0001, min(0.1, prediction))
        except:
            # Fallback to simple formula if model fails
            return 0.01 * np.sqrt(features[0][0] / 10000)  # Simple square-root impact model

    def _calculate_expected_slippage(
        self, 
        side: str, 
        size: float, 
        order_book: Dict[str, Any],
        volatility: float
    ) -> Dict[str, float]:
        """
        Calculate expected execution slippage based on order book
        
        Args:
            side: Trade side ('buy' or 'sell')
            size: Trade size
            order_book: Order book data
            volatility: Current volatility
            
        Returns:
            Dictionary with slippage estimates
        """
        try:
            # Get relevant order book side
            book_side = order_book['asks'] if side == 'buy' else order_book['bids']
            
            # Get reference price
            if side == 'buy':
                reference_price = order_book['ask_price']
            else:
                reference_price = order_book['bid_price']
            
            # Calculate immediate market order slippage
            remaining_size = size
            cost = 0.0
            
            for level in book_side:
                level_price = level['price']
                level_size = level['size']
                
                if remaining_size <= level_size:
                    cost += remaining_size * level_price
                    remaining_size = 0
                    break
                else:
                    cost += level_size * level_price
                    remaining_size -= level_size
            
            # If order book not deep enough, estimate remaining impact
            if remaining_size > 0:
                # Assume additional slippage based on volatility and remaining size
                additional_slippage = volatility * np.sqrt(remaining_size / size)
                # Apply slippage in correct direction
                if side == 'buy':
                    estimated_price = book_side[-1]['price'] * (1 + additional_slippage)
                else:
                    estimated_price = book_side[-1]['price'] * (1 - additional_slippage)
                
                cost += remaining_size * estimated_price
            
            # Calculate effective execution price
            effective_price = cost / size if size > 0 else reference_price
            
            # Calculate slippage
            if side == 'buy':
                slippage_pct = (effective_price / reference_price) - 1
            else:
                slippage_pct = 1 - (effective_price / reference_price)
            
            return {
                'percent': slippage_pct,
                'absolute': slippage_pct * reference_price
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate slippage: {e}")
            # Return conservative default
            return {
                'percent': 0.001 * volatility * np.sqrt(size / 10000),
                'absolute': order_book['mid_price'] * 0.001 * volatility * np.sqrt(size / 10000)
            }

    def _calculate_optimal_size(
        self, 
        requested_size: float, 
        liquidity: LiquidityProfile,
        impact_pct: float
    ) -> float:
        """
        Calculate optimal order size to minimize market impact
        
        Args:
            requested_size: Requested trade size
            liquidity: Liquidity profile
            impact_pct: Estimated market impact percentage
            
        Returns:
            Optimal order size
        """
        try:
            # Get liquidity metrics
            depth = liquidity.depth_factor
            
            # Base optimal size on available liquidity and acceptable impact
            # If impact is already low, keep requested size
            if impact_pct < 0.001:
                return requested_size
            
            # Calculate optimal size based on square-root formula
            # This reverses the square root impact model to find size that gives acceptable impact
            acceptable_impact = 0.001  # Target 0.1% impact
            impact_ratio = acceptable_impact / impact_pct
            
            # Apply square root formula: size ~ impact^2
            optimal_size = requested_size * (impact_ratio ** 2)
            
            # Apply depth adjustment
            optimal_size = optimal_size * depth
            
            # Ensure optimal size is not too small
            optimal_size = max(optimal_size, requested_size * 0.1)
            
            # Cap at requested size
            optimal_size = min(optimal_size, requested_size)
            
            return optimal_size
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate optimal size: {e}")
            # Return conservative default: 10% of requested size
            return max(requested_size * 0.1, 1.0)

    async def _get_risk_assessment(
        self, 
        trade_plan: Dict[str, Any], 
        impact_analysis: Dict[str, Any]
    ) -> RiskEvaluationResult:
        """
        Get risk assessment from risk manager for the trade
        
        Args:
            trade_plan: Trade plan dictionary
            impact_analysis: Impact analysis results
            
        Returns:
            Risk evaluation result
        """
        try:
            # Prepare risk context with impact analysis
            risk_context = {
                'trade_plan': trade_plan,
                'market_impact': impact_analysis,
                'timestamp': time.time()
            }
            
            # Get risk assessment asynchronously
            return await self.risk_manager.evaluate_trade_risk(risk_context)
        except Exception as e:
            self.logger.warning(f"Failed to get risk assessment: {e}")
            # Return default conservative risk assessment
            return RiskEvaluationResult(
                risk_level=0.7,  # High risk assumed as fallback
                risk_factors={
                    'impact_risk': 0.8,
                    'liquidity_risk': 0.7,
                    'volatility_risk': 0.6
                },
                approval_status='conditional',
                risk_limits={
                    'max_impact_bps': 5,
                    'max_size_pct': 0.1
                }
            )

    def _optimize_execution_parameters(
        self,
        trade_plan: Dict[str, Any],
        impact_analysis: Dict[str, Any],
        risk_assessment: RiskEvaluationResult,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize execution parameters based on impact analysis and risk assessment
        
        Args:
            trade_plan: Trade plan dictionary
            impact_analysis: Impact analysis results
            risk_assessment: Risk evaluation result
            market_data: Market data dictionary
            
        Returns:
            Optimized execution parameters
        """
        # Apply vectorized optimization using numpy for performance
        try:
            # Extract trade parameters
            asset = trade_plan['asset']
            size = trade_plan['size']
            side = trade_plan['side']
            urgency = trade_plan.get('urgency', 'normal')
            
            # Extract market conditions
            volatility = market_data['volatility_metrics']['realized_volatility']
            liquidity = market_data['liquidity_profile']
            optimal_size = impact_analysis['optimal_size']
            
            # Determine execution strategy based on conditions
            if optimal_size < size * 0.2 and urgency != 'high':
                # Very low optimal size suggests we should use a highly passive approach
                strategy = 'iceberg'
                time_horizon = min(3600, max(300, int(300 * (size / optimal_size))))
                participation_rate = 0.05  # Very low participation rate
            elif volatility > 0.03 or risk_assessment.risk_level > 0.7:
                # High volatility or risk suggests a more cautious approach
                strategy = 'twap'
                time_horizon = min(1800, max(300, int(600 * (size / optimal_size))))
                participation_rate = 0.1
            elif liquidity.depth_factor > 0.8 and size / liquidity.adv < 0.01:
                # Good liquidity and small relative size suggests more aggressive approach
                strategy = 'adaptive'
                time_horizon = min(900, max(60, int(300 * (size / optimal_size))))
                participation_rate = 0.25
            else:
                # Default balanced approach
                strategy = 'vwap'
                time_horizon = min(1200, max(180, int(500 * (size / optimal_size))))
                participation_rate = 0.15
                
            # Calculate number of child orders based on optimal size
            num_child_orders = max(1, int(np.ceil(size / optimal_size)))
            
            # Select venues based on liquidity profile
            venues = self._select_execution_venues(asset, side, size)
            
            # Apply routing camouflage if anti-frontrunning is enabled
            if self.config['security']['anti_frontrunning']:
                venues = self.order_camouflage.apply_venue_randomization(venues)
                
            return {
                'strategy': strategy,
                'time_horizon_seconds': time_horizon,
                'participation_rate': participation_rate,
                'num_child_orders': num_child_orders,
                'child_order_size': size / num_child_orders,
                'venues': venues,
                'execution_style': 'passive' if strategy in ['iceberg', 'twap'] else 'adaptive'
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to optimize execution parameters: {e}")
            # Return conservative default execution parameters
            return {
                'strategy': 'twap',
                'time_horizon_seconds': 900,  # 15 minutes
                'participation_rate': 0.1,
                'num_child_orders': 5,
                'child_order_size': trade_plan['size'] / 5,
                'venues': [{'venue': 'primary', 'allocation': 1.0}],
                'execution_style': 'passive'
            }

    def _select_execution_venues(
        self,
        asset: str,
        side: str,
        size: float
    ) -> List[Dict[str, Any]]:
        """
        Select optimal execution venues based on asset and trade characteristics
        
        Args:
            asset: Asset symbol
            side: Trade side ('buy' or 'sell')
            size: Trade size
            
        Returns:
            List of venue allocations
        """
        try:
            # Get venue recommendations from broker selector
            return BrokerSelector.get_optimal_venues(asset, side, size)
        except Exception as e:
            self.logger.warning(f"Failed to select execution venues: {e}")
            # Return default venue allocation
            return [{'venue': 'primary', 'allocation': 1.0}]

    async def _run_impact_simulation(
        self,
        trade_plan: Dict[str, Any],
        impact_analysis: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for market impact uncertainty analysis
        
        Args:
            trade_plan: Trade plan dictionary
            impact_analysis: Impact analysis results
            market_data: Market data dictionary
            
        Returns:
            Dictionary of simulation results
        """
        try:
            # Define simulation parameters
            num_simulations = 1000
            asset = trade_plan['asset']
            size = trade_plan['size']
            side = trade_plan['side']
            
            # Extract market conditions
            volatility = market_data['volatility_metrics']['realized_volatility']
            baseline_impact = impact_analysis['impact_pct']
            
            # Run simulation in a separate process to avoid blocking
            simulation_result = await asyncio.to_thread(
                self.impact_simulator.run_impact_simulation,
                asset=asset,
                size=size,
                side=side,
                baseline_impact=baseline_impact,
                volatility=volatility,
                num_simulations=num_simulations
            )
            
            return {
                'impact_confidence_intervals': simulation_result['confidence_intervals'],
                'max_adverse_impact': simulation_result['worst_case'],
                'impact_volatility': simulation_result['impact_volatility'],
                'tail_risk_5pct': simulation_result['tail_risk_5pct']
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to run impact simulation: {e}")
            # Return conservative default simulation results
            baseline_impact = impact_analysis['impact_pct']
            return {
                'impact_confidence_intervals': {
                    '50pct': baseline_impact,
                    '90pct': baseline_impact * 1.5,
                    '95pct': baseline_impact * 2.0
                },
                'max_adverse_impact': baseline_impact * 3.0,
                'impact_volatility': volatility,
                'tail_risk_5pct': baseline_impact * 2.5
            }

    def _prepare_impact_analysis_result(
        self,
        trade_plan: Dict[str, Any],
        impact_analysis: Dict[str, Any],
        risk_assessment: RiskEvaluationResult,
        execution_params: Dict[str, Any],
        uncertainty_analysis: Dict[str, Any],
        market_regime: str
    ) -> Dict[str, Any]:
        """
        Prepare final impact analysis result
        
        Args:
            trade_plan: Trade plan dictionary
            impact_analysis: Impact analysis results
            risk_assessment: Risk evaluation result
            execution_params: Execution parameters
            uncertainty_analysis: Uncertainty analysis results
            market_regime: Current market regime
            
        Returns:
            Final impact analysis result
        """
        # Combine all results into a comprehensive analysis
        return {
            'trade_details': {
                'asset': trade_plan['asset'],
                'side': trade_plan['side'],
                'size': trade_plan['size'],
                'timestamp': time.time()
            },
            'market_impact': {
                'expected_impact_pct': impact_analysis['impact_pct'],
                'expected_impact_price': impact_analysis['impact_absolute'],
                'expected_price': impact_analysis['expected_price'],
                'slippage_pct': impact_analysis['slippage_pct'],
                'slippage_price': impact_analysis['slippage_absolute'],
                'relative_size': impact_analysis['relative_size'],
                'current_price': impact_analysis['current_price']
            },
            'risk_assessment': {
                'risk_level': risk_assessment.risk_level,
                'risk_factors': risk_assessment.risk_factors,
                'approval_status': risk_assessment.approval_status,
                'risk_limits': risk_assessment.risk_limits
            },
            'execution_recommendation': {
                'strategy': execution_params['strategy'],
                'time_horizon_seconds': execution_params['time_horizon_seconds'],
                'participation_rate': execution_params['participation_rate'],
                'num_child_orders': execution_params['num_child_orders'],
                'child_order_size': execution_params['child_order_size'],
                'venues': execution_params['venues'],
                'execution_style': execution_params['execution_style']
            },
            'uncertainty_analysis': {
                'confidence_intervals': uncertainty_analysis['impact_confidence_intervals'],
                'max_adverse_impact': uncertainty_analysis['max_adverse_impact'],
                'impact_volatility': uncertainty_analysis['impact_volatility'],
                'tail_risk_5pct': uncertainty_analysis['tail_risk_5pct']
            },
            'market_conditions': {
                'regime': market_regime,
                'timestamp': time.time(),
                'analysis_duration_ms': (time.time() - self.last_execution_time) * 1000
            }
        }

    def _cache_analysis_result(self, asset: str, analysis: Dict[str, Any]) -> None:
        """Cache analysis result for potential reuse
        
        Args:
            asset: Asset symbol
            analysis: Analysis result
        """
        # Implement LRU-style caching with TTL
        self._last_analysis_results[asset] = {
            'data': analysis,
            'timestamp': time.time()
        }
        
        # Prune cache if it exceeds max size
        if len(self._last_analysis_results) > self.config['caching']['max_items']:
            # Remove oldest items
            oldest_asset = min(
                self._last_analysis_results, 
                key=lambda k: self._last_analysis_results[k]['timestamp']
            )
            del self._last_analysis_results[oldest_asset]

    def _update_execution_metrics(self, analysis: Dict[str, Any]) -> None:
        """Update execution metrics for performance tracking
        
        Args:
            analysis: Analysis result
        """
        # Calculate execution time
        execution_time = time.time() - self.last_execution_time
        self.last_execution_time = time.time()
        
        # Log the performance metric
        if self.config['logging']['performance_tracking']:
            self.logger.debug(f"Market impact analysis completed in {execution_time*1000:.2f}ms")
            
            # Alert on slow execution
            if execution_time > 0.05:  # More than 50ms is slow for HFT
                self.logger.warning(
                    f"Slow market impact analysis detected: {execution_time*1000:.2f}ms for "
                    f"{analysis['trade_details']['asset']}"
                )

    async def _handle_analysis_failure(self, trade_plan: Dict[str, Any], error: Exception) -> None:
        """Handle market impact analysis failure
        
        Args:
            trade_plan: Trade plan that failed analysis
            error: Exception that occurred
        """
        # Log error with full details
        error_details = {
            'trade_plan': trade_plan,
            'error': str(error),
            'timestamp': time.time()
        }
        
        # Log critical error
        log_error(
            subsystem="market_impact",
            error=error,
            context=error_details
        )
        
        # Send alert if critical
        if isinstance(error, (TimeoutError, ConnectionError)) or "critical" in str(error).lower():
            await self.alert_manager.send_alert(
                "Critical market impact analysis failure",
                f"Failed to analyze {trade_plan['asset']} trade: {error}",
                level="critical"
            )

    def _fallback_impact_analysis(self, trade_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Provide a conservative fallback market impact analysis when normal analysis fails
        
        Args:
            trade_plan: Trade plan dictionary
            
        Returns:
            Conservative impact analysis
        """
        # Estimate very conservative impact (high impact assumption)
        asset = trade_plan['asset']
        size = trade_plan['size']
        side = trade_plan['side']
        
        # Use simple square-root formula with conservative factor
        conservative_factor = self.config['fallback']['conservative_adjustment']
        impact_pct = 0.01 * np.sqrt(size / 10000) * conservative_factor
        
        # Use last known price or a default
        last_price = 100.0  # Default fallback price
        try:
            # Try to get cached price
            cached_data = self._last_analysis_results.get(asset, {}).get('data', {})
            if cached_data and 'market_impact' in cached_data:
                last_price = cached_data['market_impact']['current_price']
        except:
            pass
            
        # Generate conservative analysis
        return {
            'trade_details': {
                'asset': asset,
                'side': side,
                'size': size,
                'timestamp': time.time()
            },
            'market_impact': {
                'expected_impact_pct': impact_pct,
                'expected_impact_price': impact_pct * last_price,
                'expected_price': last_price * (1 + (1 if side == 'buy' else -1) * impact_pct),
                'slippage_pct': impact_pct * 1.5,
                'slippage_price': impact_pct * 1.5 * last_price,
                'relative_size': size / 10000,
                'current_price': last_price
            },
            'risk_assessment': {
                'risk_level': 0.8,  # High risk assumed
                'risk_factors': {
                    'impact_risk': 0.9,
                    'liquidity_risk': 0.8,
                    'volatility_risk': 0.7
                },
                'approval_status': 'conditional',
                'risk_limits': {
                    'max_impact_bps': 5,
                    'max_size_pct': 0.05
                }
            },
            'execution_recommendation': {
                'strategy': 'twap',  # Conservative TWAP strategy
                'time_horizon_seconds': 1800,  # 30 minutes
                'participation_rate': 0.05,  # Very low participation
                'num_child_orders': 10,
                'child_order_size': size / 10,
                'venues': [{'venue': 'primary', 'allocation': 1.0}],
                'execution_style': 'passive'
            },
            'uncertainty_analysis': {
                'confidence_intervals': {
                    '50pct': impact_pct,
                    '90pct': impact_pct * 2.0,
                    '95pct': impact_pct * 3.0
                },
                'max_adverse_impact': impact_pct * 5.0,
                'impact_volatility': 0.03,
                'tail_risk_5pct': impact_pct * 4.0
            },
            'market_conditions': {
                'regime': 'unknown',
                'timestamp': time.time(),
                'is_fallback': True
            }
        }

    async def _apply_anti_frontrunning_measures(self, trade_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Apply anti-frontrunning measures to the trade plan
        
        Args:
            trade_plan: Original trade plan
            
        Returns:
            Modified trade plan with anti-frontrunning measures
        """
        if not self.config['security']['anti_frontrunning']:
            return trade_plan
            
        # Apply order camouflage techniques
        try:
            # Create a copy of the trade plan to avoid modifying the original
            secured_plan = dict(trade_plan)
            
            # Apply jittering to size (±2% random variation)
            size_jitter = 0.98 + (np.random.random() * 0.04)  # Between 0.98 and 1.02
            secured_plan['size'] = secured_plan['size'] * size_jitter
            
            # Round to avoid obvious patterns
            if secured_plan['size'] > 100:
                # Round to random "natural" number
                rounding_factor = 10 ** np.random.randint(0, 3)  # 1, 10, or 100
                secured_plan['size'] = round(secured_plan['size'] / rounding_factor) * rounding_factor
            
            # Add order book camouflage instructions
            secured_plan['execution_flags'] = secured_plan.get('execution_flags', {})
            secured_plan['execution_flags'].update({
                'use_stealth_ordering': True,
                'randomize_venues': True,
                'use_hidden_orders': np.random.random() > 0.5,  # 50% chance of using hidden orders
                'time_randomization': True
            })
            
            return secured_plan
            
        except Exception as e:
            self.logger.warning(f"Failed to apply anti-frontrunning measures: {e}")
            return trade_plan

    def _handle_order_book_update(self, asset: str, order_book: OrderBookState) -> None:
        """Handle real-time order book updates
        
        Args:
            asset: Asset symbol
            order_book: Updated order book state
        """
        # Update order book cache with new data
        self._order_book_cache[asset] = {
            'data': order_book,
            'timestamp': time.time()
        }
        
        # If shared memory is being used, update the shared dictionary
        if self.config['performance']['use_shared_memory']:
            # Create a serializable copy of the order book
            serializable_book = {
                'bids': [(level['price'], level['size']) for level in order_book['bids']],
                'asks': [(level['price'], level['size']) for level in order_book['asks']],
                'mid_price': order_book['mid_price'],
                'spread': order_book['bid_ask_spread'],
                'timestamp': time.time()
            }
            
            self.shared_order_book[asset] = serializable_book

    def _handle_trade_update(self, asset: str, trade: Dict[str, Any]) -> None:
        """Handle real-time trade updates and update impact models
        
        Args:
            asset: Asset symbol
            trade: Trade data containing price, size, side, etc.
        """
        try:
            # Update trade history cache
            if asset not in self._trade_history_cache:
                self._trade_history_cache[asset] = deque(maxlen=1000)
            self._trade_history_cache[asset].append(trade)

            # Calculate real-time trade impact metrics
            impact_metrics = self._calculate_trade_impact(asset, trade)
            
            # Update impact model with new data point
            self._update_impact_model(asset, impact_metrics)
            
            # Check for anomalous impact
            if self._detect_anomalous_impact(impact_metrics):
                self.logger.warning(f"Anomalous trade impact detected for {asset}")
                self._trigger_impact_alert(asset, impact_metrics)
            
            # Update shared memory if enabled
            if self.config['performance']['use_shared_memory']:
                self.shared_impact_metrics[asset] = impact_metrics
                
            # Emit metrics
            self._emit_impact_metrics(asset, impact_metrics)
            
        except Exception as e:
            self.logger.error(f"Error processing trade update for {asset}: {e}")
            
    def _calculate_trade_impact(self, asset: str, trade: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact metrics for a single trade
        
        Args:
            asset: Asset symbol
            trade: Trade data
            
        Returns:
            Dict containing impact metrics
        """
        metrics = {}
        
        # Get order book snapshot
        order_book = self._order_book_cache.get(asset, {}).get('data')
        if not order_book:
            return metrics
            
        # Calculate price impact
        pre_trade_mid = (order_book['best_bid'] + order_book['best_ask']) / 2
        price_impact = abs(trade['price'] - pre_trade_mid) / pre_trade_mid
        metrics['price_impact'] = price_impact
        
        # Calculate volume impact
        daily_volume = self._get_daily_volume(asset)
        volume_impact = trade['size'] / daily_volume if daily_volume else 0
        metrics['volume_impact'] = volume_impact
        
        # Calculate spread impact
        spread_impact = abs(trade['price'] - pre_trade_mid) / order_book['bid_ask_spread']
        metrics['spread_impact'] = spread_impact
        
        # Calculate market depth impact
        depth_impact = self._calculate_depth_impact(trade, order_book)
        metrics['depth_impact'] = depth_impact
        
        return metrics
        
    def _update_impact_model(self, asset: str, metrics: Dict[str, float]) -> None:
        """Update the market impact prediction model with new data
        
        Args:
            asset: Asset symbol
            metrics: Impact metrics from latest trade
        """
        if asset not in self._impact_models:
            self._impact_models[asset] = self._initialize_impact_model()
            
        model = self._impact_models[asset]
        
        # Convert metrics to model features
        features = self._prepare_impact_features(metrics)
        
        # Update model weights using online learning
        model.partial_fit(features)
        
        # Update model metadata
        self._impact_model_metadata[asset] = {
            'last_update': time.time(),
            'num_updates': self._impact_model_metadata.get(asset, {}).get('num_updates', 0) + 1
        }
        
    def _initialize_impact_model(self) -> Any:
        """Initialize a new market impact prediction model
        
        Returns:
            New model instance
        """
        # Initialize model architecture
        model = Sequential([
            LSTM(64, input_shape=(100, 5), return_sequences=True),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def _prepare_impact_features(self, metrics: Dict[str, float]) -> np.ndarray:
        """Prepare impact metrics as model features
        
        Args:
            metrics: Raw impact metrics
            
        Returns:
            Feature array for model update
        """
        features = np.array([
            metrics.get('price_impact', 0),
            metrics.get('volume_impact', 0), 
            metrics.get('spread_impact', 0),
            metrics.get('depth_impact', 0),
            time.time() # Timestamp
        ])
        return features.reshape(1, -1)
        
    def _detect_anomalous_impact(self, metrics: Dict[str, float]) -> bool:
        """Detect if trade impact metrics are anomalous
        
        Args:
            metrics: Impact metrics to check
            
        Returns:
            True if impact is anomalous
        """
        # Check each metric against thresholds
        for metric, value in metrics.items():
            threshold = self.config['impact_thresholds'].get(metric)
            if threshold and value > threshold:
                return True
                
        return False
        
    def _trigger_impact_alert(self, asset: str, metrics: Dict[str, float]) -> None:
        """Trigger alerts for anomalous market impact
        
        Args:
            asset: Asset symbol
            metrics: Impact metrics that triggered alert
        """
        alert = {
            'asset': asset,
            'metrics': metrics,
            'timestamp': time.time(),
            'severity': 'high'
        }
        
        # Send alert to risk management
        self.risk_manager.handle_impact_alert(alert)
        
        # Log alert
        self.logger.warning(f"Market impact alert triggered for {asset}: {metrics}")
        
        # Update metrics
        self.metrics['impact_alerts'].inc()
        
    def _emit_impact_metrics(self, asset: str, metrics: Dict[str, float]) -> None:
        """Emit impact metrics to monitoring system
        
        Args:
            asset: Asset symbol
            metrics: Impact metrics to emit
        """
        for metric, value in metrics.items():
            self.metrics[f'impact_{metric}'].labels(asset=asset).set(value)
            
    def _get_daily_volume(self, asset: str) -> float:
        """Get asset's daily trading volume
        
        Args:
            asset: Asset symbol
            
        Returns:
            Daily volume in base currency
        """
        try:
            volume = self.market_data.get_daily_volume(asset)
            return float(volume)
        except Exception as e:
            self.logger.error(f"Error getting volume for {asset}: {e}")
            return 0.0
            
    def _calculate_depth_impact(self, trade: Dict[str, Any], 
                              order_book: Dict[str, Any]) -> float:
        """Calculate impact on market depth
        
        Args:
            trade: Trade data
            order_book: Current order book state
            
        Returns:
            Depth impact metric
        """
        side = trade['side']
        size = trade['size']
        
        # Get relevant side of book
        levels = order_book['asks'] if side == 'buy' else order_book['bids']
        
        # Calculate cumulative size at each level
        cum_size = 0
        for level in levels:
            cum_size += level['size']
            if cum_size >= size:
                return level['price']
                
        return 0.0
        
    async def close(self):
        """Clean up resources when shutting down"""
        # Close thread and process pools
        self.thread_pool.shutdown()
        self.process_pool.shutdown()
        
        # Close WebSocket connections
        await self.websocket_pool.close()
        
        # Release shared memory
        if hasattr(self, 'manager') and self.manager:
            self.manager.shutdown()
            
        # Log shutdown status
        self.logger.info("MarketImpactEngine shut down successfully")
        
        # Clean up any remaining metrics/gauges
        if hasattr(self, 'metrics'):
            for metric in self.metrics.values():
                metric.clear()
                
        # Close database connections if any
        if hasattr(self, 'db_conn'):
            await self.db_conn.close()
            
        # Cancel any pending tasks
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()
                
        # Wait for tasks to complete
        await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)
        
        # Final cleanup
        if hasattr(self, 'shared_order_book'):
            self.shared_order_book.clear()
            
        if hasattr(self, 'shared_impact_metrics'):
            self.shared_impact_metrics.clear()
            
        # Release any remaining system resources
        gc.collect()
        
        return True

def __del__(self):
    """Destructor to ensure cleanup"""
    if not asyncio.get_event_loop().is_closed():
        asyncio.create_task(self.close())
        
if __name__ == "__main__":
    # Example usage
    impact_engine = MarketImpactEngine(
        config={
            "performance": {"use_shared_memory": True},
            "logging": {"level": "INFO"}
        }
    )
    
    try:
        # Run the engine
        asyncio.run(impact_engine.start())
    except KeyboardInterrupt:
        # Handle graceful shutdown
        print("Shutting down MarketImpactEngine...")
        asyncio.run(impact_engine.close())
        print("Shutdown complete")
        
        # Clean up any remaining resources
        if hasattr(impact_engine, 'shared_order_book'):
            impact_engine.shared_order_book.clear()
            
        if hasattr(impact_engine, 'shared_impact_metrics'): 
            impact_engine.shared_impact_metrics.clear()
            
        # Force garbage collection
        gc.collect()
        
        sys.exit(0)