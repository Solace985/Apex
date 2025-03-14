import numpy as np
import pandas as pd
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, AsyncGenerator, Union, Set, Callable
from collections import deque
import logging
import functools
import warnings

# Internal Apex Imports
from Apex.utils.helpers.error_handler import handle_exceptions
from Apex.utils.helpers.validation import validate_market_session, secure_float
from Apex.utils.logging.structured_logger import StructuredLogger
from Apex.src.Core.data.realtime.market_data import WebSocketFeed
from Apex.src.Core.data.realtime.data_feed import DataFeed
from Apex.src.Core.data.historical_data import HistoricalData
from Apex.src.Core.trading.execution.market_impact import ImpactCalculator
from Apex.src.Core.trading.risk.risk_management import RiskEngine
from Apex.src.ai.forecasting.order_flow import OrderFlowAnalyzer, FlowPredictionModel
from Apex.src.ai.analysis.market_regime_classifier import MarketRegimeClassifier
from Apex.src.Core.trading.strategies.strategy_orchestrator import StrategyOrchestrator
from Apex.src.Core.trading.hft.liquidity_manager import LiquidityManager
from Apex.src.Core.trading.execution.broker_factory import BrokerRouter
from Apex.src.ai.analysis.institutional_clusters import InstitutionalClusterAnalyzer
from Apex.src.ai.reinforcement.q_learning.agent import QAgent
from Apex.src.ai.forecasting.sentiment_analysis import SentimentAnalyzer
from Apex.src.ai.ensembles.ensemble_coordinator import register_session_analyzer, update_model_weights
from Apex.src.ai.analysis.options_flow_analyzer import OptionsFlowAnalyzer
from Apex.src.ai.transformers.regime_forecaster import RegimeForecaster
from Apex.src.Core.trading.execution.algo_execution import AlgoExecutionRouter
from Apex.src.ai.forecasting.lstm_flow_predictor import LSTMFlowPredictor
from Apex.src.ai.ensembles.meta_trader import MetaTrader
from Apex.src.ai.models.model_weight_optimizer import ModelWeightOptimizer
from Apex.src.Core.data.event_calendar import EconomicCalendar

# Additional imports for integration fixes
from Apex.src.Core.trading.execution.order_execution import OrderExecutionManager
from Apex.src.Core.trading.risk.risk_management import RiskManager
from Apex.src.ai.ensembles.ensemble_voting import EnsembleVotingSystem
from Apex.src.Core.trading.risk.portfolio_manager import PositionSizer
from Apex.src.Core.trading.execution.market_impact import ExecutionTimingOptimizer
from Apex.src.ai.analysis.market_regime_classifier import update_regime_classification

# Performance-optimized constants
MAX_HISTORY_SIZE = 1000  # Increased for better pattern recognition
ANOMALY_THRESHOLD = 0.05
LIQUIDITY_COLLAPSE_THRESHOLD = 0.2
PARALLELISM_LIMIT = 12  # Increased parallelism for better throughput
CACHE_SIZE = 200  # Increased cache size for better hit rates
HIGH_VOLATILITY_THRESHOLD = 2.0
MAX_SESSION_TRANSITIONS = 20  # For tracking recent session transitions
ORDER_BOOK_DEPTH = 10  # Increased from 5 for better liquidity analysis
INST_PRESENCE_THRESHOLD = 0.7

# Session types enum for vectorized operations - added overlapping sessions
SESSION_TYPES = {
    "PRE_MARKET": 0,
    "REGULAR_HOURS": 1,
    "POST_MARKET": 2,
    "OVERNIGHT": 3,
    "WEEKEND": 4,
    "HOLIDAY": 5,
    "CRYPTO_24H": 6,
    "FOREX_ASIAN": 7,
    "FOREX_LONDON": 8,
    "FOREX_NY": 9,
    "LONDON_NY_OVERLAP": 10,  # High liquidity period
    "SYDNEY_TOKYO_OVERLAP": 11,  # Important Asian crossover
    "ASIA_EUROPE_OVERLAP": 12,  # Transition period
}

# Intra-session dynamics for fine-grained classification
INTRA_SESSION_TYPES = {
    "OPENING_MOMENTUM": 0,
    "MIDDAY_LULL": 1,
    "CLOSING_AUCTION": 2,
    "LUNCH_LIQUIDITY_DROP": 3,
    "POWER_HOUR": 4,
    "OVERNIGHT_DRIFT": 5,
    "PRE_NEWS_POSITIONING": 6,
    "POST_NEWS_VOLATILITY": 7,
}

# Market regimes for session-specific strategy adaptation
MARKET_REGIMES = {
    "TRENDING_UP": 0,
    "TRENDING_DOWN": 1,
    "RANGE_BOUND": 2,
    "BREAKOUT": 3,
    "REVERSAL": 4,
    "HIGH_VOLATILITY": 5,
    "LOW_VOLATILITY": 6,
    "MEAN_REVERSION": 7,
    "RISK_ON": 8,
    "RISK_OFF": 9,
}

# Execution strategies for different market conditions
EXECUTION_STRATEGIES = {
    "AGGRESSIVE_MARKET": 0,
    "PASSIVE_LIMIT": 1,
    "TWAP": 2,
    "VWAP": 3,
    "ICEBERG": 4,
    "DARK_POOL": 5,
    "SNIPER": 6,
    "PCOMP": 7,  # Participation-based execution
    "ARRIVAL_PRICE": 8,
    "IMPLEMENTATION_SHORTFALL": 9,
}

class SessionDetector:
    """Advanced Market Session Analysis & Adaptive Execution System with ML-based Classification
    
    The SessionDetector is a critical component of Apex that:
    1. Detects market sessions and regimes in real-time
    2. Analyzes order flow and liquidity patterns
    3. Identifies institutional activity and adapts execution
    4. Manages AI model weighting based on market conditions
    5. Provides optimal execution parameters to trading components
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = StructuredLogger("SessionDetector")
        
        # Real-time Data Feeds with connection pooling
        self.ws_feed = WebSocketFeed(symbol)
        self.data_feed = DataFeed(symbol)
        self.historic_data = HistoricalData(symbol)
        self.economic_calendar = EconomicCalendar()
        
        # Integrated Apex Components - initialized lazily for performance
        self._regime_classifier = None
        self._liquidity_manager = None
        self._risk_engine = None
        self._strategy_orchestrator = None
        self._broker_router = None
        self._institutional_analyzer = None
        self._sentiment_analyzer = None
        self._rl_agent = None
        self._options_flow_analyzer = None
        self._regime_forecaster = None
        self._algo_execution_router = None
        self._lstm_flow_predictor = None
        self._meta_trader = None
        self._model_weight_optimizer = None
        self._order_flow_analyzer = None
        
        # Integration components - initialized for direct system influence
        self._order_execution_manager = None
        self._risk_manager = None
        self._ensemble_voting_system = None
        self._position_sizer = None
        self._execution_timing_optimizer = None
        
        # Advanced intra-session and overlap detection
        self._setup_session_detection_components()
        
        # Session Analysis Parameters - Vectorized storage (expanded)
        self._initialize_vectorized_storage()
        
        # Real-Time State Tracking with atomic updates
        self.current_session = "REGULAR_HOURS"
        self.current_intra_session = "MIDDAY_LULL"
        self.current_market_regime = "RANGE_BOUND"
        self.current_execution_strategy = "PASSIVE_LIMIT"
        self.last_transition = datetime.utcnow()
        self.anomaly_flags = set()
        
        # Tracking recent transitions for pattern analysis
        self.session_transitions = deque(maxlen=MAX_SESSION_TRANSITIONS)
        
        # Session execution quality tracking (expanded)
        self._initialize_performance_tracking()
        
        # Real-time institutional flow tracking
        self.institutional_presence_level = 0.0
        self.institutional_flow_direction = 0.0  # -1.0 (selling) to 1.0 (buying)
        
        # Gamma exposure and options flow tracking
        self.gamma_exposure = 0.0
        self.options_volume_skew = 0.0  # Put/call ratio deviation
        
        # Market sentiment and risk classification
        self.market_sentiment = 0.0  # -1.0 (bearish) to 1.0 (bullish)
        self.risk_regime = "RISK_ON"  # Risk-on or Risk-off classification
        
        # Enhanced liquidity tracking
        self.liquidity_forecast = np.zeros(24, dtype=np.float32)  # 24-hour forecast
        self.liquidity_imbalance = 0.0  # -1.0 (ask-heavy) to 1.0 (bid-heavy)
        
        # Session-specific volatility characteristics
        self.volatility_regime = "NORMAL"  # Normal, High, Low, Extreme
        self.volatility_forecast = np.zeros(8, dtype=np.float32)  # 8-hour forecast
        
        # Concurrency control with semaphore (increased limit)
        self.semaphore = asyncio.Semaphore(PARALLELISM_LIMIT)
        
        # Performance optimization
        self.session_cache = {}
        self._last_liquidity_profile = None
        self._session_transition_lock = asyncio.Lock()
        
        # Enhanced event handling for high-impact events
        self.pending_economic_events = []
        self.high_impact_event_pending = False
        
        # Direct connections to trading components for real-time influence
        self._setup_trading_component_connections()
        
        # Register with Apex systems and initialize connections
        self._register_with_apex()
        
        # Integration fixes: Add priority queue for execution modifications
        self.execution_modification_queue = asyncio.PriorityQueue()
        
        # Integration fixes: Add risk update cooldown
        self.last_risk_update = datetime.utcnow()
        self.risk_update_cooldown = timedelta(seconds=30)  # Only update risk every 30 seconds
        self.significant_risk_change_threshold = 0.15  # Only update if risk changes by 15%
        self.last_risk_value = 1.0
        
        # Integration fixes: Add liquidity forecast consolidation
        self.consolidated_liquidity_forecast = None
        self.last_liquidity_update = datetime.utcnow()
        self.liquidity_update_cooldown = timedelta(seconds=60)  # Update liquidity forecast every minute
        
        # Integration fixes: Add execution update rate limiting
        self.last_execution_update = datetime.utcnow()
        self.execution_update_cooldown = timedelta(seconds=2)  # Minimum 2 seconds between execution updates
        self.execution_update_priority = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}  # Priority levels
        self.pending_execution_updates = []
        
        # Integration fixes: Add market regime classification smoothing
        self.regime_confidence = {}  # Store confidence levels for regime classifications
        self.regime_confirmation_count = 0  # Count of consecutive confirmations for a regime change
        self.required_regime_confirmations = 2  # Require 2 confirmations before changing regime
        self.regime_ema_factor = 0.8  # EMA factor for smoothing regime transitions
        self.regime_confidence_threshold = 0.7  # 70% confidence required for regime change
        
        # Integration fixes: Add risk update retry mechanism
        self.risk_update_retries = {}  # Track retry attempts for risk updates
        self.max_risk_retries = 3  # Maximum number of retry attempts
        self.risk_retry_delay = 5  # Seconds between retry attempts
        self.risk_update_confirmations = {}  # Track confirmations from risk manager
        
        # Integration fixes: Add model weight update filtering
        self.last_model_weights = {}  # Store last applied model weights
        self.model_weight_change_threshold = 0.05  # 5% threshold for weight updates
        self.model_weight_update_lock = asyncio.Lock()  # Lock for weight updates during volatility
        
        # Integration fixes: Add economic event immediate risk adjustment
        self.economic_event_risk_adjustments = {
            "MAJOR": 0.5,  # 50% risk reduction for major events (Rate, CPI, NFP)
            "MINOR": 0.8,  # 20% risk reduction for minor events (PMI, Retail)
            "REGULAR": 0.9  # 10% risk reduction for regular events
        }
        self.economic_event_alert_window = 60  # Minutes before event to trigger alert
        self.economic_event_immediate_window = 30  # Minutes before event for immediate adjustment
        
    def _setup_session_detection_components(self) -> None:
        """Configure advanced session detection components"""
        # Sessions timing definitions (UTC) - for precise transitions
        self.session_time_ranges = {
            "PRE_MARKET": ((14, 0), (14, 30)),     # 9:30-10:00 ET
            "REGULAR_HOURS": ((14, 30), (21, 0)),  # 9:30-16:00 ET
            "POST_MARKET": ((21, 0), (22, 30)),    # 16:00-17:30 ET
            "OVERNIGHT": ((22, 30), (14, 0)),      # 17:30-9:30 ET
            "FOREX_ASIAN": ((22, 0), (8, 0)),      # 18:00-3:00 ET
            "FOREX_LONDON": ((8, 0), (16, 0)),     # 3:00-11:00 ET
            "FOREX_NY": ((13, 0), (22, 0)),        # 8:00-17:00 ET
            # Overlapping sessions with precise definitions
            "LONDON_NY_OVERLAP": ((13, 0), (16, 0)),  # 8:00-11:00 ET
            "SYDNEY_TOKYO_OVERLAP": ((22, 0), (1, 0)), # 18:00-20:00 ET
            "ASIA_EUROPE_OVERLAP": ((7, 0), (9, 0)),   # 2:00-4:00 ET
        }
        
        # Time-based intra-session definitions for equities
        self.intra_session_times = {
            "OPENING_MOMENTUM": ((14, 30), (15, 0)),    # First 30 mins
            "MIDDAY_LULL": ((17, 0), (19, 0)),          # Midday low volatility
            "LUNCH_LIQUIDITY_DROP": ((16, 30), (17, 30)), # Lunch hour
            "POWER_HOUR": ((20, 0), (21, 0)),           # Last trading hour
            "CLOSING_AUCTION": ((20, 55), (21, 5)),     # Closing auction
        }
        
        # Session-specific market regime biases
        self.session_regime_biases = {
            "PRE_MARKET": ["TRENDING_UP", "TRENDING_DOWN", "BREAKOUT"],
            "REGULAR_HOURS": ["RANGE_BOUND", "TRENDING_UP", "TRENDING_DOWN"],
            "POST_MARKET": ["REVERSAL", "RANGE_BOUND"],
            "OVERNIGHT": ["LOW_VOLATILITY", "MEAN_REVERSION"],
            "LONDON_NY_OVERLAP": ["HIGH_VOLATILITY", "BREAKOUT", "TRENDING_UP", "TRENDING_DOWN"],
            "SYDNEY_TOKYO_OVERLAP": ["RANGE_BOUND", "LOW_VOLATILITY"],
        }
        
        # Session-specific execution strategy preferences
        self.session_execution_strategies = {
            "PRE_MARKET": ["PASSIVE_LIMIT", "ICEBERG"],
            "REGULAR_HOURS": ["AGGRESSIVE_MARKET", "VWAP", "PCOMP"],
            "POST_MARKET": ["PASSIVE_LIMIT", "ICEBERG"],
            "OVERNIGHT": ["PASSIVE_LIMIT"],
            "LONDON_NY_OVERLAP": ["VWAP", "TWAP", "AGGRESSIVE_MARKET"],
            "FOREX_ASIAN": ["PASSIVE_LIMIT", "VWAP"],
        }
        
    def _setup_trading_component_connections(self) -> None:
        """Set up direct connections to trading components for real-time influence"""
        # These connections enable the SessionDetector to directly influence
        # trading decisions and execution parameters across the system
        self.strategy_weight_callbacks = []
        self.execution_param_callbacks = []
        self.risk_adjustment_callbacks = []
        
        # Integration callbacks for direct system influence
        self.order_execution_callbacks = []
        self.risk_management_callbacks = []
        self.ensemble_voting_callbacks = []
        self.strategy_orchestration_callbacks = []
        self.position_sizing_callbacks = []
        self.execution_timing_callbacks = []
        self.meta_trader_callbacks = []
        self.market_regime_callbacks = []
        
        # Placeholder for callback registrations that will be populated
        # when the respective components are initialized
    
    def _initialize_vectorized_storage(self) -> None:
        """Initialize pre-allocated NumPy arrays for maximum performance with expanded tracking"""
        # Pre-allocate arrays for advanced vectorized operations
        self.session_history = np.zeros((MAX_HISTORY_SIZE, 5), dtype=np.float32)  
        # [timestamp, session_id, intra_session_id, market_regime_id, liquidity]
        self.session_history_index = 0
        
        # Volatility tracking with expanded dimensions
        self.volatility_history = np.zeros((MAX_HISTORY_SIZE, 5), dtype=np.float32)  
        # [timestamp, volatility, session_id, intra_session_id, market_regime_id]
        self.volatility_index = 0
        
        # Global market impact tracking (expanded)
        self.global_market_impact = np.zeros(6, dtype=np.float32)  
        # [forex, equities, commodities, crypto, bonds, volatility_indices]
        
        # Order book depth history (increased depth)
        self.depth_history = np.zeros((100, 3), dtype=np.float32)  
        # [bid_depth, ask_depth, bid_ask_imbalance]
        self.depth_index = 0
        
        # Institutional flow tracking
        self.institutional_flow = np.zeros((100, 4), dtype=np.float32)
        # [timestamp, buy_volume, sell_volume, net_flow]
        self.institutional_flow_index = 0
        
        # Options flow and gamma exposure tracking
        self.options_flow = np.zeros((100, 5), dtype=np.float32)
        # [timestamp, call_volume, put_volume, call_put_ratio, gamma_exposure]
        self.options_flow_index = 0
        
        # AI model performance tracking per session
        self.model_performance = np.zeros((len(SESSION_TYPES), 10), dtype=np.float32)
        # [precision, recall, f1, sharpe, sortino, win_rate, avg_profit, avg_loss, max_drawdown, calmar]
        
        # Market regime transition probabilities matrix
        self.regime_transitions = np.zeros((len(MARKET_REGIMES), len(MARKET_REGIMES)), dtype=np.float32)
        
        # Execution strategy performance matrix
        self.execution_performance = np.zeros((len(EXECUTION_STRATEGIES), len(SESSION_TYPES), 4), dtype=np.float32)
        # [fill_rate, price_improvement, latency, market_impact]
    
    def _initialize_performance_tracking(self) -> None:
        """Initialize performance tracking for sessions, regimes, and execution strategies"""
        # Session performance metrics
        self.session_performance = {
            session: {
                "slippage": [], 
                "fill_rate": [],
                "latency": [],
                "market_impact": [],
                "adverse_selection": [],
                "opportunity_cost": [],
                "timing_cost": [],
            } 
            for session in SESSION_TYPES
        }
        
        # Market regime performance metrics
        self.regime_performance = {
            regime: {
                "sharpe": [],
                "sortino": [],
                "win_rate": [],
                "profit_factor": [],
                "avg_return": [],
                "max_drawdown": [],
            }
            for regime in MARKET_REGIMES
        }
        
        # AI model weights per session and regime
        self.model_weights = {
            session: {
                regime: np.ones(10, dtype=np.float32) / 10  # Equal weights initially
                for regime in MARKET_REGIMES
            }
            for session in SESSION_TYPES
        }
    
    # Lazy loading properties for components to improve startup time
    @property
    def regime_classifier(self):
        if self._regime_classifier is None:
            self._regime_classifier = MarketRegimeClassifier(self.symbol)
        return self._regime_classifier
    
    @property
    def liquidity_manager(self):
        if self._liquidity_manager is None:
            self._liquidity_manager = LiquidityManager(self.symbol)
        return self._liquidity_manager
    
    @property
    def risk_engine(self):
        if self._risk_engine is None:
            self._risk_engine = RiskEngine(self.symbol)
        return self._risk_engine
    
    @property
    def strategy_orchestrator(self):
        if self._strategy_orchestrator is None:
            self._strategy_orchestrator = StrategyOrchestrator(self.symbol)
        return self._strategy_orchestrator
    
    @property
    def broker_router(self):
        if self._broker_router is None:
            self._broker_router = BrokerRouter(self.symbol)
        return self._broker_router
    
    @property
    def institutional_analyzer(self):
        if self._institutional_analyzer is None:
            self._institutional_analyzer = InstitutionalClusterAnalyzer(self.symbol)
        return self._institutional_analyzer
    
    @property
    def sentiment_analyzer(self):
        if self._sentiment_analyzer is None:
            self._sentiment_analyzer = SentimentAnalyzer(self.symbol)
        return self._sentiment_analyzer
    
    @property
    def options_flow_analyzer(self):
        if self._options_flow_analyzer is None:
            self._options_flow_analyzer = OptionsFlowAnalyzer(self.symbol)
        return self._options_flow_analyzer
    
    @property
    def regime_forecaster(self):
        if self._regime_forecaster is None:
            self._regime_forecaster = RegimeForecaster(self.symbol)
        return self._regime_forecaster
    
    @property
    def algo_execution_router(self):
        if self._algo_execution_router is None:
            self._algo_execution_router = AlgoExecutionRouter(self.symbol)
        return self._algo_execution_router
    
    @property
    def lstm_flow_predictor(self):
        if self._lstm_flow_predictor is None:
            self._lstm_flow_predictor = LSTMFlowPredictor(self.symbol)
        return self._lstm_flow_predictor
    
    @property
    def meta_trader(self):
        if self._meta_trader is None:
            self._meta_trader = MetaTrader(self.symbol)
        return self._meta_trader
    
    @property
    def model_weight_optimizer(self):
        if self._model_weight_optimizer is None:
            self._model_weight_optimizer = ModelWeightOptimizer()
        return self._model_weight_optimizer
    
    @property
    def order_flow_analyzer(self):
        if self._order_flow_analyzer is None:
            self._order_flow_analyzer = OrderFlowAnalyzer(self.symbol)
        return self._order_flow_analyzer
    
    # Integration components for direct system influence
    @property
    def order_execution_manager(self):
        if self._order_execution_manager is None:
            self._order_execution_manager = OrderExecutionManager(self.symbol)
        return self._order_execution_manager
    
    @property
    def risk_manager(self):
        if self._risk_manager is None:
            self._risk_manager = RiskManager(self.symbol)
        return self._risk_manager
    
    @property
    def ensemble_voting_system(self):
        if self._ensemble_voting_system is None:
            self._ensemble_voting_system = EnsembleVotingSystem(self.symbol)
        return self._ensemble_voting_system
    
    @property
    def position_sizer(self):
        if self._position_sizer is None:
            self._position_sizer = PositionSizer(self.symbol)
        return self._position_sizer
    
    @property
    def execution_timing_optimizer(self):
        if self._execution_timing_optimizer is None:
            self._execution_timing_optimizer = ExecutionTimingOptimizer(self.symbol)
        return self._execution_timing_optimizer
    
    @property
    def rl_agent(self):
        if self._rl_agent is None:
            # Enhanced RL agent for adaptive session/regime classification
            state_dim = 10  # Expanded state dimensions for better classification
            action_dim = len(SESSION_TYPES) + len(MARKET_REGIMES)
            self._rl_agent = QAgent(
                state_dimension=state_dim,
                action_dimension=action_dim,
                learning_rate=0.01,
                discount_factor=0.95,
                exploration_rate=0.1
            )
        return self._rl_agent
    
    def _register_with_apex(self) -> None:
        """Register with Apex core systems for integrated operations"""
        from Apex.src.Core.trading.risk.risk_registry import register_session_monitor
        
        # Register with core systems
        register_session_monitor(self.symbol, self)
        register_session_analyzer(self.symbol, self)
        
        # Set up callback registrations for real-time influence
        if self._meta_trader is not None:
            self.strategy_weight_callbacks.append(self.meta_trader.update_model_weights)
            self.meta_trader_callbacks.append(self.meta_trader.update_session_data)
        
        if self._risk_engine is not None:
            self.risk_adjustment_callbacks.append(self.risk_engine.adjust_risk_parameters)
        
        if self._broker_router is not None:
            self.execution_param_callbacks.append(self.broker_router.update_execution_parameters)
        
        if self._algo_execution_router is not None:
            self.execution_param_callbacks.append(self.algo_execution_router.update_algo_parameters)
        
        # Integration callbacks for direct system influence
        if self._order_execution_manager is not None:
            self.order_execution_callbacks.append(self.order_execution_manager.update_execution_strategy)
            self.order_execution_callbacks.append(self.order_execution_manager.adjust_position_sizing)
        
        if self._risk_manager is not None:
            self.risk_management_callbacks.append(self.risk_manager.adjust_risk_parameters)
            self.risk_management_callbacks.append(self.risk_manager.update_session_risk_profile)
        
        if self._ensemble_voting_system is not None:
            self.ensemble_voting_callbacks.append(self.ensemble_voting_system.update_model_weights)
        
        if self._strategy_orchestrator is not None:
            self.strategy_orchestration_callbacks.append(self.strategy_orchestrator.adapt_to_institutional_flow)
            self.strategy_orchestration_callbacks.append(self.strategy_orchestrator.update_session_strategy)
        
        if self._position_sizer is not None:
            self.position_sizing_callbacks.append(self.position_sizer.adjust_for_session)
            self.position_sizing_callbacks.append(self.position_sizer.adjust_for_liquidity)
        
        if self._execution_timing_optimizer is not None:
            self.execution_timing_callbacks.append(self.execution_timing_optimizer.optimize_based_on_forecast)
        
        if self._regime_classifier is not None:
            self.market_regime_callbacks.append(update_regime_classification)
        
        self.logger.info("SessionDetector registered with core systems", 
                        symbol=self.symbol,
                        components_connected=len(self.strategy_weight_callbacks) + 
                                           len(self.execution_param_callbacks) +
                                           len(self.risk_adjustment_callbacks) +
                                           len(self.order_execution_callbacks) +
                                           len(self.risk_management_callbacks) +
                                           len(self.ensemble_voting_callbacks) +
                                           len(self.strategy_orchestration_callbacks) +
                                           len(self.position_sizing_callbacks) +
                                           len(self.execution_timing_callbacks) +
                                           len(self.meta_trader_callbacks) +
                                           len(self.market_regime_callbacks))

    @handle_exceptions
    async def start_monitoring(self) -> AsyncGenerator[Dict, None]:
        """Real-time market session analysis pipeline with optimized parallelism and enhanced components"""
        # Start LSTM model warmup in background to avoid blocking
        asyncio.create_task(self._warmup_ai_models())
        
        # Initialize high-impact event monitoring
        asyncio.create_task(self._monitor_economic_events())
        
        # Integration fix: Start execution modification processor
        asyncio.create_task(self._process_execution_modifications())
        
        async for market_data in self.ws_feed.stream_market_data():
            if not self._fast_validate_data(market_data):
                continue

            # Process critical tasks concurrently with controlled parallelism
            async with self.semaphore:
                tasks = [
                    self._detect_session_transition(market_data),
                    self._analyze_microstructure(market_data),
                    self._detect_anomalies(market_data),
                    self._track_options_flow(market_data),
                    self._forecast_liquidity_trends(market_data)
                ]
                
                # Execute in parallel with gather
                results = await asyncio.gather(*tasks)
                
                # Extract session transition information from results
                session_changed = results[0]
                
                # If session or regime changed, update connected components
                if session_changed:
                    # Integration fix: Proper sequencing of updates
                    # 1. First update risk parameters
                    await self._update_risk_components()
                    # 2. Then update strategy components
                    await self._update_strategy_components()
                    # 3. Then update execution components
                    await self._update_execution_components()
                    # 4. Finally update meta trader components
                    await self._update_meta_trader_components()
                    # 5. Update all other connected components
                    await self._update_connected_components()
                
                # Less critical tasks - run with lower priority
                asyncio.create_task(self._monitor_global_markets())
                asyncio.create_task(self._update_ai_model_weights())
                asyncio.create_task(self._forecast_market_regimes())
                
                # Integration: Optimize execution timing based on liquidity forecast
                asyncio.create_task(self._optimize_execution_timing())

            # Generate enhanced session report
            yield self._generate_session_report()
    
    async def _warmup_ai_models(self) -> None:
        """Warm up AI models to avoid cold-start latency in critical paths"""
        try:
            # Pre-load and warm up models that will be needed
            if self._lstm_flow_predictor is not None:
                await self._lstm_flow_predictor.warm_up()
            
            if self._regime_forecaster is not None:
                await self._regime_forecaster.initialize_model()
            
            if self._options_flow_analyzer is not None:
                await self._options_flow_analyzer.load_model()
                
            self.logger.info("AI models warmed up successfully", symbol=self.symbol)
        except Exception as e:
            self.logger.error("Error warming up AI models", 
                             symbol=self.symbol, 
                             error=str(e))

    def _fast_validate_data(self, data: Dict) -> bool:
        """Ultra-fast data validation with early exits"""
        # Minimal validation for critical fields
        if not all(k in data for k in ('timestamp', 'bid_price', 'ask_price')):
            return False
            
        # Security validation with early exit
        if not validate_market_session(data):
            return False
            
        # Timestamp anti-spoofing check
        if data['timestamp'] > time.time() + 1:
            return False
            
        return True

    async def _detect_session_transition(self, data: Dict) -> bool:
        """Enhanced session classification with intra-session detection, regime classification,
        and session overlaps - optimized for real-time performance"""
        # Extract expanded state features for classification
        features = self._extract_session_features(data)
        
        # Check cache for faster classification
        cache_key = tuple(np.round(features[:5], 3))  # Use first 5 features for cache key
        
        # Initialize session variables
        new_session = self.current_session
        new_intra_session = self.current_intra_session
        new_market_regime = self.current_market_regime
        session_changed = False
        
        # Fast time-based classification with timestamp
        timestamp = datetime.fromtimestamp(data['timestamp'])
        time_tuple = (timestamp.hour, timestamp.minute)
        
        # First, check exact time-based sessions including overlaps
        for session, (start, end) in self.session_time_ranges.items():
            if self._is_time_in_range(time_tuple, start, end):
                new_session = session
                break
        
        # Now check intra-session times if in regular hours
        if new_session == "REGULAR_HOURS":
            for intra_session, (start, end) in self.intra_session_times.items():
                if self._is_time_in_range(time_tuple, start, end):
                    new_intra_session = intra_session
                    break
        
        # If market regime check not in cache, use AI classification
        if cache_key in self.session_cache:
            new_market_regime = self.session_cache[cache_key]
        else:
            # Use multiple signals for regime classification
            volatility = self.regime_classifier.calculate_volatility(data)
            trend_strength = self.regime_classifier.calculate_trend_strength(data)
            volume_profile = data.get('relative_volume', 1.0)
            
            # ML-based regime classification
            state_vector = np.array(features, dtype=np.float32)
            
            # Get regime classification from transformer model
            if self._regime_forecaster is not None:
                market_regime_id = await self._regime_forecaster.classify_current_regime(state_vector)
                new_market_regime = list(MARKET_REGIMES.keys())[market_regime_id]
                
                # Update cache with LRU-like behavior
                if len(self.session_cache) >= CACHE_SIZE:
                    # Clear oldest entries when cache is full
                    self.session_cache = {k: self.session_cache[k] for k in list(self.session_cache.keys())[-CACHE_SIZE//2:]}
                self.session_cache[cache_key] = new_market_regime
        
        # Check for session or regime transition with lock
        if (new_session != self.current_session or 
            new_intra_session != self.current_intra_session or 
            new_market_regime != self.current_market_regime):
            
            async with self._session_transition_lock:
                # Double-check after acquiring lock
                session_changed = (new_session != self.current_session or 
                                  new_intra_session != self.current_intra_session or
                                  new_market_regime != self.current_market_regime)
                
                if session_changed:
                    await self._handle_session_transition(new_session, new_intra_session, new_market_regime)
        
        # Vectorized update of session history with expanded dimensions
        idx = self.session_history_index % MAX_HISTORY_SIZE
        self.session_history[idx] = [
            data['timestamp'],
            SESSION_TYPES.get(new_session, 0),
            INTRA_SESSION_TYPES.get(new_intra_session, 0),
            MARKET_REGIMES.get(new_market_regime, 0),
            data.get('bid_ask_depth', 0)
        ]
        self.session_history_index += 1
        
        return session_changed

    def _is_time_in_range(self, time_tuple: Tuple[int, int], start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if current time is within the specified range with wrap-around support"""
        hour, minute = time_tuple
        start_hour, start_minute = start
        end_hour, end_minute = end
        
        # Convert to minutes for easier comparison
        current_minutes = hour * 60 + minute
        start_minutes = start_hour * 60 + start_minute
        end_minutes = end_hour * 60 + end_minute
        
        # Handle ranges that cross midnight
        if end_minutes < start_minutes:
            return current_minutes >= start_minutes or current_minutes <= end_minutes
        else:
            return start_minutes <= current_minutes <= end_minutes

    def _extract_session_features(self, data: Dict) -> List[float]:
        """Extract expanded normalized features for session and regime classification"""
        # Time-based features with efficient calculation
        timestamp = datetime.fromtimestamp(data['timestamp'])
        hour_normalized = (timestamp.hour + timestamp.minute / 60.0) / 24.0
        day_normalized = timestamp.weekday() / 6.0
        
        # Market microstructure features
        volume_normalized = min(1.0, secure_float(data.get('volume', 0)) / 
                               max(1, self.historic_data.average_volume(self.symbol)))
        
        spread = (data['ask_price'] - data['bid_price']) / max(0.0001, data['bid_price'])
        spread_normalized = min(1.0, spread / 0.01)  # Normalize to typical max spread
        
        volatility_normalized = min(1.0, self.regime_classifier.current_volatility() / 
                                   max(0.001, self.historic_data.average_volatility(self.symbol)))
        
        # Institutional activity feature
        inst_activity = min(1.0, self.institutional_presence_level)
        
        # Order book imbalance for flow prediction
        book_imbalance = 0.0
        if 'bids' in data and 'asks' in data:
            bid_size = sum(qty for _, qty in data['bids'][:ORDER_BOOK_DEPTH])
            ask_size = sum(qty for _, qty in data['asks'][:ORDER_BOOK_DEPTH])
            if bid_size + ask_size > 0:
                book_imbalance = (bid_size - ask_size) / (bid_size + ask_size)
                
        # Options activity features
        options_skew = min(1.0, max(-1.0, self.options_volume_skew))
        gamma_exposure_norm = min(1.0, max(0.0, self.gamma_exposure / 1000000))
        
        # Economic events and sentiment features
        event_pending = 1.0 if self.high_impact_event_pending else 0.0
        sentiment = min(1.0, max(-1.0, self.market_sentiment))
        
        # Return comprehensive feature vector for session/regime classification
        return [
            hour_normalized,
            day_normalized,
            volume_normalized,
            spread_normalized,
            volatility_normalized,
            inst_activity,
            book_imbalance,
            options_skew,
            gamma_exposure_norm,
            event_pending,
            sentiment
        ]

    async def _handle_session_transition(self, new_session: str, new_intra_session: str, new_market_regime: str) -> None:
        """Handle transitions between sessions and market regimes with enhanced AI model weighting and execution optimization"""
        # Record transition for pattern analysis
        transition = {
            'timestamp': datetime.utcnow(),
            'from_session': self.current_session,
            'to_session': new_session,
            'from_intra_session': self.current_intra_session,
            'to_intra_session': new_intra_session,
            'from_regime': self.current_market_regime,
            'to_regime': new_market_regime,
            'volatility': self.regime_classifier.current_volatility(),
            'liquidity': self._last_liquidity_profile,
            'institutional_presence': self.institutional_presence_level
        }
        
        # Update transition matrix with vectorized operation (for fast regime probability calculation)
        from_idx = MARKET_REGIMES.get(self.current_market_regime, 0)
        to_idx = MARKET_REGIMES.get(new_market_regime, 0)
        self.regime_transitions[from_idx, to_idx] += 1
        
        # Add to transition deque for recent pattern analysis
        self.session_transitions.append(transition)
        
        # Log transition with structured data
        self.logger.info("Session/regime transition detected", 
                        symbol=self.symbol,
                        from_session=self.current_session,
                        to_session=new_session,
                        from_regime=self.current_market_regime,
                        to_regime=new_market_regime,
                        volatility=self.regime_classifier.current_volatility())
        
        # Update current state atomically
        self.current_session = new_session
        self.current_intra_session = new_intra_session
        self.current_market_regime = new_market_regime
        self.last_transition = datetime.utcnow()
        
        # Update execution strategy based on new session and regime
        self._update_execution_strategy(new_session, new_intra_session, new_market_regime)
        
        # Dynamically adjust AI model weights based on session/regime
        await self._adjust_model_weights_for_session(new_session, new_market_regime)
        
        # Update risk parameters for new session/regime
        await self._adjust_risk_parameters_for_session(new_session, new_market_regime)
        
        # Reset anomaly flags for new session
        self.anomaly_flags = set()

    def _update_execution_strategy(self, session: str, intra_session: str, market_regime: str) -> None:
        """Dynamically adjust execution strategy based on session type, liquidity, volatility, and institutional flow."""
        # Get recommended strategies for the session
        recommended_strategies = self.session_execution_strategies.get(session, [])
        
        # If no specific recommendations, use defaults based on regime
        if not recommended_strategies:
            if market_regime in ["HIGH_VOLATILITY", "BREAKOUT"]:
                self.current_execution_strategy = "AGGRESSIVE_MARKET"
            elif market_regime in ["RANGE_BOUND", "LOW_VOLATILITY"]:
                self.current_execution_strategy = "PASSIVE_LIMIT"
            elif market_regime in ["TRENDING_UP", "TRENDING_DOWN"]:
                self.current_execution_strategy = "PCOMP"
            elif market_regime == "MEAN_REVERSION":
                self.current_execution_strategy = "ICEBERG"
            else:
                self.current_execution_strategy = "VWAP"
            return
        
        # Special cases for intra-session adjustments
        if intra_session == "OPENING_MOMENTUM" and market_regime in ["BREAKOUT", "HIGH_VOLATILITY"]:
            self.current_execution_strategy = "AGGRESSIVE_MARKET"
        elif intra_session == "CLOSING_AUCTION":
            self.current_execution_strategy = "ARRIVAL_PRICE"
        elif intra_session == "LUNCH_LIQUIDITY_DROP":
            self.current_execution_strategy = "PASSIVE_LIMIT"
        elif session in ["LONDON_NY_OVERLAP", "FOREX_LONDON"] and self.institutional_presence_level > INST_PRESENCE_THRESHOLD:
            # High institutional activity during liquid periods - use smart algos
            self.current_execution_strategy = "IMPLEMENTATION_SHORTFALL"
        else:
            # Use first recommended strategy as default
            self.current_execution_strategy = recommended_strategies[0]
            
        # New dynamic adjustment logic
        if self.institutional_presence_level > 0.7:
            self.current_execution_strategy = "ICEBERG"  # Stealth execution for institutional trading
        elif self.liquidity_imbalance > 0.6:
            self.current_execution_strategy = "PASSIVE_LIMIT"  # Reduce market impact during illiquid periods
        elif market_regime in ["TRENDING_UP", "BREAKOUT"] and self.gamma_exposure > 500000:
            self.current_execution_strategy = "AGGRESSIVE_MARKET"  # Enter aggressively if conditions are ideal
        elif self.volatility_regime == "HIGH":
            self.current_execution_strategy = "VWAP"  # Volatility is high, so execute with volume-based strategies
        else:
            self.current_execution_strategy = "PCOMP"  # Default to participation-based execution

        self.logger.info("Updated execution strategy",
                     strategy=self.current_execution_strategy,
                     session=session,
                     market_regime=market_regime,
                     institutional_presence=self.institutional_presence_level,
                     liquidity_imbalance=self.liquidity_imbalance,
                     volatility_regime=self.volatility_regime)

    async def _adjust_model_weights_for_session(self, session: str, market_regime: str) -> None:
        """Dynamically adjust AI model weights based on session and market regime"""
        # Get current model weights for this session and regime
        current_weights = self.model_weights.get(session, {}).get(market_regime, np.ones(10) / 10)
        
        # Get model performance metrics for optimization
        performance_metrics = self.model_performance[SESSION_TYPES.get(session, 0)]
        
        # Optimize weights using the model_weight_optimizer
        new_weights = await self.model_weight_optimizer.optimize_weights(
            current_weights, 
            performance_metrics,
            session=session,
            market_regime=market_regime
        )
        
        # Update weights for this session and regime
        if session in self.model_weights and market_regime in self.model_weights[session]:
            self.model_weights[session][market_regime] = new_weights
        
        # Distribute new weights to connected components via registered callbacks
        for callback in self.strategy_weight_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.symbol, new_weights, session, market_regime)
                else:
                    callback(self.symbol, new_weights, session, market_regime)
            except Exception as e:
                self.logger.error("Error updating model weights", 
                                 symbol=self.symbol, 
                                 error=str(e))
        
        # Update ensemble voting system with new weights
        update_model_weights(self.symbol, new_weights, session=session, market_regime=market_regime)

    async def _adjust_risk_parameters_for_session(self, session: str, market_regime: str) -> None:
        """Adjust risk parameters based on session and market regime"""
        # Define risk adjustment factors for sessions and regimes
        session_risk_factors = {
            "PRE_MARKET": 0.8,  # Reduce risk in pre-market
            "REGULAR_HOURS": 1.0,  # Standard risk during regular hours
            "POST_MARKET": 0.7,  # Reduce risk in post-market
            "OVERNIGHT": 0.5,  # Significantly reduce risk overnight
            "LONDON_NY_OVERLAP": 1.2,  # Increase risk during high-liquidity overlap
            "FOREX_ASIAN": 0.9,  # Slightly reduce risk during Asian session
            "FOREX_LONDON": 1.1,  # Slightly increase risk during London session
            "FOREX_NY": 1.0,  # Standard risk during NY forex session
        }
        
        regime_risk_factors = {
            "TRENDING_UP": 1.2,  # Increase risk for trending markets
            "TRENDING_DOWN": 0.9,  # Reduce risk for downtrends
            "RANGE_BOUND": 1.0,  # Standard risk for range-bound markets
            "BREAKOUT": 0.8,  # Reduce risk for breakouts due to uncertainty
            "REVERSAL": 0.7,  # Reduce risk during reversals
            "HIGH_VOLATILITY": 0.6,  # Significantly reduce risk in high volatility
            "LOW_VOLATILITY": 1.3,  # Increase risk in low volatility
            "MEAN_REVERSION": 1.1,  # Slightly increase risk for mean-reversion
            "RISK_ON": 1.2,  # Increase risk during risk-on conditions
            "RISK_OFF": 0.5,  # Significantly reduce risk during risk-off
        }
        
        # Calculate combined risk adjustment factor
        session_factor = session_risk_factors.get(session, 1.0)
        regime_factor = regime_risk_factors.get(market_regime, 1.0)
        
        # Consider institutional presence - reduce risk when institutions are active
        inst_factor = 1.0 - (self.institutional_presence_level * 0.3)
        
        # Consider options activity - adjust risk based on gamma exposure
        gamma_factor = 1.0 - (abs(self.gamma_exposure) * 0.2)
        
        # Calculate final risk adjustment factor
        risk_adjustment = session_factor * regime_factor * inst_factor * gamma_factor
        
        # Apply risk adjustment to connected components via registered callbacks
        for callback in self.risk_adjustment_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.symbol, risk_adjustment, session, market_regime)
                else:
                    callback(self.symbol, risk_adjustment, session, market_regime)
            except Exception as e:
                self.logger.error("Error adjusting risk parameters", 
                                 symbol=self.symbol, 
                                 error=str(e))

    async def _update_connected_components(self) -> None:
        """Update all connected trading components with current session information"""
        # Create comprehensive session state for connected components
        session_state = {
            'symbol': self.symbol,
            'session': self.current_session,
            'intra_session': self.current_intra_session,
            'market_regime': self.current_market_regime,
            'execution_strategy': self.current_execution_strategy,
            'institutional_presence': self.institutional_presence_level,
            'institutional_flow_direction': self.institutional_flow_direction,
            'volatility_regime': self.volatility_regime,
            'liquidity_forecast': self.liquidity_forecast.tolist(),
            'gamma_exposure': self.gamma_exposure,
            'options_volume_skew': self.options_volume_skew,
            'market_sentiment': self.market_sentiment,
            'risk_regime': self.risk_regime,
            'anomalies': list(self.anomaly_flags),
            'high_impact_event_pending': self.high_impact_event_pending
        }
        
        # Update execution parameters in connected components
        for callback in self.execution_param_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.symbol, session_state)
                else:
                    callback(self.symbol, session_state)
            except Exception as e:
                self.logger.error("Error updating connected components", 
                                 symbol=self.symbol, 
                                 error=str(e))

    async def _analyze_microstructure(self, data: Dict) -> None:
        """Analyze market microstructure for institutional activity and liquidity patterns"""
        # Extract order book data for institutional detection
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        # Check for large orders that might indicate institutional activity
        large_orders_detected = False
        institutional_direction = 0.0
        
        if bids and asks:
            # Calculate total volume in order book for reference
            total_bid_volume = sum(qty for _, qty in bids[:ORDER_BOOK_DEPTH])
            total_ask_volume = sum(qty for _, qty in asks[:ORDER_BOOK_DEPTH])
            total_volume = total_bid_volume + total_ask_volume
            
            # Check for large individual orders (potential institutional activity)
            max_bid_size = max((qty for _, qty in bids[:ORDER_BOOK_DEPTH]), default=0)
            max_ask_size = max((qty for _, qty in asks[:ORDER_BOOK_DEPTH]), default=0)
            
            # Calculate bid-ask imbalance for institutional flow direction
            if total_volume > 0:
                imbalance = (total_bid_volume - total_ask_volume) / total_volume
                self.liquidity_imbalance = imbalance
            else:
                imbalance = 0.0
            
            # Update depth history with vectorized operation
            idx = self.depth_index % 100
            self.depth_history[idx] = [total_bid_volume, total_ask_volume, imbalance]
            self.depth_index += 1
            
            # Detect large orders (potential institutional activity)
            if total_volume > 0:
                max_order_ratio = max(max_bid_size / total_volume if total_volume else 0,
                                     max_ask_size / total_volume if total_volume else 0)
                large_orders_detected = max_order_ratio > 0.2  # 20% of visible liquidity
                
                # Determine institutional flow direction
                if large_orders_detected:
                    if max_bid_size > max_ask_size:
                        institutional_direction = max_order_ratio  # Positive for buying
                    else:
                        institutional_direction = -max_order_ratio  # Negative for selling
            
            # Store liquidity profile for session transitions
            self._last_liquidity_profile = {
                'bid_volume': total_bid_volume,
                'ask_volume': total_ask_volume,
                'imbalance': imbalance,
                'max_bid_size': max_bid_size,
                'max_ask_size': max_ask_size,
                'large_orders': large_orders_detected
            }
        
        # Analyze recent trades for institutional footprints if available
        trades = data.get('trades', [])
        
        if trades:
            # Calculate institutional presence from trade sizes
            large_trade_volume = sum(size for price, size, _ in trades if size > 100)
            total_trade_volume = sum(size for _, size, _ in trades)
            
            if total_trade_volume > 0:
                large_trade_ratio = large_trade_volume / total_trade_volume
                
                # Update institutional presence with EMA
                self.institutional_presence_level = 0.9 * self.institutional_presence_level + 0.1 * large_trade_ratio
                
                # Analyze buy/sell pressure
                buy_volume = sum(size for _, size, side in trades if side == 'buy')
                sell_volume = sum(size for _, size, side in trades if side == 'sell')
                
                if buy_volume + sell_volume > 0:
                    buy_sell_ratio = (buy_volume - sell_volume) / (buy_volume + sell_volume)
                    
                    # Update institutional flow direction with EMA
                    self.institutional_flow_direction = 0.9 * self.institutional_flow_direction + 0.1 * buy_sell_ratio
        
        # If institutional analyzer is available, use it for deeper analysis
        if self._institutional_analyzer is not None:
            # Get more precise institutional activity detection
            institutional_data = await self._institutional_analyzer.detect_activity(data)
            
            if institutional_data:
                # Update institutional flow tracking with vectorized operation
                idx = self.institutional_flow_index % 100
                self.institutional_flow[idx] = [
                    data['timestamp'],
                    institutional_data.get('buy_volume', 0),
                    institutional_data.get('sell_volume', 0),
                    institutional_data.get('net_flow', 0)
                ]
                self.institutional_flow_index += 1
                
                # Update institutional presence and flow direction
                if 'presence_level' in institutional_data:
                    self.institutional_presence_level = institutional_data['presence_level']
                
                if 'flow_direction' in institutional_data:
                    self.institutional_flow_direction = institutional_data['flow_direction']
        
        # Adjust execution parameters based on institutional presence
        if large_orders_detected or self.institutional_presence_level > INST_PRESENCE_THRESHOLD:
            # If institutions are present, adjust execution strategy for stealth
            if self.current_execution_strategy == "AGGRESSIVE_MARKET":
                # Switch to more stealthy strategy
                self.current_execution_strategy = "ICEBERG"
            
            elif self.current_execution_strategy == "PASSIVE_LIMIT":
                # Switch to more adaptive strategy
                self.current_execution_strategy = "PCOMP"
            
            # Trigger immediate update of connected components
            asyncio.create_task(self._update_connected_components())

    async def _detect_anomalies(self, data: Dict) -> None:
        """Detect market structure anomalies, liquidity collapse, and unusual institutional activity"""
        anomalies_detected = False
        
        # Extract volatility from data
        volatility = self.regime_classifier.calculate_volatility(data)
        
        # Check for volatility anomalies
        if volatility > HIGH_VOLATILITY_THRESHOLD:
            self.anomaly_flags.add("VOLATILITY_SPIKE")
            self.volatility_regime = "HIGH"
            anomalies_detected = True
        
        # Check for liquidity anomalies
        if self._last_liquidity_profile:
            # Calculate normal liquidity levels for this session
            avg_liquidity = self.liquidity_manager.get_average_liquidity(
                self.symbol, 
                self.current_session, 
                self.current_intra_session
            )
            
            current_liquidity = self._last_liquidity_profile.get('bid_volume', 0) + \
                              self._last_liquidity_profile.get('ask_volume', 0)
            
            # Detect liquidity collapse
            if avg_liquidity > 0 and current_liquidity / avg_liquidity < LIQUIDITY_COLLAPSE_THRESHOLD:
                self.anomaly_flags.add("LIQUIDITY_COLLAPSE")
                anomalies_detected = True
            
            # Detect extreme order book imbalance
            if abs(self._last_liquidity_profile.get('imbalance', 0)) > 0.7:
                imbalance_type = "BID_HEAVY" if self._last_liquidity_profile.get('imbalance', 0) > 0 else "ASK_HEAVY"
                self.anomaly_flags.add(f"ORDER_BOOK_IMBALANCE_{imbalance_type}")
                anomalies_detected = True
        
        # Detect gamma squeeze conditions if options data available
        if self.gamma_exposure > 1000000 and volatility > HIGH_VOLATILITY_THRESHOLD * 0.7:
            self.anomaly_flags.add("POTENTIAL_GAMMA_SQUEEZE")
            anomalies_detected = True
        
        # Detect unusual institutional activity
        if self.institutional_presence_level > 0.8 and abs(self.institutional_flow_direction) > 0.7:
            flow_type = "BUYING" if self.institutional_flow_direction > 0 else "SELLING"
            self.anomaly_flags.add(f"HEAVY_INSTITUTIONAL_{flow_type}")
            anomalies_detected = True
        
        # Track volatility history with expanded dimensions
        idx = self.volatility_index % MAX_HISTORY_SIZE
        self.volatility_history[idx] = [
            data['timestamp'],
            volatility,
            SESSION_TYPES.get(self.current_session, 0),
            INTRA_SESSION_TYPES.get(self.current_intra_session, 0),
            MARKET_REGIMES.get(self.current_market_regime, 0)
        ]
        self.volatility_index += 1
        
        # If anomalies detected, update risk parameters and notify connected components
        if anomalies_detected:
            # Adjust risk parameters based on anomalies
            risk_multiplier = 0.5 if "LIQUIDITY_COLLAPSE" in self.anomaly_flags else 0.8
            
            # Apply risk adjustments to connected components
            for callback in self.risk_adjustment_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(self.symbol, risk_multiplier, self.current_session, self.current_market_regime)
                    else:
                        callback(self.symbol, risk_multiplier, self.current_session, self.current_market_regime)
                except Exception as e:
                    self.logger.error("Error adjusting risk for anomalies", 
                                     symbol=self.symbol, 
                                     error=str(e))
            
            # Log anomalies for monitoring
            self.logger.warning("Market anomalies detected", 
                               symbol=self.symbol,
                               anomalies=list(self.anomaly_flags),
                               session=self.current_session,
                               intra_session=self.current_intra_session)
            
            # Update connected components
            asyncio.create_task(self._update_connected_components())

    async def _track_options_flow(self, data: Dict) -> None:
        """Track options flow and gamma exposure for enhanced market regime understanding"""
        # Skip if options data not available or options flow analyzer not initialized
        if 'options' not in data or self._options_flow_analyzer is None:
            return
        
        options_data = data['options']
        
        # Analyze options data for flow and gamma exposure
        options_analysis = await self._options_flow_analyzer.analyze_flow(options_data)
        
        if options_analysis:
            # Update gamma exposure with smoothing
            self.gamma_exposure = 0.9 * self.gamma_exposure + 0.1 * options_analysis.get('gamma_exposure', 0)
            
            # Update options volume skew (put/call ratio deviation)
            self.options_volume_skew = options_analysis.get('volume_skew', 0)
            
            # Track options flow history
            idx = self.options_flow_index % 100
            self.options_flow[idx] = [
                data['timestamp'],
                options_analysis.get('call_volume', 0),
                options_analysis.get('put_volume', 0),
                options_analysis.get('call_put_ratio', 1.0),
                self.gamma_exposure
            ]
            self.options_flow_index += 1
            
            # Check for unusual options activity that might precede market moves
            if abs(options_analysis.get('volume_skew', 0)) > 0.5:
                skew_type = "CALLS" if options_analysis.get('volume_skew', 0) > 0 else "PUTS"
                self.anomaly_flags.add(f"UNUSUAL_OPTIONS_VOLUME_{skew_type}")
                
                # Log unusual options activity
                self.logger.info("Unusual options activity detected", 
                                symbol=self.symbol,
                                skew_type=skew_type,
                                volume_skew=options_analysis.get('volume_skew', 0),
                                gamma_exposure=self.gamma_exposure)

    async def _forecast_liquidity_trends(self, data: Dict) -> None:
        """Forecast liquidity trends for upcoming sessions using LSTM model"""
        # Skip if LSTM flow predictor not initialized
        if self._lstm_flow_predictor is None:
            return
        
        # Extract current liquidity data
        if self._last_liquidity_profile:
            current_liquidity = {
                'timestamp': data['timestamp'],
                'bid_volume': self._last_liquidity_profile.get('bid_volume', 0),
                'ask_volume': self._last_liquidity_profile.get('ask_volume', 0),
                'imbalance': self._last_liquidity_profile.get('imbalance', 0),
                'spread': data.get('ask_price', 0) - data.get('bid_price', 0),
                'session': SESSION_TYPES.get(self.current_session, 0),
                'intra_session': INTRA_SESSION_TYPES.get(self.current_intra_session, 0)
            }
            
            # Get liquidity forecast for next 24 hours
            forecast = await self._lstm_flow_predictor.forecast_liquidity(current_liquidity)
            
            if forecast is not None:
                # Update liquidity forecast
                self.liquidity_forecast = forecast
                
                # Log significant liquidity events in forecast
                min_liquidity = np.min(forecast)
                min_liquidity_hour = np.argmin(forecast)
                max_liquidity = np.max(forecast)
                max_liquidity_hour = np.argmax(forecast)
                
                if max_liquidity > 2.0:  # Significant liquidity spike
                    self.logger.info("High liquidity forecasted", 
                                    symbol=self.symbol,
                                    hours_ahead=max_liquidity_hour,
                                    liquidity_level=float(max_liquidity))
                
                if min_liquidity < 0.5:  # Significant liquidity drop
                    self.logger.info("Low liquidity forecasted", 
                                    symbol=self.symbol,
                                    hours_ahead=min_liquidity_hour,
                                    liquidity_level=float(min_liquidity))

    async def _monitor_economic_events(self) -> None:
        """Monitor high-impact economic events that may affect market sessions"""
        while True:
            try:
                # Get upcoming high-impact events for next 24 hours
                events = await self.economic_calendar.get_high_impact_events(hours_ahead=24)
                
                if events:
                    # Update pending events
                    self.pending_economic_events = events
                    
                    # Check for imminent high-impact events (within 1 hour)
                    imminent_events = [e for e in events if e['time_until_minutes'] <= 60]
                    
                    if imminent_events:
                        self.high_impact_event_pending = True
                        
                        # Log imminent events
                        self.logger.info("High-impact economic events upcoming", 
                                        symbol=self.symbol,
                                        events=[e['name'] for e in imminent_events],
                                        minutes_until=[e['time_until_minutes'] for e in imminent_events])
                        
                        # Adjust risk parameters for high-impact events
                        for callback in self.risk_adjustment_callbacks:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(self.symbol, 0.7, "HIGH_IMPACT_EVENT", self.current_market_regime)
                                else:
                                    callback(self.symbol, 0.7, "HIGH_IMPACT_EVENT", self.current_market_regime)
                            except Exception as e:
                                self.logger.error("Error adjusting risk for economic events", 
                                                 symbol=self.symbol, 
                                                 error=str(e))
                    else:
                        self.high_impact_event_pending = False
                
                # Update connected components if high-impact event status changed
                await self._update_connected_components()
                
                # Check every 5 minutes
                await asyncio.sleep(300)
            except Exception as e:
                self.logger.error("Error monitoring economic events", 
                                 symbol=self.symbol, 
                                 error=str(e))
                await asyncio.sleep(60)  # Retry after a minute

    async def _monitor_global_markets(self) -> None:
        """Monitor global market correlations and update impact matrix"""
        try:
            # Get global market data from data feed
            global_data = await self.data_feed.get_global_indices()
            
            if not global_data:
                return
            
            # Extract normalized returns for global markets
            forex_return = global_data.get('forex_index_return', 0)
            equity_return = global_data.get('equity_index_return', 0)
            commodity_return = global_data.get('commodity_index_return', 0)
            crypto_return = global_data.get('crypto_index_return', 0)
            bond_return = global_data.get('bond_index_return', 0)
            vix_change = global_data.get('vix_change', 0)
            
            # Update global market impact matrix with vectorized operation
            self.global_market_impact = np.array([
                forex_return,
                equity_return,
                commodity_return, 
                crypto_return,
                bond_return,
                vix_change
            ], dtype=np.float32)
            
            # Update market sentiment based on global conditions
            self.market_sentiment = 0.9 * self.market_sentiment + 0.1 * (
                0.3 * np.sign(equity_return) +
                0.2 * np.sign(forex_return) +
                0.2 * np.sign(commodity_return) +
                0.1 * np.sign(crypto_return) +
                0.1 * np.sign(-bond_return) +  # Inverse for bonds
                0.1 * np.sign(-vix_change)     # Inverse for VIX
            )
            
            # Update risk regime based on VIX and global conditions
            if vix_change > 0.1 or (equity_return < -0.01 and bond_return > 0.005):
                self.risk_regime = "RISK_OFF"
            elif vix_change < -0.05 and equity_return > 0.01:
                self.risk_regime = "RISK_ON"
            
            # Detect global risk-off events that might affect trading
            if vix_change > 0.15 and equity_return < -0.02:
                self.anomaly_flags.add("GLOBAL_RISK_OFF_EVENT")
                
                # Update connected components for emergency risk adjustment
                await self._update_connected_components()
                
        except Exception as e:
            self.logger.error("Error monitoring global markets", 
                             symbol=self.symbol, 
                             error=str(e))
    async def _update_ai_model_weights(self) -> None:
        """Update AI model weights based on recent performance in the current session/regime"""
        try:
            # Skip if no significant transitions yet
            if len(self.session_transitions) < 5:
                return
            
            # Calculate model performance for current session and regime
            session_idx = SESSION_TYPES.get(self.current_session, 0)
            regime_idx = MARKET_REGIMES.get(self.current_market_regime, 0)
            
            # Check if meta_trader is initialized
            if self._meta_trader is not None:
                # Get performance metrics for current session/regime
                performance = await self._meta_trader.get_model_performance(
                    self.symbol, 
                    self.current_session, 
                    self.current_market_regime
                )
                
                if performance:
                    # Update model performance matrix
                    self.model_performance_matrix[session_idx, regime_idx] = performance.get('score', 0.5)
                    
                    # Adjust model weights based on performance
                    if hasattr(self, 'model_weights'):
                        # Normalize performance to use as weight adjustment
                        perf_factor = min(max(performance.get('score', 0.5), 0.1), 0.9)
                        
                        # Update weights for this session/regime combination
                        self.model_weights[session_idx, regime_idx] = (
                            0.8 * self.model_weights[session_idx, regime_idx] + 
                            0.2 * perf_factor
                        )
                        
                        # Log weight updates
                        self.logger.debug(
                            "Updated model weights", 
                            symbol=self.symbol,
                            session=self.current_session,
                            regime=self.current_market_regime,
                            new_weight=self.model_weights[session_idx, regime_idx]
                        )
        except Exception as e:
            self.logger.error(
                "Error updating AI model weights",
                symbol=self.symbol,
                session=self.current_session,
                regime=self.current_market_regime,
                error=str(e)
            )
            
        finally:
            # Ensure weights are normalized across all models
            if hasattr(self, 'model_weights'):
                # Normalize weights to sum to 1.0 for each session/regime
                row_sums = np.sum(self.model_weights, axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0  # Avoid division by zero
                self.model_weights = self.model_weights / row_sums
                
                # Schedule next update
                self.last_weight_update = time.time()