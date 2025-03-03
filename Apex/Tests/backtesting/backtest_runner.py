"""
Institutional-Grade Backtesting Engine - Core Component
Location: Apex/Tests/backtesting/backtest_runner.py
Integrates with: market_data.py, meta_trader.py, order_execution.py, risk_management.py, performance_metrics.py, 
                 decision_logger.py, historical_data.py, correlation_monitor.py, simulation_engine.py, regime_detector.py
"""

import numpy as np
import pandas as pd
import json
import time
import concurrent.futures
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
from functools import lru_cache

# Core Data Components
from src.Core.data.realtime.market_data import MarketDataAPI
from src.Core.data.historical_data import HistoricalDataLoader
from src.Core.data.correlation_monitor import CorrelationMonitor
from src.Core.data.order_book_analyzer import OrderBookAnalyzer
from src.Core.data.session_detector import MarketSessionDetector

# Core Trading Components
from src.ai.ensembles.meta_trader import MetaTrader
from src.Core.trading.execution.order_execution import OrderExecution
from src.Core.trading.risk.risk_management import RiskEngine
from src.Core.trading.logging.decision_logger import DecisionLogger
from src.Core.trading.strategies.strategy_orchestrator import StrategyOrchestrator
from src.Core.trading.strategies.regime_detection import RegimeDetector

# Metrics and Reporting
from metrics.performance_metrics import PerformanceMetrics

# Backtesting Components
from Tests.backtesting.simulation_engine import SimulationEngine
from Tests.backtesting.report_generator import create_performance_report
from Tests.backtesting.performance_evaluator import PerformanceEvaluator

# Security and Validation
from src.Core.trading.security.security import SecurityValidator

# Utilities
from utils.logging.structured_logger import StructuredLogger
from src.Config.config_loader import load_backtest_config

# Constants
MAX_WORKERS = 8  # For parallel processing
CACHE_SIZE = 128  # For LRU cache

# Structured logger for backtesting
logger = StructuredLogger(__name__)


class BacktestRunner:
    """📌 Institutional-Grade Backtesting Framework
    
    This class orchestrates the entire backtesting process, integrating with various
    components of the Apex trading system to provide high-fidelity simulation of trading
    strategies across multiple timeframes and instruments.
    
    Key features:
    - Multi-asset, multi-strategy backtesting
    - Regime-aware strategy adaptation
    - Walk-forward optimization
    - Realistic market simulation (slippage, liquidity, latency)
    - Comprehensive performance analysis
    - Security validation
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize the backtesting framework with all necessary components.
        
        Args:
            config_path: Optional path to custom configuration file
        """
        # Load configuration
        self.config = load_backtest_config(config_path) if config_path else load_backtest_config()
        
        # Initialize core data components
        self.market_data = MarketDataAPI()
        self.historical_data = HistoricalDataLoader()
        self.correlation_monitor = CorrelationMonitor()
        self.order_book_analyzer = OrderBookAnalyzer()
        self.session_detector = MarketSessionDetector()
        
        # Initialize trading components
        self.meta_trader = MetaTrader(simulated=True)
        self.order_execution = OrderExecution()
        self.risk_engine = RiskEngine()
        self.strategy_orchestrator = StrategyOrchestrator()
        self.regime_detector = RegimeDetector()
        
        # Initialize analysis and logging components
        self.performance_metrics = PerformanceMetrics()
        self.simulation_engine = SimulationEngine()
        self.logger = DecisionLogger()
        self.performance_evaluator = PerformanceEvaluator()
        
        # Security validation
        self.security_validator = SecurityValidator()
        
        # Internal state tracking
        self._last_run_results = None
        self._current_simulation_state = {}
        
        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()
        
        logger.info("BacktestRunner initialized successfully")

    def run_backtest(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1D",
        strategies: Optional[List[str]] = None,
        initial_capital: float = 100000.0,
        slippage_model: str = "adaptive",
        commission_model: str = "tiered",
        enable_parallel: bool = True,
        checkpointing: bool = True,
        detailed_logging: bool = True,
    ) -> Dict[str, Any]:
        """📌 Main function to execute a full backtest across multiple symbols and strategies.
        
        Args:
            symbols: List of asset symbols to backtest
            start: Start datetime for backtest period
            end: End datetime for backtest period
            timeframe: Timeframe for analysis (e.g., "1m", "5m", "1h", "1D")
            strategies: Optional list of strategy names to include
            initial_capital: Starting capital for the backtest
            slippage_model: Model to use for slippage simulation
            commission_model: Model to use for commission calculation
            enable_parallel: Whether to use parallel processing for multi-asset testing
            checkpointing: Whether to save periodic checkpoints for recovery
            detailed_logging: Whether to log detailed information during backtest
            
        Returns:
            Dict containing comprehensive backtest results
        """
        start_time = time.time()
        
        # Input validation
        self._validate_inputs(symbols, start, end, timeframe, strategies)
        
        # Initialize results container
        results = {
            "portfolio": {
                "initial_capital": initial_capital,
                "final_capital": initial_capital,  # Will be updated
                "total_return": 0.0,              # Will be updated
                "sharpe_ratio": 0.0,              # Will be updated
                "max_drawdown": 0.0,              # Will be updated
            },
            "assets": {},
            "trades": [],
            "performance": {},
            "execution_time": 0.0,                # Will be updated
            "metadata": {
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "timeframe": timeframe,
                "slippage_model": slippage_model,
                "commission_model": commission_model,
            }
        }
        
        # Load market regime data for the entire period
        logger.info(f"🔄 Loading market regime data for period {start} to {end}")
        regime_data = self.regime_detector.detect_regimes(symbols, start, end, timeframe)
        
        # Determine strategy set to use
        strategy_set = strategies if strategies else self.config.get("default_strategies", ["momentum", "mean_reversion"])
        logger.info(f"🔄 Using strategy set: {strategy_set}")
        
        # Process assets in parallel or sequentially
        if enable_parallel and len(symbols) > 1:
            logger.info(f"🔄 Running parallel backtest for {len(symbols)} assets from {start} to {end}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(symbols))) as executor:
                # Create tasks for each asset
                future_to_symbol = {
                    executor.submit(
                        self._backtest_single_asset,
                        symbol, 
                        start, 
                        end, 
                        timeframe, 
                        strategy_set,
                        initial_capital / len(symbols),  # Allocate capital per asset
                        slippage_model,
                        commission_model,
                        regime_data.get(symbol, {}),
                        checkpointing,
                        detailed_logging
                    ): symbol for symbol in symbols
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        asset_result = future.result()
                        results["assets"][symbol] = asset_result
                        results["trades"].extend(asset_result["trades"])
                    except Exception as exc:
                        logger.error(f"❌ Asset {symbol} backtest failed with exception: {exc}")
                        logger.error(traceback.format_exc())
        else:
            # Sequential processing
            logger.info(f"🔄 Running sequential backtest for {len(symbols)} assets from {start} to {end}")
            for symbol in symbols:
                try:
                    asset_result = self._backtest_single_asset(
                        symbol, 
                        start, 
                        end, 
                        timeframe, 
                        strategy_set,
                        initial_capital / len(symbols),  # Allocate capital per asset
                        slippage_model,
                        commission_model,
                        regime_data.get(symbol, {}),
                        checkpointing,
                        detailed_logging
                    )
                    results["assets"][symbol] = asset_result
                    results["trades"].extend(asset_result["trades"])
                except Exception as exc:
                    logger.error(f"❌ Asset {symbol} backtest failed with exception: {exc}")
                    logger.error(traceback.format_exc())
        
        # Calculate portfolio-level metrics
        if results["trades"]:
            portfolio_metrics = self.performance_metrics.calculate_portfolio_performance(
                results["trades"], 
                initial_capital,
                start,
                end
            )
            results["portfolio"].update(portfolio_metrics)
        
        # Generate comprehensive performance evaluation
        results["performance"] = self.performance_evaluator.evaluate_backtest_performance(
            results["assets"],
            results["trades"],
            initial_capital,
            start,
            end,
            timeframe
        )
        
        # Record execution time
        execution_time = time.time() - start_time
        results["execution_time"] = execution_time
        logger.info(f"✅ Backtest completed in {execution_time:.2f} seconds")
        
        # Store results for later reference
        self._last_run_results = results
        
        # Generate final report
        final_results = self._generate_final_report(results)
        return final_results

    def _backtest_single_asset(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
        strategies: List[str],
        allocation: float,
        slippage_model: str,
        commission_model: str,
        regime_data: Dict[str, Any],
        enable_checkpointing: bool,
        detailed_logging: bool
    ) -> Dict[str, Any]:
        """Execute backtest for a single asset.
        
        Args:
            symbol: Asset symbol
            start: Start datetime
            end: End datetime
            timeframe: Data timeframe
            strategies: List of strategies to test
            allocation: Capital allocation for this asset
            slippage_model: Slippage simulation model
            commission_model: Commission calculation model
            regime_data: Pre-calculated market regime data
            enable_checkpointing: Whether to save checkpoints
            detailed_logging: Whether to log detailed information
            
        Returns:
            Dict containing asset-specific backtest results
        """
        if detailed_logging:
            logger.info(f"🔄 Running backtest for {symbol} from {start} to {end}")
        
        # Retrieve and validate historical data
        hist_data = self._load_validated_historical_data(symbol, start, end, timeframe)
        
        # Initialize asset-specific results container
        asset_results = {
            "symbol": symbol,
            "metrics": {},
            "trades": [],
            "equity_curve": {},
            "drawdowns": {},
            "position_history": {},
            "regime_transitions": []
        }
        
        # Initialize trading parameters
        current_position = 0
        available_capital = allocation
        trade_history = []
        equity_history = {start: allocation}  # Starting equity
        
        # Execute walk-forward optimization
        for train_data, test_data in self._walk_forward_split(hist_data, regime_data):
            # Checkpoint current state if enabled
            if enable_checkpointing:
                self._save_checkpoint(symbol, available_capital, current_position, trade_history, equity_history)
            
            # Retrain AI models on training data
            train_models = self._retrain_ai_models(symbol, train_data, strategies, regime_data)
            
            # Execute trades on test data
            for idx, row in test_data.iterrows():
                # Detect current market regime
                current_regime = regime_data.get(idx.strftime("%Y-%m-%d"), "normal")
                
                # Check for regime transition
                if asset_results["regime_transitions"] and asset_results["regime_transitions"][-1]["regime"] != current_regime:
                    asset_results["regime_transitions"].append({
                        "timestamp": idx,
                        "regime": current_regime,
                        "previous_regime": asset_results["regime_transitions"][-1]["regime"]
                    })
                elif not asset_results["regime_transitions"]:
                    asset_results["regime_transitions"].append({
                        "timestamp": idx,
                        "regime": current_regime,
                        "previous_regime": None
                    })
                
                # Prepare market data for decision making
                market_state = self._prepare_market_state(symbol, row, current_regime)
                
                # Generate trading decision
                decision = self.meta_trader.generate_signal(
                    market_state=market_state,
                    models=train_models,
                    current_position=current_position,
                    available_capital=available_capital
                )
                
                # Apply risk management rules
                risk_approved = self.risk_engine.approve_trade(
                    decision=decision,
                    current_position=current_position,
                    available_capital=available_capital,
                    market_state=market_state
                )
                
                if not risk_approved:
                    if detailed_logging:
                        logger.debug(f"⚠️ Risk management rejected trade for {symbol} at {idx}")
                    continue
                
                # Simulate trade execution
                trade_result = self.simulation_engine.simulate_trade(
                    decision=decision,
                    market_state=market_state,
                    current_position=current_position,
                    slippage_model=slippage_model,
                    commission_model=commission_model
                )
                
                # Update position and capital
                if trade_result["executed"]:
                    # Update position
                    if trade_result["action"] == "BUY":
                        current_position += trade_result["size"]
                    elif trade_result["action"] == "SELL":
                        current_position -= trade_result["size"]
                    elif trade_result["action"] == "EXIT":
                        current_position = 0
                    
                    # Update capital
                    available_capital -= trade_result["cost"] + trade_result["commission"]
                    
                    # Add trade to history
                    trade_history.append(trade_result)
                    asset_results["trades"].append(trade_result)
                    
                    # Log decision
                    self.logger.log_decision({
                        "symbol": symbol,
                        "timestamp": idx,
                        "action": trade_result["action"],
                        "size": trade_result["size"],
                        "price": trade_result["price"],
                        "cost": trade_result["cost"],
                        "commission": trade_result["commission"],
                        "slippage": trade_result["slippage"],
                        "model_weights": decision.get("model_weights", {}),
                        "risk_params": decision.get("risk_params", {})
                    })
                
                # Update equity history (mark-to-market)
                current_equity = available_capital
                if current_position != 0:
                    current_equity += current_position * row["close"]
                equity_history[idx] = current_equity
        
        # Calculate asset-specific performance metrics
        asset_results["metrics"] = self.performance_metrics.calculate_asset_performance(
            trades=asset_results["trades"],
            equity_curve=equity_history,
            initial_capital=allocation,
            symbol=symbol
        )
        
        # Convert equity history to serializable format
        asset_results["equity_curve"] = {k.strftime("%Y-%m-%d %H:%M:%S"): v for k, v in equity_history.items()}
        
        # Generate drawdown analysis
        asset_results["drawdowns"] = self.performance_metrics.calculate_drawdowns(equity_history)
        
        # Finalize position history
        asset_results["position_history"] = self._calculate_position_history(asset_results["trades"], start, end)
        
        return asset_results

    def _validate_inputs(
        self, 
        symbols: List[str], 
        start: datetime, 
        end: datetime, 
        timeframe: str, 
        strategies: Optional[List[str]]
    ) -> None:
        """Validate input parameters to ensure they meet requirements.
        
        Args:
            symbols: List of asset symbols
            start: Start datetime
            end: End datetime
            timeframe: Data timeframe
            strategies: Optional list of strategy names
            
        Raises:
            ValueError: If validation fails
        """
        # Check symbols
        if not symbols:
            raise ValueError("No symbols provided for backtest")
        
        # Check date range
        if end <= start:
            raise ValueError(f"End date {end} must be after start date {start}")
        
        # Check timeframe
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1D", "1W", "1M"]
        if timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}")
        
        # Check strategies
        if strategies:
            available_strategies = self.strategy_orchestrator.get_available_strategies()
            invalid_strategies = [s for s in strategies if s not in available_strategies]
            if invalid_strategies:
                raise ValueError(f"Invalid strategies: {invalid_strategies}. Available: {available_strategies}")
        
        # Security validation
        self.security_validator.validate_backtest_parameters(symbols, start, end, timeframe, strategies)
        
        logger.info("✅ Input validation passed")

    @lru_cache(maxsize=CACHE_SIZE)
    def _load_validated_historical_data(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime, 
        timeframe: str
    ) -> pd.DataFrame:
        """Load and validate historical data with caching.
        
        Args:
            symbol: Asset symbol
            start: Start datetime
            end: End datetime
            timeframe: Data timeframe
            
        Returns:
            DataFrame of validated historical data
            
        Raises:
            ValueError: If data validation fails
        """
        # Retrieve historical data
        hist_data = self.historical_data.get_historical_data(symbol, start, end, timeframe)
        
        # Validate data before processing
        if hist_data is None or hist_data.empty:
            logger.warning(f"⚠️ No historical data found for {symbol}. Skipping...")
            raise ValueError(f"No historical data found for {symbol}")
        
        # Check for missing values
        if hist_data.isnull().any().any():
            # Attempt to fill missing values
            logger.warning(f"⚠️ Missing values detected in {symbol} data, attempting to fix...")
            hist_data = self._handle_missing_values(hist_data)
        
        # Check for sufficient data points
        min_data_points = 30  # Arbitrary minimum
        if len(hist_data) < min_data_points:
            logger.warning(f"⚠️ Insufficient data points for {symbol}: {len(hist_data)} < {min_data_points}")
            raise ValueError(f"Insufficient data points for {symbol}")
        
        # Security validation
        hist_data = self.security_validator.validate_historical_data(hist_data)
        
        return hist_data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in historical data.
        
        Args:
            data: DataFrame with possibly missing values
            
        Returns:
            DataFrame with missing values handled
        """
        # Forward fill price data
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                data[col] = data[col].ffill()
        
        # Use zero for missing volume
        if 'volume' in data.columns:
            data['volume'] = data['volume'].fillna(0)
        
        # Drop any remaining rows with NaN values
        clean_data = data.dropna()
        
        # Log data cleaning results
        dropped_count = len(data) - len(clean_data)
        if dropped_count > 0:
            logger.warning(f"⚠️ Dropped {dropped_count} rows with missing values")
        
        return clean_data

    def _walk_forward_split(
        self, 
        data: pd.DataFrame, 
        regime_data: Dict[str, Any] = None,
        train_ratio: float = 0.7, 
        window_size: int = 30,
        min_train_size: int = 100,
        max_train_size: int = 1000
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Implements walk-forward optimization with regime awareness.
        
        Args:
            data: Full historical dataset
            regime_data: Market regime information
            train_ratio: Ratio of data to use for training (default: 0.7)
            window_size: Size of walk-forward window in bars (default: 30)
            min_train_size: Minimum training window size (default: 100)
            max_train_size: Maximum training window size (default: 1000)
            
        Yields:
            Tuple of (train_data, test_data) for each walk-forward window
        """
        # Ensure minimum viable training size
        if len(data) < min_train_size + window_size:
            logger.warning(f"⚠️ Insufficient data for walk-forward: {len(data)} < {min_train_size + window_size}")
            yield data, data  # Return full dataset as both train and test if insufficient
            return
        
        # Initial split
        train_size = max(min_train_size, min(max_train_size, int(len(data) * train_ratio)))
        
        # Walk forward
        for i in range(train_size, len(data) - window_size + 1, window_size):
            # Define window boundaries
            train_end = i
            test_end = min(i + window_size, len(data))
            
            # Check for regime transitions within window
            window_has_transition = False
            if regime_data:
                # Convert indices to strings for regime lookup
                window_dates = [d.strftime("%Y-%m-%d") for d in data.index[i:test_end]]
                regimes = [regime_data.get(d, "normal") for d in window_dates]
                window_has_transition = len(set(regimes)) > 1
            
            # Adjust window size if there's a regime transition
            if window_has_transition:
                # Find first regime transition
                prev_regime = None
                for j, date in enumerate(window_dates):
                    regime = regime_data.get(date, "normal")
                    if prev_regime and regime != prev_regime:
                        # Adjust test_end to include only up to regime transition
                        test_end = i + j
                        break
                    prev_regime = regime
            
            # Extract train and test data
            train_data = data.iloc[:train_end]
            test_data = data.iloc[train_end:test_end]
            
            yield train_data, test_data

    def _retrain_ai_models(
        self, 
        symbol: str,
        train_data: pd.DataFrame, 
        strategies: List[str],
        regime_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Retrains AI models using reinforcement learning for dynamic adaptation.
        
        Args:
            symbol: Asset symbol
            train_data: Training data
            strategies: List of strategies to train
            regime_data: Market regime information
            
        Returns:
            Dict of trained models
        """
        # Detect current market regime based on the end of training data
        current_regime = "normal"
        if regime_data:
            last_date = train_data.index[-1].strftime("%Y-%m-%d")
            current_regime = regime_data.get(last_date, "normal")
        
        logger.debug(f"🧠 Retraining models for {symbol} in {current_regime} regime")
        
        # Initialize strategy models through the orchestrator
        models = self.strategy_orchestrator.initialize_strategy_models(
            symbol=symbol,
            strategies=strategies,
            regime=current_regime
        )
        
        # Train models on historical data
        for strategy_name, model in models.items():
            try:
                # Skip training for certain fast strategies if sufficient data is not available
                if strategy_name in ["scalping", "hft"] and len(train_data) < 1000:
                    logger.debug(f"⚠️ Skipping {strategy_name} training for {symbol} due to insufficient data")
                    continue
                
                # Train model using strategy-specific method
                model = self.strategy_orchestrator.train_strategy_model(
                    strategy_name=strategy_name,
                    model=model,
                    training_data=train_data,
                    regime=current_regime
                )
                
                # Update model in the collection
                models[strategy_name] = model
                
                logger.debug(f"✅ Successfully trained {strategy_name} for {symbol}")
            except Exception as e:
                logger.error(f"❌ Error training {strategy_name} for {symbol}: {str(e)}")
                logger.error(traceback.format_exc())
        
        return models

    def _prepare_market_state(
        self, 
        symbol: str, 
        current_data: pd.Series, 
        regime: str
    ) -> Dict[str, Any]:
        """Prepare comprehensive market state for decision making.
        
        Args:
            symbol: Asset symbol
            current_data: Current market data point
            regime: Current market regime
            
        Returns:
            Dict containing market state information
        """
        # Basic market data
        market_state = {
            "symbol": symbol,
            "timestamp": current_data.name,
            "open": current_data["open"],
            "high": current_data["high"],
            "low": current_data["low"],
            "close": current_data["close"],
            "volume": current_data.get("volume", 0),
            "regime": regime
        }
        
        # Add order book data if available
        try:
            order_book = self.order_book_analyzer.get_historical_orderbook(
                symbol=symbol, 
                timestamp=current_data.name
            )
            if order_book:
                market_state["order_book"] = order_book
        except Exception:
            # Order book data is optional
            pass
        
        # Add session information
        try:
            session_info = self.session_detector.detect_session(symbol, current_data.name)
            market_state["session"] = session_info
        except Exception:
            # Session information is optional
            market_state["session"] = "unknown"
        
        # Add correlation data if available
        try:
            correlation_data = self.correlation_monitor.get_correlations(symbol, current_data.name)
            if correlation_data:
                market_state["correlations"] = correlation_data
        except Exception:
            # Correlation data is optional
            pass
        
        return market_state

    def _calculate_position_history(
        self, 
        trades: List[Dict[str, Any]], 
        start: datetime, 
        end: datetime
    ) -> Dict[str, float]:
        """Calculate position history from trades.
        
        Args:
            trades: List of executed trades
            start: Start datetime
            end: End datetime
            
        Returns:
            Dict mapping timestamps to positions
        """
        # Initialize position history
        positions = {start: 0.0}
        current_position = 0.0
        
        # Process trades
        for trade in sorted(trades, key=lambda x: x["timestamp"]):
            timestamp = trade["timestamp"]
            if trade["action"] == "BUY":
                current_position += trade["size"]
            elif trade["action"] == "SELL":
                current_position -= trade["size"]
            elif trade["action"] == "EXIT":
                current_position = 0.0
            
            positions[timestamp] = current_position
        
        # Ensure final position is recorded
        positions[end] = current_position
        
        # Convert to serializable format
        return {k.strftime("%Y-%m-%d %H:%M:%S"): v for k, v in positions.items()}

    def _save_checkpoint(
        self, 
        symbol: str, 
        capital: float, 
        position: float, 
        trades: List[Dict[str, Any]], 
        equity: Dict[datetime, float]
    ) -> None:
        """Save checkpoint for recovery.
        
        Args:
            symbol: Asset symbol
            capital: Current available capital
            position: Current position
            trades: Trade history
            equity: Equity history
        """
        checkpoint = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "capital": capital,
            "position": position,
            "trades_count": len(trades),
            "last_equity": list(equity.values())[-1] if equity else capital
        }
        
        self._current_simulation_state[symbol] = checkpoint
        
        # In a real implementation, this would save to disk or database
        logger.debug(f"💾 Checkpoint saved for {symbol}")

        def _generate_final_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final report.
        
        Args:
            results: Raw backtest results
            
        Returns:
            Dict containing comprehensive report
        """
        # Call the report generator
        report = create_performance_report(results)
        
        # Add execution summary
        report["execution_summary"] = {
            "total_assets_tested": len(results["assets"]),
            "total_trades_executed": len(results["trades"]),
            "execution_time_seconds": results["execution_time"],
            "average_time_per_asset": results["execution_time"] / max(1, len(results["assets"])),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add benchmark comparison if available
        try:
            benchmark_symbol = self.config.get("benchmark_symbol", "SPY")
            benchmark_data = self.historical_data.get_historical_data(
                benchmark_symbol,
                datetime.fromisoformat(results["metadata"]["start_time"]),
                datetime.fromisoformat(results["metadata"]["end_time"]),
                results["metadata"]["timeframe"]
            )
            
            if benchmark_data is not None and not benchmark_data.empty:
                benchmark_return = (benchmark_data["close"].iloc[-1] / benchmark_data["close"].iloc[0] - 1) * 100
                report["benchmark_comparison"] = {
                    "symbol": benchmark_symbol,
                    "return_pct": round(benchmark_return, 2),
                    "sharpe_ratio": self.performance_metrics.calculate_sharpe_ratio(benchmark_data),
                    "max_drawdown": self.performance_metrics.calculate_max_drawdown(benchmark_data["close"]),
                    "correlation_with_portfolio": self.correlation_monitor.asset_correlation(
                        benchmark_data["close"],
                        pd.Series(report["portfolio"]["equity_curve"].values()),
                        window=30
                    )
                }
                
                # Calculate alpha/beta
                portfolio_returns = pd.Series([v for v in report["portfolio"]["equity_curve"].values()]).pct_change().dropna()
                benchmark_returns = benchmark_data["close"].pct_change().dropna()
                report["portfolio"]["alpha"] = round(self.performance_metrics.calculate_alpha(
                    portfolio_returns, benchmark_returns
                ), 4)
                report["portfolio"]["beta"] = round(self.performance_metrics.calculate_beta(
                    portfolio_returns, benchmark_returns
                ), 4)
        except Exception as e:
            logger.warning(f"⚠️ Benchmark comparison failed: {str(e)}")

        # Add detailed risk analysis
        report["risk_analysis"] = {
            "value_at_risk_95": self.performance_metrics.calculate_var(
                pd.Series(report["portfolio"]["equity_curve"].values()), 
                confidence_level=0.95
            ),
            "conditional_var_95": self.performance_metrics.calculate_cvar(
                pd.Series(report["portfolio"]["equity_curve"].values()), 
                confidence_level=0.95
            ),
            "portfolio_volatility": self.performance_metrics.calculate_volatility(
                pd.Series(report["portfolio"]["equity_curve"].values())
            ),
            "worst_day_return": self.performance_metrics.worst_daily_return(
                pd.Series(report["portfolio"]["equity_curve"].values())
            ),
            "best_day_return": self.performance_metrics.best_daily_return(
                pd.Series(report["portfolio"]["equity_curve"].values())
            )
        }

        # Add strategy performance breakdown
        strategy_performance = {}
        for trade in results["trades"]:
            strategy = trade.get("strategy", "unknown")
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {
                    "total_trades": 0,
                    "profitable_trades": 0,
                    "total_pnl": 0.0,
                    "max_drawdown": 0.0
                }
            
            strategy_performance[strategy]["total_trades"] += 1
            strategy_performance[strategy]["total_pnl"] += trade["pnl"]
            if trade["pnl"] > 0:
                strategy_performance[strategy]["profitable_trades"] += 1

        # Calculate strategy-specific metrics
        for strategy, metrics in strategy_performance.items():
            metrics["win_rate"] = metrics["profitable_trades"] / metrics["total_trades"] if metrics["total_trades"] > 0 else 0
            metrics["avg_pnl_per_trade"] = metrics["total_pnl"] / metrics["total_trades"] if metrics["total_trades"] > 0 else 0
        
        report["strategy_performance"] = strategy_performance

        # Add regime analysis
        regime_stats = defaultdict(lambda: {"total_trades": 0, "profitable_trades": 0})
        for asset in results["assets"].values():
            for transition in asset.get("regime_transitions", []):
                regime = transition["regime"]
                regime_stats[regime]["total_trades"] += 1
                # Find trades in this regime period
                regime_trades = [t for t in asset["trades"] 
                               if t["timestamp"] >= transition["timestamp"] and 
                               (transition.get("next_transition") is None or 
                                t["timestamp"] < transition.get("next_transition"))]
                regime_stats[regime]["profitable_trades"] += sum(1 for t in regime_trades if t["pnl"] > 0)
        
        for regime, stats in regime_stats.items():
            stats["win_rate"] = stats["profitable_trades"] / stats["total_trades"] if stats["total_trades"] > 0 else 0
        
        report["regime_analysis"] = dict(regime_stats)

        # Add execution quality analysis
        report["execution_quality"] = {
            "total_slippage": sum(t.get("slippage", 0) for t in results["trades"]),
            "total_commission": sum(t.get("commission", 0) for t in results["trades"]),
            "avg_fill_time": np.mean([t.get("fill_time", 0) for t in results["trades"]]) if results["trades"] else 0,
            "percent_limit_orders": len([t for t in results["trades"] if t.get("order_type") == "LIMIT"]) / len(results["trades"]) if results["trades"] else 0
        }

        # Serialize datetime objects
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")

        return json.loads(json.dumps(report, default=json_serializer))