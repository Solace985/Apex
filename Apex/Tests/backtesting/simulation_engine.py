import numpy as np
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Union
import time
from datetime import datetime, timedelta

# Core imports - maintain modular architecture
from src.Core.trading.execution.order_execution import OrderExecution
from src.Core.trading.risk.risk_management import RiskManager
from src.Core.trading.execution.market_impact import MarketImpactCalculator
from src.Core.data.realtime.market_data import MarketDataHandler
from src.Core.data.realtime.websocket_handler import MarketDataWebSocket
from src.Core.trading.hft.liquidity_manager import LiquidityAnalyzer
from src.Core.trading.strategies.strategy_orchestrator import StrategyOrchestrator
from src.Core.data.correlation_monitor import CorrelationAnalyzer
from src.Core.data.order_book_analyzer import OrderBookAnalyzer
from utils.analytics.monte_carlo_simulator import AdvancedMonteCarlo
from src.ai.analysis.market_regime_classifier import MarketRegimeClassifier
from src.ai.forecasting.synthetic_market import SyntheticMarketGenerator
from src.ai.forecasting.lstm_model import LSTMForecaster
from src.ai.ensembles.meta_trader import MetaTrader
from src.Core.trading.logging.decision_logger import DecisionLogger
from Tests.backtesting.report_generator import InstitutionalReportGenerator


class SimulationEngine:
    """
    High-performance institutional-grade market simulation engine with AI integration
    for backtesting trading strategies with realistic market conditions
    """
    
    def __init__(
        self, 
        strategy, 
        market_data: pd.DataFrame, 
        initial_balance: float = 10000,
        simulation_speed: str = "real-time", 
        enable_stress_test: bool = True,
        log_level: str = "INFO"
    ):
        """
        Initialize simulation engine with advanced parameters
        
        Args:
            strategy: Trading strategy to simulate
            market_data: Historical market data for simulation
            initial_balance: Starting capital
            simulation_speed: "real-time", "accelerated", or "hft"
            enable_stress_test: Whether to run stress tests before simulation
            log_level: Logging verbosity level
        """
        # Core state variables - optimized for minimal memory footprint
        self.strategy = strategy
        self.market_data = market_data
        self.balance = initial_balance
        self.positions = []
        self.trade_history = []
        self.current_timestamp = None
        self.simulation_stats = {
            'total_trades': 0,
            'profitable_trades': 0,
            'execution_latencies': [],
            'slippage_pct': []
        }
        
        # Configuration settings - for dynamic adjustment during simulation
        self.simulation_speed = simulation_speed
        self.enable_stress_test = enable_stress_test
        self.base_execution_latency = 0.002 if simulation_speed == "hft" else 0.05  # 2ms for HFT, 50ms for standard
        self.max_execution_attempts = 3
        self.use_adaptive_latency = True
        self.thread_pool = ThreadPoolExecutor(max_workers=4)  # Thread pool for parallel processing
        
        # Core components - maintaing modular architecture
        self.order_executor = OrderExecution()
        self.risk_manager = RiskManager()
        self.market_impact = MarketImpactCalculator()
        self.market_data_handler = MarketDataHandler()
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.strategy_orchestrator = StrategyOrchestrator()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.monte_carlo = AdvancedMonteCarlo()
        self.websocket_handler = MarketDataWebSocket()
        self.order_book_analyzer = OrderBookAnalyzer()
        self.decision_logger = DecisionLogger(level=log_level)
        
        # AI components - isolated from core execution
        self.market_regime_classifier = MarketRegimeClassifier.load_from_db()
        self.synthetic_generator = SyntheticMarketGenerator()
        self.lstm_forecaster = LSTMForecaster()
        self.meta_trader = MetaTrader()
        
        # Cache for optimization
        self._regime_cache = {}
        self._liquidity_cache = {}
        self._order_book_cache = {}
        self._signal_cache = {}
        
        # Optimized data structures for vector operations
        self.signals_vector = np.zeros(len(market_data) if market_data is not None else 0)
        self.execution_quality = np.zeros(len(market_data) if market_data is not None else 0)
        
        self.decision_logger.info("Simulation engine initialized with strategy and market data")

    async def run_simulation(self) -> Dict:
        """
        Run full simulation with AI integration, optimized for low latency
        
        Returns:
            Dict: Complete simulation results and performance metrics
        """
        try:
            self.decision_logger.info("Starting simulation run")
            start_time = time.time()
            
            # Vectorized pre-simulation setup for performance
            await self._pre_simulation_setup()
            
            # Main simulation loop - optimized for minimum overhead
            trade_count = 0
            signal_generation_time = 0
            execution_time = 0
            
            async for tick in self._efficient_market_data_stream():
                self.current_timestamp = tick.get('timestamp', datetime.now())
                
                # Measure signal generation performance
                signal_start = time.time()
                signal = await self._generate_ai_signal(tick)
                signal_generation_time += time.time() - signal_start
                
                # Execute trade with latency simulation if signal exists
                if signal and signal.action != "HOLD":
                    execution_start = time.time()
                    execution_result = await self._execute_optimized_order(signal, tick)
                    execution_time += time.time() - execution_start
                    
                    if execution_result:
                        trade_count += 1
                        # Update portfolio state after successful execution
                        self._update_portfolio(execution_result)
                
                # Periodic risk assessment (not on every tick for efficiency)
                if trade_count % 10 == 0:
                    await self._adaptive_risk_management()
            
            # Post-simulation analysis and reporting
            simulation_time = time.time() - start_time
            performance_metrics = await self._post_simulation_analysis()
            
            # Performance logging
            self.decision_logger.info(f"Simulation completed in {simulation_time:.2f}s")
            self.decision_logger.info(f"Signal generation avg: {signal_generation_time/max(1, trade_count)*1000:.2f}ms")
            self.decision_logger.info(f"Order execution avg: {execution_time/max(1, trade_count)*1000:.2f}ms")
            
            return {
                'trade_history': self.trade_history,
                'performance': performance_metrics,
                'execution_stats': {
                    'signal_latency_ms': signal_generation_time/max(1, trade_count)*1000,
                    'execution_latency_ms': execution_time/max(1, trade_count)*1000,
                    'total_trades': trade_count,
                    'simulation_time_s': simulation_time
                }
            }
            
        except Exception as e:
            self.decision_logger.error(f"Simulation error: {str(e)}")
            raise

    async def _pre_simulation_setup(self):
        """Optimized pre-simulation setup with parallel processing"""
        self.decision_logger.info("Setting up simulation environment")
        
        # Run stress tests in parallel if enabled
        if self.enable_stress_test:
            stress_test_task = asyncio.create_task(self._run_advanced_stress_tests())
        
        # Initialize synthetic market data in parallel if needed
        if not self.market_data.empty:
            synthetic_data_task = asyncio.create_task(self._generate_synthetic_data())
        
        # Pre-warm AI models in parallel
        warm_up_tasks = [
            self.meta_trader.initialize(),
            self.lstm_forecaster.warm_up_model(),
            self.market_regime_classifier.preload_regimes()
        ]
        
        # Wait for all initialization tasks to complete
        await asyncio.gather(*warm_up_tasks)
        
        # Collect stress test results if enabled
        if self.enable_stress_test:
            self.stress_test_results = await stress_test_task
            
            # Update risk parameters based on stress tests
            if self.stress_test_results:
                await self.risk_manager.update_risk_parameters(
                    self.stress_test_results.get('risk_parameters', {})
                )
        
        # Get synthetic data if generated
        if not self.market_data.empty and 'synthetic_data_task' in locals():
            self.synthetic_data = await synthetic_data_task

    async def _generate_synthetic_data(self):
        """Generate synthetic market data for enhanced scenario testing"""
        return await self.thread_pool.submit(
            self.synthetic_generator.generate,
            base_data=self.market_data,
            scenarios=100  # Reduced from 1000 for performance, increase if needed
        )

    async def _efficient_market_data_stream(self):
        """
        Optimized market data streaming with minimal overhead
        Yields market ticks based on simulation speed setting
        """
        if self.simulation_speed == "hft":
            # HFT mode uses direct websocket streaming
            self.decision_logger.info("Using HFT data stream")
            async for tick in self.websocket_handler.hft_stream():
                yield tick
        else:
            # Batch processing for non-HFT modes
            batch_size = 100 if self.simulation_speed == "accelerated" else 1
            
            for i in range(0, len(self.market_data), batch_size):
                batch = self.market_data.iloc[i:i+batch_size]
                
                for idx, tick in batch.iterrows():
                    yield tick.to_dict()
                    
                    # Sleep between ticks for real-time simulation
                    if self.simulation_speed == "real-time":
                        await asyncio.sleep(self.base_execution_latency)

    async def _generate_ai_signal(self, tick):
        """
        Generate trading signals using cached AI components
        Uses efficient caching for frequently queried data
        """
        # Check cache for recent signals (within 10 seconds)
        tick_time = tick.get('timestamp', datetime.now())
        cache_key = f"{tick.get('symbol', '')}-{int(tick_time.timestamp()) // 10}"
        
        if cache_key in self._signal_cache:
            return self._signal_cache[cache_key]
            
        # Get current market regime efficiently (using cache)
        regime = await self._get_cached_regime(tick)
        
        # Generate optimized signal through MetaTrader
        signal = await self.meta_trader.generate_signal(
            market_data=tick,
            regime=regime,
            portfolio_state=self._current_portfolio_state()
        )
        
        # Cache the signal result for a short period
        self._signal_cache[cache_key] = signal
        
        # Maintain cache size by removing old entries
        if len(self._signal_cache) > 100:
            oldest_key = list(self._signal_cache.keys())[0]
            self._signal_cache.pop(oldest_key)
            
        return signal

    async def _get_cached_regime(self, tick):
        """Get market regime with efficient caching"""
        ticker = tick.get('symbol', '')
        timestamp = tick.get('timestamp', datetime.now())
        cache_key = f"{ticker}-{timestamp.hour}"
        
        if cache_key not in self._regime_cache:
            regime = await self.market_regime_classifier.current_regime(ticker, timestamp)
            self._regime_cache[cache_key] = regime
            
            # Clean up cache if too large
            if len(self._regime_cache) > 50:
                oldest_key = list(self._regime_cache.keys())[0]
                self._regime_cache.pop(oldest_key)
                
        return self._regime_cache[cache_key]

    async def _execute_optimized_order(self, signal, tick):
        """
        Execute orders with adaptive latency and smart routing
        Optimized for minimum execution time
        """
        # Skip execution if no signal or HOLD
        if not signal or signal.action == "HOLD":
            return None
            
        try:
            # Get real-time liquidity analysis for execution quality
            liquidity = await self._get_cached_liquidity(tick)
            
            # Calculate estimated market impact for this order
            impact = self.market_impact.calculate(
                order_size=signal.size,
                liquidity=liquidity,
                volatility=tick.get('volatility', 0.01)
            )
            
            # Get optimized execution latency based on market conditions
            latency = self._calculate_adaptive_latency(signal, liquidity, tick)
            
            # Simulate execution latency
            if self.simulation_speed != "accelerated":
                await asyncio.sleep(latency)
            
            # HFT-aware order routing logic
            order_book = await self._get_cached_order_book(tick)
            
            # Detect large orders and liquidity walls that might affect execution
            if order_book and self.order_book_analyzer:
                liquidity_walls = self.order_book_analyzer.detect_large_orders(order_book)
                if liquidity_walls:
                    # Adjust signal price based on liquidity walls
                    signal.price = self._adjust_price_for_liquidity(signal, liquidity_walls)
            
            # Execute through institutional-grade order execution
            execution_result = await self.order_executor.execute_institutional_order(
                signal=signal,
                impact_model=impact,
                latency=latency,
                routing_strategy=self._determine_routing_strategy(signal, tick)
            )
            
            # Record execution statistics
            if execution_result:
                self.simulation_stats['total_trades'] += 1
                if execution_result.get('profit', 0) > 0:
                    self.simulation_stats['profitable_trades'] += 1
                self.simulation_stats['execution_latencies'].append(latency)
                self.simulation_stats['slippage_pct'].append(execution_result.get('slippage_pct', 0))
                
                # Add to trade history
                self.trade_history.append({
                    'timestamp': self.current_timestamp,
                    'action': signal.action,
                    'price': execution_result.get('price'),
                    'size': signal.size,
                    'slippage': execution_result.get('slippage_pct', 0),
                    'execution_time_ms': latency * 1000,
                    'market_impact': impact.get('price_impact', 0)
                })
                
            return execution_result
            
        except Exception as e:
            self.decision_logger.error(f"Order execution error: {str(e)}")
            return None

    def _calculate_adaptive_latency(self, signal, liquidity, tick):
        """
        Dynamically calculate execution latency based on market conditions
        Uses efficient vectorized operations for performance
        """
        if not self.use_adaptive_latency:
            return self.base_execution_latency
            
        # Start with base latency for simulation type
        latency = self.base_execution_latency
        
        # Adjust for market volatility
        volatility = tick.get('volatility', 0.01)
        latency *= (1 + volatility * 10)  # Higher volatility = higher latency
        
        # Adjust for liquidity
        if liquidity:
            depth_factor = liquidity.get('depth_factor', 1.0)
            latency *= (2.0 - depth_factor)  # Lower liquidity = higher latency
            
        # Adjust for order urgency
        if hasattr(signal, 'urgency') and signal.urgency == "high":
            latency *= 0.5  # Reduce latency for urgent orders
            
        # Ensure minimum latency floor (2ms for HFT, 20ms for standard)
        min_latency = 0.002 if self.simulation_speed == "hft" else 0.02
        return max(latency, min_latency)

    async def _get_cached_liquidity(self, tick):
        """Efficient liquidity analysis with caching"""
        ticker = tick.get('symbol', '')
        cache_key = f"{ticker}-{self.current_timestamp.minute}"
        
        if cache_key not in self._liquidity_cache:
            liquidity = await self.liquidity_analyzer.analyze(tick)
            self._liquidity_cache[cache_key] = liquidity
            
            # Clean up cache if too large
            if len(self._liquidity_cache) > 50:
                oldest_key = list(self._liquidity_cache.keys())[0]
                self._liquidity_cache.pop(oldest_key)
                
        return self._liquidity_cache[cache_key]

    async def _get_cached_order_book(self, tick):
        """Efficient order book access with caching"""
        ticker = tick.get('symbol', '')
        cache_key = f"{ticker}-{self.current_timestamp.second // 5}"  # Cache per 5 seconds
        
        if cache_key not in self._order_book_cache:
            order_book = await self.market_data_handler.get_order_book(ticker)
            self._order_book_cache[cache_key] = order_book
            
            # Clean up cache if too large
            if len(self._order_book_cache) > 20:
                oldest_key = list(self._order_book_cache.keys())[0]
                self._order_book_cache.pop(oldest_key)
                
        return self._order_book_cache[cache_key]

    def _determine_routing_strategy(self, signal, tick):
        """
        Determine the optimal order routing strategy based on:
        - Order size
        - Market conditions
        - Execution urgency
        """
        if hasattr(signal, 'size') and signal.size > 1000:
            # Large orders use VWAP/TWAP execution
            return "TWAP"
        elif tick.get('volatility', 0) > 0.05:
            # High volatility uses dark pool routing
            return "DARK_POOL"
        elif hasattr(signal, 'urgency') and signal.urgency == "high":
            # Urgent orders use aggressive execution
            return "IMMEDIATE"
        else:
            # Default to smart routing
            return "SMART"

    def _adjust_price_for_liquidity(self, signal, liquidity_walls):
        """Adjust order price based on detected liquidity walls"""
        if not liquidity_walls:
            return signal.price
            
        # Find nearest liquidity wall
        nearest_wall = min(liquidity_walls, key=lambda x: abs(x - signal.price))
        
        # Adjust price to avoid liquidity walls
        if signal.action == "BUY" and nearest_wall > signal.price:
            return signal.price * 0.995  # Place slightly below liquidity wall
        elif signal.action == "SELL" and nearest_wall < signal.price:
            return signal.price * 1.005  # Place slightly above liquidity wall
            
        return signal.price

    def _update_portfolio(self, execution_result):
        """
        Update portfolio state based on execution results
        Uses vectorized operations for performance
        """
        # Early return if execution failed
        if not execution_result:
            return
            
        # Extract execution details
        action = execution_result.get('action')
        price = execution_result.get('price', 0)
        size = execution_result.get('size', 0)
        commission = execution_result.get('commission', 0)
        
        # Update balance
        if action == "BUY":
            cost = price * size + commission
            self.balance -= cost
            # Add position
            self.positions.append({
                'symbol': execution_result.get('symbol'),
                'entry_price': price,
                'size': size,
                'timestamp': self.current_timestamp
            })
        elif action == "SELL":
            revenue = price * size - commission
            self.balance += revenue
            # Remove or reduce position
            self._reduce_position(execution_result.get('symbol'), size)

    def _reduce_position(self, symbol, size):
        """Efficiently reduce or remove position"""
        remaining_size = size
        
        # Vector-optimize this by using list comprehension
        positions_to_keep = []
        
        for position in self.positions:
            if position['symbol'] == symbol and remaining_size > 0:
                if position['size'] <= remaining_size:
                    # Position completely closed
                    remaining_size -= position['size']
                else:
                    # Position partially closed
                    position['size'] -= remaining_size
                    remaining_size = 0
                    positions_to_keep.append(position)
            else:
                positions_to_keep.append(position)
                
        # Update positions list
        self.positions = positions_to_keep

    async def _adaptive_risk_management(self):
        """Efficient risk management with minimal computational overhead"""
        # Skip frequent risk calculations for performance
        if len(self.trade_history) % 5 != 0:
            return
            
        # Calculate correlation matrix (expensive operation)
        correlations = await self.correlation_analyzer.current_correlations()
        
        # Calculate portfolio risk metrics
        portfolio_risk = await self.risk_manager.calculate_portfolio_risk(
            positions=self.positions,
            correlation_data=correlations
        )
        
        # Apply dynamic risk adjustments if needed
        if portfolio_risk and portfolio_risk.get('var', 0) > self.strategy.max_var:
            # Reduce risk by adjusting strategy parameters
            await self.strategy_orchestrator.adjust_leverage(
                risk_level=portfolio_risk.get('risk_level', 'high'),
                current_exposure=portfolio_risk.get('exposure', 1.0)
            )
            
        # Check for drawdown limits
        current_drawdown = portfolio_risk.get('current_drawdown', 0)
        if current_drawdown > self.strategy.max_drawdown:
            # Temporarily halt trading if drawdown exceeds limits
            self.decision_logger.warning(f"Halting trading due to drawdown: {current_drawdown:.2f}%")
            await self.strategy_orchestrator.pause_trading(
                duration_minutes=30,
                reason=f"Drawdown limit exceeded: {current_drawdown:.2f}%"
            )

    def _current_portfolio_state(self):
        """
        Get current portfolio state for AI decision making
        Returns lightweight reference to avoid copying large objects
        """
        # Create lightweight portfolio snapshot
        portfolio_snapshot = {
            'balance': self.balance,
            'positions_count': len(self.positions),
            'position_symbols': [p['symbol'] for p in self.positions],
            'total_exposure': sum(p['size'] * p['entry_price'] for p in self.positions),
            'timestamp': self.current_timestamp
        }
        
        # Add high-level risk metrics without detailed calculations
        if hasattr(self.risk_manager, 'current_risk_profile'):
            portfolio_snapshot['risk_profile'] = self.risk_manager.current_risk_profile()
        
        return portfolio_snapshot

    async def _run_advanced_stress_tests(self):
        """Run comprehensive stress tests with parallel execution"""
        self.decision_logger.info("Running advanced stress tests")
        
        # Run multiple stress test scenarios in parallel
        stress_results = await self.monte_carlo.run_regime_aware_simulations(
            base_data=self.market_data,
            regimes=self.market_regime_classifier.known_regimes(),
            simulations=1000  # Reduced from 10000 for performance
        )
        
        # Extract key risk parameters
        if stress_results:
            max_drawdown = stress_results.get('max_drawdown', 0)
            var_95 = stress_results.get('var_95', 0)
            
            self.decision_logger.info(f"Stress test results: Max DD={max_drawdown:.2f}%, VaR(95%)={var_95:.2f}%")
            
        return stress_results

    async def _post_simulation_analysis(self):
        """
        Generate comprehensive performance reports
        Returns key performance metrics
        """
        # Generate performance metrics
        win_rate = self.simulation_stats['profitable_trades'] / max(1, self.simulation_stats['total_trades'])
        avg_latency = np.mean(self.simulation_stats['execution_latencies']) if self.simulation_stats['execution_latencies'] else 0
        avg_slippage = np.mean(self.simulation_stats['slippage_pct']) if self.simulation_stats['slippage_pct'] else 0
        
        # Generate trade analysis dataframe
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            
            # Calculate key performance metrics
            total_pnl = trades_df['price'].sum() if 'price' in trades_df else 0
            max_drawdown = self._calculate_max_drawdown(trades_df)
            sharpe_ratio = self._calculate_sharpe_ratio(trades_df)
            
            # Generate comprehensive reports
            report_generator = InstitutionalReportGenerator(
                trades=self.trade_history,
                risk_data=self.risk_manager.history if hasattr(self.risk_manager, 'history') else [],
                market_conditions=self.market_regime_classifier.history() if hasattr(self.market_regime_classifier, 'history') else []
            )
            
            # Generate reports in parallel for performance
            report_tasks = [
                report_generator.generate_json_report(),
                report_generator.generate_dashboard_feed()
            ]
            
            # Only generate PDF in full simulations
            if len(self.trade_history) > 100:
                report_tasks.append(report_generator.generate_pdf_report())
                
            # Gather reports
            reports = await asyncio.gather(*report_tasks)
            
            # Store reports for later access
            self.reports = {
                'json': reports[0],
                'dashboard': reports[1]
            }
            
            if len(reports) > 2:
                self.reports['pdf'] = reports[2]
                
            # Return performance metrics
            return {
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'avg_execution_latency_ms': avg_latency * 1000,
                'avg_slippage_pct': avg_slippage,
                'total_trades': self.simulation_stats['total_trades']
            }
            
        return {
            'win_rate': win_rate,
            'avg_execution_latency_ms': avg_latency * 1000,
            'avg_slippage_pct': avg_slippage,
            'total_trades': self.simulation_stats['total_trades']
        }

    def _calculate_max_drawdown(self, trades_df):
        """Calculate maximum drawdown using vectorized operations"""
        if 'price' not in trades_df or trades_df.empty:
            return 0
            
        # Create cumulative PNL series
        pnl_series = trades_df['price'].cumsum()
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(pnl_series)
        
        # Calculate drawdowns
        drawdowns = (running_max - pnl_series) / running_max
        
        # Return maximum drawdown as percentage
        return float(drawdowns.max() * 100) if not drawdowns.empty else 0

    def _calculate_sharpe_ratio(self, trades_df):
        """Calculate Sharpe ratio using vectorized operations"""
        if 'price' not in trades_df or trades_df.empty:
            return 0
            
        # Extract daily returns
        if 'timestamp' in trades_df:
            trades_df['date'] = pd.to_datetime(trades_df['timestamp']).dt.date
            daily_returns = trades_df.groupby('date')['price'].sum()
        else:
            # If timestamp not available, assume each trade is a separate period
            daily_returns = trades_df['price']
            
        # Calculate annualized Sharpe ratio
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        
        if std_return == 0:
            return 0
            
        # Annualize (assuming 252 trading days)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return float(sharpe)