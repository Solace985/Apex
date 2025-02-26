import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from utils.logging.structured_logger import StructuredLogger
from utils.helpers.error_handler import handle_api_error
from core.trading.execution.broker_manager import BrokerManager
from core.trading.execution.retail_core import market_data_bus
from core.trading.strategies.strategy_evaluator import StrategyEvaluator
from core.trading.strategies.strategy_selector import StrategySelector
from core.trading.risk.risk_management import validate_order
from core.trading.strategies.trend.regime_detection import RegimeDetection
from core.trading.strategies.trend.trend_following import TrendFollowingStrategy
from core.trading.strategies.mean_reversion import MeanReversionStrategy
from core.trading.strategies.trend.momentum_breakout import MomentumBreakoutStrategy

class StrategyOrchestrator:
    """Enhanced Strategy Orchestration for ApexRetail"""

    def __init__(self):
        self.logger = StructuredLogger(__name__)
        self.market_data_bus = market_data_bus
        self.broker_manager = BrokerManager()
        
        # Initialize strategy modules
        self.regime_detector = RegimeDetection()
        self.strategy_selector = StrategySelector()
        self.strategy_evaluator = StrategyEvaluator()

        # Available strategies
        self.strategies = {
            "trending": TrendFollowingStrategy(),
            "mean_reverting": MeanReversionStrategy(),
            "breakout": MomentumBreakoutStrategy()
        }

        self.last_execution_time = datetime.min  # Prevent unnecessary frequent execution

    @handle_api_error(retries=3)
    def select_best_strategy(self) -> Any:
        """Enhanced strategy selection based on multiple factors"""

        market_data = self.market_data_bus.get("latest_market_data", {})
        fundamental_data = self.market_data_bus.get("fundamentals", {})
        regime = self.regime_detector.detect_market_regime(market_data)

        # Get past strategy performance
        performance_data = self.strategy_evaluator.get_strategy_performance()

        # Choose the best strategy based on regime, fundamentals, and past performance
        best_strategy = self.strategy_selector.select_strategy(
            regime=regime,
            performance_data=performance_data,
            market_conditions=market_data,
            fundamentals=fundamental_data
        )

        self.logger.info("Best strategy selected",
                         strategy=best_strategy.__class__.__name__,
                         regime=regime,
                         performance_score=performance_data.get(best_strategy.name, 0))

        return best_strategy

    @handle_api_error()
    def execute_trades(self) -> Optional[Dict[str, Any]]:
        """Executes trade signals after risk validation"""

        current_time = datetime.utcnow()
        if current_time - self.last_execution_time < timedelta(minutes=1):
            self.logger.info("Skipping execution to prevent overtrading")
            return None

        try:
            # ✅ Fetch latest market data and fundamental insights
            market_data = self.market_data_bus.get("latest_market_data", {})
            fundamental_data = self.market_data_bus.get("fundamentals", {})
            best_strategy = self.select_best_strategy()

            # ✅ Generate trade signal
            signal = best_strategy.generate_signal(market_data, fundamental_data)
            if not signal:
                self.logger.info("No actionable trade signal generated.")
                return None

            # ✅ Validate trade signal using risk management
            if not validate_order(signal):
                self.logger.warning("Trade signal rejected due to risk constraints", signal=signal)
                return None

            # ✅ Execute trade through broker manager
            execution_result = self.broker_manager.execute_order(
                symbol=signal["symbol"],
                order_type=signal["order_type"],
                quantity=signal["quantity"],
                strategy=best_strategy.name
            )

            # ✅ Update market data bus with trade details
            self.market_data_bus.update({
                "last_trade": {
                    "signal": signal,
                    "execution": execution_result,
                    "timestamp": current_time.isoformat()
                }
            })

            self.logger.info("Trade executed successfully", execution_result=execution_result)
            self.last_execution_time = current_time

            return execution_result

        except Exception as e:
            self.logger.error("Trade execution failed", error=str(e))
            return None

    def monitor_strategy_performance(self):
        """Tracks real-time performance and updates market data bus"""

        # Fetch executed trades
        executed_trade = self.market_data_bus.get("last_trade", {})
        if not executed_trade:
            self.logger.info("No recent trades to evaluate.")
            return

        strategy_name = executed_trade["signal"]["strategy"]
        trade_outcome = executed_trade["execution"]["status"]

        # Update strategy performance metrics
        self.strategy_evaluator.update_performance(strategy_name, trade_outcome)

        self.logger.info("Updated strategy performance",
                         strategy=strategy_name,
                         trade_outcome=trade_outcome)

    def run_orchestration_cycle(self):
        """Main loop for orchestrating strategies and execution"""

        if not self.market_data_bus.get("latest_market_data"):
            self.logger.warning("Market data is outdated or unavailable.")
            return

        try:
            # ✅ Execute trades based on current strategy
            self.execute_trades()

            # ✅ Monitor strategy performance
            self.monitor_strategy_performance()

        except Exception as e:
            self.logger.critical("Orchestration cycle failed", error=str(e))
