import logging
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from utils.logging.structured_logger import StructuredLogger
from utils.helpers.error_handler import handle_api_error
from Core.trading.execution.retail_core import market_data_bus

class StrategyEvaluator:
    """Evaluates the performance of different trading strategies dynamically."""

    def __init__(self):
        self.logger = StructuredLogger(__name__)
        self.market_data_bus = market_data_bus

        # Rolling window for strategy performance tracking (last 100 trades)
        self.strategy_performance = {}
        self.trade_window = 100  # Number of trades to track per strategy

        # Thresholds for performance adjustments
        self.performance_decay_factor = 0.98  # Decay factor for older trades
        self.performance_threshold = 0.55  # Threshold to consider a strategy "performing well"
        self.min_trades_for_ranking = 10  # Minimum trades before considering adjustments

    def _update_performance(self, strategy_name: str, trade_outcome: str):
        """
        Updates strategy performance metrics based on trade execution results.
        """
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = deque(maxlen=self.trade_window)

        # Convert trade outcomes to numeric values for evaluation
        outcome_map = {"win": 1, "loss": 0, "break-even": 0.5}
        trade_score = outcome_map.get(trade_outcome, 0.5)  # Default to neutral if unknown

        # Apply decay factor to older trades
        strategy_scores = self.strategy_performance[strategy_name]
        strategy_scores.append(trade_score)

        # Adjust older scores to decay their influence over time
        decayed_scores = [score * self.performance_decay_factor for score in strategy_scores]
        strategy_scores.clear()
        strategy_scores.extend(decayed_scores)

        self.logger.info("Updated strategy performance", strategy=strategy_name, score=trade_score)

    @handle_api_error()
    def update_performance(self):
        """
        Monitors executed trades and updates performance metrics.
        """
        executed_trade = self.market_data_bus.get("last_trade", {})
        if not executed_trade:
            self.logger.info("No recent trades to evaluate.")
            return

        strategy_name = executed_trade["signal"]["strategy"]
        trade_outcome = executed_trade["execution"]["status"]

        self._update_performance(strategy_name, trade_outcome)

    def get_strategy_performance(self) -> Dict[str, float]:
        """
        Returns the latest weighted performance scores for all strategies.
        """
        performance_scores = {}
        for strategy, scores in self.strategy_performance.items():
            if len(scores) >= self.min_trades_for_ranking:
                performance_scores[strategy] = np.mean(scores)

        return performance_scores

    def adjust_strategy_weighting(self) -> Dict[str, float]:
        """
        Adjusts the weighting of strategies based on past performance.
        """
        performance_data = self.get_strategy_performance()
        if not performance_data:
            self.logger.warning("No sufficient trade history to adjust strategy weightings.")
            return {}

        # Normalize scores to get weight adjustments
        max_score = max(performance_data.values(), default=1)
        adjusted_weights = {
            strategy: score / max_score for strategy, score in performance_data.items()
        }

        self.logger.info("Adjusted strategy weightings", weights=adjusted_weights)
        return adjusted_weights

    def monitor_strategy_performance(self):
        """
        Runs periodic evaluations and updates strategy rankings.
        """
        self.update_performance()
        return self.adjust_strategy_weighting()
