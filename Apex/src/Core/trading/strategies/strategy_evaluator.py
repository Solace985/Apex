# src/core/trading/strategies/strategy_evaluator.py
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Deque
from collections import deque
from core.trading.risk.config import load_config  # Load risk and strategy config
from utils.logging.structured_logger import StructuredLogger  # Structured logging
from utils.helpers.error_handler import handle_api_error  # Error handling
from core.trading.execution.retail_core import market_data_bus  # Data flow integration
from core.trading.risk.risk_management import RiskMetrics  # Risk-adjusted performance tracking

class StrategyEvaluator:
    """Evaluates and ranks trading strategies based on real-time market performance and risk metrics."""

    def __init__(self):
        self.logger = StructuredLogger(__name__)
        self.market_data_bus = market_data_bus
        self.config = load_config()['strategy_evaluation']

        # Performance tracking parameters
        self.trade_window = self.config.get('trade_window', 100)
        self.performance_decay = self.config.get('performance_decay', 0.98)
        self.min_trades = self.config.get('min_trades', 10)

        # Strategy performance storage
        self.strategy_performance = {}

        # Risk-adjusted tracking
        self.risk_metrics = RiskMetrics()

    def _validate_trade_data(self, trade_data: Dict) -> bool:
        """Validates the structure of trade data before updating performance metrics."""
        required_keys = {'signal', 'execution', 'timestamp'}
        if not all(key in trade_data for key in required_keys):
            self.logger.warning("Invalid trade data structure")
            return False

        if not isinstance(trade_data['signal'].get('strategy'), str):
            return False

        valid_statuses = {'win', 'loss', 'break-even'}
        if trade_data['execution'].get('status') not in valid_statuses:
            return False

        return True

    @handle_api_error(retries=3)
    def _update_performance(self, strategy: str, outcome: str) -> None:
        """Updates performance scores for a strategy, applying risk-based adjustments."""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                'scores': deque(maxlen=self.trade_window),
                'risk_metrics': self.risk_metrics.initialize_strategy_metrics()
            }

        # Compute risk-adjusted score
        risk_score = self.risk_metrics.calculate_risk_adjusted_return(strategy)
        outcome_value = {'win': 1, 'loss': 0, 'break-even': 0.5}[outcome]
        weighted_score = (outcome_value * 0.7) + (risk_score * 0.3)

        # Apply decay factor to past performance scores
        decayed_scores = [
            score * self.performance_decay
            for score in self.strategy_performance[strategy]['scores']
        ]
        decayed_scores.append(weighted_score)
        self.strategy_performance[strategy]['scores'] = deque(decayed_scores, maxlen=self.trade_window)

        # Update risk metrics for strategy
        self.risk_metrics.update_strategy_metrics(
            strategy,
            self.market_data_bus.get('risk_parameters')
        )

    @handle_api_error()
    def update_performance(self) -> None:
        """Retrieves and updates strategy performance based on executed trades."""
        trade_data = self.market_data_bus.get('last_trade')
        if not trade_data or not self._validate_trade_data(trade_data):
            return

        strategy = trade_data['signal']['strategy']
        outcome = trade_data['execution']['status']
        self._update_performance(strategy, outcome)

    def get_strategy_performance(self) -> Dict[str, float]:
        """Returns average performance scores for all strategies."""
        return {
            strategy: np.mean(data['scores']) if len(data['scores']) >= self.min_trades else 0.5
            for strategy, data in self.strategy_performance.items()
        }

    def adjust_strategy_weighting(self) -> Dict[str, float]:
        """Dynamically adjusts strategy weights based on performance and market conditions."""
        performance = self.get_strategy_performance()
        if not performance:
            return {}

        # Fetch current market regime
        market_regime = self.market_data_bus.get('market_regime', 'neutral')
        regime_weights = self.config.get('regime_weights', {}).get(market_regime, {})

        # Combine strategy performance with regime influence
        combined_weights = {
            strategy: (performance[strategy] * 0.7) + (regime_weights.get(strategy, 0.3) * 0.3)
            for strategy in performance
        }

        # Normalize strategy weights to ensure they sum to 1
        total_weight = sum(combined_weights.values()) + 1e-10  # Avoid division by zero
        normalized_weights = {k: v / total_weight for k, v in combined_weights.items()}

        self.logger.info("Adjusted strategy weightings", weights=normalized_weights)
        return normalized_weights

    def monitor_strategy_performance(self) -> None:
        """Runs the performance update cycle and updates strategy weightings."""
        try:
            self.update_performance()
            strategy_weights = self.adjust_strategy_weighting()
            self.market_data_bus.update({'strategy_weights': strategy_weights})
        except Exception as e:
            self.logger.critical("Performance monitoring failed", error=str(e))
            self.market_data_bus.update({'strategy_weights': {}})
