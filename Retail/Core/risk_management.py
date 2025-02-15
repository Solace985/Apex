import logging
from typing import Dict, Any

class RiskManagement:
    """Handles risk evaluation and position sizing."""
    
    def __init__(self, max_drawdown=0.2, capital_exposure_limit=0.1):
        self.max_drawdown = max_drawdown
        self.capital_exposure_limit = capital_exposure_limit
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate_risk(self, trade_details: Dict[str, Any]) -> bool:
        """Rejects trades exceeding risk threshold."""
        risk = trade_details.get("risk", 0)
        if risk > self.capital_exposure_limit:
            self.logger.warning(f"Trade rejected due to high risk: {trade_details}")
            return False
        return True

    def adjust_position_size(self, trade_details: Dict[str, Any]) -> Dict[str, Any]:
        """Adjusts position size dynamically based on risk."""
        risk = trade_details.get("risk", 0.01)
        trade_details["position_size"] = min(1.0, self.capital_exposure_limit / risk)
        return trade_details
    
    def __init__(self):
        self.risk_to_reward_ratio = 2  # Enforce 1:2 risk-reward ratio
        self.max_slippage = 0.5  # Maximum acceptable slippage in %

    def evaluate_trade(self, entry_price, stop_loss, take_profit, slippage):
        """Evaluate trade based on risk-reward ratio & slippage constraints."""
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)

        if reward / risk < self.risk_to_reward_ratio:
            return "REJECTED: Poor Risk-Reward Ratio"
        if slippage > self.max_slippage:
            return "REJECTED: Excessive Slippage"

        return "TRADE APPROVED"

# ✅ Example Usage
risk_manager = RiskManagement()
decision = risk_manager.evaluate_trade(100, 95, 110, 0.3)

print(f"Risk Decision: {decision}")
