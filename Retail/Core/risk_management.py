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
