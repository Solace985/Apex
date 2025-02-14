import logging
from typing import Dict, Any
from brokers.broker_api import BrokerAPI

class ExecutionEngine:
    """Handles trade execution logic."""

    def __init__(self, broker: BrokerAPI):
        self.broker = broker
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute_trade(self, trade_details: Dict[str, Any]):
        """Executes a trade."""
        response = self.broker.place_order(trade_details)
        if response.get("status") == "success":
            self.logger.info(f"Trade executed: {response}")
        else:
            self.logger.error(f"Trade execution failed: {response}")
