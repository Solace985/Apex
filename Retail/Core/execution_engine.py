import logging
from typing import Dict, Any
from brokers.broker_api import BrokerAPI

class ExecutionEngine:
    """Handles trade execution logic."""

    def __init__(self, broker_api):
        self.broker_api = broker_api

    def execute_trade(self, trade_signal):
        try:
            response = self.broker_api.place_order(trade_signal)
            return response
        except Exception as e:
            logging.error(f"Order execution failed: {e}")
            return None
