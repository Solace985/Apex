import logging
import requests
from Brokers.broker_api import BrokerAPI
from typing import Dict, Any

class ZerodhaBroker(BrokerAPI):
    """Zerodha API implementation."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.zerodha.com"
        self.logger = logging.getLogger(self.__class__.__name__)

    def place_order(self, order_details: Dict[str, Any]) -> Dict[str, Any]:
        """Places an order on Zerodha."""
        endpoint = f"{self.base_url}/orders"
        response = requests.post(endpoint, json=order_details, headers={"Authorization": f"Bearer {self.api_key}"})
        return response.json()

    def fetch_account_balance(self) -> Dict[str, Any]:
        """Fetches account balance."""
        endpoint = f"{self.base_url}/balance"
        response = requests.get(endpoint, headers={"Authorization": f"Bearer {self.api_key}"})
        return response.json()

    def fetch_open_orders(self) -> Dict[str, Any]:
        """Fetches all open orders."""
        endpoint = f"{self.base_url}/open_orders"
        response = requests.get(endpoint, headers={"Authorization": f"Bearer {self.api_key}"})
        return response.json()

    def cancel_order(self, order_id: str) -> bool:
        """Cancels an order."""
        endpoint = f"{self.base_url}/orders/{order_id}"
        response = requests.delete(endpoint, headers={"Authorization": f"Bearer {self.api_key}"})
        return response.status_code == 200
