import requests
import os

class DhanBroker:
    """Handles order execution via Dhan API."""

    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key or os.getenv("DHAN_API_KEY")
        self.api_secret = api_secret or os.getenv("DHAN_API_SECRET")
        self.base_url = "https://api.dhan.co"

    def place_order(self, symbol, qty, order_type, price=None):
        """Executes a trade on Dhan."""
        endpoint = f"{self.base_url}/orders"
        payload = {
            "symbol": symbol,
            "qty": qty,
            "order_type": order_type,
            "price": price,
            "api_key": self.api_key,
            "api_secret": self.api_secret,
        }
        response = requests.post(endpoint, json=payload)
        return response.json()

    def estimate_fees(self, order_details):
        """Estimates trading fees for a given order."""
        return 0.05 * order_details["qty"]

    def get_execution_speed(self, order_details):
        """Estimates execution speed for a trade."""
        return 0.2  # Approximate execution time in seconds

    def get_liquidity(self, order_details):
        """Fetches real-time liquidity for a given asset."""
        return 1000000  # Dummy value; integrate with real API if available
