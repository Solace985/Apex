import requests
import os

class OandaBroker:
    """Handles order execution via OANDA API."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OANDA_API_KEY")
        self.base_url = "https://api-fxpractice.oanda.com/v3"

    def place_order(self, symbol, qty, order_type, price=None):
        """Executes a trade on OANDA."""
        endpoint = f"{self.base_url}/accounts/orders"
        payload = {
            "order": {
                "instrument": symbol,
                "units": qty,
                "type": order_type.upper(),
                "price": price,
            }
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(endpoint, json=payload, headers=headers)
        return response.json()

    def estimate_fees(self, order_details):
        """Estimates trading fees for a given order."""
        return 0.02 * order_details["qty"]

    def get_execution_speed(self, order_details):
        """Estimates execution speed for a trade."""
        return 0.15  # Approximate execution time in seconds

    def get_liquidity(self, order_details):
        """Fetches real-time liquidity for a given asset."""
        return 500000  # Dummy value; replace with actual API data if available
