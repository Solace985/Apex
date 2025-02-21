import requests
import os
import hashlib
import hmac
import time
import logging

logger = logging.getLogger(__name__)

class UpstoxBroker:
    """Handles order execution via Upstox API."""

    def __init__(self, api_key=None, api_secret=None, access_token=None):
        """Initialize the Upstox API with secure authentication and request signing."""
        self.api_key = api_key or os.getenv("UPSTOX_API_KEY")
        self.api_secret = api_secret or os.getenv("UPSTOX_API_SECRET")
        self.access_token = access_token or os.getenv("UPSTOX_ACCESS_TOKEN")
        self.base_url = "https://api.upstox.com/v2"
        
        if not self.api_key or not self.api_secret or not self.access_token:
            raise ValueError("❌ Upstox API Key, Secret, and Access Token are required!")

    def _sign_request(self, params: dict) -> dict:
        """Signs API requests using HMAC SHA256 for security."""
        params["timestamp"] = int(time.time() * 1000)
        query_string = "&".join([f"{key}={value}" for key, value in sorted(params.items())])
        signature = hmac.new(self.api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        params["signature"] = signature
        return params

    def get_account_balance(self):
        """Fetches account balance from Upstox API."""
        endpoint = f"{self.base_url}/funds"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        try:
            response = requests.get(endpoint, headers=headers, timeout=5)
            response_data = response.json()

            if response.status_code == 200:
                return response_data.get("data", {}).get("balances", {})

            logger.error(f"⚠️ Failed to fetch Upstox account balance: {response_data}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Upstox API request failed: {e}")

        return {}  # Return empty balance dictionary on failure

    def place_order(self, symbol, qty, order_type, price=None):
        """Executes a trade on Upstox with secure request signing and rate limit handling."""
        # ✅ Fetch Account Balance Before Order
        balance = self.get_account_balance()
        required_balance = price * qty if price else qty  # Market order uses qty, limit order uses price * qty

        if balance.get(symbol.upper(), 0) < required_balance:
            logger.warning(f"❌ Order rejected: Insufficient balance for {symbol}. Needed: {required_balance}, Available: {balance.get(symbol.upper(), 0)}")
            return {"status": "FAILED", "error": "Insufficient balance"}

        endpoint = f"{self.base_url}/order/place"
        params = {
            "instrument_token": symbol.upper(),
            "transaction_type": "BUY" if order_type.lower() == "market" else "SELL",
            "order_type": "LIMIT" if order_type.lower() == "limit" else "MARKET",
            "quantity": qty,
        }

        if price:
            params["price"] = price

        signed_params = self._sign_request(params)
        headers = {"Authorization": f"Bearer {self.access_token}"}
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = requests.post(endpoint, params=signed_params, headers=headers, timeout=5)
                response_data = response.json()

                if response.status_code == 200:
                    logger.info(f"✅ Order placed successfully: {response_data}")
                    return response_data

                elif response.status_code == 429:  # Rate limit exceeded
                    wait_time = int(response.headers.get("Retry-After", 5))
                    logger.warning(f"⚠️ Upstox API rate limit exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Order failed (Attempt {attempt+1}/{max_retries}): {response_data}")

            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Upstox API request failed (Attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2)

        logger.critical(f"❌ Order placement failed after {max_retries} attempts.")
        return {"status": "FAILED", "error": "Max retries reached"}
