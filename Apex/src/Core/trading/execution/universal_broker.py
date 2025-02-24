import requests
import json
import os
import logging
import time
from brokers.broker_manager import BrokerManager

logger = logging.getLogger(__name__)

class UniversalBroker:
    """Handles interaction with multiple brokers dynamically."""

    def __init__(self, user_id):
        """Initialize broker with dynamic configurations and OAuth authentication."""
        self.user_id = user_id
        self.broker_data = self._load_user_broker()

        if not self.broker_data:
            raise ValueError("❌ No broker selected. Please authenticate first.")

        self.broker_name = self.broker_data["broker_name"]
        self.access_token = self.broker_data["access_token"]

        # ✅ Load broker configurations dynamically
        manager = BrokerManager()
        self.config = manager.get_broker(self.broker_name)

        if not self.config:
            raise ValueError(f"❌ Broker '{self.broker_name}' is not configured! Please add it first.")

        self.base_url = self.config["base_url"]
        self.auth_type = self.config["auth_type"]
        self.headers = self._generate_headers()

    def _generate_headers(self):
        """Generates authentication headers dynamically."""
        headers = self.config.get("headers", {})
        if "{access_token}" in headers:
            headers = {key: value.replace("{access_token}", self.access_token) for key, value in headers.items()}
        return headers

    def _load_user_broker(self):
        """Loads the user's broker selection and stored OAuth token."""
        try:
            with open("config/user_broker.json", "r") as file:
                data = json.load(file)
                return data.get(self.user_id, None)
        except FileNotFoundError:
            return None

    def place_order(self, symbol, qty, order_type, price=None, retries=3):
        """Places an order using dynamic broker configuration with automatic retry."""
        endpoint = self.config["endpoints"]["place_order"]
        url = f"{self.base_url}{endpoint}"
        payload = {
            "symbol": symbol.upper(),
            "qty": qty,
            "order_type": order_type.upper(),
            "price": price,
        }

        for attempt in range(retries):
            try:
                response = requests.post(url, json=payload, headers=self.headers, timeout=5)
                response_data = response.json()

                if response.status_code == 200:
                    logger.info(f"✅ Order placed successfully: {response_data}")
                    return response_data
                else:
                    logger.warning(f"⚠️ Order failed (Attempt {attempt+1}/{retries}): {response_data}")

            except requests.exceptions.RequestException as e:
                logger.error(f"❌ API request failed (Attempt {attempt+1}/{retries}): {e}")
                time.sleep(2)  # Wait before retrying

        logger.critical(f"❌ Order placement failed after {retries} attempts.")
        return {"status": "FAILED", "error": "Max retries reached"}

    def get_account_balance(self):
        """Fetches account balance using dynamic broker configuration."""
        endpoint = self.config["endpoints"]["get_balance"]
        url = f"{self.base_url}{endpoint}"

        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            response_data = response.json()
            if response.status_code == 200:
                return response_data.get("data", {}).get("balances", {})
            logger.error(f"⚠️ Failed to fetch balance: {response_data}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API request failed: {e}")
        return {}

    def additional_function(self, user_id, broker_name):
        """Handles additional broker interaction."""
        manager = BrokerManager()
        self.config = manager.get_broker(broker_name)

        if not self.config:
            raise ValueError(f"❌ Broker '{broker_name}' is not configured! Please add it first.")

        self.base_url = self.config["base_url"]
        self.auth_type = self.config["auth_type"]
        self.headers = self._generate_headers()
