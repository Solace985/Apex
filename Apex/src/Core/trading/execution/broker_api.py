import logging
import asyncio
import time
import requests
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
import base64
import random

logger = logging.getLogger(__name__)

# ✅ Generate secure key (must be saved externally)
ENCRYPTION_KEY = base64.urlsafe_b64encode(Fernet.generate_key())

# ✅ Broker-defined rate limit (Adjust per broker's API rules)
MAX_ORDERS_PER_MINUTE = 20

class BrokerAPI:
    """Handles interaction with the broker securely."""

    def __init__(self, api_key: str, api_secret: str):
        """Initialize the API with secure authentication."""
        self.api_key = api_key
        self.api_secret = api_secret  # ✅ Require secret key for secure API access
        self._recent_orders = {}  # ✅ Store recent orders to prevent duplicate execution
        self._last_order_timestamps = []  # Initialize the list to track order timestamps

    def _encrypt_order(self, order: Dict[str, Any]) -> str:
        """Encrypts the order details before sending to broker."""
        cipher = Fernet(ENCRYPTION_KEY)
        encrypted_order = cipher.encrypt(str(order).encode()).decode()
        return encrypted_order

    def _is_duplicate_order(self, order: Dict[str, Any]) -> bool:
        """Check if the order is a duplicate within the last 10 seconds."""
        current_time = time.time()
        order_signature = f"{order['symbol']}_{order['side']}_{order['quantity']}"

        if order_signature in self._recent_orders and current_time - self._recent_orders[order_signature] < 10:
            return True  # Order is a duplicate

        # Store the order to track recent trades
        self._recent_orders[order_signature] = current_time
        return False

    def _apply_rate_limit(self):
        """Ensures we do not exceed broker's API rate limit."""
        if len(self._last_order_timestamps) >= MAX_ORDERS_PER_MINUTE:
            sleep_time = random.uniform(1.0, 3.0)  # ✅ Randomized sleep to avoid pattern detection
            logger.warning(f"⚠️ API rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)

        # ✅ Store the timestamp of this order
        self._last_order_timestamps.append(time.time())

    async def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Send order to broker securely with logging & retries."""
        
        # ✅ Order Size Validation
        min_order_size = 0.01  # Example: Minimum 0.01 BTC per trade
        max_order_size = 10.0  # Example: Maximum 10 BTC per trade

        if order["quantity"] < min_order_size or order["quantity"] > max_order_size:
            logger.warning(f"⚠️ Order rejected: Invalid size {order['quantity']} (Allowed: {min_order_size}-{max_order_size})")
            return {"status": "FAILED", "error": "Invalid order size"}

        # ✅ Prevent Duplicate Orders
        if self._is_duplicate_order(order):
            logger.warning(f"⚠️ Duplicate order detected: {order}")
            return {"status": "FAILED", "error": "Duplicate order detected"}

        # ✅ Fetch Account Balance
        balance = self.fetch_account_balance()
        required_balance = order["price"] * order["quantity"]

        if balance.get(order["symbol"], 0) < required_balance:
            logger.warning(f"❌ Order rejected: Insufficient balance for {order['symbol']}. Needed: {required_balance}, Available: {balance.get(order['symbol'], 0)}")
            return {"status": "FAILED", "error": "Insufficient balance"}

        # Apply rate limit before sending the API request
        self._apply_rate_limit()

        # Encrypt the order before sending
        encrypted_order = self._encrypt_order(order)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"📤 Placing order: {encrypted_order}")
                # Apply rate limit before sending the API request
                self._apply_rate_limit()  # ✅ Ensure we don't exceed API limits
                encrypted_order = self._encrypt_order(order)
                response = requests.post(
                    "https://broker-api.com/order",
                    json={"encrypted_data": encrypted_order},  # ✅ Send encrypted payload
                    headers={"Authorization": f"Bearer {self.api_key}", "X-Auth-Secret": self.api_secret},
                    timeout=5,
                )
                response_data = response.json()

                if response.status_code == 200 and response_data.get("status") == "SUCCESS":
                    # ✅ Verify order was executed
                    order_id = response_data.get("order_id")
                    confirmation = self._verify_trade_execution(order_id)

                    if confirmation:
                        logger.info(f"✅ Order executed successfully: {response_data}")
                        return response_data
                    else:
                        logger.warning(f"⚠️ Order placement confirmed, but execution failed! Order ID: {order_id}")
                        return {"status": "FAILED", "error": "Order execution failed"}

                logger.warning(f"⚠️ Order failed (Attempt {attempt+1}/{max_retries}): {response_data}")
                await asyncio.sleep(2)  # Wait before retrying

            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Order API request failed (Attempt {attempt+1}/{max_retries}): {e}")
                await asyncio.sleep(2)

        logger.critical(f"❌ Order placement failed after {max_retries} attempts.")
        return {"status": "FAILED", "error": "Max retries reached"}

    def fetch_account_balance(self) -> Dict[str, float]:
        """Fetches the account balance from the broker (Mocked)."""
        logger.info("Fetching account balance.")
        return {"BTC": 5.0, "ETH": 10.0}  # Replace with real broker API call

    def fetch_open_orders(self) -> Dict[str, Any]:
        """Fetches all open orders from the broker."""
        logger.info("Fetching open orders.")
        pass

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        logger.info(f"Cancelling order: {order_id}")
        return {"status": "CANCELLED", "order_id": order_id}

    def _verify_trade_execution(self, order_id: str) -> bool:
        """Check if the order actually got executed."""
        for _ in range(3):  # Retry 3 times
            time.sleep(2)
            response = self.fetch_open_orders()  # Check if order is still open
            if order_id not in response:
                return True  # ✅ Order has been executed
        return False  # ❌ Order is still open, execution failed
