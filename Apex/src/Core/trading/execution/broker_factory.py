import os
import logging
import time
import requests  # Required for IP-based country detection
import asyncio  # Added for asynchronous operations
from Apex.src.Core.execution.universal_broker import UniversalBroker
from Apex.src.Core.execution.broker_manager import BrokerManager  # Required for dynamic broker configuration
from Brokers.dummy_broker import DummyBroker
from cryptography.fernet import Fernet
from ratelimit import limits, sleep_and_retry
import base64
from tenacity import retry, stop_after_attempt, wait_fixed
from functools import lru_cache  # Added for caching

# Use a static encryption key stored securely in an environment variable
SECRET_KEY = os.getenv("ENCRYPTION_KEY")

if not SECRET_KEY:
    logging.critical("❌ ENCRYPTION_KEY is missing! API key decryption will fail.")
    raise ValueError("ENCRYPTION_KEY is required for secure API key storage.")

SECRET_KEY = base64.urlsafe_b64encode(SECRET_KEY.encode()).decode()  # Decode to string after encoding

logger = logging.getLogger(__name__)

class BrokerFactory:
    """Factory for selecting the best broker dynamically for LIVE or TEST trading."""

    def __init__(self, mode="LIVE"):
        """
        Initializes the broker factory.
        mode: "LIVE" for real trading, "TEST" for simulated trading.
        """
        self.mode = mode.upper()  # Convert to uppercase for consistency
        self.country = self._detect_user_country()  # Auto-detect country
        self.available_brokers = self._load_brokers()
        self._broker_performance_cache = {}  # Caching broker performance

        if self.mode == "TEST":
            logger.info("⚠️ Trading in TEST mode: Using Dummy Broker.")
        else:
            logger.info(f"✅ Trading in LIVE mode: Selecting best available broker for country: {self.country}")

    @sleep_and_retry
    @limits(calls=5, period=10)  # Limit to 5 calls per 10 seconds
    def _detect_user_country(self):
        """Detects user's country using multiple APIs with rate limiting."""
        geo_services = [
            "https://ipinfo.io/json",
            "https://ipapi.co/json",
            "https://geolocation-db.com/json/"
        ]
        for service in geo_services:
            try:
                response = requests.get(service, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if "country" in data:
                        return data["country"]
            except requests.exceptions.RequestException as e:
                logger.error(f"⚠️ Country detection failed: {service} - {e}")
        
        logger.warning("⚠️ Unable to determine country. Defaulting to 'Unknown'.")
        return "Unknown"

    @lru_cache(maxsize=10)  # Store the 10 most recent broker performance results
    def _get_cached_broker_performance(self, broker_name):
        return self._broker_performance_cache.get(broker_name, {})

    def load_api_key(self, env_var):
        """Load and decrypt API keys securely, with automatic rotation on failure."""
        encrypted_key = os.getenv(env_var)

        if not encrypted_key:
            logger.warning(f"⚠️ Missing API key for {env_var}. Broker may not function properly.")
            return None

        try:
            cipher = Fernet(SECRET_KEY)
            decrypted_key = cipher.decrypt(encrypted_key.encode()).decode()

            # ✅ Check if the key is still valid (API call test)
            if not self._validate_api_key(decrypted_key):
                logger.warning(f"⚠️ API key for {env_var} is invalid or expired. Attempting to rotate...")
                return self._rotate_api_key(env_var)

            return decrypted_key
        except Exception as e:
            logger.error(f"❌ Failed to decrypt API key for {env_var}. Error: {e}")
            return None

    def _load_brokers(self):
        """Dynamically loads brokers using UniversalBroker."""
        brokers = {}

        try:
            # Load user-specific broker configurations from broker manager
            manager = BrokerManager()
            available_brokers = manager.get_all_brokers()

            for broker_name in available_brokers:
                try:
                    brokers[broker_name] = UniversalBroker(user_id="default_user")  # Replace with actual user ID logic
                    logger.info(f"✅ Loaded {broker_name} via UniversalBroker.")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize {broker_name}. Error: {e}")
                    time.sleep(2)

        except Exception as e:
            logger.error(f"❌ Broker loading failed: {e}")

        return brokers

    # ✅ Max Daily Loss Protection
    MAX_DAILY_LOSS = 5.0  # Stop trading if loss exceeds 5% of account balance

    def _get_current_loss(self):
        """Fetch the current loss percentage from the trading account."""
        # Replace this with real API call to fetch account PnL
        return 3.2  # Example: Bot is currently at -3.2% loss

    def _update_broker_reliability(self, broker_name, success=True):
        """Updates broker reliability score based on execution success."""
        if broker_name not in self._broker_performance_cache:
            self._broker_performance_cache[broker_name] = {"score": 100, "fails": 0}
        
        if success:
            self._broker_performance_cache[broker_name]["score"] += 2  # Reward success
        else:
            self._broker_performance_cache[broker_name]["score"] -= 5  # Penalize failures
            self._broker_performance_cache[broker_name]["fails"] += 1
        
        self._broker_performance_cache[broker_name]["score"] = max(0, self._broker_performance_cache[broker_name]["score"])

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def get_broker(self, broker_name=None, order_details=None):
        """Returns a broker instance while logging all failures."""
        try:
            current_loss = self._get_current_loss()
            if current_loss >= self.MAX_DAILY_LOSS:
                logger.critical(f"❌ Max daily loss of {self.MAX_DAILY_LOSS}% reached. Stopping all trading.")
                return self.available_brokers["dummy"]  # Disable real trading

            if self.mode == "TEST":
                return self.available_brokers["dummy"]

            if broker_name:
                return UniversalBroker(user_id="default_user")  # Replace "default_user" with actual user logic

            if order_details:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return loop.create_task(self._select_best_broker(order_details))  # Non-blocking
                else:
                    return asyncio.run(self._select_best_broker(order_details))  # Blocking (CLI mode)

            # ✅ Select a broker dynamically based on lowest slippage
            best_broker = None
            min_slippage = float("inf")

            for name, broker in self.available_brokers.items():
                try:
                    slippage = broker.get_slippage(order_details)
                    if slippage < min_slippage:
                        min_slippage = slippage
                        best_broker = broker
                except Exception:
                    continue  # Skip failed brokers

            if best_broker:
                logger.info(f"✅ Selected broker with lowest slippage: {best_broker}")
                return best_broker

            logger.warning("⚠️ No valid brokers available! Defaulting to Dummy Broker.")
            return self.available_brokers["dummy"]

        except Exception as e:
            logger.error(f"❌ Broker selection failed. Details: {order_details}, Error: {e}")
            self._update_broker_reliability(broker_name, success=False)
            return self.available_brokers["dummy"]  # Fail-safe fallback

    async def fetch_broker_data(self, broker, order_details):
        """Asynchronously fetch broker performance data."""
        try:
            fee = await asyncio.wait_for(broker.estimate_fees(order_details), timeout=3)
            liquidity = await asyncio.wait_for(broker.get_liquidity(order_details), timeout=3)
            execution_speed = await asyncio.wait_for(broker.get_execution_speed(order_details), timeout=3)
            return (fee, liquidity, execution_speed)  # ✅ Added missing return statement
        except asyncio.TimeoutError:
            logger.error(f"⚠️ Broker API timeout for {broker}. Skipping.")
            return (float('inf'), 0, float('inf'))  # Worst-case scenario

    async def _select_best_broker(self, order_details):
        """Optimized broker selection using asynchronous calls."""
        # ✅ Check if we already have recent performance data
        broker_results = {}
        current_time = time.time()

        for name, broker in self.available_brokers.items():
            if name in self._broker_performance_cache and current_time - self._broker_performance_cache[name]["timestamp"] < 60:
                # ✅ Use cached data if it's less than 60 seconds old
                broker_results[name] = self._broker_performance_cache[name]["data"]
            else:
                # ✅ Fetch fresh data and store it in the cache
                broker_results[name] = await self.fetch_broker_data(broker, order_details)
                self._broker_performance_cache[name] = {"data": broker_results[name], "timestamp": current_time}

        broker_scores = {}
        for name, (fee, liquidity, execution_speed) in broker_results.items():
            score = fee - (0.0001 * liquidity) + (execution_speed * 0.1)
            broker_scores[name] = score

        if not broker_scores:
            logger.warning("⚠️ No brokers available for selection. Defaulting to Dummy Broker.")
            return self.available_brokers["dummy"]

        best_broker_name = min(broker_scores, key=broker_scores.get)
        logger.info("✅ Selected Secure Broker")
        return self.available_brokers[best_broker_name]

    def _validate_api_key(self, api_key):
        """Check if an API key is valid by making a test API call."""
        try:
            # Example validation (Binance ping request)
            response = requests.get("https://api.binance.com/api/v3/ping", headers={"X-MBX-APIKEY": api_key}, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _rotate_api_key(self, env_var):
        """Rotate the API key by retrieving a new one from a secure vault."""
        logger.info(f"🔄 Rotating API key for {env_var}...")
        # Fetch a new API key from a secure vault (Replace with real implementation)
        new_key = "NEW_SECURE_KEY_FROM_VAULT"

        if new_key:
            os.environ[env_var] = new_key  # Temporarily update for this session
            return new_key

        logger.error(f"❌ Failed to rotate API key for {env_var}. No backup key available.")
        return None

    def get_account_balance(self, user_id):
        """Fetch account balance dynamically from UniversalBroker."""
        broker = UniversalBroker(user_id=user_id)
        return broker.get_account_balance()
