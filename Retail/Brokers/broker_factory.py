import os
import logging
import time
import requests  # Required for IP-based country detection
import asyncio  # Added for asynchronous operations
from Brokers.zerodha import ZerodhaBroker
from Brokers.binance import BinanceBroker
from Brokers.upstox import UpstoxBroker
from Brokers.coinswitch import CoinSwitchBroker
from Brokers.dhan import DhanBroker
from Brokers.oanda import OandaBroker
from Brokers.dummy_broker import DummyBroker
from Retail.Core.Python.config import decrypt_api_key  # Importing decrypt_api_key

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

    def _detect_user_country(self):
        """Detects the user's country using multiple geolocation APIs with validation."""
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
                    if isinstance(data, dict) and "country" in data and isinstance(data["country"], str):
                        country = data["country"].strip()
                        if country:
                            return country
            except requests.exceptions.RequestException as e:
                logger.error(f"⚠️ Country detection API failed: {service} - {e}")
        
        logger.error("⚠️ Could not determine user's country after multiple attempts. Defaulting to 'Unknown'.")
        return "Unknown"

    def load_api_key(self, env_var):
        """Load and decrypt API keys securely with validation."""
        encrypted_key = os.getenv(env_var)
        
        if not encrypted_key:
            logger.warning(f"⚠️ Missing API key for {env_var}. Broker may not function properly.")
            return None

        try:
            return decrypt_api_key(encrypted_key)
        except Exception:
            logger.error(f"❌ Failed to decrypt API key for {env_var}. Check encryption settings.")
            return None

    def _load_brokers(self):
        """Dynamically loads available brokers based on encrypted API keys and user country."""
        brokers = {"dummy": DummyBroker()}  # ✅ Always include Dummy Broker

        if self.mode == "LIVE":
            if self.load_api_key("BINANCE_API_KEY") and self.load_api_key("BINANCE_API_SECRET"):
                brokers["binance"] = BinanceBroker(self.load_api_key("BINANCE_API_KEY"), self.load_api_key("BINANCE_API_SECRET"))

            if self.load_api_key("OANDA_API_KEY") and self.load_api_key("OANDA_API_SECRET"):
                brokers["oanda"] = OandaBroker(self.load_api_key("OANDA_API_KEY"), self.load_api_key("OANDA_API_SECRET"))

            # ✅ India-Specific Brokers
            if self.country == "IN":
                if self.load_api_key("ZERODHA_API_KEY") and self.load_api_key("ZERODHA_API_SECRET"):
                    brokers["zerodha"] = ZerodhaBroker(self.load_api_key("ZERODHA_API_KEY"), self.load_api_key("ZERODHA_API_SECRET"))
                
                if self.load_api_key("UPSTOX_API_KEY") and self.load_api_key("UPSTOX_API_SECRET"):
                    brokers["upstox"] = UpstoxBroker(self.load_api_key("UPSTOX_API_KEY"), self.load_api_key("UPSTOX_API_SECRET"))
                
                if self.load_api_key("DHAN_API_KEY") and self.load_api_key("DHAN_API_SECRET"):
                    brokers["dhan"] = DhanBroker(self.load_api_key("DHAN_API_KEY"), self.load_api_key("DHAN_API_SECRET"))

            # ✅ Global Crypto Brokers
            if self.load_api_key("COINSWITCH_API_KEY") and self.load_api_key("COINSWITCH_API_SECRET"):
                brokers["coinswitch"] = CoinSwitchBroker(self.load_api_key("COINSWITCH_API_KEY"), self.load_api_key("COINSWITCH_API_SECRET"))

        return brokers

    def get_broker(self, broker_name=None, order_details=None):
        """
        Returns the best available broker.

        - If `broker_name` is provided, it tries to return that broker.
        - If no `broker_name` is given, it selects the best broker dynamically based on:
          - Fees
          - Liquidity
          - Execution speed
        - If in TEST mode, it always returns `DummyBroker`.
        """
        if self.mode == "TEST":
            return self.available_brokers["dummy"]
        if broker_name and broker_name.lower() in self.available_brokers:
            return self.available_brokers[broker_name.lower()]

        if order_details:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return loop.create_task(self._select_best_broker(order_details))  # Non-blocking
            else:
                return asyncio.run(self._select_best_broker(order_details))  # Blocking (CLI mode)

        # ✅ Select a broker dynamically if Zerodha is unavailable
        available_brokers = {k: v for k, v in self.available_brokers.items() if v.api_key}
        
        if available_brokers:
            best_broker = min(available_brokers, key=lambda k: available_brokers[k].get_execution_speed({}))
            logger.info("✅ Selected Best Broker: [Hidden for security]")
            return available_brokers[best_broker]
        # ✅ If no brokers available, use Dummy Broker
        logger.warning("⚠️ No valid brokers available! Defaulting to Dummy Broker.")
        return self.available_brokers["dummy"]

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
        broker_tasks = {name: self.fetch_broker_data(broker, order_details) for name, broker in self.available_brokers.items()}
        broker_results = await asyncio.gather(*broker_tasks.values())

        broker_scores = {}
        for (name, (fee, liquidity, execution_speed)) in zip(broker_tasks.keys(), broker_results):
            score = fee - (0.0001 * liquidity) + (execution_speed * 0.1)
            broker_scores[name] = score

        if not broker_scores:
            logger.warning("⚠️ No brokers available for selection. Defaulting to Dummy Broker.")
            return self.available_brokers["dummy"]

        best_broker_name = min(broker_scores, key=broker_scores.get)
        logger.info("✅ Selected Secure Broker")
        return self.available_brokers[best_broker_name]

