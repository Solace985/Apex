import os
import logging
from Brokers.zerodha import ZerodhaBroker
from Brokers.binance import BinanceBroker
from Brokers.upstox import UpstoxBroker
from Brokers.coinswitch import CoinSwitchBroker
from Brokers.dhan import DhanBroker
from Brokers.oanda import OandaBroker
from Brokers.dummy_broker import DummyBroker

logger = logging.getLogger(__name__)

class BrokerFactory:
    """Factory for selecting the best broker dynamically for LIVE or TEST trading."""

    def __init__(self, mode="LIVE"):
        """
        Initializes the broker factory.
        mode: "LIVE" for real trading, "TEST" for simulated trading.
        """
        self.mode = mode.upper()  # Convert to uppercase for consistency
        self.available_brokers = self._load_brokers()

        if self.mode == "TEST":
            logger.info("⚠️ Trading in TEST mode: Using Dummy Broker.")
        else:
            logger.info("✅ Trading in LIVE mode: Selecting best available broker.")

    def _load_brokers(self):
        """Dynamically loads available brokers based on environment variables."""

        brokers = {}

        # ✅ Dummy Broker (for testing)
        brokers["dummy"] = DummyBroker()

        # ✅ LIVE Brokers (Only if mode is LIVE)
        if self.mode == "LIVE":
            brokers["zerodha"] = ZerodhaBroker(os.getenv("ZERODHA_API_KEY"), os.getenv("ZERODHA_API_SECRET"))
            brokers["binance"] = BinanceBroker(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))
            brokers["upstox"] = UpstoxBroker(os.getenv("UPSTOX_API_KEY"), os.getenv("UPSTOX_API_SECRET"))
            brokers["coinswitch"] = CoinSwitchBroker(os.getenv("COINSWITCH_API_KEY"), os.getenv("COINSWITCH_API_SECRET"))
            brokers["dhan"] = DhanBroker(os.getenv("DHAN_API_KEY"), os.getenv("DHAN_API_SECRET"))
            brokers["oanda"] = OandaBroker(os.getenv("OANDA_API_KEY"), os.getenv("OANDA_API_SECRET"))

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
            return self._select_best_broker(order_details)

        return self.available_brokers.get("zerodha", self.available_brokers["dummy"])  # Default to Zerodha or Dummy

    def _select_best_broker(self, order_details):
        """
        Selects the most optimal broker dynamically based on:
        - Fees
        - Liquidity
        - Execution Speed
        - Market Depth
        """
        broker_scores = {}
        
        for name, broker in self.available_brokers.items():
            try:
                fee = broker.estimate_fees(order_details)
                liquidity = broker.get_liquidity(order_details)
                execution_speed = broker.get_execution_speed(order_details)

                # Lower fees & execution time, higher liquidity is better
                score = fee - (0.0001 * liquidity) + (execution_speed * 0.1)  
                broker_scores[name] = score
            except Exception as e:
                logger.error(f"⚠️ Error evaluating broker {name}: {e}")

        # Select broker with the lowest score (best performance)
        best_broker_name = min(broker_scores, key=broker_scores.get)
        logger.info(f"✅ Selected Best Broker: {best_broker_name.upper()}")

        return self.available_brokers[best_broker_name]
