from brokers.zerodha import ZerodhaBroker
from brokers.binance import BinanceBroker
from brokers.dummy_broker import DummyBroker

class BrokerFactory:
    """Factory for dynamic broker selection."""
    
    @staticmethod
    def get_broker(broker_name, api_key, api_secret):
        if broker_name.lower() == "zerodha":
            return ZerodhaBroker(api_key, api_secret)
        elif broker_name.lower() == "binance":
            return BinanceBroker(api_key, api_secret)
        return DummyBroker()
