import requests
import numpy as np

class LiquidityManager:
    def __init__(self):
        self.order_book_data = {}

    def fetch_order_book(self, exchange: str, symbol: str):
        """Fetch real-time order book data from the exchange."""
        if exchange == "binance":
            url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=100"
        elif exchange == "kraken":
            url = f"https://api.kraken.com/0/public/Depth?pair={symbol}"
        else:
            return None

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data
        return None

    def detect_whale_activity(self, exchange: str, symbol: str):
        """Detect whale orders (large volume orders) in the order book."""
        data = self.fetch_order_book(exchange, symbol)
        if not data:
            return False

        # Extract bid/ask order book levels
        bids = np.array(data['bids'], dtype=float)
        asks = np.array(data['asks'], dtype=float)

        bid_volumes = bids[:, 1]
        ask_volumes = asks[:, 1]

        avg_bid_size = np.mean(bid_volumes)
        avg_ask_size = np.mean(ask_volumes)

        # If a single bid/ask order is 5x the average volume, flag it as whale activity
        whale_bid = any(bid_volumes > avg_bid_size * 5)
        whale_ask = any(ask_volumes > avg_ask_size * 5)

        return whale_bid or whale_ask

# ✅ Example Usage
liquidity_manager = LiquidityManager()
if liquidity_manager.detect_whale_activity("binance", "BTCUSDT"):
    print("Whale detected! Adjusting strategy...")
