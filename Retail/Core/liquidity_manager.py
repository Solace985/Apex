import requests
import numpy as np
from threading import Lock
import time

class RateLimiter:
    """Prevents exceeding API call limits."""
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = Lock()
        self.failed_order_count = 0  # Track failed orders

    def allow_request(self):
        with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.period]

            # 🚨 If too many API errors, reduce order rate
            if self.failed_order_count > 3:
                print("⚠ Too many failed orders. Reducing request rate.")  # Replace logger with print for simplicity
                self.max_calls = max(1, self.max_calls - 1)  # 🔻 Reduce allowed orders

            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            return False

class LiquidityManager:
    def __init__(self):
        self.order_book_data = {}
        self.rate_limiter = RateLimiter(10, 1)  # Initialize rate limiter

    def fetch_order_book(self, exchange: str, symbol: str):
        """Fetch real-time order book data from the exchange."""
        if exchange == "binance":
            url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=100"
        elif exchange == "kraken":
            url = f"https://api.kraken.com/0/public/Depth?pair={symbol}"
        else:
            return None

        if not self.rate_limiter.allow_request():
            print("⚠ Rate limit exceeded. Delaying request.")  # Replace logger with print for simplicity
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

class OrderBookProcessor:
    """Processes real-time order book data to track large institutional movements."""
    
    def __init__(self, exchange_api):
        self.exchange_api = exchange_api
        self.order_book = {}

    async def fetch_order_book(self, symbol):
        self.order_book = await self.exchange_api.get_order_book(symbol)
        return self.order_book

    def detect_whale_activity(self):
        """Detects unusual large orders that indicate institutional trading."""
        large_orders = [order for order in self.order_book['bids'] if order['size'] > 1000]  # Adjust threshold
        return len(large_orders) > 5  # If more than 5 large orders, assume institutional activity

# ✅ Example Usage
liquidity_manager = LiquidityManager()
if liquidity_manager.detect_whale_activity("binance", "BTCUSDT"):
    print("Whale detected! Adjusting strategy...")
