import requests
import numpy as np
from threading import Lock
import time
import asyncio

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

            # 🚨 If too many API errors, reduce order rate dynamically
            if self.failed_order_count > 3:
                print("⚠ Too many failed orders. Reducing request rate.")  
                self.max_calls = max(1, self.max_calls - 1)  # 🔻 Reduce allowed orders

            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            return False

class LiquidityManager:
    """Tracks real-time liquidity, detects institutional order flow, and prevents trading into illiquid conditions."""

    def __init__(self):
        self.order_book_data = {}
        self.rate_limiter = RateLimiter(10, 1)  # ✅ Initialize rate limiter
        self.liquidity_threshold = 500  # ✅ Minimum liquidity needed to execute large trades

    async def fetch_order_book(self, exchange: str, symbol: str):
        """Fetches real-time order book data from an exchange API."""
        if exchange == "binance":
            url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=100"
        elif exchange == "kraken":
            url = f"https://api.kraken.com/0/public/Depth?pair={symbol}"
        else:
            return None

        if not self.rate_limiter.allow_request():
            print("⚠ Rate limit exceeded. Delaying request.")  
            return None

        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None

    async def get_market_liquidity(self, exchange: str, symbol: str):
        """Analyzes market liquidity to determine whether it is safe to execute trades."""
        data = await self.fetch_order_book(exchange, symbol)
        if not data:
            return 0

        # Extract bid/ask sizes from order book
        bids = np.array(data['bids'], dtype=float)
        asks = np.array(data['asks'], dtype=float)

        total_liquidity = np.sum(bids[:, 1]) + np.sum(asks[:, 1])

        return total_liquidity

    async def detect_whale_activity(self, exchange: str, symbol: str):
        """Detects whale orders (large institutional orders) in the order book."""
        data = await self.fetch_order_book(exchange, symbol)
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

    async def adjust_trade_size_based_on_liquidity(self, trade_size, exchange, symbol):
        """Dynamically adjusts trade size based on market liquidity to reduce slippage."""
        market_liquidity = await self.get_market_liquidity(exchange, symbol)

        if market_liquidity < self.liquidity_threshold:
            print("⚠ Low liquidity detected! Reducing trade size.")  
            return max(trade_size * 0.5, 1)  # Reduce trade size in illiquid conditions

        return trade_size  # ✅ Proceed with full trade size if liquidity is sufficient
