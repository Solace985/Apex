import numpy as np
import requests
import time

# This module tracks institutional buy/sell pressure using order book depth, volume imbalances and dark pool trades.

class InstitutionalOrderFlow:
    def __init__(self):
        self.institutional_activity = []
        self.whale_threshold = 1000000  # Trades above $1M considered as whale trades

    def fetch_order_book_data(self, symbol="BTCUSD"):
        """Fetches order book data to track large institutional movements."""
        try:
            response = requests.get(f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=100")
            order_book = response.json()

            # Extract top bid & ask volume
            top_bid_volume = float(order_book["bids"][0][1])
            top_ask_volume = float(order_book["asks"][0][1])

            return top_bid_volume, top_ask_volume
        except:
            return 0, 0

    def detect_whale_activity(self, recent_trades):
        """Detects if institutional traders are accumulating or distributing assets."""
        whale_trades = [trade for trade in recent_trades if trade["amount"] > self.whale_threshold]

        if len(whale_trades) > 0:
            self.institutional_activity.append({"time": time.time(), "whale_trades": len(whale_trades)})

        return len(whale_trades) > 0

    def get_institutional_bias(self):
        """Determines if institutions are buying (bullish) or selling (bearish)."""
        if len(self.institutional_activity) < 5:
            return "neutral"  # Not enough data

        recent_activity = self.institutional_activity[-5:]
        buy_trades = sum(1 for trade in recent_activity if trade["whale_trades"] > 2)
        sell_trades = sum(1 for trade in recent_activity if trade["whale_trades"] < 2)

        return "bullish" if buy_trades > sell_trades else "bearish" if sell_trades > buy_trades else "neutral"
