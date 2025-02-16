class MeanReversionStrategy:
    """Mean Reversion Strategy: Buys when price is low, sells when price is high."""

    def compute_signal(self, market_data):
        """
        ✅ Logic:
        - If price is much lower than its average → Buy
        - If price is much higher than its average → Sell
        """
        price = market_data["price"]
        moving_avg = sum(price[-20:]) / 20  # 20-period moving average

        if price[-1] < moving_avg * 0.98:  # ✅ Price is much lower than average
            return "BUY"
        elif price[-1] > moving_avg * 1.02:  # ✅ Price is much higher than average
            return "SELL"
        return "HOLD"
