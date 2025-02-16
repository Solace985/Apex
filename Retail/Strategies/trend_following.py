import numpy as np

class TrendFollowingStrategy:
    """Trend Following Strategy: Enters trades in the direction of a strong trend."""

    def compute_signal(self, market_data, short_window=10, long_window=50):
        """
        ✅ Logic:
        - If short-term moving average > long-term moving average → BUY
        - If short-term moving average < long-term moving average → SELL
        - Else → HOLD
        """
        short_ma = np.mean(market_data["price"][-short_window:])
        long_ma = np.mean(market_data["price"][-long_window:])

        if short_ma > long_ma:
            return "BUY"
        elif short_ma < long_ma:
            return "SELL"
        return "HOLD"
