import numpy as np

class MomentumBreakoutStrategy:
    """Momentum Breakout Strategy: Trades when price breaks key resistance or support."""

    def compute_signal(self, market_data, period=20):
        """
        ✅ Logic:
        - If price breaks above the highest high in `period` days → BUY
        - If price breaks below the lowest low in `period` days → SELL
        - Else → HOLD
        """
        highs = market_data["high"][-period:]
        lows = market_data["low"][-period:]
        current_price = market_data["price"][-1]

        highest_high = np.max(highs)
        lowest_low = np.min(lows)

        if current_price > highest_high:
            return "BUY"
        elif current_price < lowest_low:
            return "SELL"
        return "HOLD"
