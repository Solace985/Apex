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
        volatility = np.std(market_data["price"][-long_window:])  # Measure market volatility

        # ✅ If volatility is high → Use shorter windows
        if volatility > np.mean(volatility):
            short_window = 5
            long_window = 25
        else:
            short_window = 10
            long_window = 50

        short_ma = np.mean(market_data["price"][-short_window:])
        long_ma = np.mean(market_data["price"][-long_window:])
        
        avg_volume = np.mean(market_data["volume"][-long_window:])
        current_volume = market_data["volume"][-1]

        # Calculate ATR
        tr = np.max([market_data["high"] - market_data["low"],
                      np.abs(market_data["high"] - market_data["price"][:-1]),
                      np.abs(market_data["low"] - market_data["price"][:-1])], axis=0)
        atr = np.mean(tr)

        rsi = self.tech_analysis.relative_strength_index(market_data["price"])

        if short_ma > long_ma and rsi < 70:  # ✅ Avoid buying in overbought conditions
            return "BUY"
        elif short_ma < long_ma and rsi > 30:  # ✅ Avoid selling in oversold conditions
            return "SELL"
        return "HOLD"
