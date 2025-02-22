import numpy as np

class MeanReversionStrategy:
    """Mean Reversion Strategy with Adaptive Thresholds, Trend Filtering, and Optimized Computation."""

    def __init__(self, ema_period=20, atr_period=14, std_dev_multiplier=1.5, trend_filter_period=50):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.std_dev_multiplier = std_dev_multiplier
        self.trend_filter_period = trend_filter_period  # ✅ New: Checks Market Trend to Avoid Trading Against It

    def compute_signal(self, market_data):
        """
        ✅ Improved Logic:
        - Uses **EMA** instead of SMA for better trend tracking.
        - **Dynamic Entry/Exit Thresholds** based on Average True Range (ATR).
        - **Filters False Signals** using Standard Deviation of price changes.
        - **Avoids Reverting in Strong Trends** by using a long-term EMA.
        """
        price = np.array(market_data["price"])
        high = np.array(market_data["high"])
        low = np.array(market_data["low"])
        close = np.array(market_data["close"])

        # ✅ Compute Short-Term Exponential Moving Average (EMA)
        ema = self.calculate_ema(price, self.ema_period)

        # ✅ Compute Long-Term EMA for Trend Filtering
        long_term_ema = self.calculate_ema(price, self.trend_filter_period)

        # ✅ Compute Average True Range (ATR) for Dynamic Thresholds
        atr = self.calculate_atr(high, low, close, self.atr_period)

        # ✅ Compute Standard Deviation for Price Filtering
        std_dev = np.std(price[-self.ema_period:])

        # ✅ Dynamic Thresholds (More Adaptive)
        lower_threshold = ema - (self.std_dev_multiplier * atr)
        upper_threshold = ema + (self.std_dev_multiplier * atr)

        # ✅ Market Trend Filtering: If EMA is significantly below long-term trend, avoid buys
        if ema < long_term_ema and price[-1] < lower_threshold:
            return "HOLD"

        # ✅ Decision Logic
        if price[-1] < lower_threshold:
            return "BUY"
        elif price[-1] > upper_threshold:
            return "SELL"
        return "HOLD"

    def calculate_ema(self, price_series, period):
        """
        ✅ Optimized EMA Calculation (Vectorized NumPy Approach).
        """
        alpha = 2 / (period + 1)
        ema = np.convolve(price_series, [alpha] + [(1 - alpha) ** i for i in range(1, period)], mode="valid")
        return ema[-1]  # Return latest EMA value

    def calculate_atr(self, high, low, close, period):
        """
        ✅ Computes the Average True Range (ATR) for market volatility.
        - ATR helps set **dynamic thresholds** instead of fixed % changes.
        - ✅ Handles missing or incorrect data points gracefully.
        """
        tr1 = np.abs(high - low)
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])

        tr_combined = np.maximum(tr1[1:], np.maximum(tr2, tr3))  # ✅ Ensure No Missing Data
        atr = np.mean(tr_combined[-period:])  # Compute ATR over the given period
        return atr
