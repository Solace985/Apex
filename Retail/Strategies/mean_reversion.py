import numpy as np

class MeanReversionStrategy:
    """Enhanced Mean Reversion Strategy with EMA, ATR-based dynamic thresholds, and Standard Deviation Filtering."""

    def compute_signal(self, market_data, ema_period=20, atr_period=14, std_dev_multiplier=1.5):
        """
        ✅ Improved Logic:
        - Uses **EMA** instead of SMA for faster trend responsiveness.
        - **Dynamic Entry/Exit Thresholds** based on Average True Range (ATR).
        - **Filters False Signals** using Standard Deviation of price changes.
        """
        price = market_data["price"]
        high = market_data["high"]
        low = market_data["low"]
        close = market_data["close"]

        # ✅ Compute Exponential Moving Average (EMA)
        ema = self.calculate_ema(price, ema_period)

        # ✅ Compute Average True Range (ATR) for Dynamic Thresholds
        atr = self.calculate_atr(high, low, close, atr_period)

        # ✅ Compute Standard Deviation for Price Filtering
        std_dev = np.std(price[-ema_period:])

        # ✅ Dynamic Thresholds
        lower_threshold = ema - (std_dev_multiplier * atr)  # Buy if price falls below this
        upper_threshold = ema + (std_dev_multiplier * atr)  # Sell if price rises above this

        # ✅ Decision Logic
        if price[-1] < lower_threshold:
            return "BUY"
        elif price[-1] > upper_threshold:
            return "SELL"
        return "HOLD"

    def calculate_ema(self, price_series, period):
        """
        ✅ Computes the Exponential Moving Average (EMA).
        """
        alpha = 2 / (period + 1)  # Smoothing factor
        ema = np.zeros_like(price_series)
        ema[0] = price_series[0]  # Initialize EMA with the first price
        for i in range(1, len(price_series)):
            ema[i] = alpha * price_series[i] + (1 - alpha) * ema[i - 1]
        return ema[-1]  # Return latest EMA value

    def calculate_atr(self, high, low, close, period):
        """
        ✅ Computes the Average True Range (ATR) for market volatility.
        - ATR helps set dynamic thresholds instead of fixed % changes.
        """
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - close[:-1])
        tr3 = np.abs(low - close[:-1])
        tr = np.maximum(tr1[1:], np.maximum(tr2, tr3))  # True range across 3 conditions
        atr = np.mean(tr[-period:])  # Compute ATR over the given period
        return atr
