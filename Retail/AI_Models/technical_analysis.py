import pandas as pd
import numpy as np

class TechnicalAnalysis:
    """Computes various technical indicators for market analysis."""

    def moving_average(self, prices, period=14):
        """Computes simple moving average (SMA)."""
        return np.mean(prices[-period:])

    def exponential_moving_average(self, prices, period=14):
        """Computes exponential moving average (EMA)."""
        return pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1]

    def relative_strength_index(self, prices, period=14):
        """Calculates the Relative Strength Index (RSI)."""
        delta = np.diff(prices)
        gain = np.maximum(delta, 0)
        loss = -np.minimum(delta, 0)

        avg_gain = np.mean(gain[:period])
        avg_loss = np.mean(loss[:period])

        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        return 100 - (100 / (1 + rs))

    def bollinger_bands(self, prices, period=20, std_factor=2):
        """Calculates Bollinger Bands (Upper and Lower)."""
        sma = np.mean(prices[-period:])
        std_dev = np.std(prices[-period:])
        upper_band = sma + std_factor * std_dev
        lower_band = sma - std_factor * std_dev
        return upper_band, lower_band

    def stochastic_oscillator(self, prices, period=14):
        """Calculates the Stochastic Oscillator."""
        highest_high = np.max(prices[-period:])
        lowest_low = np.min(prices[-period:])
        return 100 * (prices[-1] - lowest_low) / (highest_high - lowest_low + 1e-10)

    def average_true_range(self, high, low, close, period=14):
        """Calculates the Average True Range (ATR)."""
        tr = np.maximum(high - low, np.maximum(np.abs(high - close[:-1]), np.abs(low - close[:-1])))
        return np.mean(tr[-period:])

    def aroon_indicator(self, high, low, period=25):
        """Calculates Aroon Indicator (Aroon Up and Aroon Down)."""
        high_idx = np.argmax(high[-period:])
        low_idx = np.argmin(low[-period:])
        aroon_up = (period - high_idx) / period * 100
        aroon_down = (period - low_idx) / period * 100
        return aroon_up, aroon_down

    def macd(self, prices, short_period=12, long_period=26, signal_period=9):
        """Calculates Moving Average Convergence Divergence (MACD)."""
        short_ema = pd.Series(prices).ewm(span=short_period, adjust=False).mean()
        long_ema = pd.Series(prices).ewm(span=long_period, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        return macd_line.iloc[-1], signal_line.iloc[-1]

    def money_flow_index(self, high, low, close, volume, period=14):
        """Calculates the Money Flow Index (MFI)."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        pos_flow = np.sum(money_flow[1:][typical_price[1:] > typical_price[:-1]])
        neg_flow = np.sum(money_flow[1:][typical_price[1:] < typical_price[:-1]])
        money_ratio = pos_flow / (neg_flow + 1e-10)
        return 100 - (100 / (1 + money_ratio))
