import pandas as pd
import numpy as np
from Retail.Backtesting.backtest_runner import BacktestOrchestrator

class TechnicalAnalysis:
    """Advanced Technical Analysis Toolkit."""

    @staticmethod
    def moving_average(prices, period=14):
        """Simple Moving Average (SMA)."""
        return pd.Series(prices).rolling(window=period).mean().iloc[-1] if len(prices) >= period else None

    @staticmethod
    def exponential_moving_average(prices, period=14):
        """Exponential Moving Average (EMA)."""
        return pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1] if len(prices) >= period else None

    @staticmethod
    def relative_strength_index(prices, period=14):
        """Relative Strength Index (RSI)."""
        if len(prices) < period:
            return None
        delta = pd.Series(prices).diff().dropna()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=period).mean().iloc[-1]
        avg_loss = loss.rolling(window=period).mean().iloc[-1]

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def bollinger_bands(prices, period=20, std_factor=2):
        """Bollinger Bands."""
        prices_series = pd.Series(prices)
        if len(prices_series) < period:
            return None, None
        sma = prices_series.rolling(window=period).mean().iloc[-1]
        std_dev = prices_series.rolling(window=period).std().iloc[-1]
        upper_band = sma + std_factor * std_dev
        lower_band = sma - std_factor * std_dev
        return upper_band, lower_band

    @staticmethod
    def stochastic_oscillator(prices, period=14):
        """Stochastic Oscillator."""
        prices_series = pd.Series(prices)
        if len(prices_series) < period:
            return None
        lowest_low = prices_series.rolling(window=period).min().iloc[-1]
        highest_high = prices_series.rolling(window=period).max().iloc[-1]
        stoch = 100 * (prices[-1] - lowest_low) / (highest_high - lowest_low + 1e-10)
        return stoch

    @staticmethod
    def average_true_range(high, low, close, period=14):
        """Average True Range (ATR)."""
        high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
        if len(high) < period:
            return None
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        return atr

    @staticmethod
    def aroon_indicator(high, low, period=25):
        """Aroon Indicator."""
        if len(high) < period or len(low) < period:
            return None, None
        high_idx = np.argmax(high[-period:])
        low_idx = np.argmin(low[-period:])
        aroon_up = (period - high_idx) / period * 100
        aroon_down = (period - low_idx) / period * 100
        return aroon_up, aroon_down

    @staticmethod
    def macd(prices, short_period=12, long_period=26, signal_period=9):
        """Moving Average Convergence Divergence (MACD)."""
        prices_series = pd.Series(prices)
        if len(prices_series) < long_period:
            return None, None, None
        ema_short = prices_series.ewm(span=short_period, adjust=False).mean()
        ema_long = prices_series.ewm(span=long_period, adjust=False).mean()
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_line.iloc[-1], signal_line.iloc[-1], macd_histogram.iloc[-1]

    @staticmethod
    def money_flow_index(high, low, close, volume, period=14):
        """Money Flow Index (MFI)."""
        if len(high) < period or len(low) < period or len(close) < period or len(volume) < period:
            return None
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        delta_tp = typical_price.diff()

        pos_flow = money_flow.where(delta_tp > 0, 0).rolling(period).sum().iloc[-1]
        neg_flow = money_flow.where(delta_tp < 0, 0).rolling(period).sum().iloc[-1]

        money_ratio = pos_flow / (neg_flow + 1e-10)
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi

    @staticmethod
    def ichimoku_cloud(high, low, conversion_period=9, base_period=26, span_b_period=52):
        """Ichimoku Cloud."""
        if len(high) < span_b_period:
            return None, None
        conversion_line = (pd.Series(high).rolling(conversion_period).max() +
                           pd.Series(low).rolling(conversion_period).min()) / 2
        base_line = (pd.Series(high).rolling(base_period).max() +
                     pd.Series(low).rolling(base_period).min()) / 2
        span_a = ((conversion_line + base_line) / 2).shift(base_period)
        span_b = ((pd.Series(high).rolling(span_b_period).max() +
                   pd.Series(low).rolling(span_b_period).min()) / 2).shift(base_period)
        return span_a.iloc[-1], span_b.iloc[-1]

    @staticmethod
    def adx(high, low, close, period=14):
        """Average Directional Index (ADX)."""
        high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
        if len(high) < period or len(low) < period or len(close) < period:
            return None
        plus_dm = high.diff().clip(lower=0)
        minus_dm = low.diff().clip(upper=0).abs()
        tr = TechnicalAnalysis.average_true_range(high, low, close, period)

        plus_di = 100 * (plus_dm.ewm(span=period).mean().iloc[-1] / (tr + 1e-10))
        minus_di = 100 * (minus_dm.ewm(span=period).mean().iloc[-1] / (tr + 1e-10))
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        adx_value = pd.Series(dx).ewm(span=period).mean().iloc[-1]
        return adx_value

    @staticmethod
    def fibonacci_retracement(high_price, low_price):
        """Fibonacci Retracement Levels."""
        diff = high_price - low_price
        levels = {
            "23.6%": high_price - 0.236 * diff,
            "38.2%": high_price - 0.382 * diff,
            "50.0%": high_price - 0.500 * diff,
            "61.8%": high_price - 0.618 * diff,
            "78.6%": high_price - 0.786 * diff
        }
        return levels

    def evaluate_indicator_effectiveness(self, asset_class, indicator_func, historical_data):
        """Evaluate indicator effectiveness dynamically."""
        
        indicators_with = historical_data.apply(indicator_func, axis=1)
        performance_with = BacktestOrchestrator(asset_class, indicators=indicators_with)

        indicators_without = historical_data.drop(columns=[indicator_func.__name__], errors='ignore')
        performance_without = BacktestOrchestrator(asset_class, indicators=indicators_without)

        return performance_with.sharpe_ratio > performance_without.sharpe_ratio

    def adaptive_indicator_selection(self, asset_class, indicators, historical_data):
        """Adaptive selection of effective indicators."""
        effective_indicators = []
        for indicator_func in indicators:
            if self.evaluate_indicator_effectiveness(asset_class, indicator_func, historical_data):
                effective_indicators.append(indicator_func)
        return effective_indicators
