import pandas as pd
import numpy as np
from Apex.Tests.backtesting.backtest_runner import BacktestOrchestrator

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
        """Optimized ATR Calculation (reduces redundant calculations)."""
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
        return np.mean(tr[-period:]) if len(tr) >= period else None

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
        """Optimized ADX Calculation."""
        plus_dm = np.maximum(high[1:] - high[:-1], 0)
        minus_dm = np.maximum(low[:-1] - low[1:], 0)
        
        atr = TechnicalAnalysis.average_true_range(high, low, close, period)
        if atr is None:
            return None

        plus_di = 100 * (np.mean(plus_dm[-period:]) / atr)
        minus_di = 100 * (np.mean(minus_dm[-period:]) / atr)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        return np.mean(dx[-period:])

    @staticmethod
    def vwap(close_prices, volume):
        """Volume Weighted Average Price (VWAP)."""
        if len(close_prices) < 1 or len(volume) < 1:
            return None
        return np.cumsum(close_prices * volume) / np.cumsum(volume)

    @staticmethod
    def obv(close_prices, volume):
        """On-Balance Volume (OBV)."""
        obv = [0]
        for i in range(1, len(close_prices)):
            if close_prices[i] > close_prices[i - 1]:
                obv.append(obv[-1] + volume[i])
            elif close_prices[i] < close_prices[i - 1]:
                obv.append(obv[-1] - volume[i])
            else:
                obv.append(obv[-1])
        return obv[-1]

    @staticmethod
    def supertrend(high, low, close, period=10, multiplier=3):
        """Supertrend Indicator."""
        atr = TechnicalAnalysis.average_true_range(high, low, close, period)
        if atr is None:
            return None
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        return {"upper_band": upper_band[-1], "lower_band": lower_band[-1]}

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
    
    def evaluate_indicator_performance(self, asset_class, indicator_func, historical_data):
        """Evaluates the effectiveness of an indicator in past trading scenarios."""
        indicators_with = historical_data.apply(indicator_func, axis=1)
        indicators_without = historical_data.drop(columns=[indicator_func.__name__], errors="ignore")

        performance_with = BacktestOrchestrator(asset_class, indicators=indicators_with)
        performance_without = BacktestOrchestrator(asset_class, indicators=indicators_without)

        # Compute confidence score based on Sharpe Ratio comparison
        confidence_score = performance_with.sharpe_ratio - performance_without.sharpe_ratio
        return max(0, confidence_score)  # Confidence must be non-negative

    def indicator_confidence_weights(self, historical_data, rolling_window=50):
        """
        Dynamically assigns confidence weights to each indicator based on past performance.
        Uses a rolling window to continuously update confidence.
        """
        confidence_scores = {}
        for indicator_name, indicator_func in self.indicators.items():
            recent_data = historical_data.tail(rolling_window)  # Only consider recent data
            confidence_scores[indicator_name] = self.evaluate_indicator_performance("crypto", indicator_func, recent_data)

        # Normalize confidence scores
        total_confidence = sum(confidence_scores.values()) + 1e-10  # Avoid division by zero
        normalized_confidence = {key: value / total_confidence for key, value in confidence_scores.items()}

        return normalized_confidence

    def select_best_indicators(self, market_data, asset_class, ai_model, historical_data):
        """
        Uses AI reinforcement learning to dynamically select the most effective indicators.
        Adds confidence weighting based on past accuracy to avoid overfitting.
        """
        indicators = {
            "RSI": self.relative_strength_index(market_data["close"]),
            "MACD": self.macd(market_data["close"])[0],
            "Bollinger_Upper": self.bollinger_bands(market_data["close"])[0],
            "Bollinger_Lower": self.bollinger_bands(market_data["close"])[1],
            "ADX": self.adx(market_data["high"], market_data["low"], market_data["close"]),
            "VWAP": self.vwap(market_data["close"], market_data["volume"]),
            "Supertrend": self.supertrend(market_data["high"], market_data["low"], market_data["close"]),
        }

        # Normalize values to prevent extreme impact
        normalized_indicators = {key: self._normalize(value) for key, value in indicators.items() if value is not None}

        # Assign confidence scores to each indicator based on past performance
        confidence_weights = self.indicator_confidence_weights(historical_data)

        # Apply weighting to indicators before passing to AI
        weighted_indicators = {key: normalized_indicators[key] * confidence_weights.get(key, 1) for key in normalized_indicators}

        market_condition = self.detect_market_condition(market_data)

        # Assign priority to indicators based on market condition
        if market_condition == "trend":
            priority_indicators = ["ADX", "Supertrend", "Bollinger_Upper", "Bollinger_Lower"]
        elif market_condition == "range":
            priority_indicators = ["RSI", "MACD", "Stochastic_Oscillator"]
        else:
            priority_indicators = list(weighted_indicators.keys())  # Use all if uncertain

        # AI chooses which indicators to prioritize, but respects the hierarchy
        action = ai_model.select_action(np.array([weighted_indicators[ind] for ind in priority_indicators if ind in weighted_indicators]))

        # Select only the top-performing indicators based on AI decision, within priority groups
        selected_indicators = {name: weighted_indicators[name] for i, name in enumerate(priority_indicators) if name in weighted_indicators and action[i] > 0.5}

        contradiction_score = self.detect_conflicting_indicators(selected_indicators)

        # If contradiction score is too high, reduce trading confidence
        if contradiction_score > 1.5:
            selected_indicators = {key: value * 0.5 for key, value in selected_indicators.items()}

        # If contradiction is extreme, avoid making a trade
        if contradiction_score > 2.5:
            return {}  # No trade signal

        return selected_indicators

    def detect_conflicting_indicators(self, selected_indicators, historical_data):
        """
        Detects conflicts between selected indicators and resolves them based on past performance.
        Uses historical win rates to determine which indicator should be prioritized.
        """
        contradiction_score = 0
        conflict_pairs = [
            ("RSI", "MACD"),
            ("Bollinger_Upper", "ADX"),
            ("Supertrend", "VWAP"),
        ]

        indicator_win_rates = self.get_indicator_win_rates(historical_data)  # New function to get past performance

        for ind1, ind2 in conflict_pairs:
            if ind1 in selected_indicators and ind2 in selected_indicators:
                value1 = selected_indicators[ind1]
                value2 = selected_indicators[ind2]

                if (value1 > 0 and value2 < 0) or (value1 < 0 and value2 > 0):
                    contradiction_score += abs(value1 - value2)

                    # Resolve contradiction based on past win rates
                    if indicator_win_rates[ind1] > indicator_win_rates[ind2]:
                        selected_indicators[ind2] *= 0.5  # Reduce confidence in weaker indicator
                    elif indicator_win_rates[ind1] < indicator_win_rates[ind2]:
                        selected_indicators[ind1] *= 0.5  # Reduce confidence in weaker indicator
                    else:
                        selected_indicators[ind1] *= 0.75  # Reduce both slightly if they are equal
                        selected_indicators[ind2] *= 0.75

        return contradiction_score

    def _normalize(self, value):
        """Helper function to normalize indicator values to prevent skewed weight distribution."""
        return (value - np.mean(value)) / (np.std(value) + 1e-10) if isinstance(value, (int, float)) else value

    def get_indicator_win_rates(self, historical_data):
        """
        Retrieves the historical win rates of each indicator based on past trade outcomes.
        Win rate = Percentage of times the indicator correctly predicted market direction.
        """
        win_rates = {
            "RSI": 0.55,  # Example win rates (to be computed dynamically from backtesting data)
            "MACD": 0.60,
            "Bollinger_Upper": 0.50,
            "ADX": 0.65,
            "Supertrend": 0.70,
            "VWAP": 0.58,
        }

        # In a real implementation, retrieve actual win rates from historical backtesting
        return win_rates

    def detect_market_condition(self, market_data):
        """
        Determines whether the market is trending or ranging.
        Returns:
            "trend" - If trend indicators confirm a strong trend.
            "range" - If momentum indicators show range-bound movement.
            "uncertain" - If mixed signals are detected.
        """
        adx_value = self.adx(market_data["high"], market_data["low"], market_data["close"])
        rsi_value = self.relative_strength_index(market_data["close"])
        supertrend = self.supertrend(market_data["high"], market_data["low"], market_data["close"])
        bollinger_upper, bollinger_lower = self.bollinger_bands(market_data["close"])
        vwap_value = self.vwap(market_data["close"], market_data["volume"])
        avg_volume = np.mean(market_data["volume"][-14:])  # 14-period average volume

        if adx_value is None or rsi_value is None or supertrend is None or vwap_value is None:
            return "uncertain"

        # Define strong trend: ADX > 25, RSI moving away from 50, price above Bollinger bands
        is_trending = (
            adx_value > 25
            and abs(rsi_value - 50) > 10
            and market_data["close"].iloc[-1] > bollinger_upper
            and market_data["volume"].iloc[-1] > avg_volume  # Ensuring trend is backed by volume
        )

        # Define range: RSI between 40-60, ADX < 20, price inside Bollinger bands
        is_ranging = (
            40 < rsi_value < 60
            and adx_value < 20
            and bollinger_lower < market_data["close"].iloc[-1] < bollinger_upper
            and market_data["volume"].iloc[-1] < avg_volume  # Low volume means weak trend
        )

        if is_trending:
            return "trend"
        elif is_ranging:
            return "range"
        
        return "uncertain"
