import numpy as np

class RegimeDetectionStrategy:
    """Regime Detection Strategy: Determines whether the market is trending or ranging."""

    def detect_regime(self, market_data, period=50):
        """
        ✅ Logic:
        - If Average Directional Index (ADX) > 25 → Market is trending
        - If ADX < 25 → Market is ranging
        """
        highs = market_data["high"][-period:]
        lows = market_data["low"][-period:]
        closes = market_data["price"][-period:]

        tr = np.max([highs - lows, np.abs(highs - closes[:-1]), np.abs(lows - closes[:-1])], axis=0)
        atr = np.mean(tr)  # Average True Range

        adx = 100 * np.mean(atr[-14:]) / np.mean(closes[-14:])  # ADX formula approximation

        # ✅ Compute Bollinger Bands
        sma = np.mean(closes)  # 50-period SMA
        std_dev = np.std(closes)
        upper_band = sma + (2 * std_dev)
        lower_band = sma - (2 * std_dev)

        bollinger_bandwidth = (upper_band - lower_band) / sma  # ✅ Measures market compression

        # ✅ Compute Volume Trend
        volumes = market_data["volume"][-period:]
        volume_trend = np.mean(volumes[-5:]) - np.mean(volumes[:5])  # ✅ Measures recent volume change

        # ✅ Adjust Decision Logic
        if adx > 25 and bollinger_bandwidth > 0.05 and volume_trend > 0:
            return "TRENDING"
        elif adx < 25 and bollinger_bandwidth < 0.03 and volume_trend < 0:
            return "RANGING"
        else:
            return "NEUTRAL"
