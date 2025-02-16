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

        return "TRENDING" if adx > 25 else "RANGING"
