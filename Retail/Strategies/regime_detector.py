import numpy as np
from scipy.stats import norm

class MarketRegimeEngine:
    def __init__(self, lookback=30):
        self.lookback = lookback
        
    def detect_regime(self, prices):
        """
        Detects market regime (trending, mean-reverting, high volatility)
        """
        returns = np.diff(np.log(prices))
        volatility = np.std(returns) * np.sqrt(252)

        if volatility > 0.25:
            return "high_volatility"
        elif abs(np.mean(returns)) > 0.001:
            return "trending"
        return "mean_reverting"
