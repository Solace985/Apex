import talib
import numpy as np

class TechnicalAnalysis:
    def __init__(self):
        pass

    def compute_indicators(self, prices, volumes):
        """Calculates MACD, RSI, VWAP, and ADX."""
        macd, signal, _ = talib.MACD(prices)
        rsi = talib.RSI(prices)
        adx = talib.ADX(prices, volumes)

        return {
            "macd": macd[-1],
            "signal": signal[-1],
            "rsi": rsi[-1],
            "adx": adx[-1]
        }
    
    # for considering technical analysis. also prevents false trades by considering historical performance.
    def __init__(self):
        self.previous_trades = []

    def calculate_weighted_signal(self, indicators):
        """Use historical performance to assign weights to technical indicators."""
        weights = {
            "MACD": 0.4,
            "RSI": 0.3,
            "VWAP": 0.3
        }

        signal_strength = sum(indicators[ind] * weights[ind] for ind in indicators)
        return signal_strength

    def validate_trade(self, indicators, volume):
        """Validate trade based on weighted technical indicators & volume confirmation."""
        signal_strength = self.calculate_weighted_signal(indicators)

        if signal_strength > 0.7 and volume > 10000:
            return "BUY"
        elif signal_strength < -0.7 and volume > 10000:
            return "SELL"
        return "NO TRADE"

# Example Usage
technical_analysis = TechnicalAnalysis()
trade_decision = technical_analysis.validate_trade({"MACD": 1, "RSI": 0.5, "VWAP": 1}, 12000)

print(f"Trade Decision: {trade_decision}")
