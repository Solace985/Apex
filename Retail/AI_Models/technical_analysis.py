import talib
import numpy as np

class TechnicalAnalysis:
    """Handles technical indicator calculations."""

    def __init__(self):
        self.previous_trades = []

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

    def extract_technical_features(self, market_data):
        """Prepares technical indicators as a feature vector for MADDPG."""
        
        ema_fast = market_data.get("ema_fast", 0)
        ema_slow = market_data.get("ema_slow", 0)
        macd_signal = market_data.get("macd_signal", 0)
        rsi = market_data.get("rsi", 0)
        adx = market_data.get("adx", 0)
        vwap = market_data.get("vwap", 0)
        obv = market_data.get("obv", 0)

        # Convert indicators into a structured state representation for MADDPG
        state_representation = np.array([
            ema_fast, ema_slow, macd_signal, rsi, adx, vwap, obv
        ])

        return state_representation

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
