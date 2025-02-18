import yaml
import numpy as np
from typing import Dict, Any
from abc import ABC, abstractmethod
from ai_models.maddpg_model import ReinforcementLearningStrategy
from ai_models.lstm_model import LSTMPricePredictor
from core.risk_management import RiskManager
from core.market_impact import MarketImpactAnalyzer

# Load strategy configuration
def load_strategy_config():
    with open("Retail/config/strategy_config.yaml", "r") as file:
        return yaml.safe_load(file)

config = load_strategy_config()

class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trade signals based on market data."""
        pass

class StrategyManager:
    def __init__(self):
        self.strategies = {
            "mean_reversion": MeanReversionStrategy(),
            "momentum_breakout": MomentumBreakoutStrategy(),
            "volatility_scalping": VolatilityScalpingStrategy(),
            "fundamental_trend_following": FundamentalTrendStrategy(),
            "macroeconomic_trend": MacroEconomicTrendStrategy(),
        }
        self.risk_manager = RiskManager()
        self.market_impact = MarketImpactAnalyzer()
        self.reinforcement_model = ReinforcementLearningStrategy()
        self.lstm_predictor = LSTMPricePredictor()

    def select_strategy(self, asset_class: str, market_data: Dict[str, Any]):
        """Dynamically selects the best strategy for the given asset based on market conditions."""
        strategy_options = config["strategies"].get(asset_class, ["mean_reversion"])
        
        # Rank strategies using AI reinforcement learning
        best_strategy = self.reinforcement_model.rank_strategies(strategy_options, market_data)

        # Apply AI-based market condition adaptation
        volatility_level = self.market_impact.analyze_volatility(market_data)
        if volatility_level > 2.0:
            best_strategy = "volatility_scalping"

        return self.strategies.get(best_strategy, DefaultStrategy())

    def generate_trade_signal(self, asset_class: str, market_data: Dict[str, Any]):
        """Generates trade signals based on AI-driven strategy selection."""
        selected_strategy = self.select_strategy(asset_class, market_data)
        raw_signal = selected_strategy.generate_signal(market_data)

        # Adjust trade size & risk dynamically
        final_signal = self.risk_manager.adjust_trade_signal(raw_signal, asset_class, market_data)
        return final_signal

# Example Strategy Implementations
class MeanReversionStrategy(Strategy):
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mean reversion strategy based on RSI and Bollinger Bands."""
        rsi = market_data.get("rsi", 50)
        if rsi < 30:
            return {"action": "BUY", "reason": "RSI oversold, likely to revert up"}
        elif rsi > 70:
            return {"action": "SELL", "reason": "RSI overbought, likely to revert down"}
        return {"action": "HOLD", "reason": "No strong mean reversion signal"}

class MomentumBreakoutStrategy(Strategy):
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Momentum breakout strategy using volume and price action."""
        if market_data["price"] > market_data["resistance"]:
            return {"action": "BUY", "reason": "Price broke resistance, strong momentum"}
        return {"action": "HOLD", "reason": "No breakout detected"}

class VolatilityScalpingStrategy(Strategy):
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Scalping strategy in high volatility environments."""
        if market_data["volatility"] > 2.5:
            return {"action": "SELL", "reason": "High volatility detected, taking short-term profits"}
        return {"action": "HOLD", "reason": "No extreme volatility"}

class FundamentalTrendStrategy(Strategy):
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Long-term fundamental trend strategy based on economic reports."""
        if market_data["gdp_growth"] > 2.5 and market_data["inflation"] < 2.0:
            return {"action": "BUY", "reason": "Strong GDP growth and low inflation"}
        return {"action": "HOLD", "reason": "Mixed fundamental signals"}

class MacroEconomicTrendStrategy(Strategy):
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Macro-driven strategy based on CPI and unemployment rates."""
        if market_data["cpi"] > 3.0 and market_data["unemployment"] > 5.0:
            return {"action": "SELL", "reason": "Rising inflation and weak labor market"}
        return {"action": "HOLD", "reason": "No strong macro trend"}

class DefaultStrategy(Strategy):
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "HOLD", "reason": "No strategy available"}

