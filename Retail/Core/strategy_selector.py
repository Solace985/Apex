import yaml
from typing import Dict, Any
from abc import ABC, abstractmethod

# Load strategy configuration
def load_strategy_config():
    with open("Retail/Config/strategy_config.yaml", "r") as file:
        return yaml.safe_load(file)

config = load_strategy_config()

class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trade signals based on market data."""
        pass

class StrategyManager:
    def __init__(self):
        self.strategies = {}

    def select_strategy(self, asset_class: str):
        """Selects the correct strategy for the given asset class."""
        strategy_name = config["strategies"].get(asset_class, "default_strategy")
        return self.load_strategy(strategy_name)

    def load_strategy(self, strategy_name: str):
        """Dynamically loads the required strategy class."""
        strategy_classes = {
            "mean_reversion": MeanReversionStrategy(),
            "momentum_breakout": MomentumBreakoutStrategy(),
            "volatility_scalping": VolatilityScalpingStrategy(),
            "fundamental_trend_following": FundamentalTrendStrategy(),
            "macroeconomic_trend": MacroEconomicTrendStrategy(),
        }
        return strategy_classes.get(strategy_name, DefaultStrategy())

    def get_indicators(self, asset_class: str):
        """Fetches the relevant technical indicators for the asset type."""
        return config["indicators"].get(asset_class, [])

    def get_fundamental_sources(self, asset_class: str):
        """Fetches relevant fundamental analysis sources for the asset type."""
        return config["fundamental_sources"].get(asset_class, [])

# Example Strategy Implementations
class MeanReversionStrategy(Strategy):
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "BUY", "reason": "RSI oversold"}

class MomentumBreakoutStrategy(Strategy):
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "BUY", "reason": "Price broke resistance"}

class VolatilityScalpingStrategy(Strategy):
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "SELL", "reason": "High volatility detected"}

class FundamentalTrendStrategy(Strategy):
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "BUY", "reason": "Positive economic reports"}

class MacroEconomicTrendStrategy(Strategy):
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "HOLD", "reason": "High CPI, mixed signals"}

class DefaultStrategy(Strategy):
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "HOLD", "reason": "No strategy available"}