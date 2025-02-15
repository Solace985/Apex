from typing import List, Dict, Any
from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

class StrategyManager:
    def __init__(self):
        self.strategies: List[Strategy] = []

    def add_strategy(self, strategy: Strategy):
        self.strategies.append(strategy) 