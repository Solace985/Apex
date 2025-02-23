import logging
import subprocess
from Strategies.mean_reversion import MeanReversionStrategy
from Strategies.momentum_breakout import MomentumBreakoutStrategy
from Strategies.trend_following import TrendFollowingStrategy
from Strategies.regime_detection import RegimeDetection
from Strategies.strategy_selector import StrategySelector

class StrategyOrchestrator:
    """Manages and switches between different strategies dynamically."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.regime_detector = RegimeDetection()
        self.mean_reversion = MeanReversionStrategy()
        self.momentum_breakout = MomentumBreakoutStrategy()
        self.trend_following = TrendFollowingStrategy()
        self.strategy_selector = StrategySelector()

    def select_best_strategy(self, market_data):
        """Dynamically selects the best strategy based on real-time market conditions."""
        market_regime = self.regime_detector.detect_market_regime(market_data)

        strategy_options = {
            "trending": self.trend_following,
            "mean_reverting": self.mean_reversion,
            "breakout": self.momentum_breakout,
        }

        selected_strategy = strategy_options.get(market_regime, self.trend_following)
        
        self.logger.info(f"Market Regime Detected: {market_regime}. Activating {selected_strategy.__class__.__name__}.")
        return selected_strategy

    def execute_trades(self, market_data):
        """Runs the selected strategy and executes the trade."""
        strategy = self.select_best_strategy(market_data)
        trade_signal = strategy.generate_signal(market_data)

        if trade_signal:
            print(f"✅ Executing Trade: {trade_signal}")

            # 🔹 Call Rust execution system
            command = [
                "./target/release/trading_bot",  # Adjust path if needed
                "--symbol", trade_signal["symbol"],
                "--order_type", trade_signal["order_type"]
            ]
            subprocess.run(command)
            return trade_signal  # Send signal to order execution engine
        return None