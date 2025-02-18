import logging
import numpy as np
from typing import Dict, Any
from keras.models import Sequential
from keras.layers import Dense
from Retail.Core.config import load_config

config = load_config()

class RiskManager:
    """Handles AI-driven risk evaluation, dynamic stop-loss adjustment, and position sizing."""

    def __init__(self, data_feed, portfolio_state, strategy_stats):
        self.data_feed = data_feed
        self.portfolio_state = portfolio_state
        self.strategy_stats = strategy_stats
        self.logger = logging.getLogger(self.__class__.__name__)

        # ✅ AI-Powered Risk Assessment Model
        self.model = Sequential([
            Dense(128, activation="relu", input_shape=(6,)),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        self.model.compile(optimizer="adam", loss="binary_crossentropy")
        self.trade_history = []  # ✅ Store trade outcomes for continuous learning

    def train_model(self):
        """Uses past trade outcomes to improve AI-based risk analysis."""
        if len(self.trade_history) < 50:
            return  # ✅ Train only if enough trade history exists

        data = np.array([trade['features'] for trade in self.trade_history])
        labels = np.array([trade['outcome'] for trade in self.trade_history])

        self.model.fit(data, labels, epochs=10, batch_size=32)
        self.trade_history.clear()  # ✅ Reset trade history after training

    def evaluate_trade(self, order: Dict[str, Any], market_data: Dict[str, Any]) -> str:
        """Evaluates trade risk and returns decision (APPROVED/REJECTED)."""

        # ✅ AI-Powered Trade Risk Prediction
        features = np.array([
            market_data["volatility"],
            market_data["spread"],
            market_data["liquidity"],
            market_data["institutional_activity"],
            market_data["momentum"],
            market_data["trend_strength"]
        ]).reshape(1, -1)

        risk_score = self.model.predict(features)[0][0]

        if risk_score > 0.8:
            self.logger.warning("🚫 High risk trade detected. Aborting.")
            return "REJECTED: High risk environment"

        # ✅ Dynamic Stop-Loss Adjustment
        order["stop_loss"] = self.dynamic_stop_loss(market_data["volatility"])

        # ✅ Position Sizing Using Kelly Criterion
        order["position_size"] = self.calculate_position_size(order["symbol"], order["strategy_type"])

        # ✅ Risk-Reward Ratio Evaluation
        stop_loss = abs(order.get("entry_price") - order.get("stop_loss", 0))
        take_profit = abs(order.get("take_profit", 0) - order.get("entry_price"))
        risk_reward_ratio = take_profit / stop_loss if stop_loss > 0 else 0

        if risk_reward_ratio < 2:
            return "REJECTED: Poor Risk-Reward Ratio"

        # ✅ Volatility-Based Trade Rejection
        volatility = market_data.get("volatility", 0.01)
        if volatility > config.risk.volatility_threshold:
            return "REJECTED: Excessive Market Volatility"

        # ✅ Slippage-Based Trade Rejection
        slippage = market_data.get("slippage", 0.005)
        if slippage > config.risk.max_drawdown:
            return "REJECTED: Excessive Slippage"

        return "TRADE APPROVED"

    def dynamic_stop_loss(self, market_volatility):
        """Adjusts stop-loss dynamically based on volatility."""
        return max(0.005, min(config.risk.stop_loss * (1 + market_volatility / 100), 0.05))

    def calculate_position_size(self, symbol, strategy_type):
        """
        Determines position size dynamically using volatility and Kelly Criterion.
        """
        volatility = self.data_feed.get_historical_volatility(symbol)
        account_equity = self.portfolio_state.equity
        risk_capital = account_equity * config.risk.max_drawdown
        
        win_rate = self.strategy_stats[strategy_type]['win_rate']
        avg_win_loss_ratio = self.strategy_stats[strategy_type]['pnl_ratio']
        kelly_fraction = win_rate - ((1 - win_rate) / avg_win_loss_ratio)
        
        position_size = (risk_capital * kelly_fraction) / volatility
        return round(position_size, 2)
