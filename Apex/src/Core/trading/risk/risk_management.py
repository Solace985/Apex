import logging
import numpy as np
import os
from typing import Dict, Any
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from Apex.src.ai.forecasting.order_flow import InstitutionalOrderFlow
from Apex.src.ai.reinforcement.maddpg_model import ReinforcementLearningRisk
from Apex.src.Core.trading.execution.market_impact import MarketImpactAnalyzer
from Apex.src.Core.trading.ai.config import load_config

config = load_config()

class RiskManager:
    """AI-powered risk evaluation, adaptive stop-loss, and dynamic position sizing."""

    def __init__(self, data_feed, portfolio_state, strategy_stats):
        self.data_feed = data_feed
        self.portfolio_state = portfolio_state
        self.strategy_stats = strategy_stats
        self.logger = logging.getLogger(self.__class__.__name__)

        # ✅ AI-Powered Risk Assessment Model
        self.model_path = config.paths.risk_model_path  # ✅ Configurable path
        self.model = self.load_or_initialize_model()
        self.trade_history = []  # ✅ Store trade outcomes for reinforcement learning

        # ✅ Real-Time Market Analysis Tools
        self.market_impact = MarketImpactAnalyzer()
        self.order_flow = InstitutionalOrderFlow()
        self.reinforcement_risk = ReinforcementLearningRisk()

    def load_or_initialize_model(self):
        """Loads the AI risk model from a file or initializes a new one."""
        if os.path.exists(self.model_path):
            self.logger.info(f"Loading AI risk model from {self.model_path}")
            return load_model(self.model_path)
        else:
            self.logger.warning("No pre-trained risk model found. Initializing new model.")
            model = Sequential([
                Dense(128, activation="relu", input_shape=(10,)),  # ✅ Increased feature size
                Dense(64, activation="relu"),
                Dense(1, activation="sigmoid")
            ])
            model.compile(optimizer="adam", loss="binary_crossentropy")
            return model

    def save_model(self):
        """Saves the trained AI model."""
        self.logger.info(f"Saving AI risk model to {self.model_path}")
        self.model.save(self.model_path)

    def train_model(self):
        """Reinforcement learning for risk assessment based on past trade results."""
        if len(self.trade_history) < 50:
            return  # ✅ Train only if enough trade history exists

        # ✅ Dynamic training based on prediction accuracy
        current_accuracy = self.evaluate_model_accuracy()
        if current_accuracy > 0.85:  # ✅ Train only if accuracy is below 85%
            return  

        data = np.array([trade['features'] for trade in self.trade_history])
        labels = np.array([trade['outcome'] for trade in self.trade_history])

        self.logger.info("Training AI risk model...")
        self.model.fit(data, labels, epochs=10, batch_size=32, verbose=0)
        self.trade_history.clear()
        self.save_model()

    def evaluate_model_accuracy(self):
        """Evaluates AI model prediction accuracy based on recent trade results."""
        if not self.trade_history:
            return 1.0  # ✅ Default high accuracy if no data yet

        data = np.array([trade['features'] for trade in self.trade_history])
        labels = np.array([trade['outcome'] for trade in self.trade_history])
        predictions = self.model.predict(data).flatten()
        accuracy = np.mean((predictions > 0.5) == labels)

        return accuracy

    def evaluate_trade(self, order: Dict[str, Any], market_data: Dict[str, Any]) -> str:
        """Evaluates trade risk and returns decision (APPROVED/REJECTED)."""

        # ✅ AI-Powered Trade Risk Prediction with Additional Features
        features = np.array([
            market_data["volatility"],
            market_data["spread"],
            market_data["liquidity"],
            market_data["institutional_activity"],
            market_data["momentum"],
            market_data["trend_strength"],
            self.order_flow.detect_accumulation(market_data["symbol"]),
            market_data["slippage"],  # ✅ Added slippage analysis
            market_data["leverage"],  # ✅ Added leverage risk
            market_data["open_interest"]  # ✅ Added open interest tracking
        ]).reshape(1, -1)

        risk_score = self.model.predict(features)[0][0]
        self.logger.info(f"Trade risk score: {risk_score:.2f}")

        if risk_score > 0.8:
            self.logger.warning("🚫 High-risk trade detected. Aborting.")
            return "REJECTED: High risk environment"

        # ✅ Adaptive Risk-Reward Ratio Based on AI Confidence
        ai_confidence = self.reinforcement_risk.get_trade_confidence(order["symbol"])
        min_rr_ratio = 1.5 if ai_confidence > 0.9 else 2.0  # ✅ Adaptive risk-reward threshold

        stop_loss = abs(order.get("entry_price") - order.get("stop_loss", 0))
        take_profit = abs(order.get("take_profit", 0) - order.get("entry_price"))
        risk_reward_ratio = take_profit / stop_loss if stop_loss > 0 else 0

        self.logger.info(f"Trade R/R Ratio: {risk_reward_ratio:.2f}")

        if risk_reward_ratio < min_rr_ratio:
            return "REJECTED: Poor Risk-Reward Ratio"

        # ✅ Market Volatility & Order Flow-Based Trade Filtering
        volatility = market_data.get("volatility", 0.01)
        institutional_buying = self.order_flow.detect_accumulation(order["symbol"])
        
        if volatility > config.risk.volatility_threshold and not institutional_buying:
            return "REJECTED: Excessive Market Volatility"

        return "TRADE APPROVED"

    def dynamic_stop_loss(self, market_volatility):
        """Adjusts stop-loss dynamically based on volatility and AI learning."""
        return max(0.005, min(config.risk.stop_loss * (1 + market_volatility / 100), 0.05))

    def calculate_position_size(self, symbol, strategy_type):
        """
        Determines position size dynamically using volatility and adaptive Kelly Criterion.
        """
        volatility = self.data_feed.get_historical_volatility(symbol)
        account_equity = self.portfolio_state.equity
        risk_capital = account_equity * config.risk.max_drawdown
        
        win_rate = self.reinforcement_risk.estimate_win_rate(strategy_type)
        avg_win_loss_ratio = self.strategy_stats[strategy_type]['pnl_ratio']
        kelly_fraction = win_rate - ((1 - win_rate) / avg_win_loss_ratio)
        
        # ✅ Volatility-Adjusted Position Sizing
        position_size = (risk_capital * kelly_fraction) / (volatility * 1.5)  # ✅ Scales down in volatile markets
        return round(position_size, 2)

    def asset_class_risk(self, symbol: str) -> float:
        """Get risk parameters per asset class"""
        asset_config = load_asset_config()  # From retail.yaml  
        if symbol in asset_config['crypto']:  
            return 0.08  # Higher risk tolerance  
        elif symbol in asset_config['equity']:  
            return 0.03  # Conservative  
        return 0.05  # Default  
