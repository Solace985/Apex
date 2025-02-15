import logging
import numpy as np
from typing import Dict, Any
from keras.models import Sequential
from keras.layers import Dense

class RiskManagement:
    """Handles risk evaluation and position sizing."""
    
    def __init__(self, max_drawdown=0.2, capital_exposure_limit=0.1):
        self.max_drawdown = max_drawdown
        self.capital_exposure_limit = capital_exposure_limit
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate_risk(self, trade_details: Dict[str, Any]) -> bool:
        """Rejects trades exceeding risk threshold."""
        risk = trade_details.get("risk", 0)
        if risk > self.capital_exposure_limit:
            self.logger.warning(f"Trade rejected due to high risk: {trade_details}")
            return False
        return True

    def adjust_position_size(self, trade_details: Dict[str, Any]) -> Dict[str, Any]:
        """Adjusts position size dynamically based on risk."""
        risk = trade_details.get("risk", 0.01)
        trade_details["position_size"] = min(1.0, self.capital_exposure_limit / risk)
        return trade_details
    
    def __init__(self):
        self.risk_to_reward_ratio = 2  # Enforce 1:2 risk-reward ratio
        self.max_slippage = 0.5  # Maximum acceptable slippage in %

    def evaluate_trade(self, entry_price, stop_loss, take_profit, slippage):
        """Evaluate trade based on risk-reward ratio & slippage constraints."""
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)

        if reward / risk < self.risk_to_reward_ratio:
            return "REJECTED: Poor Risk-Reward Ratio"
        if slippage > self.max_slippage:
            return "REJECTED: Excessive Slippage"

        return "TRADE APPROVED"

class AdaptiveRiskManagement:
    """Dynamically adjusts risk based on market volatility & liquidity."""

    def __init__(self, max_drawdown=0.2, capital_exposure_limit=0.1):
        self.max_drawdown = max_drawdown  # Default drawdown limit
        self.capital_exposure_limit = capital_exposure_limit  # % of capital to risk per trade
        self.volatility_threshold = 0.03  # Default volatility threshold (3%)
        self.liquidity_threshold = 5000  # Minimum liquidity needed for safe trade execution
        self.logger = logging.getLogger(self.__class__.__name__)

    def adjust_risk_parameters(self, market_data: Dict[str, Any]):
        """
        Dynamically adjust drawdown & exposure limits based on real-time volatility & liquidity.
        """
        volatility = np.std(market_data.get("price_series", []))  # Calculate standard deviation as volatility
        liquidity = market_data.get("order_book_liquidity", 0)  # Fetch liquidity data
        
        # If volatility is high, reduce drawdown & risk exposure
        if volatility > self.volatility_threshold:
            self.max_drawdown = 0.1  # Reduce drawdown limit
            self.capital_exposure_limit = 0.05  # Lower position size
            self.logger.warning(f"High Volatility Detected: Reducing Risk Exposure (Max Drawdown: {self.max_drawdown}, Position Size: {self.capital_exposure_limit})")

        # If liquidity is low, avoid trading
        if liquidity < self.liquidity_threshold:
            self.logger.warning("⚠ Low Liquidity: Avoiding trade due to high slippage risk.")
            return False

        return True

    def evaluate_trade(self, entry_price, stop_loss, take_profit, slippage, market_data):
        """
        Evaluate trades dynamically based on real-time risk conditions.
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)

        # Adjust risk parameters dynamically
        if not self.adjust_risk_parameters(market_data):
            return "REJECTED: Market Conditions Not Favorable"

        if reward / risk < 2:  # Maintain at least 1:2 risk-reward ratio
            return "REJECTED: Poor Risk-Reward Ratio"

        if slippage > self.max_drawdown:
            return "REJECTED: Excessive Slippage"

        return "TRADE APPROVED"

class ReinforcementRiskManagement:
    """
    AI-powered risk management that continuously improves using reinforcement learning.
    """

    def __init__(self):
        self.model = Sequential([
            Dense(128, activation="relu", input_shape=(6,)),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        self.model.compile(optimizer="adam", loss="binary_crossentropy")
        self.trade_history = []  # ✅ Store trade outcomes for continuous training
        self.logger = logging.getLogger(self.__class__.__name__)

    def train_model(self):
        """
        Uses trade history to train reinforcement learning model.
        """
        if len(self.trade_history) < 50:
            return  # Train only if enough data exists

        data = np.array([trade['features'] for trade in self.trade_history])
        labels = np.array([trade['outcome'] for trade in self.trade_history])

        self.model.fit(data, labels, epochs=10, batch_size=32)
        self.trade_history.clear()  # Reset history after training

    def evaluate_trade(self, market_data):
        """
        Predicts trade risk dynamically using real-time reinforcement learning.
        """

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
        
        return "TRADE APPROVED"
