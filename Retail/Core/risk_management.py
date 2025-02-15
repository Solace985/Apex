import logging
import numpy as np
from typing import Dict, Any
from keras.models import Sequential
from keras.layers import Dense

class RiskManagement:
    """Handles risk evaluation and position sizing."""
    
    def __init__(self, data_feed, portfolio_state, strategy_stats, config):
        self.data_feed = data_feed
        self.portfolio_state = portfolio_state
        self.strategy_stats = strategy_stats
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate_risk(self, trade_details: Dict[str, Any]) -> bool:
        """Rejects trades exceeding risk threshold."""
        risk = trade_details.get("risk", 0)
        if risk > self.config.risk.max_drawdown:
            self.logger.warning(f"Trade rejected due to high risk: {trade_details}")
            return False
        return True

    def adjust_position_size(self, trade_details: Dict[str, Any]) -> Dict[str, Any]:
        """Adjusts position size dynamically based on risk."""
        risk = trade_details.get("risk", 0.01)
        trade_details["position_size"] = min(1.0, self.config.risk.capital_exposure_limit / risk)
        return trade_details

    def calculate_position_size(self, symbol, strategy_type):
        """
        Dynamically determines position size using volatility and Kelly Criterion.
        """
        volatility = self.data_feed.get_historical_volatility(symbol)
        account_equity = self.portfolio_state.equity
        risk_capital = account_equity * self.config.risk.max_drawdown
        
        win_rate = self.strategy_stats[strategy_type]['win_rate']
        avg_win_loss_ratio = self.strategy_stats[strategy_type]['pnl_ratio']
        kelly_fraction = win_rate - ((1 - win_rate) / avg_win_loss_ratio)
        
        position_size = (risk_capital * kelly_fraction) / volatility
        return round(position_size, 2)

class AdaptiveRiskManagement:
    """Dynamically adjusts risk based on market volatility & liquidity."""

    def __init__(self, data_feed, portfolio_state, strategy_stats, config):
        self.data_feed = data_feed
        self.portfolio_state = portfolio_state
        self.strategy_stats = strategy_stats
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def adjust_risk_parameters(self, market_data: Dict[str, Any]):
        """
        Dynamically adjust drawdown & exposure limits based on real-time volatility & liquidity.
        """
        volatility = np.std(market_data.get("price_series", []))  # Calculate standard deviation as volatility
        liquidity = market_data.get("order_book_liquidity", 0)  # Fetch liquidity data
        
        # If volatility is high, reduce drawdown & risk exposure
        if volatility > self.config.risk.volatility_threshold:
            self.config.risk.max_drawdown = 0.1  # Reduce drawdown limit
            self.config.risk.capital_exposure_limit = 0.05  # Lower position size
            self.logger.warning(f"High Volatility Detected: Reducing Risk Exposure (Max Drawdown: {self.config.risk.max_drawdown}, Position Size: {self.config.risk.capital_exposure_limit})")

        # If liquidity is low, avoid trading
        if liquidity < 5000:  # Assuming a fixed liquidity threshold
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

        if slippage > self.config.risk.max_drawdown:
            return "REJECTED: Excessive Slippage"

        return "TRADE APPROVED"

class AdaptiveRiskManager:
    """AI-driven risk management with dynamic stop-loss adjustment."""

    def __init__(self, max_drawdown=0.02):
        self.max_drawdown = max_drawdown

    def dynamic_stop_loss(self, market_volatility):
        """
        Adjusts stop-loss dynamically based on volatility.
        """
        return max(0.005, min(self.max_drawdown * (1 + market_volatility / 100), 0.05))

    def evaluate_risk(self, trade_details: Dict[str, Any], market_volatility: float) -> bool:
        """
        AI-driven risk evaluation before executing trade.
        """
        stop_loss = self.dynamic_stop_loss(market_volatility)
        trade_risk = trade_details.get("risk", 0)
        return trade_risk <= stop_loss

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

import numpy as np

class RiskManager:
    def __init__(self, max_drawdown=0.02):
        self.max_drawdown = max_drawdown

    def calculate_position_size(self, symbol, strategy_type, portfolio_state, strategy_stats, data_feed):
        """
        Dynamically determines position size using volatility and Kelly Criterion.
        """
        volatility = data_feed.get_historical_volatility(symbol)
        account_equity = portfolio_state.equity
        risk_capital = account_equity * self.max_drawdown
        
        # Kelly Criterion-based position sizing
        win_rate = strategy_stats[strategy_type]['win_rate']
        avg_win_loss_ratio = strategy_stats[strategy_type]['pnl_ratio']
        kelly_fraction = win_rate - ((1 - win_rate) / avg_win_loss_ratio)
        
        position_size = (risk_capital * kelly_fraction) / volatility
        return round(position_size, 2)
