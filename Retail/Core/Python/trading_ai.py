from Retail.Core.Python.config import load_config

config = load_config()

from Retail.AI_Models.Timeseries.lstm_model import LSTMModel
from Retail.AI_Models.technical_analysis import TechnicalAnalysis
from Retail.AI_Models.sentiment_analysis import SentimentAnalyzer
from Retail.AI_Models.Reinforcement.maddpg_model import MADDPG

class TradingAI:
    """AI-driven trade signal generation using LSTM, Technical, Sentiment, and ML analysis."""

    def __init__(self):
        self.lstm_model = LSTMModel(time_steps=60, features=1)
        self.technical_analysis = TechnicalAnalysis()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.maddpg = MADDPG(state_dim=10, action_dim=1)
        self.enable_live_adaptation = config.ai_trading.enable_live_adaptation
        self.strategy_selection = config.ai_trading.strategy_selection

    def get_trade_signal(self, market_data):
        """
        Uses AI models (LSTM, Technical, Sentiment) to generate a trade signal (BUY, SELL, HOLD).
        """
        # Preprocess the data for LSTM
        scaled_data = self.lstm_model.scale_data(market_data)

        # Predict future price movement
        predicted_price = self.lstm_model.predict(scaled_data)

        # Compute technical indicators
        rsi = self.technical_analysis.relative_strength_index(market_data["price"])
        macd, signal = self.technical_analysis.macd(market_data["price"])
        mfi = self.technical_analysis.money_flow_index(market_data["high"], market_data["low"], market_data["price"], market_data["volume"])

        # Aggregate the final technical signal
        tech_signal = (rsi * 0.3) + (macd * 0.3) + (mfi * 0.4)

        sentiment_score = self.sentiment_analyzer.get_sentiment(market_data)
        ai_decision = self.maddpg.select_action(tech_signal)

        # Weighted Decision Making
        final_decision = (ai_decision * 0.4) + (tech_signal * 0.3) + (sentiment_score * 0.2) + (predicted_price * 0.1)

        if final_decision > 0.6:
            return {"symbol": market_data["symbol"], "action": "BUY", "confidence": final_decision}
        elif final_decision < 0.4:
            return {"symbol": market_data["symbol"], "action": "SELL", "confidence": final_decision}
        else:
            return None
