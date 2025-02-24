import time
import logging
import numpy as np
from Retail.Core.Python.config import load_config
from Retail.AI_Models.Timeseries.lstm_model import LSTMModel
from Retail.AI_Models.technical_analysis import TechnicalAnalysis
from Retail.AI_Models.sentiment_analysis import SentimentAnalyzer
from Retail.AI_Models.Reinforcement.maddpg_model import MADDPG

logger = logging.getLogger(__name__)
config = load_config()

class TradingAI:
    """AI-driven trade signal generation using LSTM, Technical, Sentiment, and ML analysis."""

    def __init__(self):
        self.lstm_model = LSTMModel(time_steps=60, features=1)
        self.technical_analysis = TechnicalAnalysis()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.maddpg = MADDPG(state_dim=10, action_dim=1)
        self.enable_live_adaptation = config.ai_trading.enable_live_adaptation
        self.strategy_selection = config.ai_trading.strategy_selection
        self.retrain_interval = config.ai_trading.retrain_interval  # Retrain every X days
        self.last_retrain = None
        self.market_data_buffer = []  # Initialize market data buffer
        self.last_prediction = None  # Initialize cache for AI predictions

    def get_trade_signal(self, market_data):
        """Uses AI models (LSTM, Technical, Sentiment) to generate a trade signal (BUY, SELL, HOLD)."""
        
        # ✅ Validate input data
        required_keys = ["symbol", "price", "high", "low", "volume"]
        for key in required_keys:
            if key not in market_data or market_data[key] is None:
                logger.error(f"❌ Missing required market data: {key}")
                return {"error": f"❌ Missing required market data: {key}"}

        try:
            market_data = {k: float(v) for k, v in market_data.items() if k in required_keys}
        except ValueError:
            logger.error("❌ Invalid market data type (must be numeric).")
            return {"error": "❌ Invalid market data type (must be numeric)."}

        # Store incoming data
        self.market_data_buffer.append(market_data["price"])  # ✅ Append only price data
        if len(self.market_data_buffer) >= self.lstm_model.time_steps:
            logger.info("🔄 Retraining AI models with accumulated data...")
            try:
                batch_data = np.array(self.market_data_buffer[-self.lstm_model.time_steps:])  
                self.lstm_model.train(batch_data)
                self.maddpg.train(batch_data)
                self.last_retrain = time.time()
                self.market_data_buffer = []  # ✅ Clear buffer after training
            except Exception as e:
                logger.error(f"❌ Model Retraining Error: {e}")
                self.market_data_buffer = []  # Clear buffer even if training fails

        # ✅ Check for sufficient historical data for LSTM prediction
        if not isinstance(market_data["price"], (list, np.ndarray)) or len(market_data["price"]) < self.lstm_model.time_steps:
            logger.error("❌ Insufficient historical data for LSTM prediction.")
            return {"error": "❌ Not enough historical data for AI prediction."}

        # ✅ Generate AI-driven trading signals
        try:
            scaled_data = self.lstm_model.scale_data(market_data)

            # ✅ Cache AI Predictions for Performance Optimization
            if self.last_prediction is None or self.last_retrain is None or (time.time() - self.last_retrain > self.retrain_interval):
                try:
                    self.last_prediction = self.lstm_model.predict([scaled_data])
                except Exception as e:
                    logger.error(f"❌ LSTM Prediction Failed: {e}")
                    self.last_prediction = 0  # Fallback to neutral prediction

            predicted_price = self.last_prediction  # Use cached prediction
            predicted_price = np.clip(predicted_price, np.percentile(predicted_price, 5), np.percentile(predicted_price, 95))  # Remove outliers
        except Exception as e:
            logger.error(f"❌ LSTM Model Prediction Error: {e}")
            predicted_price = 0  # Default to no predicted movement if AI fails

        try:
            rsi = self.technical_analysis.relative_strength_index(market_data["price"])
            macd, signal = self.technical_analysis.macd(market_data["price"])
            mfi = self.technical_analysis.money_flow_index(
                market_data["high"], market_data["low"], market_data["price"], market_data["volume"]
            )
            tech_signal = (rsi * 0.3) + (macd * 0.3) + (mfi * 0.4)
        except Exception as e:
            logger.error(f"❌ Technical Indicator Error: {e}")
            tech_signal = 0  # ✅ Default to no technical signal if calculations fail

        try:
            sentiment_score = self.sentiment_analyzer.get_sentiment(market_data)
        except Exception as e:
            logger.error(f"❌ Sentiment Analysis Error: {e}")
            sentiment_score = 0  # Default to neutral sentiment if an error occurs
        try:
            ai_decision = np.clip(self.maddpg.select_action(tech_signal), -0.8, 0.8)  # ✅ More stable decision range
        except Exception as e:
            logger.error(f"❌ Reinforcement Learning Decision Error: {e}")
            ai_decision = 0  # ✅ Default to HOLD if AI fails

        # ✅ Dynamic weighted decision (Prevent division by zero)
        total_confidence = max(1e-6, abs(ai_decision) + abs(tech_signal) + abs(sentiment_score) + abs(predicted_price) + 1e-6)

        if total_confidence == 0:
            logger.warning("⚠️ All indicators are zero, defaulting to HOLD.")
            return {"symbol": market_data["symbol"], "action": "HOLD", "confidence": 0.5}

        final_decision = (
            (ai_decision / total_confidence) +
            (tech_signal / total_confidence) +
            (sentiment_score / total_confidence) +
            (predicted_price / total_confidence)
        )

        decision_reason = (
            f"🔍 Trade Decision -> {market_data['symbol']}, "
            f"AI: {ai_decision:.4f}, RSI: {rsi:.2f}, MACD: {macd:.2f}, "
            f"MFI: {mfi:.2f}, Sentiment: {sentiment_score:.2f}, "
            f"Predicted: {predicted_price:.4f}, Final Decision: {final_decision:.4f}"
        )

        if final_decision > 0.6:
            decision_reason += " 📈 AI sees a strong uptrend → BUY"
        elif final_decision < 0.4:
            decision_reason += " 📉 AI detects a strong downtrend → SELL"
        else:
            decision_reason += " 🤔 Market uncertainty → HOLD"

        logger.info(decision_reason)

        return {
            "symbol": market_data["symbol"],
            "action": "BUY" if final_decision > 0.6 else "SELL" if final_decision < 0.4 else "HOLD",
            "confidence": final_decision
        }
