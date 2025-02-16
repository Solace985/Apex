from Retail.AI_Models.technical_analysis import TechnicalAnalysis
from Retail.AI_Models.sentiment_analysis import SentimentAnalyzer
from Retail.AI_Models.maddpg_model import MADDPG

class TradingAI:
    """AI-driven trade signal generation using technical, sentiment, and ML analysis."""

    def __init__(self):
        self.technical_analysis = TechnicalAnalysis()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.maddpg = MADDPG(state_dim=10, action_dim=1)

    def get_trade_signal(self, market_data):
        """Uses AI models to generate a trade signal (BUY, SELL, HOLD)."""
        tech_signal = self.technical_analysis.analyze(market_data)
        sentiment_score = self.sentiment_analyzer.get_sentiment(market_data)
        ai_decision = self.maddpg.select_action(tech_signal)

        final_decision = (ai_decision * 0.5) + (tech_signal * 0.3) + (sentiment_score * 0.2)

        if final_decision > 0.6:
            return {"symbol": market_data["symbol"], "action": "BUY", "confidence": final_decision}
        elif final_decision < 0.4:
            return {"symbol": market_data["symbol"], "action": "SELL", "confidence": final_decision}
        else:
            return None
