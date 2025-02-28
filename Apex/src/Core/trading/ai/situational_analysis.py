import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from collections import deque
from Apex.src.Core.data.market_data import MarketDataFeed
from Apex.src.Core.trading.execution.meta_trader import MetaTrader
from Apex.src.Core.trading.risk.risk_management import RiskManager
from Apex.src.Core.trading.ai.trading_ai import TradingAI
from Apex.src.AI.ensembles.ensemble_voting import EnsembleVoting
from Apex.utils.helpers import secure_float, validate_inputs

# Define common situational patterns
SITUATIONAL_PATTERNS = {
    "Monday Low Test": "If Friday’s High is lower than Thursday’s High, Monday often retests Friday’s Low.",
    "Thursday Revisit": "If Wednesday’s High is lower than Monday’s High, Thursday tends to revisit Wednesday’s Low.",
    "Inside Day Breakout": "If today’s High/Low is inside yesterday’s range, a breakout often follows."
}

class SituationalAnalysis:
    """📌 AI-powered market structure and pattern recognition system for trade optimization."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.market_data = MarketDataFeed(symbol)
        self.meta_trader = MetaTrader()
        self.risk_manager = RiskManager()
        self.trading_ai = TradingAI()
        self.ensemble_voter = EnsembleVoting()

        # Store historical OHLC data for pattern recognition
        self.price_history = deque(maxlen=30)  # Store last 30 trading days

        # Cache for situational pattern tracking
        self.detected_patterns = {}

    async def analyze_market_conditions(self) -> Dict[str, Any]:
        """📌 Detects key price-action patterns and generates market context insights."""
        market_data = await self.market_data.get_historical_data(days=30)
        if not market_data:
            return {"error": "No market data available."}

        # Parse historical OHLC data
        self.price_history.clear()
        for day in market_data:
            self.price_history.append({
                "date": day["date"],
                "open": secure_float(day["open"]),
                "high": secure_float(day["high"]),
                "low": secure_float(day["low"]),
                "close": secure_float(day["close"])
            })

        # Identify market patterns
        identified_patterns = self._detect_patterns()
        risk_adjustments = self._analyze_risk_factors(identified_patterns)

        # Adjust AI model weightings if needed
        await self._adjust_ai_model_weights(identified_patterns)

        return {
            "identified_patterns": identified_patterns,
            "risk_adjustments": risk_adjustments
        }

    def _detect_patterns(self) -> Dict[str, Any]:
        """📌 Identifies repeating price-action patterns based on historical market data."""
        patterns = {}
        if len(self.price_history) < 10:
            return {"error": "Not enough historical data to detect patterns."}

        for i in range(1, len(self.price_history) - 1):
            today = self.price_history[i]
            yesterday = self.price_history[i - 1]
            day_before = self.price_history[i - 2]

            # Check for "Monday Low Test" Pattern
            if i >= 5 and self.price_history[i - 3]["high"] < self.price_history[i - 4]["high"]:
                patterns["Monday Low Test"] = {
                    "prediction": f"{self.symbol} likely to revisit {self.price_history[i-1]['low']} on Monday.",
                    "confidence_score": 80
                }

            # Check for "Thursday Revisit" Pattern
            if i >= 3 and self.price_history[i - 2]["high"] < self.price_history[i - 5]["high"]:
                patterns["Thursday Revisit"] = {
                    "prediction": f"{self.symbol} may revisit {self.price_history[i-1]['low']} on Thursday.",
                    "confidence_score": 75
                }

            # Check for "Inside Day Breakout"
            if yesterday["high"] > today["high"] and yesterday["low"] < today["low"]:
                patterns["Inside Day Breakout"] = {
                    "prediction": f"Potential breakout for {self.symbol} as inside-day detected.",
                    "confidence_score": 85
                }

        self.detected_patterns = patterns
        return patterns

    def _analyze_risk_factors(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """📌 Adjusts risk parameters based on situational analysis insights."""
        risk_adjustments = {}
        for pattern, data in patterns.items():
            if pattern == "Inside Day Breakout":
                risk_adjustments["Inside Day Breakout"] = {
                    "stop_loss": "10% wider",
                    "take_profit": "15% closer",
                    "trade_confidence": "+5%"
                }
            elif pattern == "Monday Low Test":
                risk_adjustments["Monday Low Test"] = {
                    "stop_loss": "5% tighter",
                    "take_profit": "Normal",
                    "trade_confidence": "-5%"
                }
            elif pattern == "Thursday Revisit":
                risk_adjustments["Thursday Revisit"] = {
                    "stop_loss": "7% tighter",
                    "take_profit": "Slightly closer",
                    "trade_confidence": "-3%"
                }

        return risk_adjustments

    async def _adjust_ai_model_weights(self, patterns: Dict[str, Any]):
        """📌 Adjusts AI model weighting if situational analysis suggests strong patterns."""
        if not patterns:
            return

        for pattern, data in patterns.items():
            if data["confidence_score"] > 80:
                adjustment = {
                    "TA": 0.4,
                    "FA": 0.2,
                    "SA": 0.2,
                    "RL": 0.2
                }
                await self.ensemble_voter.update_model_weights(adjustment)

    async def get_situational_context(self) -> Dict[str, Any]:
        """📌 Provides a real-time API for other modules to query situational insights."""
        return {
            "market_context": await self.analyze_market_conditions(),
            "patterns_detected": self.detected_patterns
        }

    async def integrate_with_trade_execution(self, trade_data: Dict[str, Any]):
        """📌 Modifies trade execution based on situational insights."""
        patterns = await self.analyze_market_conditions()
        if not patterns["identified_patterns"]:
            return trade_data  # No adjustments if no patterns detected

        adjustments = patterns["risk_adjustments"]
        trade_data["adjustments"] = adjustments
        return trade_data
