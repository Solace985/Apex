# src/AI/analysis/market_regime_classifier.py
import numpy as np
from typing import Dict, Tuple, Any
from datetime import datetime
from sklearn.ensemble import IsolationForest
from Apex.utils.helpers import validate_inputs, secure_float
from Apex.src.Core.data.realtime.market_data import MarketDataFeed
from Apex.src.ai.ensembles.transformers_lstm import TimeSeriesTransformer
from Apex.src.Core.trading.strategies.regime_detection import RegimeParameters

class QuantumRegimeClassifier:
    """AI-powered market regime classification with 6-layer detection"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.data_feed = MarketDataFeed(symbol)
        self.transformer = TimeSeriesTransformer()
        self._load_models()
        self._setup_thresholds()
        
    def _load_models(self):
        """Load pre-trained models without reimplementation"""
        self.lstm_model = self._load_lstm_model()
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.rf_classifier = self._load_rf_classifier()

    def _setup_thresholds(self):
        """Dynamic thresholds from config"""
        self.thresholds = {
            'trend_strength': secure_float(RegimeParameters.trend_threshold),
            'volatility': secure_float(RegimeParameters.volatility_cutoff),
            'sentiment': secure_float(RegimeParameters.sentiment_boundary)
        }

    async def classify_regime(self) -> Tuple[str, float]:
        """Real-time regime classification pipeline"""
        raw_data = await self.data_feed.get_realtime_metrics()
        processed = self._preprocess_data(raw_data)
        
        # Feature engineering
        features = self._create_feature_vector(processed)
        
        # Anomaly detection
        if self._detect_anomalies(features):
            return "volatile", 1.0
            
        # AI predictions
        lstm_pred = self._predict_regime_shifts(features)
        transformer_embeddings = self.transformer.analyze(features)
        
        # Final classification
        return self._final_classification(lstm_pred, transformer_embeddings)

    def _preprocess_data(self, data: Dict) -> Dict:
        """Secure data normalization pipeline"""
        return {
            'prices': self._normalize(data['prices']),
            'volume': secure_float(data['volume']),
            'sentiment': self._get_sentiment_score(),
            'fundamentals': self._get_fundamental_metrics()
        }

    def _create_feature_vector(self, data: Dict) -> np.array:
        """Multi-source feature engineering"""
        technical = [
            self._calculate_adx(data['prices']),
            self._calculate_atr(data['prices']),
            self._calculate_rsi(data['prices'])
        ]
        return np.concatenate([
            technical,
            [data['sentiment']],
            data['fundamentals']
        ])

    def _final_classification(self, lstm_pred: float, embeddings: np.array) -> Tuple[str, float]:
        """Ensemble classification with 4 regime outputs"""
        combined = np.concatenate([[lstm_pred], embeddings])
        regime_code = self.rf_classifier.predict([combined])[0]
        confidence = np.max(self.rf_classifier.predict_proba([combined]))
        
        return self._decode_regime(regime_code), confidence

    async def get_model_weights(self, regime: str) -> Dict[str, float]:
        """Dynamic weight adjustments for ensemble voting"""
        return {
            'ta': self._get_ta_weight(regime),
            'fa': self._get_fa_weight(regime),
            'sentiment': self._get_sentiment_weight(regime),
            'rl': self._get_rl_weight(regime)
        }

    def _get_ta_weight(self, regime: str) -> float:
        """Technical analysis weighting logic"""
        weights = {
            'bull': 0.7,
            'bear': 0.3,
            'sideways': 0.4,
            'volatile': 0.2
        }
        return weights.get(regime, 0.4)

    # Security-hardened methods
    @validate_inputs
    def _load_lstm_model(self):
        """Secure model loading implementation"""
        try:
            return load_model('lstm_regime.h5') 
        except Exception as e:
            raise RuntimeError(f"Model load failed: {str(e)}")

    @validate_inputs
    def _load_rf_classifier(self):
        """Load pre-trained classifier"""
        return joblib.load('regime_classifier.pkl')

    # Integration points
    def _get_sentiment_score(self) -> float:
        """Integrated sentiment analysis"""
        from Apex.src.Core.trading.ai.sentiment_analysis import get_market_sentiment
        return secure_float(get_market_sentiment(self.symbol))

    def _get_fundamental_metrics(self) -> list:
        """Fundamental data integration"""
        from Apex.src.Core.fundamental.fundamental_engine import get_key_metrics
        return list(get_key_metrics(self.symbol).values())

    # Adaptive learning
    def update_with_feedback(self, trade_outcome: Dict):
        """Reinforcement learning feedback loop"""
        if trade_outcome['success']:
            self._reinforce_regime(trade_outcome['regime'])
        else:
            self._penalize_regime(trade_outcome['regime'])

    def _reinforce_regime(self, regime: str):
        """Strengthen successful regime detection"""
        pass  # Implementation using online learning

    def _penalize_regime(self, regime: str):
        """Weaken poor regime detection"""
        pass  # Implementation using online learning

    def _analyze_order_flow(self, order_flow: Dict[str, Any]) -> float:
        """📌 Analyzes order book data to determine buy/sell pressure bias."""
        if not order_flow:
            return 0.0
        
        bid_volume = sum(qty for price, qty in order_flow["bids"])
        ask_volume = sum(qty for price, qty in order_flow["asks"])

        return round((bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-6), 4)