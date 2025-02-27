# Apex/src/ai/ensembles/ensemble_voting.py
import numpy as np
import xgboost as xgb
from typing import Dict, Any
from datetime import datetime
from utils.logging.structured_logger import StructuredLogger
from utils.helpers.error_handler import handle_api_error
from Core.data.historical_data import HistoricalDataLoader
from Core.trading.strategies.regime_detection import RegimeDetector

class EnsembleVoter:
    """ML-powered dynamic weighting system for trade signal aggregation"""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__)
        self.regime_detector = RegimeDetector()
        self.historical_data = HistoricalDataLoader()
        self.model = self._initialize_model()
        self.performance_tracker = {}
        
        # Load historical data for initial training
        self._load_training_data()

    def _initialize_model(self):
        """Initialize XGBoost model with secure configuration"""
        return xgb.XGBClassifier(
            objective='multi:softprob',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            enable_categorical=True
        )

    @handle_api_error(retries=3)
    def _load_training_data(self):
        """Load and preprocess historical trading data"""
        raw_data = self.historical_data.load_ensemble_training_data()
        self.features = raw_data[['technical', 'fundamental', 'sentiment', 
                                'rl', 'correlation', 'regime', 'volatility']]
        self.labels = raw_data['successful_trade']
        self.asset_types = raw_data['asset_class']
        
        # Preprocess data
        self._preprocess_features()

    def _preprocess_features(self):
        """Secure feature preprocessing pipeline"""
        # Handle missing data
        self.features = self.features.fillna(method='ffill').fillna(0)
        
        # Normalize numerical features
        numerical_cols = ['technical', 'fundamental', 'sentiment', 
                        'rl', 'correlation', 'volatility']
        self.features[numerical_cols] = (self.features[numerical_cols] - 
                                       self.features[numerical_cols].mean()) / \
                                      self.features[numerical_cols].std()
        
        # Encode categorical features
        self.features = pd.get_dummies(self.features, columns=['regime', 'asset_class'])

    def _train_model(self):
        """Train the weighting model with cross-validation"""
        if len(np.unique(self.labels)) < 2:
            self.logger.error("Insufficient training data diversity")
            return

        # Split data preserving temporal order
        split_idx = int(len(self.features) * 0.8)
        X_train, X_test = self.features[:split_idx], self.features[split_idx:]
        y_train, y_test = self.labels[:split_idx], self.labels[split_idx:]

        self.model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      early_stopping_rounds=10,
                      verbose=False)

    def _calculate_performance_scores(self):
        """Track recent model performance metrics"""
        for model in ['technical', 'fundamental', 'sentiment', 'rl', 'correlation']:
            recent_perf = self.historical_data.get_recent_performance(model)
            self.performance_tracker[model] = np.mean(recent_perf[-50:]) if recent_perf else 0.5

    def get_weights(self, current_regime: str, volatility: float) -> Dict[str, float]:
        """Get dynamically optimized model weights"""
        self._calculate_performance_scores()
        
        # Create feature vector for prediction
        features = {
            'technical': self.performance_tracker['technical'],
            'fundamental': self.performance_tracker['fundamental'],
            'sentiment': self.performance_tracker['sentiment'],
            'rl': self.performance_tracker['rl'],
            'correlation': self.performance_tracker['correlation'],
            'regime': current_regime,
            'volatility': volatility,
            'asset_class': self._get_current_asset_class()
        }
        
        # Convert to DataFrame for model input
        features_df = pd.DataFrame([features])
        features_processed = pd.get_dummies(features_df)
        
        # Align columns with training data
        missing_cols = set(self.features.columns) - set(features_processed.columns)
        for col in missing_cols:
            features_processed[col] = 0
        features_processed = features_processed[self.features.columns]

        # Get model predictions
        probabilities = self.model.predict_proba(features_processed)[0]
        
        # Map probabilities to model weights
        weight_mapping = {
            'technical': probabilities[0],
            'fundamental': probabilities[1],
            'sentiment': probabilities[2],
            'rl': probabilities[3],
            'correlation': probabilities[4]
        }
        
        # Apply volatility adjustment
        return self._adjust_for_volatility(weight_mapping, volatility)

    def _adjust_for_volatility(self, weights: Dict[str, float], volatility: float) -> Dict[str, float]:
        """Adjust weights based on current market volatility"""
        volatility_factor = 1 + (volatility / 0.2)  # Scale volatility to 0-2 range
        adjusted = {
            'technical': weights['technical'] * (1 / volatility_factor),
            'fundamental': weights['fundamental'],
            'sentiment': weights['sentiment'] * volatility_factor,
            'rl': weights['rl'] * volatility_factor,
            'correlation': weights['correlation'] * (1 - (volatility / 2))
        }
        return self._normalize_weights(adjusted)

    def _get_current_asset_class(self) -> str:
        """Get asset class from market data"""
        from core.data.realtime.market_data import get_current_asset_type
        return get_current_asset_type()

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Ensure weights sum to 1 with numerical stability"""
        total = sum(weights.values())
        if total == 0:
            return {k: 0.2 for k in weights}  # Fallback equal weights
        return {k: v/total for k, v in weights.items()}

    def update_model(self):
        """Periodic model retraining with new data"""
        new_data = self.historical_data.load_recent_trades()
        if len(new_data) > 100:
            self._load_training_data()
            self._train_model()
            self.logger.info("Model successfully retrained")