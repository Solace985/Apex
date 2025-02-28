# Apex/src/ai/ensembles/transformers_lstm.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Any
from utils.logging.structured_logger import StructuredLogger
from utils.helpers.error_handler import handle_api_error
from Core.data.historical_data import HistoricalDataLoader
from Core.trading.strategies.regime_detection import RegimeDetector
from Core.trading.risk.risk_management import RiskManager

class HybridModel(nn.Module):
    """Transformer-LSTM hybrid model for multi-scale feature extraction with adaptive attention fusion"""

    def __init__(self, config: Dict):
        super().__init__()
        self.logger = StructuredLogger(__name__)
        self.config = config

        # Bi-directional LSTM for short-term patterns
        self.lstm = nn.LSTM(
            input_size=config['input_size'],
            hidden_size=config['lstm_hidden'],
            num_layers=config['lstm_layers'],
            bidirectional=True,
            batch_first=True
        )

        # Transformer encoder for long-term dependencies
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config['input_size'],
                nhead=config['n_heads'],
                dim_feedforward=config['ff_dim']
            ),
            num_layers=config['transformer_layers']
        )

        # Attention-based fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=config['lstm_hidden'] * 2,
            num_heads=config['n_heads']
        )

        # Regime-aware adaptation
        self.regime_weights = nn.ParameterDict({
            regime: nn.Parameter(torch.rand(1))
            for regime in config['regime_types']
        })

        # Final dense layers
        self.fc = nn.Sequential(
            nn.Linear(config['lstm_hidden'] * 2 + config['input_size'], 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, config['output_size'])
        )

    def forward(self, x: torch.Tensor, regime: str) -> torch.Tensor:
        """Multi-scale feature processing with regime adaptation and self-attention fusion"""
        # Process short-term features with LSTM
        lstm_out, _ = self.lstm(x)

        # Process long-term features with Transformer
        transformer_out = self.transformer(x)

        # Adaptive regime weighting
        regime_weight = self.regime_weights[regime]
        weighted_out = regime_weight * lstm_out + (1 - regime_weight) * transformer_out

        # Temporal attention fusion with Transformer-encoded features
        attn_input = weighted_out.permute(1, 0, 2)  # Reshape for multihead attention
        attn_out, _ = self.attention(attn_input, attn_input, attn_input)
        attn_out = attn_out.permute(1, 0, 2)  # Reshape back

        # Combine LSTM-Transformer fused output with the attention-enhanced features
        combined = torch.cat([attn_out, transformer_out], dim=-1)
        return self.fc(combined)

class FeatureEngineer:
    """Advanced feature engineering pipeline with secure preprocessing and risk-aware trade filtering"""

    def __init__(self, market_data_bus: Any):
        self.logger = StructuredLogger(__name__)
        self.market_data = market_data_bus
        self.regime_detector = RegimeDetector()
        self.risk_manager = RiskManager()
        self.config = self._load_config()

        # Initialize hybrid model
        self.model = HybridModel(self.config)
        self._load_pretrained_weights()

        # Initialize data processors
        from Core.data.realtime.data_feed import DataNormalizer
        self.normalizer = DataNormalizer()

    def _load_config(self) -> Dict:
        """Load model configuration from central config"""
        from Core.trading.ai.config import load_model_config
        return load_model_config('transformer_lstm')

    @handle_api_error()
    def _load_pretrained_weights(self):
        """Load pre-trained model weights securely"""
        try:
            state_dict = torch.load(self.config['model_weights'])
            self.model.load_state_dict(state_dict)
        except Exception as e:
            self.logger.critical("Failed loading model weights", error=str(e))
            raise

    @handle_api_error()
    def _prepare_inputs(self) -> torch.Tensor:
        """Enhanced input preparation with strict validation and outlier detection"""
        raw_data = self.market_data.get_feature_data()

        # Validate input structure
        validated = self._validate_data(raw_data)

        # Handle missing values
        filled_data = {k: (v if v is not None else 0) for k, v in validated.items()}

        # Outlier detection using Z-score normalization
        mean = np.mean(list(filled_data.values()))
        std = np.std(list(filled_data.values()))
        processed_data = {
            k: v if abs((v - mean) / (std + 1e-8)) < 3 else mean
            for k, v in filled_data.items()
        }

        # Normalize features securely
        normalized = self.normalizer.transform(processed_data)

        return torch.tensor(normalized).float()

    def _validate_data(self, data: Dict) -> Dict:
        """Security-focused data validation"""
        from utils.helpers.validation import validate_feature_data
        if not validate_feature_data(data):
            self.logger.error("Invalid feature data detected")
            raise ValueError("Malformed input data")
        return data

    def _get_current_regime(self) -> str:
        """Get market regime with fallback"""
        regime = self.regime_detector.current_regime()
        return regime if regime else 'neutral'

    def _calculate_confidence(self, features: torch.Tensor) -> float:
        """Calculate trade signal confidence dynamically based on volatility-adjusted variance"""
        feature_var = torch.var(features).item()
        market_volatility = self.market_data.get_volatility()

        # Scale confidence dynamically based on volatility
        confidence = max(0.1, 1 - (feature_var / self.config['max_variance']))
        adjusted_confidence = confidence * (1 - market_volatility / 2)

        return round(adjusted_confidence, 3)  # Keep precision within three decimal places

    def process_features(self) -> Dict[str, torch.Tensor]:
        """Main feature processing pipeline with adaptive risk filtering"""
        try:
            # Prepare inputs
            inputs = self._prepare_inputs()

            # Get market regime
            regime = self._get_current_regime()

            # Generate features
            with torch.no_grad():
                features = self.model(inputs.unsqueeze(0), regime)

            # Post-process outputs
            processed_features = self._postprocess_features(features.squeeze())

            # Apply trade confidence threshold
            if processed_features['confidence'] < 0.55:  # Dynamic threshold
                self.logger.warning("⚠️ Low-confidence trade signal detected. Ignoring trade.")
                return {'trade_action': 'HOLD', 'confidence': processed_features['confidence'], 'reason': 'Low confidence'}

            return processed_features

        except Exception as e:
            self.logger.error("Feature processing failed", error=str(e))
            return {}

    def _postprocess_features(self, features: torch.Tensor) -> Dict:
        """Convert tensor to trading system format"""
        return {
            'short_term': features[:self.config['short_term_dim']].numpy(),
            'long_term': features[self.config['short_term_dim']:].numpy(),
            'confidence': self._calculate_confidence(features)
        }

    def update_model(self):
        """Periodic model update with new data"""
        try:
            new_data = HistoricalDataLoader().load_training_samples()
            if len(new_data) > self.config['retrain_threshold']:
                self._retrain_model(new_data)
        except Exception as e:
            self.logger.warning("Model update failed", error=str(e))

    def _retrain_model(self, data: Dict):
        """Secure model retraining procedure"""
        from core.trading.ai.training import SafeTrainer
        trainer = SafeTrainer(self.model, data)
        trainer.train()
        torch.save(self.model.state_dict(), self.config['model_weights'])
        self.logger.info("Model successfully retrained")
