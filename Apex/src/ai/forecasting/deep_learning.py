import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Conv1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from pandas_ta import volatility, momentum, volume
from typing import Dict, Tuple, List

class DeepLearningFactory:
    """Factory for creating different deep learning architectures with TA integration"""
    
    def __init__(self, asset_class: str):
        self.asset_class = asset_class
        self.technical_indicators = {
            'crypto': ['RSI_14', 'EMA_20', 'OBV', 'ATR_14'],
            'equity': ['SMA_50', 'MACD', 'VWAP', 'STOCH_14_3'],
            'forex': ['EMA_12', 'EMA_26', 'ADX_14', 'CCI_20']
        }
        
    def create_model(self, model_type: str, input_shape: Tuple) -> Model:
        model_builder = {
            'lstm': self._build_lstm,
            'transformer': self._build_transformer,
            'cnn': self._build_cnn,
            'hybrid': self._build_hybrid
        }
        return model_builder[model_type](input_shape)
    
    def _build_lstm(self, input_shape: Tuple) -> Model:
        inputs = Input(shape=input_shape)
        x = LSTM(128, return_sequences=True)(inputs)
        x = LSTM(64)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(3, activation='softmax')(x)  # Buy, Hold, Sell
        return Model(inputs, outputs)
    
    def _build_transformer(self, input_shape: Tuple) -> Model:
        inputs = Input(shape=input_shape)
        x = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
        x = LayerNormalization(epsilon=1e-6)(x + inputs)
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(3, activation='softmax')(x)
        return Model(inputs, outputs)
    
    def _add_ta_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators specific to asset class"""
        ti_config = {
            'crypto': [
                ('RSI_14', lambda x: momentum.rsi(x['close'], length=14)),
                ('EMA_20', lambda x: x['close'].ewm(span=20).mean()),
                ('OBV', lambda x: volume.obv(x['close'], x['volume'])),
                ('ATR_14', lambda x: volatility.atr(x['high'], x['low'], x['close'], length=14))
            ],
            'equity': [
                ('SMA_50', lambda x: x['close'].rolling(50).mean()),
                ('MACD', lambda x: momentum.macd(x['close'])),
                ('VWAP', lambda x: volume.vwap(x['high'], x['low'], x['close'], x['volume'])),
                ('STOCH_14_3', lambda x: momentum.stoch(x['high'], x['low'], x['close'], 14, 3))
            ],
            'forex': [
                ('EMA_12', lambda x: x['close'].ewm(span=12).mean()),
                ('EMA_26', lambda x: x['close'].ewm(span=26).mean()),
                ('ADX_14', lambda x: volatility.adx(x['high'], x['low'], x['close'], length=14)),
                ('CCI_20', lambda x: momentum.cci(x['high'], x['low'], x['close'], length=20))
            ]
        }
        
        for name, func in ti_config[self.asset_class]:
            data[name] = func(data)
        return data.dropna()

class MultiAssetDeepLearner:
    """Deep learning model handling multiple asset classes"""
    
    def __init__(self, asset_classes: List[str]):
        self.models = {
            ac: DeepLearningFactory(ac).create_model('hybrid', (30, 8)) 
            for ac in asset_classes
        }
        self.compile_models()
        
    def compile_models(self, learning_rate=0.001):
        for model in self.models.values():
            model.compile(
                optimizer=Adam(learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
    
    def train_multi_asset(self, data: Dict[str, pd.DataFrame], epochs=50):
        for asset, df in data.items():
            processed = self.models[asset]._add_ta_features(df)
            X, y = self._prepare_data(processed)
            self.models[asset].fit(X, y, epochs=epochs, verbose=0)
# Initialize multi-asset learner
learner = MultiAssetDeepLearner(['crypto', 'equity', 'forex'])
learner.train_multi_asset(training_data)

def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    features = data.drop(columns=['target']).values
    targets = pd.get_dummies(data['target']).values
    return features.reshape(-1, 30, features.shape[1]), targets

class TemporalFusionTransformer(Model):
    """Advanced time series transformer with cross-asset attention"""
    def __init__(self, num_features: int, num_assets: int):
        super().__init__()
        self.asset_embedding = Dense(16)
        self.temporal_attention = MultiHeadAttention(num_heads=4, key_dim=64)
        self.cross_asset_attention = MultiHeadAttention(num_heads=4, key_dim=64)
        self.decoder = Dense(3, activation='softmax')
        
    def call(self, inputs):
        # Input shape: (batch_size, seq_len, num_features + asset_id)
        asset_embedded = self.asset_embedding(inputs[..., -1])
        temporal = self.temporal_attention(inputs[..., :-1], inputs[..., :-1])
        cross_asset = self.cross_asset_attention(asset_embedded, temporal)
        return self.decoder(cross_asset)