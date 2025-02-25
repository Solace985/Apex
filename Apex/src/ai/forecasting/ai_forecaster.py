import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import TimeSeriesSplit
from flask import Flask, request, jsonify
import os
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

# ✅ GPU Memory Optimization
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def load_data(filepath, time_steps=30):
    data = pd.read_csv(filepath)['correlation'].values
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X).reshape(-1, time_steps, 1), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32, activation='tanh'),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_save_model(data_path, model_path, splits=3, epochs=50):
    X, y = load_data(data_path)
    tscv = TimeSeriesSplit(n_splits=splits)
    val_losses = []

    for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        logging.info(f"Fold {i+1}/{splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model((X.shape[1], 1))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, verbose=1),
            ModelCheckpoint(f"{model_path.split('.h5')[0]}_fold_{i+1}.h5", save_best_only=True, monitor='val_loss', verbose=1)
        ]

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                            validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)

        val_losses.append(min(history.history['val_loss']))
        logging.info(f"Fold {i+1} val_loss: {val_losses[-1]}")

    logging.info(f"Avg val_loss: {np.mean(val_losses)}")

    # Save the best model
    best_fold = np.argmin(val_losses) + 1
    best_model_path = f"{model_path.split('.h5')[0]}_fold_{best_fold}.h5"
    best_model = load_model(best_model_path)
    best_model.save(model_path)
    logging.info(f"✅ Best model from fold {best_fold} saved as {model_path}")

# ✅ Main Training Execution (Run separately to avoid API downtime)
if __name__ == "__main__":
    if not os.path.exists("lstm_correlation_final_model.h5"):
        if not os.path.exists("historical_correlations.csv"):
            logging.error("historical_correlations.csv not found!")
            exit(1)

        train_and_save_model("historical_correlations.csv", "lstm_correlation_final_model.h5")

# ✅ Flask Prediction API
app = Flask(__name__)
model = None  # initially set to None at the top-level

@app.before_request
def limit_json_size():
    if request.content_length > 1024:  # limit request payload to 1KB
        return jsonify({"error": "Request too large."}), 413

@app.route('/predict_correlation', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    try:
        series = np.array(request.json["correlation_series"], dtype=float)
        if len(series) != 30:
            return jsonify({"error": "Series length must be exactly 30."}), 400

        if np.isnan(series).any():
            return jsonify({"error": "Series contains NaN or invalid values."}), 400

        prediction = model.predict(series.reshape(1, 30, 1))
        return jsonify({"predicted_correlation": float(prediction[0, 0])})

    except (KeyError, TypeError, ValueError) as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Invalid input format or data."}), 400

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Explicitly trigger training only with: python ai_forecaster.py train
        if not os.path.exists("historical_correlations.csv"):
            logging.error("historical_correlations.csv not found!")
            exit(1)
        train_and_save_model("historical_correlations.csv", "lstm_correlation_final_model.h5")
    else:
        # Default to API mode
        try:
            model = load_model("lstm_correlation_final_model.h5")
            logging.info("✅ Model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            model = None
        app.run(host="0.0.0.0", port=5000)
