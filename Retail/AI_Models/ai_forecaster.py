# ai_forecaster.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from flask import Flask, request, jsonify
import joblib

# ✅ Load historical correlation dataset
data = pd.read_csv("historical_correlations.csv")  # (Assumed dataset)

# ✅ Preprocess Data
def create_dataset(series, time_steps=30):
    X, y = [], []
    for i in range(len(series) - time_steps):
        X.append(series[i:i + time_steps])
        y.append(series[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 30
X, y = create_dataset(data['correlation'].values, time_steps)
X = np.expand_dims(X, axis=-1)  # Reshape for LSTM

# ✅ Build LSTM Model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(time_steps, 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# ✅ Train Model
model.fit(X, y, epochs=50, batch_size=32)

# ✅ Save Model
model.save("lstm_correlation_model.h5")

# ✅ Flask API to serve predictions
app = Flask(__name__)
model = tf.keras.models.load_model("lstm_correlation_model.h5")

@app.route('/predict_correlation', methods=['POST'])
def predict():
    data = request.json
    time_series = np.array(data["correlation_series"]).reshape(1, time_steps, 1)
    prediction = model.predict(time_series)
    return jsonify({"predicted_correlation": float(prediction[0, 0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
