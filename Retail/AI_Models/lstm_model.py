import os
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Multiply, Permute, Reshape, Activation, RepeatVector
from sklearn.preprocessing import MinMaxScaler

class LSTMModel:
    def __init__(self, time_steps=60, features=1, model_path="Retail/Models/lstm_trained.h5"):
        """
        Initializes the LSTM model.
        - If a trained model exists, load it.
        - If no model exists, create a new one.
        """
        self.time_steps = time_steps
        self.features = features
        self.model_path = model_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Scaling data for better LSTM performance
        self.memory = []  # ✅ Stores past market conditions
        self.memory_size = 1000  # ✅ Max past experiences stored

        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)  # Load the trained model
        else:
            # Building the Model
            inputs = Input(shape=(self.time_steps, self.features))

            lstm_out = LSTM(100, return_sequences=True)(inputs)
            lstm_out = Dropout(0.2)(lstm_out)
            lstm_out = LSTM(100, return_sequences=True)(lstm_out)
            
            # Attention Layer
            attention = Dense(1, activation='tanh')(lstm_out)
            attention = Activation('softmax')(attention)
            attention = RepeatVector(100)(attention)
            attention = Permute([2, 1])(attention)
            context_vector = Multiply()([lstm_out, attention])
            context_vector = LSTM(50, return_sequences=False)(context_vector)

            # Fully Connected Layer
            dense_out = Dense(50, activation='relu')(context_vector)
            dense_out = Dropout(0.2)(dense_out)
            output = Dense(1, activation='linear')(dense_out)

            # Final Model
            self.model = Model(inputs, output)
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def preprocess_data(self, dataset):
        """
        Scales the dataset using MinMaxScaler.
        """
        return self.scaler.fit_transform(dataset)

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_data=None):
        """
        Trains the LSTM model and saves it for future use.
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
        self.model.save(self.model_path)  # ✅ Save trained model after training
        print(f"✅ Model trained and saved at {self.model_path}")

    def predict(self, X_test):
        """
        Predicts future prices.
        """
        return self.model.predict(X_test)

    def scale_data(self, data):
        """
        Scales input data using the same scaler.
        """
        return self.scaler.transform(data)

    def inverse_scale(self, data):
        """
        Converts predictions back to original price scale.
        """
        return self.scaler.inverse_transform(data)

    def update_model(self, new_data, true_values, epochs=5):
        """
        ✅ Live Training with Memory Buffer
        - Instead of only learning from the most recent mistake, the model will also learn from past mistakes.
        - It will randomly sample old market conditions and train on them.
        """
        new_data = self.scale_data(new_data)  # Scale new input data
        true_values = self.scale_data(true_values)  # Scale actual price data

        # ✅ Track past prediction errors
        recent_error = abs(self.predict(new_data)[-1] - true_values[-1])
        
        # ✅ Adjust learning rate based on error magnitude
        base_lr = 0.001  # Default learning rate
        if recent_error > 0.05:  # If the error is high, increase learning rate
            new_lr = min(base_lr * 2, 0.01)  # Cap max learning rate
        elif recent_error < 0.01:  # If error is very low, decrease learning rate
            new_lr = max(base_lr / 2, 0.0001)  # Set minimum learning rate
        else:
            new_lr = base_lr

        # ✅ Apply the new learning rate
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
        print(f"📈 Adjusted learning rate to: {new_lr}")

        # ✅ Train the model with new data
        self.model.fit(new_data, true_values, epochs=epochs, batch_size=8, verbose=0)

        # ✅ Periodically train on past mistakes from memory buffer
        if len(self.memory) > 10:  # Don't train if memory buffer is empty
            print("🧠 Training on past mistakes from memory buffer...")
            past_data, past_predictions, past_actuals = zip(*self.memory[-10:])  # Get last 10 mistakes
            past_data = self.scale_data(np.array(past_data))
            past_actuals = self.scale_data(np.array(past_actuals))

            self.model.fit(past_data, past_actuals, epochs=epochs, batch_size=8, verbose=0)

        self.model.save(self.model_path)  # ✅ Save updated model
        print("📈 Live model updated with new market data.")
    def get_trade_signal(self, market_data, true_price=None):
        """
        Uses AI models (LSTM, Technical, Sentiment) to generate a trade signal (BUY, SELL, HOLD).
        If `true_price` is provided, the bot **remembers past trades and learns from mistakes.**
        """
        # Preprocess data
        scaled_data = self.scale_data(market_data)

        # Predict future price
        predicted_price = self.predict(scaled_data)[-1]

        # Compute trade signals
        tech_signal = self.technical_analysis.analyze(market_data)
        sentiment_score = self.sentiment_analyzer.get_sentiment(market_data)
        ai_decision = self.maddpg.select_action(tech_signal)

        # Weighted Decision
        final_decision = (
            (predicted_price * 0.3) + 
            (tech_signal * 0.3) + 
            (sentiment_score * 0.2) + 
            (ai_decision * 0.2)
        )

        if final_decision > 0.6:
            action = "BUY"
        elif final_decision < 0.4:
            action = "SELL"
        else:
            action = "HOLD"

        # ✅ Store market conditions in memory
        self.memory.append((market_data, predicted_price, true_price))
        if len(self.memory) > self.memory_size:  # Keep memory buffer size in check
            self.memory.pop(0)

        # ✅ If the bot made a mistake, update the model & adjust learning rate
        if true_price is not None:
            self.update_model(market_data, [true_price])

        return {"symbol": market_data["symbol"], "action": action, "confidence": round(final_decision, 4)}
