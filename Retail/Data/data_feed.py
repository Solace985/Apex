import asyncio
import logging
import requests
import websockets
from typing import Dict, Any
from polygon import WebSocketClient
from AI_Models.order_flow import InstitutionalOrderFlow
from Retail.Core.config import load_config
import aiohttp
import json
import optuna
import pandas as pd
import numpy as np

config = load_config()

class DataFeed:
    """Handles real-time market data fetching via WebSockets and institutional order flow tracking."""

    def __init__(self, interval: int = 1):
        self.market_data: Dict[str, Any] = {}
        self.interval = interval
        self.running = False
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.polygon_key = config.websocket.polygon_key
        self.symbols = config.websocket.symbols
        
        # Polygon WebSocket Setup
        self.ws_client = WebSocketClient(
            api_key=self.polygon_key,
            subscriptions=[f"T.{symbol}" for symbol in self.symbols]
        )
        
        # Institutional Order Flow Tracker
        self.institutional_tracker = InstitutionalOrderFlow()

    async def start_stream(self):
        """Starts WebSocket stream for real-time data."""
        self.running = True
        async with self.ws_client as ws:
            async for msg in ws:
                self.process_message(msg)

    def process_message(self, msg):
        """Processes WebSocket messages and updates market data."""
        structured_data = {
            "symbol": msg.symbol,
            "price": msg.price,
            "volume": msg.volume,
            "timestamp": msg.timestamp
        }
        # Fetch additional market insights
        structured_data.update(self.fetch_order_flow_data(msg.symbol))
        self.market_data[msg.symbol] = structured_data

    async def start(self):
        """Starts both WebSocket streaming and order flow tracking."""
        await asyncio.gather(
            self.start_stream(),  # WebSocket Data Feed
            self.fetch_data()  # Order Flow Fetching
        )

    async def fetch_data(self):
        """Fetches institutional order flow asynchronously at fixed intervals."""
        while self.running:
            self.logger.info("Fetching market order flow data...")
            for symbol in self.market_data.keys():
                self.market_data[symbol].update(self.fetch_order_flow_data(symbol))
            await asyncio.sleep(self.interval)

    def fetch_order_flow_data(self, symbol):
        """Fetches top bid, top ask & institutional bias for a given symbol."""
        try:
            top_bid, top_ask = self.institutional_tracker.fetch_order_book_data(symbol)
            institutional_bias = self.institutional_tracker.get_institutional_bias()
            return {"top_bid": top_bid, "top_ask": top_ask, "institutional_bias": institutional_bias}
        except Exception as e:
            self.logger.error(f"❌ Error fetching order flow data: {e}")
            return {}

    def stop(self):
        """Stops the data feed."""
        self.running = False

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    tau = trial.suggest_uniform('tau', 0.01, 0.1)
    # Initialize and train your model with these hyperparameters
    # Return a metric to optimize, e.g., validation loss
    return validation_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

class FeatureEngineer:
    def __init__(self):
        pass

    def process(self, market_data):
        df = pd.DataFrame(market_data)
        df['moving_average'] = df['price'].rolling(window=5).mean()
        df['price_change'] = df['price'].pct_change()
        df['volatility'] = df['price'].rolling(window=5).std()
        return df.dropna().values

def augment_data(data):
    augmented_data = []
    for sample in data:
        noise = np.random.normal(0, 0.01, sample.shape)
        augmented_data.append(sample + noise)
    return np.array(augmented_data)
