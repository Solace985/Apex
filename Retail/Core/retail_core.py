import concurrent.futures
import aiohttp
import json
import os
import logging
import threading
import asyncio
import sqlite3
import yaml
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import unittest
from logging.handlers import RotatingFileHandler
from aiohttp import web
from kafka import KafkaConsumer
from sklearn.ensemble import RandomForestClassifier
import gym
from stable_baselines3 import PPO
from pydantic import BaseModel
import aiosqlite
from keras.models import Sequential
from keras.layers import LSTM, Dense
from AI_Models.maddpg_model import MADDPG
import numpy as np
from Retail.utils.logger import setup_logger
from Retail.utils.data_feed import DataFeed as UtilsDataFeed
from Retail.utils.strategy_manager import StrategyManager
from Retail.utils.risk_manager import RiskManager as UtilsRiskManager
from Retail.AI_Models.machine_learning import MachineLearningModel
from joblib import Parallel, delayed
from Retail.Core.liquidity_manager import LiquidityManager
import requests
from textblob import TextBlob
from AI_Models.fundamental_analysis import FundamentalAnalysis
from AI_Models.sentiment_analysis import SentimentAnalyzer
from Core.liquidity_manager import LiquidityManager as AIModulesLiquidityManager
from AI_Models.technical_analysis import TechnicalAnalysis
from AI_Models.maddpg_model import MADDPG  # ✅ Added MADDPG for hierarchical indicator weighting
from Retail.Core.risk_management import AdaptiveRiskManagement
import time
from Retail.Core.data_feed import DataFeed
from Retail.Core.trading_ai import TradingAI
from Retail.Core.risk_management import RiskManager
from Retail.Core.execution_engine import ExecutionEngine
from Retail.Metrics.performance_metrics import PerformanceTracker
from Retail.Brokers.broker_factory import BrokerFactory
from Retail.Strategies.trend_following import TrendFollowingStrategy
from Retail.Strategies.mean_reversion import MeanReversionStrategy
from concurrent.futures import ProcessPoolExecutor


# This file is the core of the bot that is responsible for handling the strategy and execution of the trades.

# --------------------------
# ✅ Configurations
# --------------------------
class Config:
    DATA_FEED_INTERVAL = 1  # Fetch data every second
    RISK_THRESHOLD = 0.02  # Max risk per trade
    BROKER_API_KEYS = {
        "zerodha": os.getenv("ZERODHA_API_KEY"),
        "binance": os.getenv("BINANCE_API_KEY"),
    }
    DATABASE_PATH = "storage/trade_history.db"

    def __init__(self, config_file='config.yaml'):
        self.config = self._load_config(config_file)
        self.validate_config()
        self.watch_config_changes()
        
    def _load_config(self, config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
            
    def validate_config(self):
        # Add config validation logic
        schema = self.load_config_schema()
        jsonschema.validate(self.config, schema)

# --------------------------
# ✅ Logger Setup
# --------------------------
logger = logging.getLogger('RetailBot')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('retailbot.log', maxBytes=2000, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# --------------------------
# ✅ Database Storage
# --------------------------
class AsyncTradeStorage:
    async def __aenter__(self):
        self.conn = await aiosqlite.connect(self.db_path)
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        await self.conn.close()

class TradeStorage:
    def __init__(self, db_path):
        self.db_path = db_path
        self.create_table()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def create_table(self):
        with self.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY,
                    trade_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    trade_type TEXT,
                    price REAL,
                    volume INTEGER,
                    strategy TEXT
                )
            """)

    def store_trade(self, trade_details):
        with self.connect() as conn:
            conn.execute("""
                INSERT INTO trades (trade_type, price, volume, strategy)
                VALUES (?, ?, ?, ?)
            """, (trade_details['trade_type'], trade_details['price'],
                  trade_details['volume'], trade_details['strategy']))
            conn.commit()

# --------------------------
# ✅ Abstract Base Classes
# --------------------------
class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on market data."""
        pass

class BrokerAPI(ABC):
    """Abstract base class for broker integrations."""
    @abstractmethod
    def place_order(self, order_details: Dict[str, Any]) -> None:
        """Place a synchronous order."""
        pass

    @abstractmethod
    async def async_place_order(self, order_details: Dict[str, Any]) -> None:
        """Place an asynchronous order."""
        pass

class SyncBroker(BrokerAPI):
    """Synchronous broker integration."""
    def place_order(self, order_details: Dict[str, Any]) -> None:
        print(f"Sync order placed: {order_details}")

    async def async_place_order(self, order_details: Dict[str, Any]) -> None:
        raise NotImplementedError("Async orders not supported.")

class AsyncBroker(BrokerAPI):
    """Asynchronous broker integration."""
    async def async_place_order(self, order_details: Dict[str, Any]) -> None:
        print(f"Async order placed: {order_details}")

    def place_order(self, order_details: Dict[str, Any]) -> None:
        raise NotImplementedError("Only async orders supported.")

class RiskManager(ABC):
    @abstractmethod
    def evaluate_risk(self, trade_details: Dict[str, Any]) -> bool:
        pass

# --------------------------
# ✅ Data Processing Engine
# --------------------------
class DataFeed:
    def __init__(self, websocket_url: str):
        self.websocket_url = websocket_url
        self.market_data = {}
        self.running = False
        self.reconnect_attempts = 0

    async def stream_data(self):
        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(self.websocket_url) as ws:
                        self.reconnect_attempts = 0
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                self.market_data = json.loads(msg.data)
                                logging.info(f"Live Market Data: {self.market_data}")
            except aiohttp.ClientError as e:
                self.reconnect_attempts += 1
                wait_time = min(2 ** self.reconnect_attempts, 60)
                logging.error(f"WebSocket connection failed: {e}. Reconnecting in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

    def start(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.stream_data())

    def stop(self):
        self.running = False

# --------------------------
# ✅ Strategy Management System
# --------------------------
class StrategyManager:
    def __init__(self, data_feed, risk_manager, execution_engine):
        self.strategies: List[Strategy] = self._load_strategies()
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine
        self.logger = logging.getLogger(self.__class__.__name__)
        
        data_feed.subscribe(self.on_market_data)  # Connects StrategyManager to DataFeed

    def on_market_data(self, data):
        for strategy in self.strategies:
            signal = strategy.generate_signal(data)
            if signal:
                self.logger.info(f"Strategy {strategy.__class__.__name__} generated signal: {signal}")
                if self.risk_manager.validate(signal):
                    self.execution_engine.execute(signal)  # Trade execution

    def add_strategy(self, strategy: Strategy):
        self.strategies.append(strategy)

    def evaluate_strategies(self, market_data: Dict[str, Any]):
        signals = []
        with concurrent.futures.ProcessPoolExecutor() as executor:  # Uses multiple CPU cores
            futures = {executor.submit(strategy.generate_signal, market_data): strategy for strategy in self.strategies}
            for future in concurrent.futures.as_completed(futures):
                signal = future.result()
                if signal:
                    signals.append(signal)
        return signals

    def _load_strategies(self):
        strategy_classes = [TrendFollowingStrategy, MeanReversionStrategy]
        return [strategy() for strategy in strategy_classes if strategy.is_active]

class ExecutionEngine:
    def __init__(self, broker_api):
        self.broker_api = broker_api

    async def execute_order(self, order):
        try:
            response = await self.broker_api.place_order(order)
            return response
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return None

class AdvancedRiskManager(RiskManager):
    def __init__(self):
        super().__init__()
        self.ai_model = self.load_ai_model()

    def adjust_risk_parameters(self, market_data):
        risk_adjustments = self.ai_model.predict(market_data)
        self.max_drawdown = risk_adjustments['max_drawdown']
        self.capital_exposure_limit = risk_adjustments['capital_exposure_limit']

class AdaptiveRiskManager(RiskManager):
    """Adjusts risk dynamically based on market conditions."""
    
    def __init__(self):
        self.max_drawdown = 0.2  # Default value

    def adjust_risk_parameters(self, market_data):
        """Modify risk thresholds dynamically based on volatility levels."""
        volatility = self.calculate_volatility(market_data)
        if volatility > 0.05:  # Adjust risk in volatile markets
            self.max_drawdown = 0.1
        else:
            self.max_drawdown = 0.2

    def calculate_volatility(self, market_data):
        """Compute market volatility (e.g., ATR or standard deviation)."""
        return np.std(market_data['price_series'])

class Logger:
    def __init__(self):
        self.logger = logging.getLogger('RetailBot')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def log_trade(self, trade):
        self.logger.info(f"Trade executed: {trade}")

    def log_error(self, error):
        self.logger.error(f"Error: {error}")

class RetailBotCore:
    """Main controller for the trading bot, integrating data, AI, risk management, and execution."""

    def __init__(self):
        # Initialize essential components
        self.data_feed = DataFeed()
        self.trading_ai = TradingAI()
        self.risk_manager = RiskManager()
        self.execution_engine = ExecutionEngine()
        self.performance_tracker = PerformanceTracker()

        # Select broker (change to preferred broker)
        self.broker = BrokerFactory.create_broker("binance")
        self.execution_engine.initialize_broker(self.broker)

        # Trading Settings
        self.polling_interval = 1  # Time in seconds between each loop

    def run(self):
        """Main trading loop - runs continuously until stopped."""
        print("🚀 Trading Bot Started...")

        while True:
            try:
                # Step 1: Fetch Market Data
                market_data = self.data_feed.get_market_data()
                if not market_data:
                    print("⚠ Market data not available. Skipping this cycle.")
                    time.sleep(self.polling_interval)
                    continue
                
                # Step 2: Generate Trade Signal
                trade_signal = self.trading_ai.get_trade_signal(market_data)
                if trade_signal is None:
                    print("ℹ No valid trade signal generated. Waiting for the next cycle.")
                    time.sleep(self.polling_interval)
                    continue
                
                # Step 3: Validate Risk & Market Conditions
                if not self.risk_manager.is_safe_trade(trade_signal, market_data):
                    print("🚨 Trade rejected due to risk constraints.")
                    time.sleep(self.polling_interval)
                    continue

                # Step 4: Execute Trade
                execution_result = self.execution_engine.execute_trade(trade_signal)
                if execution_result:
                    print(f"✅ Trade executed: {execution_result}")

                # Step 5: Track Performance & Adapt Strategy
                self.performance_tracker.record_trade(execution_result)

            except Exception as e:
                print(f"❌ Error in trading loop: {e}")

        time.sleep(self.polling_interval)  # Wait before fetching new market data

    def __init__(self, execution_engine, risk_manager):
        self.execution_engine = execution_engine
        self.risk_manager = risk_manager
        self.fundamental_analysis = FundamentalAnalysis()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.liquidity_manager = LiquidityManager()
        self.technical_analysis = TechnicalAnalysis()
        self.maddpg = MADDPG(state_dim=10, action_dim=1)
        self.machine_learning_model = MachineLearningModel()  # ✅ Add ML Model

    def should_trade(self, market_data):
        """Determines if a trade should be executed based on all AI layers, including MADDPG."""

        # ✅ Step 1: Institutional Liquidity Check
        if not self.liquidity_manager.detect_whale_activity(market_data):
            print("No significant institutional liquidity detected. Avoiding trade.")
            return False

        # ✅ Step 2: Fundamental & Sentiment Analysis Check
        fundamental_score = self.fundamental_analysis.analyze_fundamentals()
        sentiment_score = self.sentiment_analyzer.get_sentiment_score()

        if fundamental_score < 0 and sentiment_score < -0.5:
            print("Fundamentals & sentiment are bearish. Avoiding trade.")
            return False

        # ✅ Step 3: Technical Analysis Validation Using MADDPG
        technical_signals = self.technical_analysis.extract_technical_features(market_data)
        maddpg_decision = self.maddpg.select_action(technical_signals)

        if maddpg_decision < 0.5:  # Threshold for confidence in trade execution
            print("MADDPG does not confirm trade. Avoiding trade.")
            return False

        # ✅ Step 4: Market Depth Analysis
        if not self.analyze_market_depth(market_data):
            return False

        print("Trade validated by all layers, including MADDPG. Executing trade.")
        return True

    def execute_trade(self, market_data):
        """Executes trade if all AI layers confirm, using adaptive risk management & parallel order execution."""

        if not self.analyze_market_depth(market_data):
            return  # Skip trade if poor liquidity

        selected_strategy = self.choose_best_strategy([market_data])  # Use batch processing
        if not selected_strategy:
            self.logger.warning("🚫 Trade skipped due to lack of a valid strategy.")
            return

        trade_details = {
            "price": market_data["price"],
            "volume": market_data["volume"],
            "strategy": selected_strategy[0].name  # Extract strategy from batch result
        }

        risk_decision = self.risk_manager.evaluate_trade(
            entry_price=trade_details["price"],
            stop_loss=trade_details["price"] * 0.98,
            take_profit=trade_details["price"] * 1.04,
            slippage=market_data.get("slippage", 0),
            market_data=market_data
        )

        if "REJECTED" in risk_decision:
            self.logger.warning(risk_decision)
            return

        # ✅ Execute multiple orders in parallel to reduce latency
        Parallel(n_jobs=2)([
            delayed(self.execution_engine.execute_trade)(trade_details),
            delayed(self.feedback_loop)(trade_details)
        ])

    def choose_strategy(self, market_data):
        """Use ML model to select the best trading strategy dynamically."""
        prediction, confidence = self.ml_model.predict(market_data)
        if confidence > 0.7:  # Confidence threshold
            strategy_index = prediction
            return self.strategy_manager.strategies[strategy_index]
        else:
            self.logger.warning("Low confidence in prediction, using default strategy.")
            return self.strategy_manager.default_strategy()

    def choose_best_strategy(self, market_data_batch):
        """
        Selects the best strategy dynamically for multiple trade opportunities at once.
        Uses AI models (MADDPG, PPO, RandomForest) for decision-making.
        """

        # ✅ Step 1: Batch Process Fundamental & Sentiment Analysis
        fundamental_scores = [self.fundamental_analysis.analyze_fundamentals(market) for market in market_data_batch]
        sentiment_scores = [self.sentiment_analyzer.get_sentiment_score(market) for market in market_data_batch]
        technical_signals = [self.technical_analysis.extract_technical_features(market) for market in market_data_batch]

        # ✅ Step 2: Filter High-Risk Trades Before AI Processing
        valid_trades = []
        for idx, market_data in enumerate(market_data_batch):
            if not self.risk_manager.evaluate_trade(
                entry_price=market_data["price"],
                stop_loss=market_data["price"] * 0.98,
                take_profit=market_data["price"] * 1.04,
                slippage=market_data.get("slippage", 0),
                market_data=market_data
            ):
                continue  # Skip trade if high risk
            valid_trades.append(market_data)

        if not valid_trades:
            self.logger.warning("🚫 No valid trades due to high risk constraints.")
            return None

        # ✅ Step 3: AI-Based Strategy Prediction for All Valid Trades
        strategy_predictions, confidences = self.machine_learning_model.predict_batch(valid_trades)

        selected_strategies = []
        for i, market_data in enumerate(valid_trades):
            if confidences[i] > 0.7:
                selected_strategies.append(self.strategy_manager.strategies[strategy_predictions[i]])
            else:
                self.logger.warning(f"⚠ Low confidence in strategy selection for {market_data['symbol']}. Using default strategy.")
                selected_strategies.append(self.strategy_manager.default_strategy())

        return selected_strategies

    def start_trading(self):
        """Main trading loop with adaptive strategy selection."""
        while True:
            market_data = self.data_feed.get_market_data()

            # ✅ Step 1: Adaptive Strategy Selection
            selected_strategy = self.choose_best_strategy(market_data)
            if not selected_strategy:
                self.logger.warning("❌ No suitable strategy found. Skipping trade.")
                continue  # Skip trade if no strategy is selected

            # ✅ Step 2: Generate Trade Signal
            trade_signal = selected_strategy.generate_signal(market_data)
            if not trade_signal:
                self.logger.info("No trade signal generated. Waiting for next opportunity.")
                continue  # Skip trade if no valid signal is generated

            # ✅ Step 3: Execute Trade
            Parallel(n_jobs=2)([
                delayed(self.execution_engine.execute_trade)(trade_signal),
                delayed(self.feedback_loop)(trade_signal)
            ])

    def initialize_broker_api(self):
        # Placeholder for initializing broker API
        pass

    def backtest(self, df):
        # Placeholder for backtesting logic
        pass

    async def safe_execute_trade(self, trade):
        try:
            if self.circuit_breaker.is_open():
                self.logger.warning("Circuit breaker active, skipping trade")
                return
                
            for attempt in range(self.retry_attempts):
                try:
                    return await self.execution_engine.execute_order(trade)
                except TemporaryError as e:
                    if attempt == self.retry_attempts - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.logger.error(f"Trade execution failed: {e}")

    def feedback_loop(self, trade_signal):
        """Update ML model based on trade outcomes."""
        # Logic to update model based on trade outcomes
        pass

    def monitor_system(self):
        """Real-time monitoring and alerting."""
        # Implement monitoring logic
        pass

    def analyze_market_depth(self, market_data):
        """
        Detects institutional order flow manipulation, iceberg orders, and spoofing attacks.
        """
        order_book = self.data_feed.get_order_book(market_data["symbol"])
        
        bid_ask_spread = order_book["ask_price"] - order_book["bid_price"]
        order_book_depth = sum(order_book["bid_volume"]) - sum(order_book["ask_volume"])

        if self.detect_spoofing(order_book):
            self.logger.warning(f"🚨 Spoofing detected in {market_data['symbol']}. Skipping trade.")
            return False

        if self.detect_dark_pool_activity(market_data):
            self.logger.warning(f"🚨 Dark Pool activity detected in {market_data['symbol']}. Skipping trade.")
            return False

        if bid_ask_spread > 0.02 * market_data["price"]:  
            self.logger.warning(f"🚫 High bid-ask spread detected for {market_data['symbol']}. Skipping trade.")
            return False

        return True

    def detect_spoofing(self, order_book):
        """
        Identifies spoofing by detecting large orders that disappear quickly.
        """
        recent_orders = self.data_feed.get_recent_orders(order_book["symbol"])
        suspicious_orders = [o for o in recent_orders if o["size"] > 1000 and o["duration"] < 0.5]

        return len(suspicious_orders) > 5  # If multiple spoof orders are detected, return True

    def detect_iceberg_orders(self, order_book):
        """
        Detects iceberg orders by tracking hidden order execution patterns.
        """
        iceberg_levels = [level for level in order_book["bid_levels"] if level["volume"] > 5000]
        return iceberg_levels[0]["price"] if iceberg_levels else None

    # to call liquidity manager and track whale movements before making trades
    def __init__(self, config):
        self.config = config
        self.liquidity_manager = LiquidityManager()
        self.technical_analysis = TechnicalAnalysis()
        self.sentiment_analysis = SentimentAnalysis()

    def should_trade(self, exchange, symbol):
        """Determine if a trade should be executed based on liquidity."""
        if self.liquidity_manager.detect_whale_activity(exchange, symbol):
            print(f"🚨 Whale detected in {symbol}. Adjusting strategy...")
            return False  # Avoid trade if unusual whale activity is detected
        return True

    def start_trading(self):
        """Main trading loop with dynamic strategy selection."""
        while True:
            market_data = self.data_feed.get_market_data()
            if self.should_trade(market_data['exchange'], market_data['symbol']):
                selected_strategy = self.choose_strategy(market_data)
                trade_signal = selected_strategy.generate_signal(market_data)

                if trade_signal:
                    Parallel(n_jobs=2)([
                        delayed(self.execution_engine.execute_trade)(trade_signal),
                        delayed(self.feedback_loop)(trade_signal)
                    ])
            else:
                print("⚠ Skipping trade due to whale detection")


# Modular Configuration Management

class Config:
    def __init__(self, config_file='config.yaml'):
        self.config = self._load_config(config_file)
        self.validate_config()
        self.watch_config_changes()
        
    def _load_config(self, config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
            
    def validate_config(self):
        # Add config validation logic
        schema = self.load_config_schema()
        jsonschema.validate(self.config, schema)

# Performance Monitoring and Metrics

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'latency': [],
            'win_rate': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }
        self.prometheus_client = PrometheusClient()  # For metrics export

    async def track_execution(self, trade_start, trade_end):
        latency = trade_end - trade_start
        self.metrics['latency'].append(latency)
        await self.prometheus_client.push_metric('trade_latency', latency)

    def calculate_performance_metrics(self):
        # Calculate various trading performance metrics
        pass

# Basic Machine Learning Integration

class MLStrategy(Strategy):
    def __init__(self):
        self.model = RandomForestClassifier()
        self.feature_engineer = FeatureEngineer()
        
    def train(self, X, y):
        self.model.fit(X, y)

    async def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        features = self.feature_engineer.process(market_data)
        prediction = self.model.predict(features)
        return {'signal': prediction}

# Market Impact Analysis(for better execution)

class MarketImpactAnalyzer:
    def __init__(self):
        self.order_book_analyzer = OrderBookAnalyzer()
        self.liquidity_calculator = LiquidityCalculator()
        
    async def analyze_impact(self, trade_size, market_data):
        liquidity = await self.liquidity_calculator.get_market_liquidity()
        order_book_depth = await self.order_book_analyzer.get_depth()
        
        return {
            'expected_slippage': self.calculate_slippage(trade_size, liquidity),
            'market_impact_cost': self.estimate_impact_cost(trade_size, order_book_depth),
            'optimal_execution_schedule': self.generate_execution_schedule(trade_size, liquidity)
        }

class TestStrategyManager(unittest.TestCase):
    def test_add_strategy(self):
        manager = StrategyManager()
        strategy = MockStrategy()
        manager.add_strategy(strategy)
        self.assertIn(strategy, manager.strategies)

async def health_check(request):
    return web.Response(text="OK")

app = web.Application()
app.router.add_get('/health', health_check)
web.run_app(app)

if __name__ == '__main__':
    unittest.main()

consumer = KafkaConsumer('market_data', bootstrap_servers='localhost:9092')
for message in consumer:
    process_market_data(message.value)

env = gym.make('TradingEnv-v0')
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

class TradeDetails(BaseModel):
    trade_type: str
    price: float
    volume: int
    strategy: str

    def build_lstm_model(input_shape):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

class FundamentalAnalysis:
    def __init__(self):
        self.news_sources = [
            "https://finnhub.io/api/v1/news?category=general&token=YOUR_API_KEY",
            "https://newsapi.org/v2/top-headlines?category=business&apiKey=YOUR_API_KEY"
        ]
        self.macroeconomic_factors = {
            "interest_rate": None,
            "inflation_rate": None,
            "gdp_growth": None
        }

    def fetch_latest_news(self):
        """Fetch latest macroeconomic news headlines."""
        articles = []
        for url in self.news_sources:
            response = requests.get(url)
            if response.status_code == 200:
                articles.extend(response.json())
        return articles

    def analyze_news_sentiment(self):
        """Perform sentiment analysis on financial news headlines."""
        articles = self.fetch_latest_news()
        sentiment_scores = [TextBlob(article['title']).sentiment.polarity for article in articles]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        return avg_sentiment

    def get_macro_data(self):
        """Fetch macroeconomic data (e.g., interest rates, GDP growth)."""
        response = requests.get("https://api.tradingeconomics.com/markets/forex?c=YOUR_API_KEY")
        if response.status_code == 200:
            self.macroeconomic_factors.update(response.json())
        return self.macroeconomic_factors

    def overall_fundamental_score(self):
        """Combine macroeconomic data & news sentiment into a single weighted score."""
        sentiment = self.analyze_news_sentiment()
        macro = self.get_macro_data()

        # Weighted Scoring System
        weighted_score = (sentiment * 0.3) + \
                         (macro["interest_rate"] * 0.4) + \
                         (macro["gdp_growth"] * 0.3)

        return weighted_score

# ✅ Example Usage
fundamental_analysis = FundamentalAnalysis()
fundamental_score = fundamental_analysis.overall_fundamental_score()

if fundamental_score < -0.3:
    print("⚠ Avoid trading due to negative macro conditions.")
elif fundamental_score > 0.3:
    print("✅ Favorable trading conditions detected.")

class RetailBotCore:
    """Main trading bot logic integrating all AI models & risk management."""

    def __init__(self, execution_engine):
        self.execution_engine = execution_engine
        self.risk_manager = AdaptiveRiskManagement()  # ✅ Replacing static risk management with adaptive risk control
        self.fundamental_analysis = FundamentalAnalysis()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.liquidity_manager = LiquidityManager()
        self.technical_analysis = TechnicalAnalysis()
        self.maddpg = MADDPG(state_dim=10, action_dim=1)

    class MarketImpactModel:
        """
        Predicts the impact of bot's own orders before executing trades.
        """

        def predict_impact(self, trade_size, market_data):
            order_book = self.data_feed.get_order_book(market_data["symbol"])
            liquidity = sum(order_book["bid_volume"]) + sum(order_book["ask_volume"])

            expected_slippage = trade_size / liquidity * 100  # Estimate slippage in %

            return expected_slippage

    def execute_trade(self, market_data):
        """Executes trade if all AI layers confirm, using adaptive risk management and market impact prediction."""

        trade_size = market_data["volume"]
        slippage_prediction = self.market_impact_model.predict_impact(trade_size, market_data)

        if slippage_prediction > 0.5:  # If predicted slippage > 0.5%, avoid trade
            self.logger.warning(f"⚠ High expected slippage ({slippage_prediction:.2f}%). Skipping trade.")
            return

        if self.should_trade(market_data):
            trade_details = {
                "price": market_data["price"],
                "volume": market_data["volume"],
                "strategy": "AI-Layered Decision with Adaptive Risk Control"
            }

            risk_decision = self.risk_manager.evaluate_trade(
                entry_price=trade_details["price"],
                stop_loss=trade_details["price"] * 0.98,  # Auto-set stop loss at 2% below entry
                take_profit=trade_details["price"] * 1.04,  # Auto-set take profit at 4% above entry
                slippage=market_data.get("slippage", 0),
                market_data=market_data
            )

            if "REJECTED" in risk_decision:
                print(risk_decision)  # Log reason for rejection
                return

            self.execution_engine.execute_trade(trade_details)
            print("✅ Trade Executed with Minimal Market Impact and Dynamic Risk Adjustments")

class MetaAICoordinator:
    """
    Combines outputs from MADDPG, ML Model, and Technical Analysis 
    to ensure optimal trade decision-making.
    """

    def __init__(self):
        self.maddpg = MADDPG(state_dim=10, action_dim=1)
        self.machine_learning_model = MachineLearningModel()
        self.technical_analysis = TechnicalAnalysis()

    def decide_trade(self, market_data):
        """
        Uses a weighted ensemble model to determine final trade decision.
        """

        # Get predictions from all models
        maddpg_decision = self.maddpg.select_action(
            self.technical_analysis.extract_technical_features(market_data)
        )
        ml_prediction, ml_confidence = self.machine_learning_model.predict(market_data)

        # **Ensemble Weighting Logic**
        final_decision = (
            (maddpg_decision * 0.5) + 
            (ml_prediction * 0.3) + 
            (self.technical_analysis.validate_trade(market_data) * 0.2)
        )

        return final_decision > 0.6  # ✅ Only trade if confidence is 60%+

    def evaluate_trade_risk(self, market_data):
        """
        Evaluates the risk of a trade using additional AI models.
        """
        risk_score = self.machine_learning_model.evaluate_risk(market_data)
        return risk_score < 0.7  # Proceed only if risk is below 70%
