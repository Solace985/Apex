import logging
from logging.handlers import RotatingFileHandler  # ✅ Import log rotation handler

from typing import Dict, Any
from brokers.broker_api import BrokerAPI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import RequestException, Timeout, HTTPError
import asyncio
import random
import time
import websockets
import json
from threading import Lock
import uuid
from Core.liquidity_manager import RateLimiter
from Core.logger import get_logger
from Core.risk_management import RiskManager
from Brokers.broker_api import BrokerAPI
from Core.hft import HFTExecutionEngine
from Brokers.websocket_handler import WebSocketExecutionEngine
from tenacity import retry, stop_after_attempt
from Core.config import load_config  # Import load_config
from session_detector import TradingSessionDetector
from Core.strategy_selector import StrategyManager
from AI_Models.sentiment_analysis import SentimentAnalyzer
from Core.market_impact import MarketImpactAnalyzer
from Core.liquidity_manager import LiquidityManager

config = load_config()  # Load configuration

logger = get_logger("execution_engine")

# This file routes the trades to brokers and executes them.

class ExecutionEngine:
    """Handles trade execution with AI, risk validation, and rate limiting."""

    def __init__(self, broker_api, risk_manager):
        self.broker_api = broker_api
        self.risk_manager = risk_manager  # ✅ Integrated risk manager
        self.logger = logging.getLogger(__name__)
        self.rate_limiter = RateLimiter(10, 1)  # Allow max 10 orders per second
        self.failed_order_count = 0  # Track failed orders
        self.executed_orders = set()  # 🛑 Track executed order IDs
        self.max_failures = 5  # ⛔ Stop execution if more than 5 consecutive failures
        self.failure_cooldown = 30  # ⏳ Pause execution for 30 seconds if circuit breaker is triggered
        
        # New attributes from config
        self.trade_confirmation = config.execution.trade_confirmation_required
        self.slippage_protection = config.execution.slippage_protection
        self.session_detector = TradingSessionDetector()  # ✅ Initialize session detector
        self.strategy_selector = StrategyManager()  # ✅ Initialize strategy selector
        self.sentiment_analyzer = SentimentAnalyzer()  # ✅ Initialize sentiment analysis
        self.market_impact_analyzer = MarketImpactAnalyzer()  # ✅ Initialize market impact analysis
        self.liquidity_manager = LiquidityManager()  # ✅ Initialize liquidity manager

    @retry(stop=stop_after_attempt(3))
    async def execute_order(self, order):
        """Executes an order only after checking liquidity, market impact, strategy selection, and sentiment analysis."""

        if not order['symbol'] in self.config['ALLOWED_SYMBOLS']:
            raise InvalidOrderError(f"Symbol {order['symbol']} not allowed")

        # 🛑 Prevent duplicate execution
        if order["id"] in self.executed_orders:
            self.logger.warning(f"⚠ Order {order['id']} has already been executed. Skipping.")
            return False

        # 💡 Split large orders into smaller ones
        max_order_size = 1000  # Adjust based on liquidity
        if order["amount"] > max_order_size:
            num_parts = order["amount"] // max_order_size
            self.logger.info(f"🔄 Splitting large order into {num_parts} smaller orders.")
            for _ in range(num_parts):
                small_order = order.copy()
                small_order["amount"] = max_order_size
                await self.execute_order(small_order)
            return True  # ✅ Skip sending the original large order

        # ✅ 1️⃣ Check Liquidity Before Executing Trade
        market_liquidity = await self.liquidity_manager.get_market_liquidity(order["exchange"], order["symbol"])
        whale_activity = await self.liquidity_manager.detect_whale_activity(order["exchange"], order["symbol"])

        self.logger.info(f"📊 Market Liquidity: {market_liquidity}")
        self.logger.info(f"🐋 Whale Activity Detected: {whale_activity}")

        if whale_activity:
            self.logger.warning("🚫 Trade skipped due to large institutional orders.")
            return False

        # ✅ 2️⃣ Adjust Trade Size Based on Liquidity
        adjusted_trade_size = await self.liquidity_manager.adjust_trade_size_based_on_liquidity(order["amount"], order["exchange"], order["symbol"])
        order["amount"] = adjusted_trade_size

        if adjusted_trade_size < 1:
            self.logger.warning("🚫 Trade size too small after liquidity adjustment. Skipping trade.")
            return False

        self.logger.info(f"✅ Adjusted Trade Size: {adjusted_trade_size}")

        # ✅ 3️⃣ Detect Trading Session
        session = self.session_detector.get_current_session()
        self.logger.info(f"🕰️ Active Trading Session: {session}")

        # ✅ 4️⃣ Select the Best Strategy for Asset Class
        asset_class = order.get("asset_class", "Unknown")
        strategy = self.strategy_selector.select_strategy(asset_class)
        indicators = self.strategy_selector.get_indicators(asset_class)
        fundamentals = self.strategy_selector.get_fundamental_sources(asset_class)

        self.logger.info(f"🔍 Selected strategy: {strategy.__class__.__name__}")
        self.logger.info(f"📊 Using indicators: {indicators}")
        self.logger.info(f"📑 Fundamental sources: {fundamentals}")

        # ✅ 5️⃣ Perform Sentiment Analysis Before Executing Trade
        sentiment_score = self.sentiment_analyzer.get_sentiment_score()
        sentiment_data = self.sentiment_analyzer.integrate_sentiment_with_trading(order)

        self.logger.info(f"📰 Sentiment Score: {sentiment_score}")
        self.logger.info(f"📈 Sentiment-Based Trade Signal: {sentiment_data['sentiment_signal']}")

        if sentiment_data['sentiment_signal'] == "hold":
            self.logger.warning(f"🚫 Trade skipped due to neutral sentiment ({sentiment_score}).")
            return False

        # ✅ 6️⃣ Check Market Impact Before Executing Trade
        market_impact = await self.market_impact_analyzer.analyze_impact(order["amount"], order["market_data"])
        self.logger.info(f"💰 Expected Slippage: {market_impact['expected_slippage']}")
        self.logger.info(f"📊 Market Impact Cost: {market_impact['market_impact_cost']}")
        self.logger.info(f"⚡ Optimal Execution Schedule: {market_impact['optimal_execution_schedule']}")

        if market_impact['market_impact_cost'] > 0.05:
            self.logger.warning("🚫 Trade skipped due to high market impact cost.")
            return False

        # ✅ AI-Based Risk Assessment Before Trade Execution
        risk_evaluation = self.risk_manager.evaluate_trade(order, order["market_data"])
        if "REJECTED" in risk_evaluation:
            self.logger.warning(f"🚫 Trade rejected: {risk_evaluation}")
            return False

        # ✅ 7️⃣ Dynamic Stop-Loss & Position Sizing Before Execution
        order["stop_loss"] = self.risk_manager.dynamic_stop_loss(order["market_data"]["volatility"])
        order["position_size"] = self.risk_manager.calculate_position_size(order["symbol"], order["strategy_type"])

        self.logger.info(f"📉 Adjusted Stop-Loss: {order['stop_loss']}")
        self.logger.info(f"📊 AI-Optimized Position Size: {order['position_size']}")

        # ✅ 8️⃣ Adjust Trade Parameters Based on Session (Without Overriding AI Stop-Loss)
        if session == "New York Session":
            order['take_profit'] = order['take_profit'] * 1.2
        elif session == "Tokyo Session":
            order['amount'] = order['amount'] * 0.5
        elif session == "London Session":
            order['risk_reward'] = order.get('risk_reward', 1) * 1.5

        # 🚨 AI Confidence Check
        if "confidence" in order and order["confidence"] < 0.6:
            self.logger.warning(f"⚠ Trade confidence too low ({order['confidence']}). Skipping execution.")
            return False

        # 🛑 Circuit breaker: Stop if too many failures
        if self.failed_order_count >= self.max_failures:
            self.logger.critical("🚨 Too many failed orders. Stopping execution for 30 seconds.")
            await asyncio.sleep(self.failure_cooldown)
            self.failed_order_count = 0

        # 🚨 Rate-Limiting
        if not self.rate_limiter.allow_request():
            self.logger.warning("⚠ Rate limit exceeded. Delaying execution.")
            time.sleep(1)

        # 📊 Adaptive Retry Delay
        market_volatility = self.get_market_volatility()
        retry_delay = 1 if market_volatility > 0.8 else 3 if market_volatility > 0.5 else 5
        self.logger.warning(f"⚠ Market volatility: {market_volatility}. Retrying in {retry_delay} seconds.")
        await asyncio.sleep(retry_delay)

        # ✅ 9️⃣ Proceed with Trade Execution
        try:
            self.logger.info(f"✅ Executing Order {order['id']}")
            self.failed_order_count = 0
            self.executed_orders.add(order["id"])
            return self.broker_api.place_order(
                symbol=order['symbol'],
                qty=order['amount'],
                order_type="LIMIT",
                price=order['price']
            )
        except Exception as e:
            self.failed_order_count += 1
            self.logger.error(f"❌ Trade execution failed: {e}")
            return None

    async def execute_orders_concurrently(self, orders):
        """Execute multiple orders concurrently."""
        tasks = [self.execute_order(order) for order in orders]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for order, result in zip(orders, results):
            if isinstance(result, Exception):
                self.logger.error(f"Order {order['id']} failed with exception: {result}")
            else:
                self.logger.info(f"Order {order['id']} executed successfully with result: {result}")

    def validate_order(self, order):
        """Validate order before execution."""
        if 'id' not in order or 'amount' not in order:
            raise ValueError("Order must contain 'id' and 'amount' fields")
        if order['amount'] <= 0:
            raise ValueError("Order amount must be greater than zero")

    async def log_execution(self, order_id, response, execution_time):
        """Log execution details asynchronously."""
        self.logger.info(f"Order {order_id} executed successfully in {execution_time:.2f} seconds: {response}")
        # Add additional logging or monitoring integration here

    def monitor_order_execution(self, order_id, response):
        """Integrate with monitoring system to track order execution."""
        self.logger.info(f"Monitoring order {order_id}: {response}")
        # Add integration with a monitoring system here

    def rate_limit(self):
        """Implement rate limiting to prevent exceeding API call limits."""
        # Example: Use a token bucket algorithm or similar approach
        pass

    def execute_trade(self, order_details, order_type="market"):
        """Executes market, limit, or stop orders with rate-limiting checks."""
        
        # Enforce rate limiting
        if not self.rate_limiter.allow_request():
            self.logger.warning("⚠ Rate limit exceeded. Delaying execution.")
            time.sleep(1)

        valid_order_types = ["market", "limit", "stop"]
        if order_type not in valid_order_types:
            raise ValueError(f"❌ Invalid order type: {order_type}")

        return self.broker_api.place_order(order_details)

    def cancel_order(self, order_id):
        """Cancels an order with retry logic if the first attempt fails."""
        max_retries = 3
        attempt = 0

        while attempt < max_retries:
            try:
                order_status = self.broker_api.get_order_status(order_id)

                if order_status == "FILLED":
                    self.logger.warning(f"⚠ Order {order_id} is already filled. Cannot cancel.")
                    return None

                self.logger.info(f"✅ Order {order_id} cancelled successfully")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to cancel order {order_id}: {e}. Retrying...")
                attempt += 1
                time.sleep(2)  # Wait before retrying

        self.logger.error(f"🚨 Order {order_id} could not be cancelled after multiple attempts.")
        return None

class SmartOrderRouter:
    """Optimizes order execution across multiple brokers to minimize costs."""

    def __init__(self, broker_list):
        self.broker_list = broker_list

    def choose_best_broker(self, trade_details):
        """Selects the broker with the best combination of execution cost, speed, and market depth."""
        best_broker = min(
            self.broker_list,
            key=lambda broker: (
                broker.get_execution_cost(trade_details)
                + broker.get_execution_latency(trade_details)
                - broker.get_order_book_depth(trade_details)
            )
        )
        return best_broker
