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

config = load_config()  # Load configuration

logger = get_logger("execution_engine")

# This file routes the trades to brokers and executes them.

class ExecutionEngine:
    """Handles trade execution with AI, risk validation, and rate limiting."""

    def __init__(self, broker_api, risk_manager, logger=None):
        self.broker_api = broker_api
        self.risk_manager = risk_manager  # ✅ Integrated risk manager
        self.logger = logger or logging.getLogger(__name__)
        self.rate_limiter = RateLimiter(10, 1)  # Allow max 10 orders per second
        self.failed_order_count = 0  # Track failed orders
        self.executed_orders = set()  # 🛑 Track executed order IDs
        self.max_failures = 5  # ⛔ Stop execution if more than 5 consecutive failures
        self.failure_cooldown = 30  # ⏳ Pause execution for 30 seconds if circuit breaker is triggered
        
        # New attributes from config
        self.trade_confirmation = config.execution.trade_confirmation_required
        self.slippage_protection = config.execution.slippage_protection
        self.session_detector = TradingSessionDetector()  # Initialize session detector

    @retry(stop=stop_after_attempt(3))
    async def execute_order(self, order):
        """Executes an order only if AI confidence, risk evaluation, and rate limiting pass."""

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

        # 🚨 AI Confidence Check (New Addition)
        if "confidence" in order and order["confidence"] < 0.6:
            self.logger.warning(f"⚠ Trade confidence too low ({order['confidence']}). Skipping execution.")
            return False

        # 🛑 Circuit breaker: Stop if too many failures
        if self.failed_order_count >= self.max_failures:
            self.logger.critical("🚨 Too many failed orders. Stopping execution for 30 seconds.")
            await asyncio.sleep(self.failure_cooldown)
            self.failed_order_count = 0  # Reset after cooldown

        # 🚨 Rate-Limiting
        if not self.rate_limiter.allow_request():
            self.logger.warning("⚠ Rate limit exceeded. Delaying execution.")
            time.sleep(1)

        # If more than 3 consecutive failures, introduce a cooldown
        if self.failed_order_count >= 3:
            self.logger.warning("⚠ High failure rate detected. Cooling down for 5 seconds...")
            await asyncio.sleep(5)
            self.failed_order_count = 0  # Reset count

        # 📊 Adaptive Retry Delay
        market_volatility = self.get_market_volatility()  # 🔍 Get market conditions

        if market_volatility > 0.8:
            retry_delay = 1  # 🚀 Retry fast in high volatility
        elif market_volatility > 0.5:
            retry_delay = 3  # Moderate retry speed
        else:
            retry_delay = 5  # 🐢 Slow retry in stable market

        self.logger.warning(f"⚠ Market volatility: {market_volatility}. Retrying in {retry_delay} seconds.")
        await asyncio.sleep(retry_delay)

        # Risk evaluation
        risk_result = self.risk_manager.evaluate_trade(order, market_data)

        if "REJECTED" in risk_result:
            self.logger.warning(f"🚨 Trade rejected: {risk_result}")
            return False

        # New session-based trade adjustments
        session = self.session_detector.get_current_session()
        self.logger.info(f"🕰️ Active Trading Session: {session}")

        if session == "New York Session":
            # Trade aggressively, use breakout strategies
            order['stop_loss'] = order['stop_loss'] * 0.8  # Tighten stop loss
            order['take_profit'] = order['take_profit'] * 1.2  # Increase TP
        elif session == "Tokyo Session":
            # Reduce trade volume, avoid momentum strategies
            order['amount'] = order['amount'] * 0.5
        elif session == "London Session":
            # High volatility, increase risk-reward ratio
            order['risk_reward'] = order.get('risk_reward', 1) * 1.5

        try:
            self.logger.info(f"✅ Order {order['id']} executed successfully")
            self.failed_order_count = 0  # Reset failure count on success
            self.executed_orders.add(order["id"])  # Mark order as executed
            
            return self.broker_api.place_order(
                symbol=order['symbol'],
                qty=order['amount'],
                order_type="LIMIT",
                price=order['price'] * 1.001  # Avoid slippage
            )
        except Exception as e:
            self.failed_order_count += 1  # Increment failure count
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
