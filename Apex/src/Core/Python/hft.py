import time
import asyncio
import logging
import concurrent.futures
from core.hft_engine import RustHFTEngine

logger = logging.getLogger(__name__)

# ✅ Dummy FPGA Engine for Testing (Replace with actual FPGA Class)
class FakeFPGAEngine:
    def send_order(self, order_details):
        logger.info(f"⚡ Simulating FPGA order execution: {order_details}")
        return {"status": "EXECUTED", "order_id": order_details.get("id", "UNKNOWN")}

def connect_to_fpga():
    """Connect to FPGA-based execution engine."""
    logger.info("🔌 FPGA Engine Connected")
    return FakeFPGAEngine()  # Replace with actual FPGA engine

class HFTTrader:
    """Handles high-frequency trading (HFT) operations."""

    def __init__(self, broker_api):
        self.broker_api = broker_api
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)  # ✅ Use a thread pool

    async def read_order_book(self, stop_event, timeout=60):
        """
        Fetches order book updates in real-time with an exit condition.
        
        :param stop_event: Async event to stop the loop.
        :param timeout: Maximum time (seconds) to auto-exit.
        """
        start_time = time.time()
        while not stop_event.is_set():
            try:
                order_book = await self.broker_api.get_order_book()
                logger.info(f"📊 Order Book: {order_book}")
            except Exception as e:
                logger.error(f"❌ Error reading order book: {e}")

            await asyncio.sleep(0.01)  # Read every 10 milliseconds

            if time.time() - start_time > timeout:
                logger.warning("⏳ Order book monitoring timed out.")
                break

    async def execute_hft_trade(self, max_retries=3):
        """
        Executes high-frequency trades in microseconds with async safety & retry logic.
        """
        buy_order = {"side": "buy", "quantity": 1}
        sell_order = {"side": "sell", "quantity": 1}

        start_time = time.time()

        async def safe_place_order(order):
            """Retries order placement with async execution in a background thread."""
            for attempt in range(max_retries):
                try:
                    return await asyncio.get_running_loop().run_in_executor(self.executor, self.broker_api.place_order, order)
                except Exception as e:
                    logger.error(f"❌ Attempt {attempt+1}: Failed to place {order['side']} order: {e}")
                    await asyncio.sleep(0.01)

            return {"status": "FAILED", "error": f"Max retries reached for {order['side']} order."}

        # ✅ Execute buy & sell orders with retry logic
        try:
            buy_result, sell_result = await asyncio.gather(
                safe_place_order(buy_order),
                safe_place_order(sell_order),
                return_exceptions=True  # ✅ Prevents one failure from blocking the entire process
            )

            # ✅ Handle individual failures separately
            if isinstance(buy_result, Exception):
                logger.error(f"❌ Buy order failed: {buy_result}")
            if isinstance(sell_result, Exception):
                logger.error(f"❌ Sell order failed: {sell_result}")

        except Exception as e:
            logger.error(f"❌ High-Frequency Trade execution failed: {e}")

        execution_time = time.time() - start_time
        logger.info(f"⚡ Trade executed in {execution_time:.6f} seconds")

        return {"buy": buy_result, "sell": sell_result}

    def detect_arbitrage(self):
        prices = self.get_prices()

        min_price_broker = min(prices, key=prices.get)
        max_price_broker = max(prices, key=prices.get)
        spread = prices[max_price_broker] - prices[min_price_broker]

        # ✅ Volatility check before placing trades
        if self.is_market_too_volatile(spread, min_price_broker, max_price_broker):
            logger.warning("🚨 Market too volatile! Skipping arbitrage trade.")
            return

        if spread > profit_threshold:
            logger.info(f"✅ Arbitrage Opportunity: Buy on {min_price_broker}, Sell on {max_price_broker}")
            self.execute_trade(min_price_broker, max_price_broker, prices[min_price_broker])

    def is_market_too_volatile(self, spread, buy_broker, sell_broker):
        """
        Determines if the market is too volatile to trade.
        """
        volatility_index = abs(spread / ((self.get_prices()[buy_broker] + self.get_prices()[sell_broker]) / 2))

        if volatility_index > 0.05:  # Example: 5% threshold for volatility
            return True
        return False

class HFTExecutionEngine:
    """Handles ultra-low latency order placement using FPGA & Rust."""

    def __init__(self):
        self.fpga_engine = connect_to_fpga()
        self.rust_engine = RustHFTEngine()

    def execute_order(self, order_details):
        """Executes an order using FPGA/Rust for low-latency trading."""
        try:
            self.fpga_engine.send_order(order_details)  # Ensure FPGA processing
            result = self.rust_engine.execute_trade(order_details)  # Call Rust engine
            return {"status": "HFT_ORDER_EXECUTED", "order_id": order_details.get("id", "UNKNOWN"), "result": result}
        except Exception as e:
            logger.error(f"❌ Error executing HFT order: {e}")
            return {"status": "FAILED", "error": str(e)}

    def place_order(self, order_details):
        """
        Places trades at sub-millisecond speeds using direct market access.
        Ensures confirmation of execution before marking as success.
        """
        try:
            response = self.fpga_engine.send_order(order_details)

            if response.get("status") == "EXECUTED":
                # ✅ Verify order is actually in the exchange's order book
                confirmed = self.confirm_order_execution(response["order_id"])
                if confirmed:
                    return {"status": "SUCCESS", "order_id": response["order_id"]}
                else:
                    logger.warning("⚠️ Order execution was marked EXECUTED but not found in the order book.")
                    return {"status": "FAILED", "error": "Execution not confirmed"}

        except Exception as e:
            logger.error(f"❌ FPGA order placement failed: {e}")
            return {"status": "FAILED", "error": str(e)}

    def confirm_order_execution(self, order_id):
        """
        Confirms that the order was successfully placed and exists in the order book.
        """
        for _ in range(3):  # Retry up to 3 times
            status = self.fpga_engine.check_order_status(order_id)
            if status == "FILLED":
                return True
            time.sleep(0.01)  # Small delay before retrying
        return False

    async def process_order_book(self, data):
        """
        Processes order book updates asynchronously using Rust engine.
        """
        return await asyncio.to_thread(self.rust_engine.handle_book, data)
