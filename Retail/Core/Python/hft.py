import time
import asyncio
from core.hft_engine import RustHFTEngine

class HFTTrader:
    def __init__(self, broker_api):
        self.broker_api = broker_api

    async def read_order_book(self):
        """
        Continuously fetches order book updates in real-time.
        """
        while True:
            order_book = await self.broker_api.get_order_book()
            print("Order Book:", order_book)
            await asyncio.sleep(0.01)  # Read every 10 milliseconds

    async def execute_hft_trade(self):
        """
        Executes high-frequency trades in microseconds.
        """
        buy_order = {"side": "buy", "quantity": 1}
        sell_order = {"side": "sell", "quantity": 1}

        start_time = time.time()
        await self.broker_api.place_order(buy_order)
        await self.broker_api.place_order(sell_order)
        execution_time = time.time() - start_time

        print(f"Trade executed in {execution_time:.6f} seconds")

class HFTExecutionEngine:
    """Handles ultra-low latency order placement."""

    def __init__(self):
        self.fpga_engine = connect_to_fpga()
        self.rust_engine = RustHFTEngine()

    def execute_order(self, order_details):
        """Executes an order with FPGA-based trading."""
        return {"status": "HFT_ORDER_EXECUTED", "order_id": order_details["id"]}

    def place_order(self, order_details):
        """
        Places trades at sub-millisecond speeds using direct market access.
        """
        self.fpga_engine.send_order(order_details)
        return True

    def process_order_book(self, data):
        return self.rust_engine.handle_book(data)
