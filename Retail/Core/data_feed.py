import asyncio
import logging
from typing import Dict, Any
from AI_Models.order_flow import InstitutionalOrderFlow

class DataFeed:
    """Handles market data fetching asynchronously."""

    def __init__(self, interval: int = 1):
        self.market_data: Dict[str, Any] = {}
        self.interval = interval
        self.running = False
        self.logger = logging.getLogger(self.__class__.__name__)

    async def fetch_data(self):
        """Fetches market data asynchronously."""
        while self.running:
            self.logger.info("Fetching market data...")
            # Simulate fetching market data (Replace this with real API call)
            self.market_data = {"price": 100.0, "volume": 5000}
            await asyncio.sleep(self.interval)

    async def start(self):
        """Starts the data feed."""
        self.running = True
        await self.fetch_data()

    def stop(self):
        """Stops the data feed."""
        self.running = False


    def __init__(self):
        self.institutional_tracker = InstitutionalOrderFlow()

    def fetch_market_data(self, symbol="BTCUSD"):
        """Fetches price, volume & institutional bias."""
        price = self.fetch_price(symbol)
        volume = self.fetch_volume(symbol)
        top_bid, top_ask = self.institutional_tracker.fetch_order_book_data(symbol)
        institutional_bias = self.institutional_tracker.get_institutional_bias()

        return {"price": price, "volume": volume, "top_bid": top_bid, "top_ask": top_ask, "institutional_bias": institutional_bias}
