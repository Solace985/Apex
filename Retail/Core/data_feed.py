import asyncio
import logging
from typing import Dict, Any

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
