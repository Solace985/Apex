import asyncio
import logging
import requests
from typing import Dict, Any
from AI_Models.order_flow import InstitutionalOrderFlow

class DataFeed:
    """Handles market data fetching asynchronously and fetches real-time market data from exchange APIs."""

    def __init__(self, interval: int = 1):
        self.market_data: Dict[str, Any] = {}
        self.interval = interval
        self.running = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_url = "https://api.binance.com/api/v3/ticker/price"
        self.institutional_tracker = InstitutionalOrderFlow()

    async def fetch_data(self):
        """Fetches market data asynchronously."""
        while self.running:
            self.logger.info("Fetching market data...")
            # Simulate fetching market data (Replace this with real API call)
            self.market_data = self.get_market_data()
            await asyncio.sleep(self.interval)

    async def start(self):
        """Starts the data feed."""
        self.running = True
        await self.fetch_data()

    def stop(self):
        """Stops the data feed."""
        self.running = False

    def get_market_data(self, symbol="BTCUSDT"):
        """Fetches latest market data for the given trading pair."""
        try:
            response = requests.get(self.api_url, params={"symbol": symbol})
            if response.status_code == 200:
                data = response.json()
                return {
                    "symbol": symbol,
                    "price": float(data["price"]),
                    "volume": 1  # Placeholder, should be replaced with actual volume
                }
            else:
                self.logger.error(f"❌ API error: {response.status_code}")
                return None
        except Exception as e:
            self.logger.error(f"❌ Error fetching market data: {e}")
            return None

    def fetch_market_data(self, symbol="BTCUSD"):
        """Fetches price, volume & institutional bias."""
        market_data = self.get_market_data(symbol)
        if market_data:
            top_bid, top_ask = self.institutional_tracker.fetch_order_book_data(symbol)
            institutional_bias = self.institutional_tracker.get_institutional_bias()
            market_data.update({"top_bid": top_bid, "top_ask": top_ask, "institutional_bias": institutional_bias})
        return market_data
