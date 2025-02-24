class DummyBroker:
    """A dummy broker for testing trade execution without real orders."""
    
    def place_order(self, symbol, qty, order_type, price):
        """Simulates order execution."""
        return {
            "status": "executed",
            "symbol": symbol,
            "qty": qty,
            "order_type": order_type,
            "price": price
        }

    def estimate_fees(self, order_details):
        """Simulates fee estimation (fixed small amount)."""
        return 0.01 * order_details.get("qty", 1)  # 1% simulated fee

    def get_liquidity(self, order_details):
        """Returns simulated liquidity (high)."""
        return 100000  # Arbitrary high liquidity

    def get_execution_speed(self, order_details):
        """Simulates fast execution speed."""
        return 0.01  # 10ms execution speed

    def get_slippage(self, order_details):
        """Simulates slippage (very low for dummy trades)."""
        return 0.0001  # Almost no slippage

    def get_market_depth(self, order_details):
        """Simulates market depth."""
        return 1000000  # Arbitrary deep market

    def __str__(self):
        return "DummyBroker"
