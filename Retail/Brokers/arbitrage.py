import requests

class ArbitrageTrader:
    def __init__(self, broker_apis):
        self.broker_apis = broker_apis  # List of different broker APIs

    def get_prices(self):
        """
        Fetch real-time prices from all brokers.
        """
        prices = {}
        for broker in self.broker_apis:
            prices[broker] = self.broker_apis[broker].fetch_market_price()
        return prices

    def detect_arbitrage(self):
        """
        Find arbitrage opportunities across exchanges.
        """
        prices = self.get_prices()
        min_price_broker = min(prices, key=prices.get)
        max_price_broker = max(prices, key=prices.get)

        if prices[max_price_broker] - prices[min_price_broker] > 0.5:  # Threshold for profit
            print(f"Arbitrage Opportunity: Buy on {min_price_broker}, Sell on {max_price_broker}")
            self.execute_trade(min_price_broker, max_price_broker, prices[min_price_broker])

    def execute_trade(self, buy_broker, sell_broker, buy_price):
        """
        Execute arbitrage trades between two brokers.
        """
        buy_order = {"side": "buy", "quantity": 1, "price": buy_price}
        sell_order = {"side": "sell", "quantity": 1}

        self.broker_apis[buy_broker].place_order(buy_order)
        self.broker_apis[sell_broker].place_order(sell_order)
