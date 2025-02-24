import requests
import logging
import time

logger = logging.getLogger(__name__)

MAX_RETRIES = 3  # ✅ Retry failed API calls up to 3 times
MAX_WAIT_TIME = 10  # ✅ Maximum wait time for orders to fill (in seconds)
MAX_FILL_RETRIES = 20  # ✅ Maximum times to check if an order fully fills

class ArbitrageTrader:
    def __init__(self, broker_apis):
        self.broker_apis = broker_apis  # List of different broker APIs

    def get_prices(self):
        """
        Fetch real-time prices from all brokers with retries and rate limiting.
        """
        prices = {}
        for broker in self.broker_apis:
            for attempt in range(MAX_RETRIES):
                try:
                    time.sleep(0.1)  # ✅ Rate limiting to avoid API bans
                    price = self.broker_apis[broker].fetch_market_price()
                    if price is not None:
                        prices[broker] = price
                        break  # ✅ Success, exit retry loop
                except Exception as e:
                    logger.error(f"❌ Attempt {attempt+1}: Error fetching price from {broker}: {e}")
                    time.sleep(0.5)  # ✅ Small delay before retrying
        return prices

    def detect_arbitrage(self):
        """
        Find arbitrage opportunities across exchanges.
        """
        prices = self.get_prices()

        if any(price is None for price in prices.values()):
            logger.warning("⚠️ Some brokers returned None prices. Skipping arbitrage check.")
            return

        if len(prices) < 2:
            logger.warning("⚠️ Not enough brokers available for arbitrage trading.")
            return

        min_price_broker = min(prices, key=prices.get)
        max_price_broker = max(prices, key=prices.get)

        transaction_cost = 0.002  # Example: 0.2% transaction fee
        profit_threshold = (prices[max_price_broker] * transaction_cost) * 2  # Account for buy & sell fees

        if prices[max_price_broker] <= prices[min_price_broker]:
            logger.warning(f"⚠️ No arbitrage opportunity: Buy price (${prices[min_price_broker]}) equals or exceeds sell price (${prices[max_price_broker]}).")
            return

        if (prices[max_price_broker] - prices[min_price_broker]) > profit_threshold:
            logger.info(f"✅ Arbitrage Opportunity: Buy on {min_price_broker} (${prices[min_price_broker]}), "
                        f"Sell on {max_price_broker} (${prices[max_price_broker]})")
            self.execute_trade(min_price_broker, max_price_broker, prices[min_price_broker], prices[max_price_broker], transaction_cost)

    def execute_trade(self, buy_broker, sell_broker, buy_price, sell_price, transaction_cost):
        """
        Execute arbitrage trades between two brokers.
        """
        # ✅ Ensure sufficient liquidity on the sell side
        sell_liquidity = self.broker_apis[sell_broker].get_liquidity({"symbol": "BTC/USD"})
        if sell_liquidity < 1:
            logger.warning(f"⚠️ Insufficient liquidity on {sell_broker}. Skipping trade.")
            return

        buy_order = {"side": "buy", "quantity": 1, "price": buy_price}
        buy_response = self.broker_apis[buy_broker].place_order(buy_order)

        # ✅ Ensure buy order is completely filled before proceeding
        retry_count = 0
        start_time = time.time()
        while buy_response.get("status") != "FILLED":
            if time.time() - start_time > MAX_WAIT_TIME or retry_count >= MAX_FILL_RETRIES:
                logger.error(f"❌ Buy order on {buy_broker} did not fill in time. Aborting.")
                return

            time.sleep(0.5)  # ✅ Small wait before rechecking order status
            buy_response = self.broker_apis[buy_broker].get_order_status(buy_response["order_id"])
            retry_count += 1

        filled_qty = buy_response.get("filled_quantity", 1)  # ✅ Adjust for partial fills
        sell_order = {"side": "sell", "quantity": filled_qty}
        logger.info(f"✅ Buy order on {buy_broker} filled at ${buy_price}. Placing sell order on {sell_broker}...")

        sell_response = self.broker_apis[sell_broker].place_order(sell_order)
        if sell_response.get("status") == "FILLED":
            profit = (sell_price - buy_price) - (sell_price * transaction_cost)
            logger.info(f"✅ Arbitrage trade successful! Estimated Profit: ${profit:.2f}")
        else:
            logger.warning(f"⚠️ Warning: Sell order on {sell_broker} did not fill! Manual action may be required.")
