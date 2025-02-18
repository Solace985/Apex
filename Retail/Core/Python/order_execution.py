import logging
from Retail.Brokers.broker_factory import BrokerFactory

logger = logging.getLogger(__name__)

class OrderExecution:
    """Handles trade execution by routing orders to the best available broker."""

    def __init__(self, mode="LIVE"):
        """
        Initializes the execution system with a broker factory.
        
        :param mode: "LIVE" for real trading, "TEST" for simulated trading.
        """
        self.mode = mode.upper()
        self.broker_factory = BrokerFactory(mode=self.mode)

    def execute_trade(self, order_details):
        """
        Routes a trade to the best available broker and executes it.
        
        :param order_details: Dictionary containing trade details (symbol, amount, price, order_type).
        :return: Execution result or error message.
        """
        try:
            # Select the best broker based on order details
            broker = self.broker_factory.get_best_broker(order_details, mode=self.mode)
            
            logger.info(f"✅ Executing trade through {broker.__class__.__name__}...")

            # Execute the order
            result = broker.place_order(
                symbol=order_details["symbol"],
                qty=order_details["amount"],
                order_type=order_details.get("order_type", "LIMIT"),
                price=order_details["price"]
            )

            logger.info(f"✅ Trade executed successfully: {result}")
            return result

        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return {"status": "FAILED", "error": str(e)}

