import logging
import asyncio
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

    async def get_broker_async(self, order_details):
        """Asynchronously gets the best broker."""
        return await self.broker_factory.get_broker(order_details=order_details)

    def execute_trade(self, order_details):
        """
        Routes a trade to the best available broker and executes it.
        
        :param order_details: Dictionary containing trade details (symbol, amount, price, order_type).
        :return: Execution result or error message.
        """
        try:
            # Validate order details
            required_keys = ["symbol", "amount", "price"]
            missing_keys = [key for key in required_keys if key not in order_details]

            if missing_keys:
                error_msg = f"❌ Missing required order details: {', '.join(missing_keys)}"
                logger.error(error_msg)
                return {"status": "FAILED", "error": error_msg}

            # Select the best broker asynchronously
            loop = asyncio.get_event_loop()
            if loop.is_running():
                broker = loop.create_task(self.get_broker_async(order_details))  # ✅ Non-blocking in async envs
            else:
                broker = asyncio.run(self.get_broker_async(order_details))  # ✅ Works in CLI mode

            if not broker:
                logger.error("❌ No available broker for this trade. Execution aborted.")
                return {"status": "FAILED", "error": "No broker available"}

            # Log trade details for debugging
            logger.info(f"📌 Trade Details: {order_details}")
            logger.info(f"✅ Executing trade through {broker.__class__.__name__}...")

            # Execute the order with proper error handling
            try:
                result = broker.place_order(
                    symbol=order_details["symbol"],
                    qty=order_details["amount"],
                    order_type=order_details.get("order_type", "LIMIT"),
                    price=order_details["price"]
                )

                if not result or isinstance(result, dict) and ("error" in result or "status" in result and result["status"].upper() == "FAILED"):
                    raise ValueError(f"❌ Trade execution failed: {result}")


                logger.info(f"✅ Trade executed successfully: {result}")
                return result

            except Exception as e:
                logger.error(f"❌ Trade execution failed: {e}")
                return {"status": "FAILED", "error": str(e)}

        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return {"status": "FAILED", "error": str(e)}
