import websockets
import asyncio
import json
import logging
import uuid

class WebSocketExecutionEngine:
    """Handles order execution via WebSocket."""
    
    FATAL_ERRORS = ["INSUFFICIENT_FUNDS", "INVALID_SYMBOL", "ACCOUNT_BANNED"]

    def __init__(self, broker_api, broker_ws_url="wss://your-broker-websocket-endpoint"):
        self.broker_api = broker_api
        self.websocket_url = broker_ws_url
        self.websocket = None
        self.logger = logging.getLogger(self.__class__.__name__)

    async def connect(self):
        """Attempts to establish a WebSocket connection with proper cleanup."""
        if self.websocket:
            await self.websocket.close()  # ✅ Close any existing WebSocket before reconnecting

        retries = 0
        max_retries = 10
        while retries < max_retries:
            try:
                self.websocket = await websockets.connect(self.websocket_url)
                self.logger.info("✅ Connected to broker WebSocket.")
                asyncio.create_task(self.send_heartbeat())
                return
            except Exception as e:
                self.logger.error(f"❌ WebSocket connection failed (Attempt {retries+1}/{max_retries}): {e}")
                retries += 1
                await asyncio.sleep(2 ** retries)

        self.logger.critical("🚨 WebSocket failed to reconnect after multiple attempts. Switching to REST API.")
        self.websocket = None  # Ensure websocket is set to None after failure

    async def send_order(self, order_details):
        """Sends an order via WebSocket with a unique request ID and timeout mechanism."""
        if not self.websocket or not self.websocket.open:
            self.logger.warning("⚠ WebSocket disconnected. Waiting for reconnection...")
            await asyncio.sleep(5)
            await self.connect()

        order_details["request_id"] = str(uuid.uuid4())
        order_payload = json.dumps(order_details)
        max_retries = 3
        attempt = 0
        failure_count = 0  # Track the number of failed orders

        while attempt < max_retries:
            try:
                self.logger.info(f"🔄 Attempt {attempt+1}/{max_retries} - Sending order {order_details['id']} via WebSocket...")
                await asyncio.wait_for(self.websocket.send(order_payload), timeout=5)
                response = await asyncio.wait_for(self.websocket.recv(), timeout=5)
                response_data = json.loads(response)

                if response_data.get("error_code") in self.FATAL_ERRORS:
                    self.logger.critical(f"🚨 Fatal error for order {order_details['id']}: {response_data['error_code']}. Not retrying.")
                    return None  # Stop retrying for fatal errors

                if response_data.get("status") == "FILLED":
                    self.logger.info(f"✅ Order {order_details['id']} successfully filled.")
                    return response_data
                elif response_data.get("status") == "PARTIALLY_FILLED":
                    remaining_qty = order_details["amount"] - response_data["filled_quantity"]
                    if remaining_qty > 0:
                        order_details["amount"] = remaining_qty
                        self.logger.warning(f"⚠ Order {order_details['id']} partially filled. Adjusting remaining quantity and resending...")
                        return await self.send_order(order_details)  # 🔁 Retry with remaining order
                elif response_data.get("status") == "REJECTED":
                    self.logger.error(f"❌ Order {order_details['id']} rejected. Retrying...")
                    attempt += 1
                    failure_count += 1  # Increment failure count
                    await asyncio.sleep(2)
                else:
                    return response_data

            except websockets.exceptions.ConnectionClosed as e:
                self.logger.error(f"❌ WebSocket send failed: {e}. Reconnecting...")
                await self.connect()
                return await self.send_order(order_details)  # 🔁 Retry sending order after reconnecting
            except asyncio.TimeoutError:
                self.logger.error(f"⏳ Order {order_details['id']} timed out. Retrying...")
                attempt += 1
                failure_count += 1  # Increment failure count
                await asyncio.sleep(2)
            except Exception as e:
                self.logger.error(f"❌ Unexpected error sending WebSocket order: {e}. Retrying in 5 seconds...")
                attempt += 1  # 🔄 Increment attempt count
                failure_count += 1
                await asyncio.sleep(5)  # ⏳ Delay before retrying

            # ⏳ WebSocket failed, switching to REST API
            if self.websocket is None or not self.websocket.open:
                self.logger.warning("⚠ WebSocket is down. Switching to REST API for order execution.")
                return await self.broker_api.place_order(order_details)  # ⏩ Use REST API instead

        if failure_count >= 5:
            self.logger.critical("🚨 Too many failed orders. Stopping execution.")
            return None

    async def send_heartbeat(self):
        """Sends periodic pings to keep WebSocket connection alive."""
        while True:
            try:
                if self.websocket and self.websocket.open:
                    await self.websocket.ping()
                    self.logger.info("💓 WebSocket heartbeat sent.")
                else:
                    self.logger.warning("⚠ WebSocket disconnected. Attempting to reconnect...")
                    await self.connect()
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                self.logger.info("🛑 WebSocket heartbeat task cancelled.")
                break
            except Exception as e:
                self.logger.warning(f"⚠ WebSocket heartbeat failed: {e}")

    async def close_connection(self):
        """Closes WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
