import logging
from pythonjsonlogger import jsonlogger

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(levelname)s %(message)s %(module)s %(funcName)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_order(self, order):
        self.logger.info(
            "Order executed",
            extra={
                "order_id": order.id,
                "symbol": order.symbol,
                "slippage": order.slippage,
                "latency": order.latency
            }
        )
