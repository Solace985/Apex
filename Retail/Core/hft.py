import time

class HFTTrader:
    def __init__(self, broker_api):
        self.broker_api = broker_api

    def read_order_book(self):
        """
        Continuously fetches order book updates in real-time.
        """
        while True:
            order_book = self.broker_api.get_order_book()
            print("Order Book:", order_book)
            time.sleep(0.01)  # Read every 10 milliseconds

    def execute_hft_trade(self):
        """
        Executes high-frequency trades in microseconds.
        """
        buy_order = {"side": "buy", "quantity": 1}
        sell_order = {"side": "sell", "quantity": 1}

        start_time = time.time()
        self.broker_api.place_order(buy_order)
        self.broker_api.place_order(sell_order)
        execution_time = time.time() - start_time

        print(f"Trade executed in {execution_time:.6f} seconds")
