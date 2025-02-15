import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, strategy, historical_data):
        self.strategy = strategy
        self.data = historical_data

    def run_backtest(self):
        for index, row in self.data.iterrows():
            signal = self.strategy.generate_signal(row)
            if signal:
                print(f"Trade Signal at {row['timestamp']}: {signal}")

    def add_slippage(self, order):
        bid_ask_spread = self.data[order.symbol]['ask'] - self.data[order.symbol]['bid']
        slippage = bid_ask_spread * 0.1 if order.side == 'BUY' else bid_ask_spread * 0.15
        return slippage

    def add_latency(self, order):
        latency = np.random.exponential(scale=0.05)  # 50ms avg latency
        return order.timestamp + pd.Timedelta(latency, 'ms')


# allows users to simulate trades without using real money.