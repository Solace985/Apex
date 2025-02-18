import numpy as np
import pandas as pd

class BacktestingEngine:
    def __init__(self, strategy, historical_data):
        self.strategy = strategy
        self.data = historical_data

    def run_backtest(self):
        for index, row in self.data.iterrows():
            signal = self.strategy.generate_signal(row)
            if signal:
                print(f"Trade Signal at {row['timestamp']}: {signal}")

    def add_slippage(self, order):
        """
        Simulates real-world bid-ask spread slippage.
        """
        bid_ask_spread = self.data[order.symbol]['ask'] - self.data[order.symbol]['bid']
        slippage = bid_ask_spread * (0.1 if order.side == 'BUY' else 0.15)
        order.execution_price += slippage
        return slippage

    def add_latency(self, order):
        """
        Adds execution latency simulation.
        """
        latency = np.random.exponential(scale=0.05)  # 50ms average latency
        order.execution_time += pd.Timedelta(latency, 'ms')
        return latency

# allows users to simulate trades without using real money.