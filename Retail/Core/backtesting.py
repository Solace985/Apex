import pandas as pd

class Backtester:
    def __init__(self, strategy, historical_data):
        self.strategy = strategy
        self.data = historical_data

    def run_backtest(self):
        for index, row in self.data.iterrows():
            signal = self.strategy.generate_signal(row)
            if signal:
                print(f"Trade Signal at {row['timestamp']}: {signal}")


# allows users to simulate trades without using real money.