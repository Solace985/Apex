import numpy as np
import pandas as pd
from backtesting.order_execution import OrderExecution

class SimulationEngine:
    """Simulates trading execution in a backtesting environment."""

    def __init__(self, strategy, market_data, initial_balance=10000):
        self.strategy = strategy
        self.market_data = market_data
        self.balance = initial_balance
        self.positions = []  # Stores active trades
        self.trade_history = []  # Stores closed trades
        self.order_executor = OrderExecution()

    def run_simulation(self):
        """Runs backtest and executes simulated trades."""
        for index, row in self.market_data.iterrows():
            signal = self.strategy.generate_signal(row)
            
            if signal == "BUY":
                order = self.order_executor.execute_order("BUY", row["close"], self.balance)
                self.positions.append(order)
            elif signal == "SELL" and self.positions:
                buy_price = self.positions[-1]["price"]
                profit = row["close"] - buy_price
                self.balance += profit
                order = self.order_executor.execute_order("SELL", row["close"], self.balance, profit)
                self.trade_history.append(order)
                self.positions.pop()  # Close position

        return self.trade_history
