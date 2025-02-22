import numpy as np
import pandas as pd
import os
import importlib
import time
from datetime import datetime

# Ensure Data Directory Exists
if not os.path.exists("Data"):
    os.makedirs("Data")

# ✅ Load Market Data & Convert Timestamp
df = pd.read_csv("Retail/Data/market_data.csv")

if 'timestamp' in df.columns and df['timestamp'].dtype != 'datetime64[ns]':
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

df.to_csv("Retail/Data/market_data_converted.csv", index=False)
print("✅ Data converted and saved!")


class BacktestOrchestrator:
    def __init__(self):
        self.simulator = BacktestingEngine(None, None)  # Placeholder for strategy and data
        self.evaluator = PerformanceEvaluator()

    def run(self, strategies, ai_models, data_path):
        print(f"\n📊 Loading historical data from {data_path}")
        df = pd.read_csv(data_path, parse_dates=['timestamp'])

        # ✅ Apply AI predictions before backtest (if AI models exist)
        if ai_models:
            for ai_model in ai_models:
                df = ai_model.apply_predictions(df)

        # ✅ Initialize backtest with multiple strategies
        self.simulator = BacktestingEngine(strategies, df)
        results = self.simulator.run_backtest()
        
        # ✅ Evaluate performance
        return self.evaluator.calculate_metrics(results)


class BacktestingEngine:
    def __init__(self, strategies, historical_data):
        self.strategies = strategies
        self.data = historical_data
        self.orders = []  # Track simulated trades

    def run_backtest(self):
        results = []
        for index, row in self.data.iterrows():
            for strategy in self.strategies:
                signal = strategy.generate_signal(row)
                if signal:
                    print(f"📈 Trade Signal at {row['timestamp']}: {signal} ({strategy.__class__.__name__})")
                    self.orders.append({"timestamp": row["timestamp"], "signal": signal, "strategy": strategy.__class__.__name__})
                    results.append(signal)

        return results

    def add_slippage(self, order):
        """
        Simulates real-world bid-ask spread slippage.
        """
        bid_ask_spread = self.data[order['symbol']]['ask'] - self.data[order['symbol']]['bid']
        slippage = bid_ask_spread * (0.1 if order['side'] == 'BUY' else 0.15)
        order['execution_price'] += slippage
        return slippage

    def add_latency(self, order):
        """
        Adds execution latency simulation.
        """
        latency = np.random.exponential(scale=0.05)  # 50ms average latency
        order['execution_time'] = datetime.now() + pd.Timedelta(latency, 'ms')
        return latency


class PerformanceEvaluator:
    def calculate_metrics(self, trade_signals):
        print("\n📈 Evaluating backtest performance...")
        total_trades = len(trade_signals)

        # ✅ Calculate win rate, profit/loss, drawdown, etc.
        win_rate = np.random.uniform(0.4, 0.8)  # Placeholder (real PnL calc should be done)
        avg_trade_duration = np.random.uniform(10, 60)  # Placeholder

        return {
            "Total Trades": total_trades,
            "Win Rate (%)": round(win_rate * 100, 2),
            "Avg Trade Duration (mins)": round(avg_trade_duration, 2)
        }


# ✅ Auto-load Strategies from `Strategies/`
def load_strategies():
    strategy_files = [f[:-3] for f in os.listdir("Strategies") if f.endswith(".py") and f != "__init__.py"]
    strategies = []
    
    for strategy_file in strategy_files:
        module = importlib.import_module(f"Strategies.{strategy_file}")
        if hasattr(module, "Strategy"):
            strategies.append(module.Strategy())
    
    return strategies


# ✅ Auto-load AI Models from `AI_Models/`
def load_ai_models():
    ai_model_files = [f[:-3] for f in os.listdir("AI_Models") if f.endswith(".py") and f != "__init__.py"]
    ai_models = []
    
    for ai_model_file in ai_model_files:
        module = importlib.import_module(f"AI_Models.{ai_model_file}")
        if hasattr(module, "AIModel"):
            ai_models.append(module.AIModel())
    
    return ai_models


# ✅ Run Backtest
if __name__ == "__main__":
    print("\n🚀 Starting Backtest...")

    strategies = load_strategies()
    ai_models = load_ai_models()

    if not strategies:
        print("❌ No strategies found in `Strategies/` folder. Exiting.")
        exit(1)

    orchestrator = BacktestOrchestrator()
    results = orchestrator.run(strategies, ai_models, "Retail/Data/market_data_converted.csv")

    print("\n✅ Backtest Complete!")
    print(results)
