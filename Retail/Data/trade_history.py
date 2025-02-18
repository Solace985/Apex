import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any

class TradeHistory:
    """Stores and manages past trade data for AI learning & strategy evaluation."""

    def __init__(self, history_file="Retail/Data/trade_history.csv"):
        self.history_file = history_file
        self.columns = ["trade_id", "symbol", "entry_price", "exit_price", "pnl", "timestamp", "strategy"]
        
        # ✅ Ensure file exists
        if not os.path.exists(self.history_file):
            pd.DataFrame(columns=self.columns).to_csv(self.history_file, index=False)

    def log_trade(self, trade_data: Dict[str, Any]):
        """Logs completed trades to CSV for AI training."""
        trade_data["timestamp"] = datetime.utcnow().isoformat()

        df = pd.DataFrame([trade_data], columns=self.columns)
        df.to_csv(self.history_file, mode="a", header=False, index=False)

    def load_trade_history(self):
        """Loads trade history as a DataFrame."""
        if os.path.exists(self.history_file):
            return pd.read_csv(self.history_file)
        return pd.DataFrame(columns=self.columns)

    def get_recent_trades(self, n=50):
        """Returns the last `n` trades."""
        df = self.load_trade_history()
        return df.tail(n)
