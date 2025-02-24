import logging
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, Any

class TradeMonitor:
    """Tracks active trades, execution performance, and real-time PnL."""

    def __init__(self, trade_history_path="Retail/Data/trade_history.json"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.trade_history_path = trade_history_path
        self.active_trades = {}  # Stores active trades in memory

        # ✅ Load past trade history
        self.load_trade_history()

    def load_trade_history(self):
        """Loads past trade history into memory."""
        if os.path.exists(self.trade_history_path):
            with open(self.trade_history_path, "r") as file:
                self.active_trades = json.load(file)
        else:
            self.logger.warning("No past trade history found. Starting fresh.")

    def save_trade_history(self):
        """Saves trade history to disk."""
        with open(self.trade_history_path, "w") as file:
            json.dump(self.active_trades, file, indent=4)

    def track_trade(self, trade_id: str, trade_data: Dict[str, Any]):
        """Adds a new trade to active monitoring."""
        self.active_trades[trade_id] = {
            "symbol": trade_data["symbol"],
            "entry_price": trade_data["entry_price"],
            "position_size": trade_data["position_size"],
            "timestamp": datetime.utcnow().isoformat(),
            "status": "OPEN"
        }
        self.logger.info(f"Tracking new trade: {trade_data['symbol']} @ {trade_data['entry_price']}")
        self.save_trade_history()

    def update_trade(self, trade_id: str, execution_price: float, status: str):
        """Updates trade status with execution price & profit/loss."""
        if trade_id in self.active_trades:
            trade = self.active_trades[trade_id]
            trade["exit_price"] = execution_price
            trade["status"] = status
            trade["pnl"] = (execution_price - trade["entry_price"]) * trade["position_size"]
            self.logger.info(f"Trade {trade_id} closed with PnL: {trade['pnl']:.2f}")
            self.save_trade_history()
        else:
            self.logger.warning(f"Trade {trade_id} not found!")

    def get_active_trades(self):
        """Returns currently open trades."""
        return {tid: tdata for tid, tdata in self.active_trades.items() if tdata["status"] == "OPEN"}

    def get_trade_by_id(self, trade_id: str):
        """Fetches trade details by ID."""
        return self.active_trades.get(trade_id, None)
