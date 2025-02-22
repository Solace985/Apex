import pandas as pd

class ReportGenerator:
    """Generates reports from backtesting results."""

    @staticmethod
    def generate_report(trades):
        """Creates a summary of trading performance."""
        df = pd.DataFrame(trades)
        total_profit = df["profit"].sum()
        win_rate = (df["profit"] > 0).mean() * 100
        max_drawdown = df["profit"].min()

        return {
            "Total Profit": total_profit,
            "Win Rate (%)": win_rate,
            "Max Drawdown": max_drawdown
        }
