class PerformanceEvaluator:
    """Calculates backtesting performance metrics."""

    @staticmethod
    def calculate_win_rate(trades):
        """Calculates win rate of trades."""
        wins = sum(1 for trade in trades if trade.get("profit", 0) > 0)
        return wins / len(trades) if trades else 0

    @staticmethod
    def calculate_max_drawdown(trades):
        """Calculates maximum drawdown (biggest loss)."""
        profits = [trade.get("profit", 0) for trade in trades]
        max_loss = min(profits) if profits else 0
        return abs(max_loss)

    @staticmethod
    def calculate_total_profit(trades):
        """Calculates total profit from all trades."""
        return sum(trade.get("profit", 0) for trade in trades)
