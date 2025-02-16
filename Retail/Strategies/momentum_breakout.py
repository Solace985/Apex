import numpy as np

class MomentumBreakoutStrategy:
    """Momentum Breakout Strategy: Trades only in high volatility breakouts."""

    def compute_signal(self, market_data, period=20):
        """
        ✅ Improved Logic:
        - If price breaks above resistance & ATR is high → BUY
        - If price breaks below support & ATR is high → SELL
        - Else → HOLD
        """
        highs = market_data["high"][-period:]
        lows = market_data["low"][-period:]
        closes = market_data["price"][-period:]
        volumes = market_data["volume"][-period:]
        current_price = market_data["price"][-1]
        current_volume = market_data["volume"][-1]

        highest_high = np.max(highs)
        lowest_low = np.min(lows)
        avg_volume = np.mean(volumes)

        # ✅ ATR Calculation
        tr = np.max([highs - lows, np.abs(highs - closes[:-1]), np.abs(lows - closes[:-1])], axis=0)
        atr = np.mean(tr)

        if current_price > highest_high and current_volume > avg_volume and atr > np.mean(atr):
            return "BUY"
        elif current_price < lowest_low and current_volume > avg_volume and atr > np.mean(atr):
            return "SELL"
        # ✅ Compute Moving Average (Trend Filter)
        moving_avg = np.mean(market_data["price"][-50:])  # 50-period Simple Moving Average

        # Print trend confirmation
        print(f"Trend Confirmation: Current Price={current_price}, Moving Avg={moving_avg}")

        # ✅ Adjust Buy/Sell Conditions Based on Moving Average
        if current_price > highest_high and current_volume > avg_volume and atr > np.mean(atr) and current_price > moving_avg:
            return "BUY"
        elif current_price < lowest_low and current_volume > avg_volume and atr > np.mean(atr) and current_price < moving_avg:
            return "SELL"
        closes = market_data["price"][-(period+2):]  # ✅ Need 2 extra closing prices
        # ✅ Check if last 2 closes confirm breakout
        if closes[-2] > highest_high and closes[-1] > highest_high:
            return "BUY"
        elif closes[-2] < lowest_low and closes[-1] < lowest_low:
            return "SELL"
        
        return "HOLD"
