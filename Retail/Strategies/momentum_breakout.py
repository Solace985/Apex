import numpy as np
import logging

logger = logging.getLogger(__name__)

class MomentumBreakoutStrategy:
    """Momentum Breakout Strategy: Trades only in high volatility breakouts."""

    def __init__(self, period=20, atr_period=14, trend_filter_period=50, confirmation_candles=2):
        self.period = period  # Lookback period for breakouts
        self.atr_period = atr_period  # ATR calculation period
        self.trend_filter_period = trend_filter_period  # Moving average period for trend confirmation
        self.confirmation_candles = confirmation_candles  # Candles required to confirm breakout

    def compute_signal(self, market_data):
        """
        ✅ Improved Logic:
        - If price breaks above resistance & ATR is high → BUY
        - If price breaks below support & ATR is high → SELL
        - Else → HOLD
        """
        # Validate market data length
        required_length = max(self.period, self.atr_period, self.trend_filter_period) + self.confirmation_candles
        if len(market_data["price"]) < required_length:
            logger.warning("Not enough data to compute strategy. Returning HOLD signal.")
            return "HOLD"

        # Convert market data to numpy arrays
        highs = np.array(market_data["high"][-self.period:])
        lows = np.array(market_data["low"][-self.period:])
        closes = np.array(market_data["price"][-self.period:])
        volumes = np.array(market_data["volume"][-self.period:])
        current_price = market_data["price"][-1]
        current_volume = market_data["volume"][-1]

        # Compute breakout levels
        highest_high = np.max(highs)
        lowest_low = np.min(lows)
        avg_volume = np.mean(volumes)

        # ✅ ATR Calculation (Volatility Filter)
        atr = self.calculate_atr(np.array(market_data["high"]), np.array(market_data["low"]), np.array(market_data["price"]), self.atr_period)

        # ✅ Trend Filter - Moving Average
        moving_avg = np.mean(market_data["price"][-self.trend_filter_period:])

        logger.debug(f"Current Price: {current_price}, Highest High: {highest_high}, Lowest Low: {lowest_low}")
        logger.debug(f"ATR: {atr}, Moving Average: {moving_avg}, Avg Volume: {avg_volume}, Current Volume: {current_volume}")

        # ✅ Breakout Confirmation
        closes = market_data["price"][-(self.confirmation_candles+1):]  # Last few candles to confirm breakout
        breakout_confirmed_up = all(closes[i] > highest_high for i in range(1, len(closes)))
        breakout_confirmed_down = all(closes[i] < lowest_low for i in range(1, len(closes)))

        # ✅ Entry Decision Based on Breakout, Volume, ATR, and Trend
        if breakout_confirmed_up and current_volume > avg_volume and atr > np.mean(atr) and current_price > moving_avg:
            logger.info("Signal: BUY")
            return "BUY"
        elif breakout_confirmed_down and current_volume > avg_volume and atr > np.mean(atr) and current_price < moving_avg:
            logger.info("Signal: SELL")
            return "SELL"

        logger.info("Signal: HOLD")
        return "HOLD"

    def calculate_atr(self, high, low, close, period):
        """
        Computes the Average True Range (ATR) to measure volatility.
        """
        if len(close) < period + 1:
            logger.warning("Not enough data for ATR calculation. Returning 0 ATR.")
            return 0.0

        # Compute true range
        prev_close = close[:-1]
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - prev_close)
        tr3 = np.abs(low[1:] - prev_close)
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))

        # Compute ATR using the last `period` true range values
        if len(true_range) < period:
            atr = np.mean(true_range)
        else:
            atr = np.mean(true_range[-period:])
        return atr
