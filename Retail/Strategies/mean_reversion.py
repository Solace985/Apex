import numpy as np
import logging

logger = logging.getLogger(__name__)

class MeanReversionStrategy:
    """Mean Reversion Strategy with Adaptive Thresholds, Trend Filtering, and Optimized Computation."""

    def __init__(self, ema_period=20, atr_period=14, std_dev_multiplier=1.5, trend_filter_period=50):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.std_dev_multiplier = std_dev_multiplier
        self.trend_filter_period = trend_filter_period  # Checks market trend to avoid trading against it

    def compute_signal(self, market_data):
        """
        Improved Logic:
        - Computes a short-term EMA for responsiveness.
        - Uses ATR for dynamic threshold setting.
        - Filters noise using the standard deviation of recent prices.
        - Applies a long-term EMA to filter out trades in strong trends.
        Returns one of "BUY", "SELL", or "HOLD".
        """
        try:
            price = np.array(market_data["price"])
            high = np.array(market_data["high"])
            low = np.array(market_data["low"])
            close = np.array(market_data["close"])
        except KeyError as e:
            logger.error(f"Missing key in market data: {e}")
            return "HOLD"

        # Ensure sufficient data is available for all indicators.
        required_length = max(self.ema_period, self.trend_filter_period, self.atr_period) + 1
        if len(price) < required_length:
            logger.warning("Not enough data to compute indicators. Returning HOLD signal.")
            return "HOLD"

        # Calculate indicators
        ema = self.calculate_ema(price, self.ema_period)
        long_term_ema = self.calculate_ema(price, self.trend_filter_period)
        atr = self.calculate_atr(high, low, close, self.atr_period)
        std_dev = np.std(price[-self.ema_period:])

        # Dynamic thresholds: adjust based on volatility and noise
        lower_threshold = ema - (self.std_dev_multiplier * atr)
        upper_threshold = ema + (self.std_dev_multiplier * atr)

        logger.debug(f"Short-term EMA: {ema}")
        logger.debug(f"Long-term EMA: {long_term_ema}")
        logger.debug(f"ATR: {atr}")
        logger.debug(f"Std Dev (last {self.ema_period}): {std_dev}")
        logger.debug(f"Lower Threshold: {lower_threshold}, Upper Threshold: {upper_threshold}")

        # Trend filtering: if the market is in a strong downtrend (short-term EMA significantly below long-term EMA)
        # avoid generating a BUY signal.
        if ema < long_term_ema and price[-1] < lower_threshold:
            logger.debug("Trend filter active: Market in downtrend. Holding.")
            return "HOLD"

        # Signal decision based on current price relative to thresholds
        if price[-1] < lower_threshold:
            logger.info("Signal: BUY")
            return "BUY"
        elif price[-1] > upper_threshold:
            logger.info("Signal: SELL")
            return "SELL"
        else:
            logger.info("Signal: HOLD")
            return "HOLD"

    def calculate_ema(self, price_series, period):
        """
        Optimized EMA Calculation using a vectorized weighted approach.
        If insufficient data is available, returns the last price.
        """
        if len(price_series) < period:
            logger.warning("Not enough data for EMA calculation. Returning last available price.")
            return price_series[-1]

        alpha = 2 / (period + 1)
        # Create a kernel of weights: more recent prices have higher weight.
        weights = np.array([alpha * (1 - alpha) ** i for i in range(period)])
        weights = weights[::-1]  # Reverse so that the most recent price aligns with the end of the kernel.
        # Convolve the price series with the weights.
        ema_values = np.convolve(price_series, weights, mode="valid")
        # Normalize the result by the sum of weights (for safety, though it should sum to ~1).
        ema_values = ema_values / weights.sum()
        return ema_values[-1]

    def calculate_atr(self, high, low, close, period):
        """
        Computes the Average True Range (ATR) for market volatility.
        Returns 0.0 if insufficient data is available.
        """
        if len(close) < period + 1:
            logger.warning("Not enough data for ATR calculation. Returning 0 ATR.")
            return 0.0

        # Calculate true ranges for each period:
        prev_close = close[:-1]
        current_high = high[1:]
        current_low = low[1:]

        tr1 = current_high - current_low
        tr2 = np.abs(current_high - prev_close)
        tr3 = np.abs(current_low - prev_close)
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))

        # Use the last 'period' true range values
        if len(true_range) < period:
            atr = np.mean(true_range)
        else:
            atr = np.mean(true_range[-period:])
        return atr
