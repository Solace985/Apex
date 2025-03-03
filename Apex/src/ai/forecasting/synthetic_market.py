import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import asyncio
import time
from functools import lru_cache

# Apex internal imports
from apex.src.ai.forecasting.lstm_model import LSTMForecaster
from apex.src.Core.data.market_data import MarketDataAPI
from apex.src.ai.analysis.market_regime_classifier import MarketRegimeClassifier
from apex.src.Core.data.order_book_analyzer import OrderBookAnalyzer
from apex.src.ai.ensembles.meta_trader import MetaTrader
from apex.utils.analytics.monte_carlo_simulator import AdvancedMonteCarlo
from apex.src.Core.trading.hft.liquidity_manager import LiquidityAnalyzer
from apex.src.ai.forecasting.sentiment_analysis import NewsImpactSimulator
from apex.utils.logging.structured_logger import StructuredLogger
from apex.src.ai.ensembles.transformers_lstm import TransformerLSTM
from apex.src.Core.data.asset_validator import validate_market_data

class SyntheticMarketGenerator:
    """Institutional-grade synthetic market data generator for training, testing and simulating trading systems"""
    
    def __init__(self, cache_size: int = 128, num_workers: int = 4):
        """Initialize the synthetic market generator with performance optimizations
        
        Args:
            cache_size: Size of LRU cache for expensive operations
            num_workers: Number of parallel workers for generation tasks
        """
        self.logger = StructuredLogger("SyntheticMarket")
        self.market_data = MarketDataAPI()
        self.lstm = LSTMForecaster()
        self.transformer_lstm = TransformerLSTM()  # Enhanced AI model
        self.regime_classifier = MarketRegimeClassifier()
        self.monte_carlo = AdvancedMonteCarlo()
        self.order_book_sim = OrderBookAnalyzer()
        self.liquidity_model = LiquidityAnalyzer()
        self.news_simulator = NewsImpactSimulator()
        self.meta_trader = MetaTrader()
        
        # Performance optimization settings
        self.cache_size = cache_size
        self.num_workers = num_workers
        self._cached_regimes = {}  # In-memory cache for fast regime lookups
        
        # Vectorized computation matrices (pre-allocated for performance)
        self._volatility_scaling = np.linspace(0.5, 3.0, 10)
        self._correlation_matrix = np.eye(10)  # Default identity matrix
        
        # Execution performance tracking
        self.last_generation_time = 0
        
        self.logger.info("Initialized SyntheticMarketGenerator with optimized settings")

    async def generate_dataset(
        self,
        base_data: pd.DataFrame,
        scenarios: int = 1000,
        regime: Optional[str] = None,
        with_microstructure: bool = True,
        with_event_shocks: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Generate synthetic market datasets with regime awareness
        
        Args:
            base_data: Historical market data to base synthetic data on
            scenarios: Number of Monte Carlo scenarios to generate
            regime: Optional market regime to enforce (bull, bear, choppy, volatile)
            with_microstructure: Whether to add order book dynamics
            with_event_shocks: Whether to inject market shocks
            
        Returns:
            Dictionary of generated synthetic datasets
        """
        start_time = time.time()
        validated_data = await self._validate_input_data(base_data)
        batch_size = max(1, scenarios // self.num_workers)
        
        # Use asyncio.gather for better parallelism than ThreadPoolExecutor
        tasks = []
        for i in range(0, scenarios, batch_size):
            batch_count = min(batch_size, scenarios - i)
            tasks.append(self._generate_scenario_batch(
                validated_data, batch_count, regime, 
                with_microstructure, with_event_shocks
            ))
            
        batch_results = await asyncio.gather(*tasks)
        
        # Merge batch results
        results = {}
        for i, batch in enumerate(batch_results):
            for key, df in batch.items():
                results[f"scenario_{i}_{key}"] = df
        
        self.last_generation_time = time.time() - start_time
        self.logger.info(f"Generated {scenarios} scenarios in {self.last_generation_time:.2f}s")
        
        return results

    async def _generate_scenario_batch(
        self,
        base_data: pd.DataFrame,
        count: int,
        regime: Optional[str],
        with_microstructure: bool,
        with_event_shocks: bool
    ) -> Dict[str, pd.DataFrame]:
        """Generate a batch of synthetic market scenarios in parallel
        
        Args:
            base_data: Historical data to base scenarios on
            count: Number of scenarios to generate in this batch
            regime: Optional market regime to enforce
            with_microstructure: Whether to add order book dynamics
            with_event_shocks: Whether to inject market shocks
            
        Returns:
            Dictionary of generated scenarios
        """
        # Pre-calculate volatility and regime parameters for consistent batch generation
        batch_volatility = self._calculate_batch_volatility(base_data, count)
        
        # Detect regime if not specified
        if not regime:
            regime = await self._detect_regime_cached(base_data)
        
        # Get regime parameters (vectorized for performance)
        regime_params = self.regime_classifier.get_regime_parameters(regime)
        
        # Generate base scenarios using vectorized operations
        trend_predictions = await self._predict_trends_batch(base_data, count)
        
        # Process each scenario with optimized parallelism
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(count):
                futures.append(executor.submit(
                    self._process_single_scenario,
                    base_data=base_data,
                    trend=trend_predictions[i], 
                    volatility=batch_volatility[i],
                    regime=regime,
                    regime_params=regime_params,
                    with_microstructure=with_microstructure,
                    with_event_shocks=with_event_shocks
                ))
            
            # Collect results as they complete
            results = {}
            for i, future in enumerate(futures):
                results[f"scenario_{i}"] = future.result()
                
        return results
    
    def _process_single_scenario(
        self,
        base_data: pd.DataFrame,
        trend: pd.DataFrame,
        volatility: float,
        regime: str,
        regime_params: Dict[str, float],
        with_microstructure: bool,
        with_event_shocks: bool
    ) -> pd.DataFrame:
        """Process a single scenario with all required components
        
        This is a non-async method optimized for parallel execution in thread pools
        
        Args:
            base_data: Base historical data
            trend: Predicted trend for this scenario
            volatility: Volatility scaling for this scenario
            regime: Market regime to simulate
            regime_params: Pre-calculated regime parameters
            with_microstructure: Whether to add order book dynamics
            with_event_shocks: Whether to inject market shocks
            
        Returns:
            Fully processed synthetic market scenario
        """
        # Generate correlated noise (vectorized)
        noise = self._generate_correlated_noise_vectorized(base_data, volatility)
        
        # Combine components with regime parameters
        synthetic = self._combine_components_vectorized(
            base_data, trend, noise, regime_params
        )
        
        # Add market microstructure if requested (expensive operation)
        if with_microstructure:
            synthetic = self._add_microstructure_vectorized(synthetic)
        
        # Add event shocks if requested
        if with_event_shocks:
            synthetic = self._apply_shocks_vectorized(synthetic, regime)
            
        return synthetic
    
    async def _predict_trends_batch(
        self, 
        base_data: pd.DataFrame, 
        count: int
    ) -> List[pd.DataFrame]:
        """Predict multiple trend scenarios in batch for better performance
        
        Args:
            base_data: Base historical data
            count: Number of trend scenarios to generate
            
        Returns:
            List of trend prediction dataframes
        """
        # Use hybrid of LSTM and Transformer for better predictions
        lstm_predictions = []
        transformer_predictions = []
        
        # Create batches for efficient processing
        for _ in range(count):
            lstm_predictions.append(await self.lstm.predict_future_trend(base_data))
            transformer_predictions.append(await self.transformer_lstm.predict_trend(base_data))
        
        # Ensemble the predictions with varying weights based on context
        combined_predictions = []
        for i in range(count):
            # Adaptive weighting based on scenario
            lstm_weight = np.random.uniform(0.3, 0.7)
            transformer_weight = 1.0 - lstm_weight
            
            # Vectorized combination
            combined = pd.DataFrame(index=lstm_predictions[i].index)
            combined['close'] = (
                lstm_weight * lstm_predictions[i]['close'].values +
                transformer_weight * transformer_predictions[i]['close'].values
            )
            combined_predictions.append(combined)
            
        return combined_predictions

    def _calculate_batch_volatility(self, base_data: pd.DataFrame, count: int) -> np.ndarray:
        """Calculate batch volatility for multiple scenarios vectorized
        
        Args:
            base_data: Base historical data
            count: Number of scenarios
            
        Returns:
            Array of volatility values for each scenario
        """
        # Calculate base volatility using vectorized operations
        base_returns = base_data['close'].pct_change().dropna().values
        base_vol = np.std(base_returns) * np.sqrt(252)  # Annualized
        
        # Generate regime-appropriate volatility scenarios
        return base_vol * np.random.choice(
            self._volatility_scaling,
            size=count,
            p=np.array([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.03, 0.02])
        )

    def _generate_correlated_noise_vectorized(
        self, 
        base_data: pd.DataFrame, 
        volatility: float
    ) -> pd.DataFrame:
        """Generate correlated noise with volatility clustering (vectorized)
        
        Args:
            base_data: Base historical data
            volatility: Target volatility scaling
            
        Returns:
            Noise dataframe with realistic properties
        """
        n_points = len(base_data)
        
        # Use numpy for vectorized GARCH-like noise with volatility clustering
        # This is a simplified but high-performance version
        random_noise = np.random.normal(0, 1, n_points)
        
        # Apply volatility clustering (higher volatility tends to cluster)
        volatility_scaling = np.ones(n_points)
        for i in range(1, n_points):
            volatility_scaling[i] = 0.9 * volatility_scaling[i-1] + 0.1 * (
                1 + 0.2 * random_noise[i-1]**2
            )
        
        # Scale final noise by target volatility and clustering
        final_noise = volatility * random_noise * volatility_scaling
        
        return pd.DataFrame({
            'timestamp': base_data.index,
            'close': final_noise,
        }).set_index('timestamp')

    def _combine_components_vectorized(
        self,
        base: pd.DataFrame,
        trend: pd.DataFrame,
        noise: pd.DataFrame,
        regime_params: Dict[str, float]
    ) -> pd.DataFrame:
        """Combine synthetic data components based on regime (vectorized)
        
        Args:
            base: Base historical data
            trend: Trend prediction component
            noise: Noise component
            regime_params: Parameters for the market regime
            
        Returns:
            Combined synthetic dataframe
        """
        # Vectorized combination using numpy operations
        synthetic = base.copy()
        
        # Extract parameters (ensures single lookup)
        trend_weight = regime_params['trend_weight']
        noise_weight = regime_params['noise_weight']
        mean_reversion = regime_params.get('mean_reversion', 0.05)
        
        # Vectorized price computation
        synthetic['close'] = (
            trend_weight * trend['close'].values +
            noise_weight * noise['close'].values +
            mean_reversion * (base['close'].mean() - base['close'])
        )
        
        # Calculate derived metrics (vectorized)
        synthetic['returns'] = synthetic['close'].pct_change().fillna(0)
        synthetic['volatility'] = synthetic['returns'].rolling(
            window=24, min_periods=1
        ).std().fillna(synthetic['returns'].std())
        
        return synthetic

    def _add_microstructure_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features (vectorized)
        
        Args:
            data: Base synthetic price data
            
        Returns:
            Enhanced data with microstructure features
        """
        # Fast vectorized computation of bid-ask spread based on volatility
        volatility = data['volatility'].values
        volume = data['volume'].values if 'volume' in data else np.ones(len(data))
        
        # Vectorized spread calculation (related to volatility and inverse to volume)
        relative_spread = 0.0001 + 0.05 * volatility / np.sqrt(volume + 1)
        
        # Apply microstructure features (all vectorized)
        data['bid_price'] = data['close'] * (1 - relative_spread/2)
        data['ask_price'] = data['close'] * (1 + relative_spread/2)
        data['spread'] = data['ask_price'] - data['bid_price']
        
        # Volume profile based on price movement (vectorized)
        data['volume'] = volume if 'volume' in data else (
            1000 * (1 + 5 * np.abs(data['returns'].values))
        )
        
        # Bid/ask sizes with some imbalance based on momentum
        momentum = data['returns'].rolling(5).mean().fillna(0).values
        data['bid_size'] = data['volume'] * (0.5 - 0.3 * momentum)
        data['ask_size'] = data['volume'] * (0.5 + 0.3 * momentum)
        
        # Price impact estimation (institutional feature)
        data['price_impact'] = 0.2 * data['spread'] / np.sqrt(data['volume'])
        
        return data

    def _apply_shocks_vectorized(self, data: pd.DataFrame, regime: str) -> pd.DataFrame:
        """Apply market shocks vectorized based on regime
        
        Args:
            data: Synthetic market data
            regime: Current market regime
            
        Returns:
            Data with realistic market shocks
        """
        # Fast shock application using vectorized operations
        n_points = len(data)
        
        # Determine shock probability based on regime
        if regime == 'volatile':
            shock_prob = 0.05  # More frequent in volatile regimes
        elif regime == 'bear':
            shock_prob = 0.03  # More frequent in bear markets
        else:
            shock_prob = 0.01  # Less frequent in bull/normal markets
            
        # Generate random shock points (efficient)
        shock_points = np.random.random(n_points) < shock_prob
        shock_indices = np.where(shock_points)[0]
        
        if len(shock_indices) == 0:
            return data  # No shocks to apply
            
        # Apply shocks vectorized where needed
        for idx in shock_indices:
            # Determine shock duration (shorter is more realistic)
            duration = min(30, n_points - idx)
            if duration <= 0:
                continue
                
            # Create shock profile (exponential decay)
            shock_impact = np.random.uniform(-0.05, 0.05)  # Shock magnitude
            shock_profile = shock_impact * np.exp(-np.arange(duration)/10)
            
            # Apply to price (vectorized)
            end_idx = idx + duration
            data.iloc[idx:end_idx, data.columns.get_loc('close')] *= (
                1 + shock_profile[:end_idx-idx]
            )
            
            # Apply to volume (vectorized)
            if 'volume' in data:
                data.iloc[idx:end_idx, data.columns.get_loc('volume')] *= (
                    1 + np.abs(shock_profile[:end_idx-idx]) * 3
                )
                
            # Apply to volatility (vectorized)
            if 'volatility' in data:
                data.iloc[idx:end_idx, data.columns.get_loc('volatility')] *= (
                    1 + np.abs(shock_profile[:end_idx-idx]) * 2
                )
        
        return data

    @lru_cache(maxsize=32)
    async def _detect_regime_cached(self, base_data_key: str) -> str:
        """Cached regime detection for performance
        
        Args:
            base_data_key: String key representing the dataset
            
        Returns:
            Detected market regime
        """
        # Convert DataFrame to hashable key for cache lookup
        if isinstance(base_data_key, pd.DataFrame):
            # Use start/end dates and basic stats as cache key
            stats = base_data_key['close'].describe().to_dict()
            key = f"{base_data_key.index[0]}_{base_data_key.index[-1]}_{stats['mean']:.2f}"
            
            # Check if in memory cache
            if key in self._cached_regimes:
                return self._cached_regimes[key]
            
            # Detect and cache
            regime = await self.regime_classifier.detect_current_regime(base_data_key)
            self._cached_regimes[key] = regime
            return regime
        else:
            # If already a string key, use as is
            return await self.regime_classifier.detect_current_regime(base_data_key)

    async def _validate_input_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure input data meets quality standards with optimized validation
        
        Args:
            data: Input market data
            
        Returns:
            Validated market data
        """
        # Ensure index is datetime for time series operations
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            else:
                data.index = pd.to_datetime(data.index)
        
        # Fast validation of required columns
        required_columns = ['close']
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Use the asset validator for more complex validation
        return await validate_market_data(data)

    async def generate_realtime_synthetic(
        self,
        realtime_feed: pd.DataFrame,
        lookback: int = 24
    ) -> pd.DataFrame:
        """Generate real-time synthetic market updates optimized for low latency
        
        Args:
            realtime_feed: Recent market data feed
            lookback: Number of periods to include
            
        Returns:
            Updated synthetic market data
        """
        # Fast regime detection (cached for performance)
        regime = await self._detect_regime_cached(realtime_feed)
        
        # Predict next tick efficiently
        prediction = await self.meta_trader.predict_next_period(realtime_feed)
        
        # Generate single synthetic tick (optimized for speed)
        synthetic_update = self._generate_synthetic_tick_fast(prediction, regime)
        
        # Combine with historical data efficiently (avoid expensive concat)
        result = pd.concat([realtime_feed, synthetic_update]).iloc[-lookback:]
        
        # Return shallow copy to avoid modifying original data
        return result.copy()

    def _generate_synthetic_tick_fast(self, prediction: Dict, regime: str) -> pd.DataFrame:
        """Generate single synthetic tick optimized for low latency
        
        Args:
            prediction: Price and volume prediction
            regime: Current market regime
            
        Returns:
            Single-row dataframe with synthetic tick
        """
        # Faster than simulating complete order book for a single tick
        price = prediction['price']
        
        # Spread based on regime (wider in volatile/bear markets)
        if regime == 'volatile':
            spread_factor = 2.0
        elif regime == 'bear':
            spread_factor = 1.5
        else:
            spread_factor = 1.0
            
        # Fast tick generation
        base_spread = 0.0001 * price * spread_factor
        
        # Create single-row dataframe (faster than DataFrame constructor for simple case)
        tick = pd.DataFrame(index=[datetime.now()])
        tick['open'] = price
        tick['high'] = price * 1.0005
        tick['low'] = price * 0.9995
        tick['close'] = price
        tick['volume'] = prediction['volume']
        tick['bid_price'] = price - base_spread/2
        tick['ask_price'] = price + base_spread/2
        tick['bid_size'] = prediction['volume'] * 0.5 * (1 - 0.1 * (1 if regime == 'bear' else -1))
        tick['ask_size'] = prediction['volume'] * 0.5 * (1 + 0.1 * (1 if regime == 'bear' else -1))
        tick['regime'] = regime
        
        return tick

    async def generate_stress_scenario(
        self,
        base_data: pd.DataFrame,
        scenario_type: str
    ) -> pd.DataFrame:
        """Generate specific stress test scenarios optimized for institutional use
        
        Args:
            base_data: Base market data
            scenario_type: Type of stress scenario
            
        Returns:
            Synthetic data with stress scenario applied
        """
        # Predefine stress scenario parameters for vectorized application
        scenario_params = {
            'flash_crash': {
                'price_impact': np.random.uniform(-0.30, -0.15),
                'vol_multiplier': 3.0,
                'duration_pct': 0.2  # Percent of data affected
            },
            'liquidity_crisis': {
                'spread_multiplier': 5.0,
                'volume_multiplier': 0.2,
                'impact_multiplier': 10.0
            },
            'volatility_spike': {
                'vol_multiplier': 3.5,
                'vol_window': 6
            }
        }
        
        if scenario_type not in scenario_params:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
            
        # Get scenario parameters
        params = scenario_params[scenario_type]
        
        # Apply scenario vectorized
        if scenario_type == 'flash_crash':
            return self._apply_flash_crash_vectorized(base_data, params)
        elif scenario_type == 'liquidity_crisis':
            return self._apply_liquidity_crisis_vectorized(base_data, params)
        else:  # volatility_spike
            return self._apply_volatility_spike_vectorized(base_data, params)

    def _apply_flash_crash_vectorized(
        self, 
        data: pd.DataFrame, 
        params: Dict[str, float]
    ) -> pd.DataFrame:
        """Apply flash crash scenario using vectorized operations
        
        Args:
            data: Market data
            params: Scenario parameters
            
        Returns:
            Data with flash crash applied
        """
        result = data.copy()
        n_points = len(result)
        
        # Determine crash start point
        crash_idx = int(n_points * 0.7)  # Crash after 70% of the data
        
        # Create crash profile (fast crash, slow recovery)
        crash_length = int(n_points * params['duration_pct'])
        crash_profile = np.zeros(n_points)
        
        # Apply exponential crash profile
        crash_range = np.arange(crash_length)
        crash_profile[crash_idx:crash_idx+crash_length] = params['price_impact'] * (
            np.exp(-crash_range/10) / np.exp(0)  # Normalize to start at full impact
        )
        
        # Apply to price (vectorized)
        result['close'] = result['close'] * (1 + crash_profile)
        
        # Apply to volatility if present
        if 'volatility' in result:
            vol_profile = np.ones(n_points)
            vol_profile[crash_idx:] = params['vol_multiplier']
            result['volatility'] *= vol_profile
            
        return result

    def _apply_liquidity_crisis_vectorized(
        self, 
        data: pd.DataFrame, 
        params: Dict[str, float]
    ) -> pd.DataFrame:
        """Apply liquidity crisis scenario using vectorized operations
        
        Args:
            data: Market data
            params: Scenario parameters
            
        Returns:
            Data with liquidity crisis applied
        """
        result = data.copy()
        
        # Apply liquidity impact vectors
        if 'spread' in result:
            result['spread'] *= params['spread_multiplier']
            
        if 'volume' in result:
            result['volume'] *= params['volume_multiplier']
            
        if 'price_impact' in result:
            result['price_impact'] *= params['impact_multiplier']
        else:
            # Create price impact if not present
            result['price_impact'] = 0.01 * params['impact_multiplier'] * (
                result['close'] / result['close'].mean()
            )
            
        # Add bid-ask if not present (important for liquidity crisis)
        if 'bid_price' not in result:
            spread = result['spread'] if 'spread' in result else (
                result['close'] * 0.002 * params['spread_multiplier']
            )
            result['bid_price'] = result['close'] - spread/2
            result['ask_price'] = result['close'] + spread/2
            
        return result

    def _apply_volatility_spike_vectorized(
        self, 
        data: pd.DataFrame, 
        params: Dict[str, float]
    ) -> pd.DataFrame:
        """Apply volatility spike scenario using vectorized operations
        
        Args:
            data: Market data
            params: Scenario parameters
            
        Returns:
            Data with volatility spike applied
        """
        result = data.copy()
        
        # Calculate or update returns
        result['returns'] = result['close'].pct_change().fillna(0)
        
        # Apply volatility spike with specified window
        window = params['vol_window']
        result['volatility'] = result['returns'].rolling(window).std().fillna(
            result['returns'].std()
        ) * params['vol_multiplier']
        
        # Apply impact to prices (volatility drives price swings)
        # Create random directional swings based on heightened volatility
        np.random.seed(42)  # For reproducibility
        random_swings = np.random.normal(0, 1, len(result))
        result['close'] *= (1 + random_swings * result['volatility'])
        
        return result