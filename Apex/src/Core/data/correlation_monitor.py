import numpy as np
import pandas as pd
import numba as nb
from typing import Dict, List, Tuple, Optional, Union
import asyncio
import websockets
import json
from datetime import datetime, timedelta
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import coint
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
try:
    import dask.dataframe as dd
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

# Apex internal imports
from apex.src.Core.data.market_data import MarketDataAPI
from apex.src.Core.trading.risk.risk_management import RiskManager
from apex.src.Core.trading.execution.order_execution import OrderExecutor
from apex.src.Core.trading.strategies.strategy_orchestrator import StrategyOrchestrator
from apex.src.ai.ensembles.meta_trader import MetaTrader
from apex.src.ai.analysis.market_regime_classifier import MarketRegimeClassifier
from apex.src.Core.data.order_book_analyzer import OrderBookAnalyzer
from apex.src.Core.trading.execution.arbitrage import ArbitrageDetector
from apex.utils.analytics.monte_carlo_simulator import StressScenarioGenerator
from apex.utils.logging.structured_logger import StructuredLogger
from apex.utils.helpers.validation import validate_dataframe

# Optimization constants
MAX_THREADS = 12
REALTIME_WINDOW = 100
LOOKBACK_WINDOWS = {"short": 20, "medium": 50, "long": 100, "extreme": 250}
CORRELATION_UPDATE_INTERVAL = 0.5  # in seconds
CACHE_SIZE = 1024

class CorrelationMonitor:
    """
    Institutional-grade correlation analysis system with multi-timeframe capabilities,
    GPU acceleration, and real-time arbitrage signal generation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = StructuredLogger("CorrelationMonitor")
        self._setup_config(config or {})
        self._initialize_data_structures()
        self._initialize_integrations()
        self._setup_correlation_engine()
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
        self.processing_lock = asyncio.Lock()
        self.update_queue = asyncio.Queue(maxsize=1000)
        self.is_running = False

    def _setup_config(self, config: Dict) -> None:
        """Set up configuration parameters"""
        self.windows = config.get("windows", LOOKBACK_WINDOWS)
        self.update_interval = config.get("update_interval", CORRELATION_UPDATE_INTERVAL)
        self.use_gpu = config.get("use_gpu", HAS_GPU)
        self.use_dask = config.get("use_dask", HAS_DASK)
        self.max_assets = config.get("max_assets", 1000)
        self.coint_threshold = config.get("coint_threshold", 0.05)
        self.arbitrage_threshold = config.get("arbitrage_threshold", 0.02)

    def _initialize_data_structures(self) -> None:
        """Initialize data structures for correlation analysis"""
        self.asset_prices = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.correlation_matrices = {window: pd.DataFrame() for window in self.windows.values()}
        self.tail_correlations = pd.DataFrame()
        self.volatile_correlations = pd.DataFrame()
        self.market_regime_correlations = {}
        self.last_update_time = datetime.now()
        self.last_full_recalc_time = datetime.now()
        self.tracked_pairs = set()
        self.active_arbitrage_signals = {}

    def _initialize_integrations(self) -> None:
        """Initialize connections to other Apex components"""
        self.market_data = MarketDataAPI()
        self.risk_manager = RiskManager()
        self.order_executor = OrderExecutor()
        self.strategy_orchestrator = StrategyOrchestrator()
        self.meta_trader = MetaTrader()
        self.regime_classifier = MarketRegimeClassifier()
        self.arbitrage_detector = ArbitrageDetector()
        self.stress_generator = StressScenarioGenerator()
        self.order_book_analyzer = OrderBookAnalyzer()

    def _setup_correlation_engine(self) -> None:
        """Set up optimized correlation calculation engine"""
        if self.use_gpu and HAS_GPU:
            self.logger.info("Using GPU acceleration for correlation calculations")
            self._correlation_engine = self._gpu_correlation
        else:
            self.logger.info("Using CPU-based Numba optimization for correlation calculations")
            self._correlation_engine = self._numba_correlation

    async def initialize(self) -> None:
        """Initialize with historical data and prepare for real-time monitoring"""
        self.logger.info("Initializing correlation monitor")
        try:
            await self._load_historical_data()
            await self._calculate_initial_correlations()
            await self._warm_up_models()
            self.is_running = True
            self.logger.info("Correlation monitor initialized successfully")
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            raise

    async def _load_historical_data(self) -> None:
        """Load historical price data for initial correlation baseline"""
        self.logger.info("Loading historical price data")
        max_window = max(self.windows.values())
        lookback_days = max(30, int(max_window / 20))  # Ensure enough data for longest window
        
        hist_data = await self.market_data.get_historical_prices(
            symbols=await self.market_data.get_tracked_assets(),
            lookback_days=lookback_days
        )
        
        # Use dask for large datasets if available
        if self.use_dask and HAS_DASK and len(hist_data) > 10000:
            dask_df = dd.from_pandas(hist_data, npartitions=MAX_THREADS)
            pivot_df = dask_df.map_partitions(
                lambda df: df.pivot_table(index='timestamp', columns='symbol', values='close')
            ).compute()
            self.asset_prices = pivot_df
        else:
            self.asset_prices = hist_data.pivot_table(
                index='timestamp', columns='symbol', values='close'
            )
        
        # Calculate returns once to avoid repeated calculations
        self.returns = self.asset_prices.pct_change().dropna()
        self.logger.info(f"Loaded historical data for {self.asset_prices.shape[1]} assets")

    async def _calculate_initial_correlations(self) -> None:
        """Calculate initial correlation matrices for all timeframes"""
        self.logger.info("Calculating initial correlation matrices")
        for name, window in self.windows.items():
            window_data = self.returns.tail(window)
            if len(window_data) >= window:
                self.correlation_matrices[window] = self._calculate_correlation_matrix(window_data.values)
        
        # Calculate tail correlations (extreme market movements)
        self._calculate_tail_correlations()
        self.logger.info("Initial correlation matrices calculated")

    async def _warm_up_models(self) -> None:
        """Warm up AI models and classifiers"""
        await self.meta_trader.warmup_correlation_models()
        self.current_regime = await self.regime_classifier.classify_current_regime(self.returns)
        self.logger.info(f"Models warmed up, current market regime: {self.current_regime}")

    @staticmethod
    @nb.jit(nopython=True, parallel=True, cache=True)
    def _numba_correlation(data: np.ndarray) -> np.ndarray:
        """Numba-optimized correlation matrix calculation"""
        n = data.shape[1]
        corr_matrix = np.eye(n)  # Initialize with diagonal of ones
        
        for i in nb.prange(n):
            x = data[:, i]
            x_norm = x - np.mean(x)
            x_std = np.std(x)
            if x_std == 0:  # Handle constant arrays
                continue
                
            for j in range(i+1, n):
                y = data[:, j]
                y_norm = y - np.mean(y)
                y_std = np.std(y)
                
                if y_std == 0:  # Handle constant arrays
                    continue
                    
                corr = np.sum(x_norm * y_norm) / (x_std * y_std * len(x))
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr  # Symmetry
                
        return corr_matrix

    def _gpu_correlation(self, data: np.ndarray) -> np.ndarray:
        """GPU-accelerated correlation matrix calculation using CuPy"""
        if not HAS_GPU:
            return self._numba_correlation(data)
            
        # Transfer data to GPU
        gpu_data = cp.asarray(data)
        # Standardize the data
        gpu_data = (gpu_data - cp.mean(gpu_data, axis=0)) / cp.std(gpu_data, axis=0, ddof=1)
        # Calculate correlation matrix
        corr_matrix = cp.dot(gpu_data.T, gpu_data) / (len(data) - 1)
        # Transfer back to CPU and return as numpy array
        return cp.asnumpy(corr_matrix)

    def _calculate_correlation_matrix(self, data: np.ndarray) -> pd.DataFrame:
        """Calculate correlation matrix with proper indexing"""
        if data.size == 0 or np.isnan(data).all():
            return pd.DataFrame()
            
        # Replace NaNs with column means to avoid computation errors
        cleaned_data = np.nan_to_num(data, nan=0.0)
        
        # Calculate correlation matrix using the selected engine
        corr_array = self._correlation_engine(cleaned_data)
        
        # Convert to DataFrame with proper indices
        asset_names = self.asset_prices.columns
        return pd.DataFrame(corr_array, index=asset_names, columns=asset_names)

    def _calculate_tail_correlations(self) -> None:
        """Calculate correlations during extreme market conditions"""
        # Filter for extreme downside moves (5% worst returns)
        extreme_threshold = self.returns.quantile(0.05)
        extreme_days = (self.returns < extreme_threshold).any(axis=1)
        
        if extreme_days.sum() > 10:  # Need enough data points
            extreme_returns = self.returns[extreme_days]
            self.tail_correlations = self._calculate_correlation_matrix(extreme_returns.values)
            self.volatile_correlations = extreme_returns.rolling(20).corr().mean()

    async def update_correlations(self) -> None:
        """Update correlation matrices with latest market data"""
        now = datetime.now()
        time_since_update = (now - self.last_update_time).total_seconds()
        
        # Only update at specified intervals to avoid excessive computation
        if time_since_update < self.update_interval:
            return
            
        async with self.processing_lock:
            try:
                self.last_update_time = now
                
                # Get latest prices
                new_data = await self.market_data.get_latest_prices()
                if new_data.empty:
                    return
                    
                # Update price and returns dataframes
                self.asset_prices = pd.concat([self.asset_prices, new_data]).tail(max(self.windows.values()) * 2)
                self.returns = self.asset_prices.pct_change().dropna()
                
                # Perform incremental updates for all timeframes
                await self._incremental_correlation_update()
                
                # Full recalculation less frequently
                time_since_full_recalc = (now - self.last_full_recalc_time).total_seconds()
                if time_since_full_recalc > 60:  # Full recalc every minute
                    self.last_full_recalc_time = now
                    await self._full_correlation_recalculation()
                    
                # Update regime-specific correlations
                current_regime = await self.regime_classifier.classify_current_regime(self.returns)
                if current_regime != self.current_regime:
                    self.current_regime = current_regime
                    self.logger.info(f"Market regime changed to: {current_regime}")
                    self.market_regime_correlations[current_regime] = self.correlation_matrices[self.windows["medium"]]
                
                # Generate trading signals
                await self._generate_signals()
                
            except Exception as e:
                self.logger.error(f"Error updating correlations: {str(e)}")

    async def _incremental_correlation_update(self) -> None:
        """Efficiently update correlation matrices without full recalculation"""
        # For short and medium windows, update incrementally
        for name, window in self.windows.items():
            if name in ["short", "medium"]:  # Only use incremental for shorter timeframes
                window_data = self.returns.tail(window)
                if len(window_data) >= window:
                    # Use a separate thread for each window to parallelize
                    loop = asyncio.get_event_loop()
                    self.correlation_matrices[window] = await loop.run_in_executor(
                        self.thread_pool,
                        self._calculate_correlation_matrix,
                        window_data.values
                    )

    async def _full_correlation_recalculation(self) -> None:
        """Perform full recalculation of all correlation matrices"""
        # Update tail correlations
        self._calculate_tail_correlations()
        
        # For longer windows, do full recalculation
        for name, window in self.windows.items():
            if name in ["long", "extreme"]:  # Full recalc for longer timeframes
                window_data = self.returns.tail(window)
                if len(window_data) >= window:
                    # Use a separate thread for each window to parallelize
                    loop = asyncio.get_event_loop()
                    self.correlation_matrices[window] = await loop.run_in_executor(
                        self.thread_pool,
                        self._calculate_correlation_matrix,
                        window_data.values
                    )

    async def _generate_signals(self) -> None:
        """Generate trading signals based on correlation analysis"""
        # Parallel signal generation
        arbitrage_task = asyncio.create_task(self._detect_arbitrage_opportunities())
        portfolio_task = asyncio.create_task(self._optimize_portfolio_weights())
        risk_task = asyncio.create_task(self._update_risk_parameters())
        
        # Wait for all tasks to complete
        signals = await asyncio.gather(arbitrage_task, portfolio_task, risk_task)
        
        # Send compiled signals to strategy orchestrator
        arbitrage_signals, portfolio_weights, risk_params = signals
        await self.strategy_orchestrator.process_correlation_insights({
            "arbitrage_signals": arbitrage_signals,
            "portfolio_weights": portfolio_weights,
            "risk_parameters": risk_params,
            "current_regime": self.current_regime
        })

    async def _detect_arbitrage_opportunities(self) -> Dict:
        """Detect statistical arbitrage opportunities using correlation analysis"""
        if self.correlation_matrices[self.windows["medium"]].empty:
            return {}
            
        # Find cointegrated pairs
        coint_pairs = await self._find_cointegrated_pairs()
        
        # Detect cross-exchange arbitrage
        cross_arb = await self.arbitrage_detector.detect_opportunities(
            self.correlation_matrices[self.windows["short"]]
        )
        
        # Detect synthetic mispricing
        synth_mispricing = await self._detect_synthetic_mispricing()
        
        # Compile all arbitrage signals
        arbitrage_signals = {
            "cointegration_pairs": coint_pairs,
            "cross_exchange": cross_arb,
            "synthetic_mispricing": synth_mispricing
        }
        
        # Update active signals
        self.active_arbitrage_signals = arbitrage_signals
        return arbitrage_signals

    @lru_cache(maxsize=CACHE_SIZE)
    async def _find_cointegrated_pairs(self) -> List[Dict]:
        """Find statistically cointegrated asset pairs for pair trading"""
        pairs = []
        symbols = self.asset_prices.columns.tolist()
        
        # Only test a manageable number of pairs
        max_pairs = min(1000, len(symbols) * (len(symbols) - 1) // 2)
        pair_count = 0
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                if pair_count >= max_pairs:
                    break
                    
                asset1, asset2 = symbols[i], symbols[j]
                pair_key = f"{asset1}_{asset2}"
                
                # Skip if correlation is too low (efficiency optimization)
                medium_corr = self.correlation_matrices[self.windows["medium"]]
                if not medium_corr.empty and abs(medium_corr.loc[asset1, asset2]) < 0.3:
                    continue
                
                # Skip previously processed pairs
                if pair_key in self.tracked_pairs:
                    continue
                    
                try:
                    asset1_data = self.asset_prices[asset1].dropna()
                    asset2_data = self.asset_prices[asset2].dropna()
                    
                    # Skip if not enough data points
                    if len(asset1_data) < 30 or len(asset2_data) < 30:
                        continue
                        
                    # Run cointegration test
                    _, pvalue, _ = coint(asset1_data, asset2_data)
                    
                    if pvalue < self.coint_threshold:
                        # Calculate hedge ratio using OLS
                        spread = asset1_data - (np.cov(asset1_data, asset2_data)[0, 1] / 
                                              np.var(asset2_data)) * asset2_data
                        # z-score of spread
                        z_score = (spread - spread.mean()) / spread.std()
                        
                        pairs.append({
                            "pair": (asset1, asset2),
                            "pvalue": float(pvalue),
                            "hedge_ratio": float(np.cov(asset1_data, asset2_data)[0, 1] / np.var(asset2_data)),
                            "current_zscore": float(z_score.iloc[-1] if len(z_score) > 0 else 0),
                            "correlation": float(medium_corr.loc[asset1, asset2]) if not medium_corr.empty else 0
                        })
                        
                        # Add to tracked pairs
                        self.tracked_pairs.add(pair_key)
                        
                except Exception as e:
                    self.logger.debug(f"Error testing cointegration for {asset1}/{asset2}: {str(e)}")
                    
                pair_count += 1
                
        return pairs

    async def _detect_synthetic_mispricing(self) -> Dict:
        """Detect mispricing between synthetic assets and underlying components"""
        mispricing = {}
        
        try:
            # Get list of synthetic assets (ETFs, indices, etc.)
            synthetic_assets = await self.market_data.get_synthetic_assets()
            
            for synth in synthetic_assets:
                # Skip if asset price not available
                if synth not in self.asset_prices.columns:
                    continue
                    
                # Get components and weights
                components = await self.market_data.get_synthetic_components(synth)
                weights = await self.market_data.get_synthetic_weights(synth)
                
                # Skip if components or weights not available
                if not components or not weights:
                    continue
                    
                # Skip if any component prices missing
                if not all(comp in self.asset_prices.columns for comp in components):
                    continue
                    
                # Get current prices
                synth_price = self.asset_prices[synth].iloc[-1]
                component_prices = self.asset_prices[components].iloc[-1]
                
                # Calculate theoretical price
                theoretical_price = sum(component_prices[comp] * weights[comp] for comp in components)
                
                # Calculate spread
                spread = synth_price - theoretical_price
                spread_pct = spread / theoretical_price
                
                # Check if spread exceeds threshold
                if abs(spread_pct) > self.arbitrage_threshold:
                    mispricing[synth] = {
                        "synthetic": synth,
                        "observed_price": float(synth_price),
                        "theoretical_price": float(theoretical_price),
                        "spread": float(spread),
                        "spread_percent": float(spread_pct),
                        "components": components
                    }
        except Exception as e:
            self.logger.error(f"Error detecting synthetic mispricing: {str(e)}")
            
        return mispricing

    async def _optimize_portfolio_weights(self) -> Dict:
        """Generate optimized portfolio weights based on correlation structure"""
        if self.correlation_matrices[self.windows["medium"]].empty:
            return {}
            
        try:
            # Use medium-term correlation matrix for optimization
            corr_matrix = self.correlation_matrices[self.windows["medium"]].values
            
            # Use risk parity approach - weights inversely proportional to risk contribution
            # Calculate asset volatilities
            vols = self.returns.std().values
            
            # Ensure positive-definite correlation matrix
            min_eig = np.min(np.linalg.eigvals(corr_matrix))
            if min_eig < 0:
                # Add small constant to diagonal to ensure positive-definiteness
                corr_matrix += abs(min_eig) * np.eye(len(corr_matrix)) * 1.1
            
            # Calculate risk contribution matrix
            risk_contrib = np.outer(vols, vols) * corr_matrix
            
            # Calculate row sums (total risk contribution)
            risk_sums = risk_contrib.sum(axis=1)
            
            # Inverse risk weighting (risk parity)
            raw_weights = 1 / (risk_sums + 1e-8)  # Add small epsilon to avoid division by zero
            
            # Normalize weights
            weights = raw_weights / raw_weights.sum()
            
            # Convert to dictionary
            asset_names = self.asset_prices.columns
            weight_dict = {asset: float(weight) for asset, weight in zip(asset_names, weights)}
            
            return {"risk_parity_weights": weight_dict}
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio weights: {str(e)}")
            return {}

    async def _update_risk_parameters(self) -> Dict:
        """Update risk parameters based on correlation analysis"""
        if self.correlation_matrices[self.windows["medium"]].empty:
            return {}
            
        try:
            # Calculate regime-aware betas
            betas = await self._calculate_regime_betas()
            
            # Calculate portfolio concentration metrics
            concentration = await self._calculate_concentration_metrics()
            
            # Generate stress scenarios
            stress_scenarios = await self.stress_generator.generate_scenarios(
                self.correlation_matrices[self.windows["medium"]], 
                self.tail_correlations
            )
            
            # Compile risk parameters
            risk_params = {
                "regime_betas": betas,
                "concentration_metrics": concentration,
                "stress_scenarios": stress_scenarios,
                "tail_correlations": self.tail_correlations.to_dict() if not self.tail_correlations.empty else {}
            }
            
            # Update risk manager
            await self.risk_manager.update_correlation_risk(risk_params)
            
            return risk_params
            
        except Exception as e:
            self.logger.error(f"Error updating risk parameters: {str(e)}")
            return {}

    async def _calculate_regime_betas(self) -> Dict:
        """Calculate regime-sensitive beta coefficients"""
        betas = {}
        
        try:
            # Get benchmark for current regime
            benchmark = await self.regime_classifier.get_regime_benchmark(self.current_regime)
            
            # Skip if benchmark not available
            if benchmark not in self.returns.columns:
                return {}
                
            benchmark_returns = self.returns[benchmark]
            benchmark_var = benchmark_returns.var()
            
            # Skip if benchmark variance is zero
            if benchmark_var == 0:
                return {}
                
            # Calculate beta for each asset
            for asset in self.returns.columns:
                if asset != benchmark:
                    # Calculate covariance
                    asset_returns = self.returns[asset]
                    cov = asset_returns.cov(benchmark_returns)
                    
                    # Calculate beta
                    beta = cov / benchmark_var
                    betas[asset] = float(beta)
                    
        except Exception as e:
            self.logger.debug(f"Error calculating regime betas: {str(e)}")
            
        return betas

    async def _calculate_concentration_metrics(self) -> Dict:
        """Calculate portfolio concentration metrics based on correlation structure"""
        if self.correlation_matrices[self.windows["medium"]].empty:
            return {}
            
        try:
            # Calculate eigenvalues of correlation matrix
            corr_matrix = self.correlation_matrices[self.windows["medium"]].values
            eig_vals = np.linalg.eigvals(corr_matrix)
            
            # Sort eigenvalues in descending order
            eig_vals = np.sort(eig_vals)[::-1]
            
            # Calculate concentration metrics
            explained_variance = []
            cumulative_var = 0
            
            for val in eig_vals:
                var_explained = val / sum(eig_vals)
                cumulative_var += var_explained
                explained_variance.append(float(cumulative_var))
                
            # Determine how many principal components explain 90% of variance
            num_components_90 = next((i+1 for i, x in enumerate(explained_variance) if x >= 0.9), len(explained_variance))
            
            # Calculate effective rank (entropy of eigenvalue distribution)
            normalized_eigs = eig_vals / sum(eig_vals)
            entropy = -sum(e * np.log(e) for e in normalized_eigs if e > 0)
            effective_rank = np.exp(entropy)
            
            return {
                "top_eigenvalue_pct": float(eig_vals[0] / sum(eig_vals)),
                "effective_rank": float(effective_rank),
                "num_components_90_pct": int(num_components_90),
                "avg_correlation": float(np.mean(corr_matrix[np.triu_indices(len(corr_matrix), k=1)]))
            }
            
        except Exception as e:
            self.logger.debug(f"Error calculating concentration metrics: {str(e)}")
            return {}

    async def real_time_monitoring(self) -> None:
        """Start real-time correlation monitoring pipeline"""
        if not self.is_running:
            await self.initialize()
            
        self.logger.info("Starting real-time correlation monitoring")
        
        ws_endpoint = self.market_data.get_websocket_endpoint()
        update_task = None
        
        try:
            async with websockets.connect(ws_endpoint) as ws:
                while self.is_running:
                    try:
                        # Start periodic correlation update task
                        if update_task is None or update_task.done():
                            update_task = asyncio.create_task(self._periodic_updates())
                            
                        # Receive market data
                        message = await asyncio.wait_for(ws.recv(), timeout=1)
                        data = json.loads(message)
                        
                        # Process data
                        await self._process_realtime_data(data)
                        
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        self.logger.error(f"WebSocket monitoring error: {str(e)}")
                        await asyncio.sleep(5)  # Backoff and retry
                        
        except Exception as e:
            self.logger.error(f"Real-time monitoring failed: {str(e)}")
            self.is_running = False
            raise
            
    async def _periodic_updates(self) -> None:
        """Periodically update correlations and generate signals"""
        while self.is_running:
            try:
                await self.update_correlations()
                
                # Detect market regime changes
                new_regime = await self.regime_classifier.classify_current_regime(self.returns)
                if new_regime != self.current_regime:
                    self.current_regime = new_regime
                    self.logger.info(f"Market regime changed to: {self.current_regime}")
                    await self._handle_regime_change(new_regime)
                    
            except Exception as e:
                self.logger.error(f"Periodic update error: {str(e)}")
                
            await asyncio.sleep(self.update_interval)

    async def _process_realtime_data(self, data: Dict) -> None:
        """Process real-time market data update"""
        try:
            if "type" not in data or data["type"] != "price_update":
                return
                
            symbol = data.get("symbol")
            price = data.get("price")
            timestamp = datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
            
            if not symbol or price is None:
                return
                
            # Create new row for the data
            new_row = pd.DataFrame({symbol: [price]}, index=[timestamp])
            
            # Update asset prices efficiently
            if symbol in self.asset_prices.columns:
                # In-place update for existing columns is more efficient
                self.asset_prices.loc[timestamp, symbol] = price
            else:
                # Merge for new columns
                self.asset_prices = pd.concat([self.asset_prices, new_row], axis=1)
                self.logger.info(f"Added new asset to monitoring: {symbol}")
                
            # Sort by timestamp and remove duplicates
            self.asset_prices = self.asset_prices.sort_index().iloc[-max(self.windows.values()*2):]
            
        except Exception as e:
            self.logger.debug(f"Error processing real-time data: {str(e)}")

    async def _handle_regime_change(self, new_regime: str) -> None:
        """Handle market regime change"""
        # Store correlation matrix for this regime
        self.market_regime_correlations[new_regime] = self.correlation_matrices[self.windows["medium"]]
        
        # Update risk parameters for new regime
        risk_params = await self._update_risk_parameters()
        
        # Notify strategy orchestrator
        await self.strategy_orchestrator.handle_regime_change({
            "new_regime": new_regime,
            "risk_parameters": risk_params,
            "correlation_matrix": self.correlation_matrices[self.windows["medium"]].to_dict()
        })

    async def get_correlation_report(self) -> Dict:
        """Generate a comprehensive correlation analysis report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "current_regime": self.current_regime,
            "correlation_matrices": {
                name: matrix.to_dict()