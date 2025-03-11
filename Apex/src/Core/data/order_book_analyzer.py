import numpy as np
import pandas as pd
from decimal import Decimal
from typing import Dict, List, Tuple, Optional, Generator, Any
import asyncio
import time
from collections import deque
import logging
from numba import jit

# Internal Apex imports
from Apex.utils.helpers.error_handler import handle_exceptions
from Apex.utils.helpers.validation import validate_orderbook, validate_trade_size
from Apex.utils.logging.structured_logger import StructuredLogger
from Apex.src.Core.data.realtime.market_data import WebSocketFeed
from Apex.src.Core.data.realtime.data_feed import DataFeed
from Apex.src.Core.trading.execution.market_impact import ImpactCalculator
from Apex.src.Core.trading.risk.risk_management import RiskParameters
from Apex.src.Core.trading.execution.conflict_resolver import ConflictResolver
from Apex.src.ai.forecasting.spread_forecaster import SpreadPredictor
from Apex.src.ai.analysis.market_maker_patterns import MarketMakerTracker
from Apex.src.ai.forecasting.order_flow import OrderFlowPredictor
from Apex.src.ai.ensembles.ensemble_voting import EnsembleVoter
from Apex.src.Core.trading.hft.liquidity_manager import LiquidityManager


class OrderBookAnalyzer:
    """
    High-performance order book analyzer for real-time market liquidity assessment.
    Optimized for ultra-low latency trading with institutional-grade liquidity detection.
    """

    def __init__(self, 
                 symbol: str, 
                 depth: int = 20, 
                 history_size: int = 500,
                 risk_params: Optional[RiskParameters] = None):
        """
        Initialize order book analyzer with optimal parameters for HFT trading.
        
        Args:
            symbol: Trading instrument symbol
            depth: Order book depth to analyze
            history_size: Number of order book snapshots to retain in memory
            risk_params: Risk management parameters
        """
        self.symbol = symbol
        self.depth = depth
        self.logger = StructuredLogger("order_book_analyzer")
        
        # Core data streams
        self.websocket_feed = WebSocketFeed(symbol)
        self.data_feed = DataFeed(symbol)
        
        # Liquidity analysis components
        self.liquidity_manager = LiquidityManager(symbol)
        self.market_maker_tracker = MarketMakerTracker(symbol)
        self.impact_calculator = ImpactCalculator()
        self.spread_predictor = SpreadPredictor(lookback_window=50)
        self.order_flow_predictor = OrderFlowPredictor(symbol)
        self.conflict_resolver = ConflictResolver()
        
        # Performance-optimized data buffers
        self.order_book_cache = deque(maxlen=history_size)
        self.spread_history = np.zeros(history_size, dtype=np.float32)
        self.spread_idx = 0
        self.imbalance_history = np.zeros(history_size, dtype=np.float32)
        self.imbalance_idx = 0
        
        # Liquidity pools detection
        self.liquidity_pools = {'bids': {}, 'asks': {}}
        self.dark_pool_activity = {}
        
        # HFT optimization flags
        self.is_spoofing_detected = False
        self.is_high_volatility = False
        self.last_process_time = time.time_ns()
        
        # Risk management integration
        self.risk_params = risk_params or RiskParameters(symbol)
        self.max_allowed_impact = 0.05  # 5 basis points default
        
        # Key metrics for trade execution
        self.current_spread = 0.0
        self.current_imbalance = 0.0
        self.current_bid_pressure = 0.0
        self.current_ask_pressure = 0.0
        self.spoofing_likelihood = 0.0
        
        # Initialize integration with other Apex components
        self._initialize_system_integration()
        
    def _initialize_system_integration(self) -> None:
        """Set up connections to other Apex system components."""
        # Register with ensemble voter for AI model integration
        try:
            from Apex.src.ai.ensembles.ensemble_voting import register_analyzer
            register_analyzer(self.symbol, self)
            
            # Connect to risk management system
            from Apex.src.Core.trading.risk.risk_management import register_liquidity_monitor
            register_liquidity_monitor(self.symbol, self)
            
            self.logger.info("Successfully integrated with Apex subsystems", 
                             data={"symbol": self.symbol})
        except Exception as e:
            self.logger.error("Failed to integrate with Apex subsystems", 
                              data={"symbol": self.symbol, "error": str(e)})
        
    @handle_exceptions
    async def start_analysis(self) -> None:
        """Initialize and start real-time order book analysis."""
        try:
            self.logger.info("Starting order book analysis", data={"symbol": self.symbol})
            await self.websocket_feed.connect()
            asyncio.create_task(self._continuous_analysis())
        except Exception as e:
            self.logger.error("Failed to start order book analysis", 
                             data={"symbol": self.symbol, "error": str(e)})
            raise
    
    async def _continuous_analysis(self) -> None:
        """Continuously analyze order book updates with minimal latency."""
        async for order_book in self.websocket_feed.stream_order_book():
            start_time = time.time_ns()
            
            # Process order book data
            self._update_order_book(order_book)
            
            # Run all analysis in parallel for minimal latency
            await asyncio.gather(
                self._analyze_spread_and_imbalance(),
                self._detect_liquidity_pools(),
                self._identify_market_maker_activity(),
                self._detect_spoofing_and_manipulation(),
                self._predict_short_term_liquidity()
            )
            
            # Provide execution quality metrics to relevant systems
            await self._update_execution_systems()
            
            # Calculate processing latency for monitoring
            process_time = (time.time_ns() - start_time) / 1_000_000  # in milliseconds
            if process_time > 10:  # Log if processing takes > 10ms
                self.logger.warning("Order book analysis latency exceeded threshold", 
                                   data={"latency_ms": process_time, "symbol": self.symbol})
    
    def _update_order_book(self, order_book: Dict) -> None:
        """Update order book cache and key metrics with validation."""
        if not validate_orderbook(order_book):
            self.logger.warning("Invalid order book data received", 
                               data={"symbol": self.symbol})
            return
            
        self.order_book_cache.append(order_book)
        
        # Extract and validate top of book data
        try:
            if len(order_book['bids']) > 0 and len(order_book['asks']) > 0:
                best_bid = float(order_book['bids'][0][0])
                best_ask = float(order_book['asks'][0][0])
                
                # Update spread with bounds checking
                if best_ask > best_bid:  # Validate spread is positive
                    self.current_spread = best_ask - best_bid
                    self.spread_history[self.spread_idx] = self.current_spread
                    self.spread_idx = (self.spread_idx + 1) % len(self.spread_history)
        except (KeyError, IndexError, ValueError) as e:
            self.logger.warning("Error processing order book data", 
                               data={"symbol": self.symbol, "error": str(e)})
    
    @jit(nopython=True)
    def _calculate_weighted_imbalance(self, bids: np.ndarray, asks: np.ndarray) -> float:
        """
        Numba-optimized calculation of order book imbalance using volume-weighted approach.
        
        Args:
            bids: Array of bid prices and quantities
            asks: Array of ask prices and quantities
            
        Returns:
            Float representing volume-weighted order imbalance [-1.0 to 1.0]
        """
        # Calculate total bid and ask volume with depth weighting
        weighted_bid_vol = 0.0
        weighted_ask_vol = 0.0
        
        for i in range(min(len(bids), 10)):  # Analyze top 10 levels
            weight = 1.0 / (1.0 + i * 0.1)  # More weight to levels closer to mid price
            weighted_bid_vol += bids[i, 1] * weight
            
        for i in range(min(len(asks), 10)):
            weight = 1.0 / (1.0 + i * 0.1)
            weighted_ask_vol += asks[i, 1] * weight
            
        # Calculate imbalance ratio [-1.0 to 1.0]
        total_vol = weighted_bid_vol + weighted_ask_vol
        if total_vol > 0:
            return (weighted_bid_vol - weighted_ask_vol) / total_vol
        return 0.0
    
    async def _analyze_spread_and_imbalance(self) -> None:
        """Analyze current spread and order book imbalance."""
        try:
            # Get current order book
            if not self.order_book_cache:
                return
                
            current_ob = self.order_book_cache[-1]
            
            # Convert to numpy arrays for vectorized computation
            bids_array = np.array([[float(p), float(q)] for p, q in current_ob['bids'][:self.depth]], 
                                  dtype=np.float32)
            asks_array = np.array([[float(p), float(q)] for p, q in current_ob['asks'][:self.depth]], 
                                  dtype=np.float32)
            
            # Calculate weighted order imbalance
            self.current_imbalance = self._calculate_weighted_imbalance(bids_array, asks_array)
            self.imbalance_history[self.imbalance_idx] = self.current_imbalance
            self.imbalance_idx = (self.imbalance_idx + 1) % len(self.imbalance_history)
            
            # Calculate bid/ask pressure (not JIT-optimized for clarity)
            self.current_bid_pressure = np.sum(bids_array[:, 1])
            self.current_ask_pressure = np.sum(asks_array[:, 1])
            
            # Predict spread changes using ML model
            self.predicted_spread = await self._predict_spread_changes()
            
        except Exception as e:
            self.logger.error("Error in spread and imbalance analysis", 
                             data={"symbol": self.symbol, "error": str(e)})
    
    async def _predict_spread_changes(self) -> float:
        """Predict near-term spread changes using ML model."""
        # Use only filled portion of history array for prediction
        valid_spread_history = np.trim_zeros(self.spread_history)
        if len(valid_spread_history) < 10:  # Need minimum data points
            return self.current_spread
            
        try:
            # Get spread prediction from ML model
            return await self.spread_predictor.predict_async(valid_spread_history)
        except Exception:
            # Fallback to simple moving average if model fails
            return np.mean(valid_spread_history[-10:])
    
    async def _detect_liquidity_pools(self) -> Dict:
        """Identify significant liquidity pools and institutional order zones."""
        if not self.order_book_cache:
            return {}
            
        try:
            current_ob = self.order_book_cache[-1]
            
            # Convert to numpy arrays for faster processing
            bids_array = np.array([[float(p), float(q)] for p, q in current_ob['bids'][:self.depth]], 
                                  dtype=np.float32)
            asks_array = np.array([[float(p), float(q)] for p, q in current_ob['asks'][:self.depth]], 
                                  dtype=np.float32)
            
            # Find liquidity clusters (price levels with significantly higher volume)
            bid_clusters = self._find_liquidity_clusters(bids_array, side='bid')
            ask_clusters = self._find_liquidity_clusters(asks_array, side='ask')
            
            # Update liquidity pools
            self.liquidity_pools = {
                'bids': bid_clusters,
                'asks': ask_clusters
            }
            
            # Detect potential dark pool activity by analyzing trade prints vs order book
            await self._detect_dark_pool_activity()
            
            return self.liquidity_pools
            
        except Exception as e:
            self.logger.error("Error detecting liquidity pools", 
                             data={"symbol": self.symbol, "error": str(e)})
            return {}
    
    def _find_liquidity_clusters(self, price_qty_array: np.ndarray, side: str) -> Dict[float, float]:
        """Find price levels with abnormally high liquidity (potential institutional orders)."""
        if len(price_qty_array) < 3:
            return {}
            
        # Calculate median quantity and identify outliers
        median_qty = np.median(price_qty_array[:, 1])
        std_qty = np.std(price_qty_array[:, 1])
        
        # Identify significant liquidity levels (> 2 standard deviations from median)
        significant_threshold = median_qty + (2 * std_qty)
        
        # Find price levels with significant liquidity
        significant_levels = {}
        for price, qty in price_qty_array:
            if qty > significant_threshold:
                significant_levels[float(price)] = float(qty)
                
        return significant_levels
    
    async def _detect_dark_pool_activity(self) -> None:
        """Detect off-exchange and dark pool trading activity."""
        try:
            # Get recent trades from data feed
            recent_trades = await self.data_feed.get_recent_trades(limit=100)
            if not recent_trades:
                return
                
            # Convert to DataFrame for easier processing
            trades_df = pd.DataFrame(recent_trades)
            
            # Calculate trade volume distribution
            volume_distribution = trades_df.groupby('venue')['volume'].sum()
            
            # Identify significant off-exchange volume
            total_volume = volume_distribution.sum()
            if total_volume > 0:
                dark_pool_ratio = volume_distribution.get('OFF', 0) / total_volume
                
                self.dark_pool_activity = {
                    'ratio': dark_pool_ratio,
                    'is_significant': dark_pool_ratio > 0.15,  # 15% threshold
                    'total_volume': float(total_volume)
                }
        except Exception as e:
            self.logger.error("Error detecting dark pool activity", 
                             data={"symbol": self.symbol, "error": str(e)})
    
    async def _identify_market_maker_activity(self) -> Dict:
        """Identify and track market maker behavior patterns."""
        if len(self.order_book_cache) < 10:
            return {}
            
        try:
            # Use market maker tracker to identify patterns
            market_maker_data = await self.market_maker_tracker.analyze_patterns(list(self.order_book_cache))
            
            # Check if market makers are pulling liquidity
            is_pulling_liquidity = market_maker_data.get('is_pulling_liquidity', False)
            
            if is_pulling_liquidity:
                self.logger.info("Market makers pulling liquidity detected", 
                                data={"symbol": self.symbol})
                
            return market_maker_data
            
        except Exception as e:
            self.logger.error("Error identifying market maker activity", 
                             data={"symbol": self.symbol, "error": str(e)})
            return {}
    
    async def _detect_spoofing_and_manipulation(self) -> Dict:
        """Detect market manipulation like spoofing, layering and quote stuffing."""
        if len(self.order_book_cache) < 20:
            return {'spoofing_detected': False}
            
        try:
            # Calculate order appearance/cancellation rates
            cancel_rates = await self._calculate_order_cancel_rates()
            
            # Update spoofing likelihood based on multiple factors
            self.spoofing_likelihood = self._estimate_spoofing_likelihood(cancel_rates)
            
            # Set flag if spoofing likelihood is high
            self.is_spoofing_detected = self.spoofing_likelihood > 0.7
            
            if self.is_spoofing_detected:
                self.logger.warning("Potential spoofing activity detected", 
                                   data={"symbol": self.symbol, "likelihood": self.spoofing_likelihood})
                
            return {
                'spoofing_detected': self.is_spoofing_detected,
                'spoofing_likelihood': self.spoofing_likelihood,
                'cancel_rates': cancel_rates
            }
            
        except Exception as e:
            self.logger.error("Error detecting manipulation", 
                             data={"symbol": self.symbol, "error": str(e)})
            return {'spoofing_detected': False}
    
    async def _calculate_order_cancel_rates(self) -> Dict:
        """Calculate order placement and cancellation rates."""
        # This is a simplified implementation
        # Real implementation would track individual orders
        return {
            'bid_cancel_rate': 0.0,
            'ask_cancel_rate': 0.0
        }
    
    def _estimate_spoofing_likelihood(self, cancel_rates: Dict) -> float:
        """Estimate likelihood of spoofing based on order book patterns."""
        # Simplified implementation
        # Real implementation would use ML model
        return 0.0
    
    async def _predict_short_term_liquidity(self) -> Dict:
        """Predict short-term liquidity changes using order flow analysis."""
        try:
            # Pass order book data to order flow predictor
            valid_imbalance_history = self.imbalance_history[self.imbalance_history != 0]
            
            if len(valid_imbalance_history) < 10:
                return {}
                
            prediction = await self.order_flow_predictor.predict_liquidity_shift(valid_imbalance_history)
            
            return prediction
            
        except Exception as e:
            self.logger.error("Error predicting liquidity", 
                             data={"symbol": self.symbol, "error": str(e)})
            return {}
    
    async def _update_execution_systems(self) -> None:
        """Update trade execution systems with liquidity analysis."""
        try:
            # Prepare liquidity data for execution systems
            liquidity_data = {
                'symbol': self.symbol,
                'timestamp': time.time_ns(),
                'current_spread': self.current_spread,
                'predicted_spread': getattr(self, 'predicted_spread', self.current_spread),
                'imbalance': self.current_imbalance,
                'bid_pressure': self.current_bid_pressure,
                'ask_pressure': self.current_ask_pressure,
                'spoofing_detected': self.is_spoofing_detected,
                'liquidity_pools': self.liquidity_pools,
                'dark_pool_activity': self.dark_pool_activity
            }
            
            # Send liquidity data to meta_trader
            from Apex.src.ai.ensembles.meta_trader import update_liquidity_data
            await update_liquidity_data(liquidity_data)
            
            # Send liquidity data to risk management
            from Apex.src.Core.trading.risk.risk_management import update_liquidity_risk
            await update_liquidity_risk(liquidity_data)
            
        except Exception as e:
            self.logger.error("Error updating execution systems", 
                             data={"symbol": self.symbol, "error": str(e)})
    
    # ===== Public API Methods =====
    
    @validate_trade_size
    async def estimate_market_impact(self, order_size: float, side: str) -> Dict:
        """
        Estimate market impact for a given order size.
        
        Args:
            order_size: Size of the order
            side: 'buy' or 'sell'
            
        Returns:
            Dict containing impact estimates
        """
        if not self.order_book_cache:
            return {'impact_bps': 0.0, 'is_executable': True}
            
        try:
            # Get current order book
            current_ob = self.order_book_cache[-1]
            
            # Determine which side of the book to analyze
            levels = current_ob['asks'] if side.lower() == 'buy' else current_ob['bids']
            
            # Convert to numpy for faster calculation
            price_qty = np.array([[float(p), float(q)] for p, q in levels[:self.depth]], 
                                dtype=np.float32)
            
            # Calculate cumulative available quantity
            cum_qty = np.cumsum(price_qty[:, 1])
            
            # Determine price levels needed to fill the order
            if order_size <= cum_qty[-1]:
                # Find the index where cumulative quantity exceeds order size
                idx = np.searchsorted(cum_qty, order_size)
                if idx < len(price_qty):
                    avg_price = np.sum(price_qty[:idx+1, 0] * price_qty[:idx+1, 1]) / np.sum(price_qty[:idx+1, 1])
                    
                    # Calculate impact relative to best price
                    best_price = price_qty[0, 0]
                    impact_bps = abs(avg_price - best_price) / best_price * 10000  # in basis points
                    
                    return {
                        'impact_bps': float(impact_bps),
                        'avg_price': float(avg_price),
                        'best_price': float(best_price),
                        'is_executable': impact_bps <= self.max_allowed_impact * 10000
                    }
            
            # If not enough liquidity to execute
            return {
                'impact_bps': float('inf'),
                'is_executable': False,
                'reason': 'insufficient_liquidity'
            }
            
        except Exception as e:
            self.logger.error("Error estimating market impact", 
                             data={"symbol": self.symbol, "error": str(e), "order_size": order_size})
            return {'impact_bps': 0.0, 'is_executable': False, 'reason': 'calculation_error'}
    
    @validate_trade_size
    async def optimize_order_size(self, target_size: float, side: str, max_impact_bps: float = 25.0) -> float:
        """
        Optimize order size to limit market impact.
        
        Args:
            target_size: Desired order size
            side: 'buy' or 'sell'
            max_impact_bps: Maximum allowed market impact in basis points
            
        Returns:
            Optimized order size
        """
        if not self.order_book_cache:
            return 0.0
            
        try:
            # Binary search to find optimal size
            low, high = 0.0, float(target_size)
            optimal_size = 0.0
            
            while high - low > target_size * 0.01:  # 1% precision
                mid = (low + high) / 2
                impact = await self.estimate_market_impact(mid, side)
                
                if impact['impact_bps'] <= max_impact_bps:
                    # We can increase size
                    optimal_size = mid
                    low = mid
                else:
                    # We need to decrease size
                    high = mid
            
            return float(optimal_size)
            
        except Exception as e:
            self.logger.error("Error optimizing order size", 
                             data={"symbol": self.symbol, "error": str(e)})
            return 0.0
    
    async def get_execution_recommendation(self, order_size: float, side: str) -> Dict:
        """
        Get execution recommendations for an order.
        
        Args:
            order_size: Size of the order
            side: 'buy' or 'sell'
            
        Returns:
            Dict containing execution recommendations
        """
        try:
            # Get market impact estimates
            impact = await self.estimate_market_impact(order_size, side)
            
            # Check if dark pools should be used
            use_dark_pool = order_size > 0 and self.dark_pool_activity.get('is_significant', False)
            
            # Check if order should be split
            should_split = impact.get('impact_bps', 0) > 10.0 or order_size > 0
            
            # Calculate optimal order types and venue distribution
            optimal_venues = await self._calculate_optimal_venues(order_size, side)
            
            return {
                'market_impact_bps': impact.get('impact_bps', 0),
                'is_executable': impact.get('is_executable', False),
                'should_use_dark_pool': use_dark_pool,
                'should_split_order': should_split,
                'optimal_venues': optimal_venues,
                'iceberg_recommended': order_size > 0,
                'time_in_force': 'IOC' if self.is_high_volatility else 'GTC',
                'current_spread_bps': self.current_spread * 10000,
                'spoofing_detected': self.is_spoofing_detected
            }
            
        except Exception as e:
            self.logger.error("Error getting execution recommendation", 
                             data={"symbol": self.symbol, "error": str(e)})
            return {'is_executable': False, 'reason': 'calculation_error'}
    
    async def _calculate_optimal_venues(self, order_size: float, side: str) -> Dict:
        """Calculate optimal venue distribution for order execution."""
        # This is a simplified implementation
        return {
            'lit_markets': 0.7,
            'dark_pools': 0.3
        }
    
    async def get_current_liquidity_snapshot(self) -> Dict:
        """
        Get current liquidity snapshot for strategy decisions.
        
        Returns:
            Dict containing liquidity metrics
        """
        return {
            'symbol': self.symbol,
            'timestamp': time.time_ns(),
            'current_spread': self.current_spread,
            'imbalance': self.current_imbalance,
            'bid_pressure': self.current_bid_pressure,
            'ask_pressure': self.current_ask_pressure,
            'spoofing_detected': self.is_spoofing_detected,
            'has_significant_liquidity_pools': bool(self.liquidity_pools['bids'] or self.liquidity_pools['asks']),
            'dark_pool_activity_ratio': self.dark_pool_activity.get('ratio', 0.0)
        }
    
    def __del__(self) -> None:
        """Clean up resources when object is destroyed."""
        try:
            asyncio.create_task(self.websocket_feed.disconnect())
            self.logger.info("Order book analyzer resources cleaned up", 
                           data={"symbol": self.symbol})
        except Exception:
            pass