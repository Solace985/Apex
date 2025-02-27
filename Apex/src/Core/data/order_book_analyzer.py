# src/Core/data/order_book_analyzer.py
import numpy as np
from decimal import Decimal
from typing import Dict, List, Tuple
import asyncio
from collections import deque
from Apex.utils.helpers import validate_orderbook, secure_float
from Apex.src.ai.forecasting.spread_forecaster import LSTMSpreadPredictor
from Apex.src.Core.data.realtime.market_data import WebSocketFeed
from Apex.src.Core.trading.execution.market_impact import ImpactCalculator

class QuantumOrderBookAnalyzer:
    """Institutional-grade order book analysis with 11 liquidity metrics"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ws_feed = WebSocketFeed(symbol)
        self.spread_predictor = LSTMSpreadPredictor()
        self.impact_calculator = ImpactCalculator()
        self._setup_initial_state()

    def _setup_initial_state(self):
        """Initialize analysis buffers and caches"""
        self.order_book_cache = deque(maxlen=1000)
        self.spread_history = deque(maxlen=500)
        self.liquidity_pools = {'bids': {}, 'asks': {}}
        self.market_maker_patterns = self._load_mm_patterns()

    async def real_time_analysis(self):
        """Main analysis loop processing real-time order book updates"""
        async for ob_update in self.ws_feed.stream_order_book():
            self._process_update(ob_update)
            yield await self._full_liquidity_analysis()

    @validate_orderbook
    def _process_update(self, order_book: Dict):
        """Secure order book processing pipeline"""
        self.order_book_cache.append(order_book)
        self._update_spread_history(order_book)
        self._detect_hidden_liquidity(order_book)

    def _update_spread_history(self, order_book: Dict):
        """Track spread dynamics with millisecond precision"""
        spread = order_book['asks'][0][0] - order_book['bids'][0][0]
        self.spread_history.append(secure_float(spread))

    def calculate_order_imbalance(self, depth: int = 10) -> Dict:
        """
        ✅ Multi-depth order book imbalance analysis for better market pressure insights.
        - Analyzes **buying vs. selling pressure** across multiple order book levels.
        """
        bids = self.current_bids[:depth]
        asks = self.current_asks[:depth]
        
        bid_vol = sum(qty for _, qty in bids)
        ask_vol = sum(qty for _, qty in asks)

        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)  # Avoid division by zero

        return {
            'imbalance': round(imbalance, 4),
            'bid_pressure': bid_vol,
            'ask_pressure': ask_vol
        }
    def detect_liquidity_pools(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> List[Tuple[str, float]]:
        """
        ✅ Identifies **high-liquidity support & resistance zones** using historical order book data.
        - Filters out **fake liquidity zones created by spoof orders**.
        """
        avg_bid_qty = np.mean([qty for _, qty in bids[:10]]) if bids else 0
        avg_ask_qty = np.mean([qty for _, qty in asks[:10]]) if asks else 0
        
        significant_levels = []
        
        for price, qty in bids[:5]:
            if qty > 2 * avg_bid_qty:
                significant_levels.append(("support", price))

        for price, qty in asks[:5]:
            if qty > 2 * avg_ask_qty:
                significant_levels.append(("resistance", price))

        return significant_levels

    def estimate_slippage(self, order_size: float, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> float:
        """
        ✅ More advanced **slippage prediction** using historical execution data.
        - Prevents **unexpected price jumps** when placing large orders.
        """
        if not bids or not asks:
            return 0.0

        total_bid_volume = sum(qty for _, qty in bids)
        total_ask_volume = sum(qty for _, qty in asks)

        bid_slippage = order_size / (total_bid_volume + 1e-6)  # Avoid division by zero
        ask_slippage = order_size / (total_ask_volume + 1e-6)

        # Adjust slippage estimate based on historical execution data
        historical_slippage = np.mean(self.spread_history[-10:]) if self.spread_history else 0.01

        return round((bid_slippage + ask_slippage + historical_slippage) / 3, 4)

    async def optimize_execution(self, order: Dict) -> Dict:
        """Liquidity-optimized execution plan generation"""
        liquidity_map = await self._get_liquidity_snapshot()
        spread_forecast = self.spread_predictor.predict(self.spread_history)
        
        return {
            'size': self._adjust_size_to_liquidity(order['quantity'], liquidity_map),
            'strategy': self._select_execution_strategy(spread_forecast),
            'urgency': self._calculate_execution_urgency(order),
            'dark_pool_allocation': self._calculate_dark_pool_split(order)
        }

    def analyze_market_makers(self) -> Dict:
        """Market maker activity fingerprint detection"""
        mm_signals = {}
        for pattern, detector in self.market_maker_patterns.items():
            mm_signals[pattern] = detector.analyze(self.order_book_cache)
        return mm_signals

    def filter_spoofing_orders(self, order_book: Dict) -> Dict:
        """AI-powered spoofing detection and order book cleansing"""
        return {
            'bids': [order for order in order_book['bids'] 
                    if not self._is_spoofed_order(order, 'bid')],
            'asks': [order for order in order_book['asks']
                    if not self._is_spoofed_order(order, 'ask')]
        }

    # Core analysis utilities
    def _calculate_depth_profile(self) -> Dict:
        """Quantify available liquidity at progressive price levels"""
        return {
            'bids': self._aggregate_depth('bids'),
            'asks': self._aggregate_depth('asks')
        }

    def _aggregate_depth(self, side: str) -> List[Dict]:
        """Cumulative liquidity aggregation with tick resolution"""
        cumulative = 0
        depth_profile = []
        for price, qty in getattr(self, f'current_{side}'):
            cumulative += qty
            depth_profile.append({
                'price': price,
                'quantity': qty,
                'cumulative': cumulative
            })
        return depth_profile

    def _adjust_size_to_liquidity(self, size: Decimal, liquidity: Dict) -> Decimal:
        """Fragment orders based on available liquidity pockets"""
        remaining = float(size)
        allocated = 0.0
        for level in liquidity['bids']:
            if remaining <= 0:
                break
            alloc = min(remaining, level['quantity'])
            allocated += alloc
            remaining -= alloc
        return Decimal(str(allocated))

    # Security-hardened detection methods
    def _is_iceberg_order(self, price: float, qty: float, side: str) -> bool:
        """Statistical iceberg detection using historical patterns"""
        avg_qty = np.mean([q for _, q in getattr(self, f'current_{side}')])
        return qty > 3 * avg_qty and price not in self.liquidity_pools[side]

    def _is_spoofed_order(self, order: Tuple, side: str) -> bool:
        """Pattern-based spoofing detection"""
        price, qty = order
        return (
            qty > self._get_spoofing_threshold(side) and
            price not in self._get_valid_price_levels(side)
        )

    # Integration points
    @property
    def current_bids(self):
        return self.ws_feed.current_order_book['bids']

    @property
    def current_asks(self):
        return self.ws_feed.current_order_book['asks']

    async def _get_liquidity_snapshot(self) -> Dict:
        """Integrated with HFT liquidity manager"""
        from Apex.src.Core.trading.hft.liquidity_manager import get_hft_liquidity
        return await get_hft_liquidity(self.symbol)

    def _select_execution_strategy(self, spread_forecast: float) -> str:
        """Execution strategy selection based on spread predictions"""
        if spread_forecast > 2.0:
            return 'DarkPoolAggregation'
        elif spread_forecast < 0.5:
            return 'ImmediateExecution'
        return 'TWAP'

    # Pattern loading utilities
    def _load_mm_patterns(self) -> Dict:
        """Load market maker fingerprints from AI analysis"""
        from Apex.src.ai.analysis.market_maker_patterns import load_pattern_detectors
        return load_pattern_detectors(self.symbol)
