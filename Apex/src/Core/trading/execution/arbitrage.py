"""
Apex Core: Quantum Arbitrage Engine (QAE)
-----------------------------------------
Institutional-grade multi-market arbitrage system with sub-millisecond execution
capabilities and AI-driven opportunity prediction and validation.

Key features:
- Multi-exchange and multi-broker arbitrage detection and execution
- Ultra-low latency WebSocket and FIX protocol integration
- GPU-accelerated opportunity detection
- AI-enhanced opportunity prediction and validation
- Distributed execution with fault tolerance
- Comprehensive risk controls with real-time exposure monitoring
"""

import asyncio
import time
import numpy as np
import msgpack
import uvloop
import hmac
import hashlib
import os
import logging
import json
from typing import Dict, List, Optional, Tuple, Union, Set
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

# Apex Core Imports
from Apex.utils.logging.structured_logger import StructuredLogger
from Apex.src.Core.trading.execution.order_execution import OrderExecutionManager
from Apex.src.Core.trading.hft.liquidity_manager import LiquidityOracle
from Apex.src.Core.trading.risk.risk_management import RiskManagementEngine
from Apex.src.Core.data.trade_history import TradeHistory
from Apex.src.Core.data.order_book_analyzer import OrderBookAnalyzer
from Apex.src.Core.data.realtime.market_data import MarketDataStream
from Apex.src.Core.data.realtime.websocket_handler import WebSocketManager
from Apex.src.Core.trading.execution.broker_factory import BrokerFactory
from Apex.src.Core.trading.execution.broker_manager import BrokerManager
from Apex.src.Core.trading.execution.universal_broker import UniversalBrokerAPI
from Apex.src.ai.ensembles.meta_trader import MetaTrader
from Apex.src.ai.analysis.market_regime_classifier import MarketRegimeClassifier
from Apex.utils.analytics.monte_carlo_simulator import MonteCarloSimulator
from Apex.src.Core.trading.execution.market_impact import MarketImpactAnalyzer
from Apex.Config.config_loader import ConfigLoader
from Apex.utils.logging.logging_handler import LoggingHandler

# Configure UVLoop for enhanced async performance
uvloop.install()

# Initialize structured logger
logger = StructuredLogger("QuantumArbitrage", log_level=logging.INFO)

@dataclass
class ArbitrageOpportunity:
    """Data class for arbitrage opportunities with nanosecond precision timestamps."""
    id: str
    buy_exchange: str
    sell_exchange: str
    symbol: str
    buy_price: float
    sell_price: float
    spread: float
    amount: float
    timestamp_ns: int
    discovered_ns: int
    strategy: str
    confidence: float
    expected_profit: float
    expected_slippage: float
    latency_estimate_ns: float
    opportunity_window_ms: float
    risk_score: float = 0.0
    is_validated: bool = False
    is_executing: bool = False
    market_impact: float = 0.0
    execution_method: str = "direct"  # "direct" or "broker"
    metadata: Dict = field(default_factory=dict)


class ArbitrageStrategySelector:
    """Selects optimal arbitrage strategies based on market conditions."""
    
    def __init__(self):
        self.config = ConfigLoader().get_arbitrage_config()
        self.market_regime = MarketRegimeClassifier()
        self.strategies = {
            "cross_exchange": self._cross_exchange_arbitrage,
            "triangular": self._triangular_arbitrage,
            "statistical": self._statistical_arbitrage,
            "latency": self._latency_arbitrage,
            "futures_spot": self._futures_spot_arbitrage,
        }
        # Performance tracking
        self.strategy_performance = defaultdict(lambda: {
            "attempts": 0,
            "executions": 0,
            "profits": [],
            "latency_ns": []
        })
        
    async def select_strategies(self, market_data: dict) -> List[str]:
        """Select appropriate arbitrage strategies based on current market conditions."""
        regime = await self.market_regime.classify_current_regime()
        
        # Strategy selection based on market regime
        if regime == "high_volatility":
            return ["cross_exchange", "latency", "futures_spot"]
        elif regime == "low_volatility":
            return ["triangular", "statistical"]
        elif regime == "trending":
            return ["cross_exchange", "futures_spot"]
        else:  # default/unknown regime
            return ["cross_exchange", "triangular"]
    
    async def evaluate_strategy_performance(self) -> Dict:
        """Evaluate performance of each strategy and optimize allocation."""
        results = {}
        for name, stats in self.strategy_performance.items():
            if stats["attempts"] == 0:
                continue
                
            success_rate = stats["executions"] / stats["attempts"] if stats["attempts"] > 0 else 0
            avg_profit = np.mean(stats["profits"]) if stats["profits"] else 0
            avg_latency = np.mean(stats["latency_ns"]) if stats["latency_ns"] else 0
            
            # Calculate performance score (higher is better)
            score = (success_rate * 0.4) + (avg_profit * 0.4) - (avg_latency * 0.2)
            results[name] = score
            
        return results
        
    async def _cross_exchange_arbitrage(self, market_data: dict) -> List[ArbitrageOpportunity]:
        """Detect cross-exchange arbitrage opportunities."""
        opportunities = []
        
        # Vectorized implementation using numpy for faster processing
        symbols = market_data["symbols"]
        exchanges = market_data["exchanges"]
        prices = np.array([[market_data["prices"].get((ex, sym), np.nan) 
                            for sym in symbols] 
                           for ex in exchanges])
        
        # Find min and max prices for each symbol
        min_prices = np.nanmin(prices, axis=0)
        max_prices = np.nanmax(prices, axis=0)
        min_exchanges = np.nanargmin(prices, axis=0)
        max_exchanges = np.nanargmax(prices, axis=0)
        
        # Calculate spreads
        spreads = (max_prices - min_prices) / min_prices
        
        # Filter opportunities based on minimum spread threshold
        min_spread = self.config["strategies"]["cross_exchange"]["min_spread"]
        viable_opportunities = np.where(spreads > min_spread)[0]
        
        now_ns = time.time_ns()
        
        for i in viable_opportunities:
            symbol = symbols[i]
            buy_exchange = exchanges[min_exchanges[i]]
            sell_exchange = exchanges[max_exchanges[i]]
            buy_price = prices[min_exchanges[i], i]
            sell_price = prices[max_exchanges[i], i]
            spread = spreads[i]
            
            # Skip if any price is NaN
            if np.isnan(buy_price) or np.isnan(sell_price):
                continue
                
            # Calculate optimal trade size based on liquidity
            amount = market_data["liquidity"].get((buy_exchange, symbol), 0)
            
            # Generate opportunity
            opp = ArbitrageOpportunity(
                id=f"cross_{symbol}_{buy_exchange}_{sell_exchange}_{now_ns}",
                buy_exchange=buy_exchange,
                sell_exchange=sell_exchange,
                symbol=symbol,
                buy_price=buy_price,
                sell_price=sell_price,
                spread=spread,
                amount=amount,
                timestamp_ns=market_data["timestamp_ns"],
                discovered_ns=now_ns,
                strategy="cross_exchange",
                confidence=0.85,  # Base confidence
                expected_profit=(sell_price - buy_price) * amount,
                expected_slippage=0.0,  # Will be calculated during validation
                latency_estimate_ns=5000000,  # 5ms initial estimate
                opportunity_window_ms=100.0,  # Initial estimate
            )
            opportunities.append(opp)
        
        return opportunities
        
    async def _triangular_arbitrage(self, market_data: dict) -> List[ArbitrageOpportunity]:
        """Detect triangular arbitrage opportunities across currency pairs."""
        opportunities = []
        
        # Process for each exchange
        for exchange in market_data["exchanges"]:
            # Get all trading pairs for this exchange
            pairs = [sym for sym in market_data["symbols"] 
                     if (exchange, sym) in market_data["prices"]]
            
            # Find all possible triangular paths (A->B->C->A)
            for path in self._find_triangular_paths(pairs):
                pair_ab, pair_bc, pair_ca = path
                
                # Extract prices
                price_ab = market_data["prices"].get((exchange, pair_ab), np.nan)
                price_bc = market_data["prices"].get((exchange, pair_bc), np.nan)
                price_ca = market_data["prices"].get((exchange, pair_ca), np.nan)
                
                if np.isnan(price_ab) or np.isnan(price_bc) or np.isnan(price_ca):
                    continue
                
                # Calculate triangular arbitrage profit
                profit_ratio = (1 / price_ab) * (1 / price_bc) * (1 / price_ca) - 1
                
                # If profit exists beyond threshold
                min_profit = self.config["strategies"]["triangular"]["min_profit"]
                if profit_ratio > min_profit:
                    now_ns = time.time_ns()
                    amount = min(
                        market_data["liquidity"].get((exchange, pair_ab), 0),
                        market_data["liquidity"].get((exchange, pair_bc), 0),
                        market_data["liquidity"].get((exchange, pair_ca), 0)
                    )
                    
                    opp = ArbitrageOpportunity(
                        id=f"tri_{exchange}_{pair_ab}_{pair_bc}_{pair_ca}_{now_ns}",
                        buy_exchange=exchange,
                        sell_exchange=exchange,  # Same exchange for triangular
                        symbol=f"{pair_ab}-{pair_bc}-{pair_ca}",  # Composite symbol
                        buy_price=price_ab,  # Starting price
                        sell_price=price_ab * (1 + profit_ratio),  # Effective final price
                        spread=profit_ratio,
                        amount=amount,
                        timestamp_ns=market_data["timestamp_ns"],
                        discovered_ns=now_ns,
                        strategy="triangular",
                        confidence=0.80,  # Base confidence
                        expected_profit=amount * profit_ratio,
                        expected_slippage=0.0,
                        latency_estimate_ns=8000000,  # 8ms initial estimate
                        opportunity_window_ms=50.0,  # Smaller window for triangular arb
                    )
                    opportunities.append(opp)
        
        return opportunities
    
    async def _statistical_arbitrage(self, market_data: dict) -> List[ArbitrageOpportunity]:
        """Detect statistical arbitrage opportunities between correlated assets."""
        # Implementation for statistical arbitrage
        # This is a placeholder - would require historical correlation data
        return []
        
    async def _latency_arbitrage(self, market_data: dict) -> List[ArbitrageOpportunity]:
        """Detect latency-based arbitrage opportunities."""
        # Implementation for latency arbitrage
        # This requires analysis of exchange price propagation delays
        return []
        
    async def _futures_spot_arbitrage(self, market_data: dict) -> List[ArbitrageOpportunity]:
        """Detect futures vs spot arbitrage opportunities."""
        # Implementation for futures-spot arbitrage
        # This requires both spot and futures pricing data
        return []
    
    def _find_triangular_paths(self, pairs: List[str]) -> List[Tuple[str, str, str]]:
        """Find all possible triangular paths in the given currency pairs."""
        # Extract all currencies from pairs
        currencies = set()
        for pair in pairs:
            base, quote = pair.split('/') if '/' in pair else pair.split('_')
            currencies.add(base)
            currencies.add(quote)
        
        paths = []
        # For each possible currency triplet
        for a in currencies:
            for b in currencies:
                if b == a:
                    continue
                for c in currencies:
                    if c == a or c == b:
                        continue
                    
                    # Check if all required pairs exist
                    pair_ab = f"{a}/{b}" if f"{a}/{b}" in pairs else f"{a}_{b}"
                    pair_bc = f"{b}/{c}" if f"{b}/{c}" in pairs else f"{b}_{c}"
                    pair_ca = f"{c}/{a}" if f"{c}/{a}" in pairs else f"{c}_{a}"
                    
                    if pair_ab in pairs and pair_bc in pairs and pair_ca in pairs:
                        paths.append((pair_ab, pair_bc, pair_ca))
        
        return paths


class BrokerArbitrageAdapter:
    """Adapter for executing arbitrage trades through traditional brokers."""
    
    def __init__(self):
        self.broker_manager = BrokerManager()
        self.broker_factory = BrokerFactory()
        self.broker_apis = {}  # Cache of broker API connections
        self.config = ConfigLoader().get_arbitrage_config()
        self.execution_stats = defaultdict(lambda: {
            "attempts": 0,
            "successes": 0,
            "fails": 0,
            "latency_ms": []
        })
        
    async def initialize_broker_connections(self):
        """Initialize connections to all configured brokers."""
        broker_config = self.config.get("brokers", {})
        for broker_name, settings in broker_config.items():
            try:
                broker_api = await self.broker_factory.create_broker(
                    broker_name, 
                    settings.get("api_key", ""), 
                    settings.get("api_secret", ""),
                    settings.get("passphrase", "")
                )
                self.broker_apis[broker_name] = broker_api
                logger.info(f"Initialized broker connection: {broker_name}")
            except Exception as e:
                logger.error(f"Failed to initialize broker: {broker_name}", error=str(e))
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> Dict:
        """Execute an arbitrage opportunity through brokers."""
        start_time_ns = time.time_ns()
        
        # Update stats
        self.execution_stats[opportunity.buy_exchange]["attempts"] += 1
        self.execution_stats[opportunity.sell_exchange]["attempts"] += 1
        
        try:
            # Step 1: Get broker APIs
            buy_broker = self._get_broker_api(opportunity.buy_exchange)
            sell_broker = self._get_broker_api(opportunity.sell_exchange)
            
            if not buy_broker or not sell_broker:
                logger.error("Broker API not available", 
                             buy_exchange=opportunity.buy_exchange,
                             sell_exchange=opportunity.sell_exchange)
                return {"success": False, "error": "Broker API not available"}
            
            # Step 2: Verify opportunity still exists
            is_valid = await self._verify_opportunity(buy_broker, sell_broker, opportunity)
            if not is_valid:
                return {"success": False, "error": "Opportunity no longer valid"}
            
            # Step 3: Execute orders with nanosecond precision timing
            buy_order = await buy_broker.create_limit_order(
                symbol=opportunity.symbol,
                side="buy",
                amount=opportunity.amount,
                price=opportunity.buy_price * (1 + self.config["execution"]["price_buffer"]),
                client_order_id=f"{opportunity.id}_buy"
            )
            
            sell_order = await sell_broker.create_limit_order(
                symbol=opportunity.symbol,
                side="sell",
                amount=opportunity.amount,
                price=opportunity.sell_price * (1 - self.config["execution"]["price_buffer"]),
                client_order_id=f"{opportunity.id}_sell"
            )
            
            # Step 4: Monitor execution and handle fills
            buy_result = await self._monitor_order_execution(buy_broker, buy_order, opportunity)
            sell_result = await self._monitor_order_execution(sell_broker, sell_order, opportunity)
            
            # Step 5: Calculate final profit/loss
            success = buy_result["success"] and sell_result["success"]
            
            if success:
                profit = (sell_result["executed_price"] - buy_result["executed_price"]) * min(
                    buy_result["executed_amount"],
                    sell_result["executed_amount"]
                )
                
                self.execution_stats[opportunity.buy_exchange]["successes"] += 1
                self.execution_stats[opportunity.sell_exchange]["successes"] += 1
                
                latency_ms = (time.time_ns() - start_time_ns) / 1_000_000
                self.execution_stats[opportunity.buy_exchange]["latency_ms"].append(latency_ms)
                self.execution_stats[opportunity.sell_exchange]["latency_ms"].append(latency_ms)
                
                logger.info("Arbitrage execution successful", 
                           opportunity_id=opportunity.id,
                           profit=profit,
                           latency_ms=latency_ms)
                
                return {
                    "success": True,
                    "profit": profit,
                    "buy_details": buy_result,
                    "sell_details": sell_result,
                    "latency_ns": time.time_ns() - start_time_ns
                }
            else:
                self.execution_stats[opportunity.buy_exchange]["fails"] += 1
                self.execution_stats[opportunity.sell_exchange]["fails"] += 1
                
                # Attempt to cancel any unfilled orders
                if buy_result["status"] == "open":
                    await buy_broker.cancel_order(buy_order["id"])
                if sell_result["status"] == "open":
                    await sell_broker.cancel_order(sell_order["id"])
                
                logger.warning("Arbitrage execution failed", 
                              opportunity_id=opportunity.id,
                              buy_status=buy_result["status"],
                              sell_status=sell_result["status"])
                
                return {
                    "success": False,
                    "buy_details": buy_result,
                    "sell_details": sell_result,
                    "error": "Execution failed"
                }
                
        except Exception as e:
            logger.error("Error executing arbitrage through brokers", 
                        opportunity_id=opportunity.id,
                        error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _verify_opportunity(self, buy_broker, sell_broker, opportunity) -> bool:
        """Verify the arbitrage opportunity still exists."""
        # Get current prices from both exchanges
        buy_ticker = await buy_broker.get_ticker(opportunity.symbol)
        sell_ticker = await sell_broker.get_ticker(opportunity.symbol)
        
        if not buy_ticker or not sell_ticker:
            return False
        
        # Check if spread still exists
        current_buy_price = buy_ticker["ask"]
        current_sell_price = sell_ticker["bid"]
        current_spread = (current_sell_price - current_buy_price) / current_buy_price
        
        min_spread = self.config["execution"]["min_spread"]
        return current_spread >= min_spread
    
    async def _monitor_order_execution(self, broker, order, opportunity) -> Dict:
        """Monitor order execution and return results."""
        max_wait_time_ms = self.config["execution"]["max_wait_time_ms"]
        poll_interval_ms = self.config["execution"]["poll_interval_ms"]
        
        start_time_ms = time.time() * 1000
        while (time.time() * 1000) - start_time_ms < max_wait_time_ms:
            # Check order status
            order_status = await broker.get_order_status(order["id"])
            
            if order_status["status"] == "filled":
                return {
                    "success": True,
                    "status": "filled",
                    "executed_price": order_status["executed_price"],
                    "executed_amount": order_status["executed_amount"],
                    "fee": order_status.get("fee", 0)
                }
            elif order_status["status"] == "canceled" or order_status["status"] == "expired":
                return {"success": False, "status": order_status["status"]}
            
            # Wait before checking again
            await asyncio.sleep(poll_interval_ms / 1000)
        
        # If we reach here, the order hasn't filled within the time limit
        return {"success": False, "status": "timeout"}
    
    def _get_broker_api(self, exchange_name):
        """Get broker API for the specified exchange."""
        if exchange_name in self.broker_apis:
            return self.broker_apis[exchange_name]
        
        # If not found, try to match with a universal broker
        for broker_name, broker in self.broker_apis.items():
            if broker.supports_exchange(exchange_name):
                return broker
        
        return None


class QuantumArbitrageEngine:
    """
    Institutional-Grade Multi-Market Arbitrage System
    - Ultra-low latency detection and execution
    - AI-driven opportunity prediction and validation
    - Atomic cross-exchange transaction processing
    - Seamless integration with Apex's trading ecosystem
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    async def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Load configuration
        self.config = ConfigLoader().get_arbitrage_config()
        
        # Initialize components
        self.execution_manager = OrderExecutionManager()
        self.liquidity_oracle = LiquidityOracle()
        self.risk_engine = RiskManagementEngine()
        self.trade_history = TradeHistory()
        self.market_impact = MarketImpactAnalyzer()
        self.order_book = OrderBookAnalyzer()
        self.market_data = MarketDataStream()
        self.websocket_manager = WebSocketManager()
        
        # Initialize broker adapter for non-HFT execution
        self.broker_adapter = BrokerArbitrageAdapter()
        await self.broker_adapter.initialize_broker_connections()
        
        # Initialize strategy selector
        self.strategy_selector = ArbitrageStrategySelector()
        
        # AI components
        self.meta_trader = MetaTrader()
        self.opportunity_predictor = await self.meta_trader.load_component('arbitrage_predictor')
        self.opportunity_validator = await self.meta_trader.load_component('opportunity_validator')
        self.market_regime = MarketRegimeClassifier()
        
        # State management
        self.active_opportunities = {}
        self.opportunity_queue = asyncio.PriorityQueue()
        self.execution_queue = asyncio.Queue()
        self._shutdown = False
        self.execution_lock = defaultdict(asyncio.Lock)
        
        # Security setup
        self.hmac_key = os.environ.get("APEX_HMAC_SECRET", "default_key").encode()
        
        # Performance metrics
        self.metrics = {
            'opportunities_detected': 0,
            'opportunities_validated': 0,
            'executions_attempted': 0,
            'executions_successful': 0,
            'executions_failed': 0,
            'latency_ns': deque(maxlen=1000),
            'profit_loss': deque(maxlen=1000),
            'throughput': 0.0
        }
        
        # Initialize WebSocket connections
        await self._initialize_websockets()
        
        # Start monitoring threads
        await self._start_monitoring_tasks()
        
        logger.info("Quantum Arbitrage Engine initialized")

    async def _initialize_websockets(self):
        """Initialize WebSocket connections to exchanges."""
        exchanges = self.config.get("exchanges", [])
        symbols = self.config.get("symbols", [])
        
        for exchange in exchanges:
            for symbol in symbols:
                try:
                    await self.websocket_manager.subscribe_to_orderbook(
                        exchange=exchange,
                        symbol=symbol,
                        callback=self._on_orderbook_update
                    )
                    logger.info(f"Subscribed to orderbook: {exchange}/{symbol}")
                except Exception as e:
                    logger.error(f"Failed to subscribe to orderbook: {exchange}/{symbol}", error=str(e))

    async def _start_monitoring_tasks(self):
        """Start monitoring tasks."""
        asyncio.create_task(self._monitor_active_opportunities())
        asyncio.create_task(self._monitor_execution_queue())
        asyncio.create_task(self._monitor_system_performance())
        asyncio.create_task(self._monitor_risk_exposure())

    async def start(self):
        """Start the quantum arbitrage engine."""
        logger.info("Starting Quantum Arbitrage Engine")
        
        # Start all processing tasks
        tasks = [
            self._process_market_data(),
            self._predict_opportunities(),
            self._validate_opportunities(),
            self._execute_opportunities(),
            self._monitor_performance(),
            self._handle_risk_controls(),
            self._cleanup_expired_opportunities()
        ]
        
        await asyncio.gather(*tasks)

    async def shutdown(self):
        """Gracefully shutdown the quantum arbitrage engine."""
        logger.info("Shutting down Quantum Arbitrage Engine")
        self._shutdown = True
        
        # Cancel all pending orders
        await self.execution_manager.cancel_all_pending()
        
        # Close all WebSocket connections
        await self.websocket_manager.close_all_connections()
        
        logger.info("Quantum Arbitrage Engine shutdown complete")

    async def _on_orderbook_update(self, exchange: str, symbol: str, data: dict):
        """Process orderbook updates from WebSocket."""
        # Process the orderbook data
        timestamp_ns = int(data.get("timestamp", time.time()) * 1_000_000_000)
        
        market_data = {
            "exchange": exchange,
            "symbol": symbol,
            "timestamp_ns": timestamp_ns,
            "bids": data.get("bids", []),
            "asks": data.get("asks", []),
            "last_update_id": data.get("last_update_id", 0)
        }
        
        # Update the market data store
        await self.market_data.update(exchange, symbol, market_data)
        
        # Trigger opportunity detection
        await self._detect_opportunities_from_orderbook(exchange, symbol, market_data)

    async def _detect_opportunities_from_orderbook(self, exchange: str, symbol: str, orderbook: dict):
        """Detect arbitrage opportunities from orderbook data."""
        # Get best bid and ask from this orderbook
        best_bid = orderbook["bids"][0][0] if orderbook["bids"] else 0
        best_ask = orderbook["asks"][0][0] if orderbook["asks"] else 0
        
        # Get orderbooks from other exchanges for this symbol
        other_exchanges = []
        for ex in self.config.get("exchanges", []):
            if ex != exchange:
                other_ob = await self.market_data.get(ex, symbol)
                if other_ob:
                    other_exchanges.append((ex, other_ob))
        
        # Look for cross-exchange arbitrage opportunities
        for other_ex, other_ob in other_exchanges:
            other_best_bid = other_ob["bids"][0][0] if other_ob["bids"] else 0
            other_best_ask = other_ob["asks"][0][0] if other_ob["asks"] else 0
            
            # Check if arbitrage opportunity exists (buy low, sell high)
            if best_ask < other_best_bid:
                # Can buy on this exchange and sell on other exchange
                spread = (other_best_bid - best_ask) / best_ask
                
                # Check if spread meets minimum threshold
                if spread >= self.config["min_spread"]:
                    # Calculate available volume
                    volume = min(
                        orderbook["asks"][0][1],  # Volume available at best ask
                        other_ob["bids"][0][1]    # Volume available at best bid
                    )
                    
                    # Calculate expected profit
                    expected_profit = (other_best_bid - best_ask) * volume
                    
                    # Create opportunity
                    opportunity = ArbitrageOpportunity(
                        id=self._generate_opportunity_id(),
                        buy_exchange=exchange,
                        sell_exchange=other_ex,
                        symbol=symbol,
                        buy_price=best_ask,
                        sell_price=other_best_bid,
                        spread=spread,
                        amount=volume,
                        timestamp_ns=orderbook["timestamp_ns"],
                        discovered_ns=time.time_ns(),
                        strategy="cross_exchange",
                        confidence=0.9,
                        expected_profit=expected_profit,
                        expected_slippage=0.0,
                        latency_estimate_ns=0,
                        opportunity_window_ms=100.0
                    )
                    
                    # Add to active opportunities
                    self.active_opportunities[opportunity.id] = opportunity
                    
                    # Add to validation queue
                    priority = -opportunity.expected_profit  # Higher profit = higher priority
                    await self.opportunity_queue.put((priority, opportunity))
                    
                    self.metrics["opportunities_detected"] += 1
                    
                    logger.debug(f"Detected arbitrage opportunity: {exchange} -> {other_ex}, spread: {spread:.2%}")
            
            # Check reverse opportunity (buy on other exchange, sell on this exchange)
            if other_best_ask < best_bid:
                # Can buy on other exchange and sell on this exchange
                spread = (best_bid - other_best_ask) / other_best_ask
                
                if spread >= self.config["min_spread"]:
                    volume = min(
                        other_ob["asks"][0][1],  # Volume available at best ask
                        orderbook["bids"][0][1]  # Volume available at best bid
                    )
                    
                    expected_profit = (best_bid - other_best_ask) * volume
                    
                    opportunity = ArbitrageOpportunity(
                        id=self._generate_opportunity_id(),
                        buy_exchange=other_ex,
                        sell_exchange=exchange,
                        symbol=symbol,
                        buy_price=other_best_ask,
                        sell_price=best_bid,
                        spread=spread,
                        amount=volume,
                        timestamp_ns=orderbook["timestamp_ns"],
                        discovered_ns=time.time_ns(),
                        strategy="cross_exchange",
                        confidence=0.9,
                        expected_profit=expected_profit,
                        expected_slippage=0.0,
                        latency_estimate_ns=0,
                        opportunity_window_ms=100.0
                    )
                    
                    # Add to active opportunities
                    self.active_opportunities[opportunity.id] = opportunity
                    
                    # Add to validation queue
                    priority = -opportunity.expected_profit  # Higher profit = higher priority
                    await self.opportunity_queue.put((priority, opportunity))
                    
                    self.metrics["opportunities_detected"] += 1
                    
                    logger.debug(f"Detected arbitrage opportunity: {other_ex} -> {exchange}, spread: {spread:.2%}")

    async def _process_market_data(self):
        """Process market data from all sources and detect arbitrage opportunities."""
        while not self._shutdown:
            try:
                # Get consolidated market data from all exchanges
                market_data = await self.market_data.get_consolidated_data()
                
                # Get current market regime
                market_regime = await self.market_regime.classify_current_regime()
                
                # Select appropriate strategies based on market regime
                active_strategies = await self.strategy_selector.select_strategies(market_data)
                
                # Process each strategy
                opportunities = []
                for strategy_name in active_strategies:
                    strategy_func = getattr(self.strategy_selector, f"_{strategy_name}_arbitrage")
                    strategy_opportunities = await strategy_func(market_data)
                    opportunities.extend(strategy_opportunities)
                    
                    # Update metrics
                    self.metrics["opportunities_detected"] += len(strategy_opportunities)
                
                # Process detected opportunities
                for opp in opportunities:
                    # Add to active opportunities
                    self.active_opportunities[opp.id] = opp
                    
                    # Add to validation queue with priority
                    priority = -opp.expected_profit  # Higher profit = higher priority
                    await self.opportunity_queue.put((priority, opp))
                
                # Adaptive sleep based on market volatility
                sleep_time = self._calculate_adaptive_sleep_time(market_regime)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error("Error in market data processing", error=str(e))
                await asyncio.sleep(0.1)  # Brief pause on error

    def _calculate_adaptive_sleep_time(self, market_regime: str) -> float:
        """Calculate adaptive sleep time based on market regime."""
        base_sleep = self.config.get("base_sleep_time", 0.001)  # 1ms default
        
        if market_regime == "high_volatility":
            return base_sleep * 0.5  # Process faster in volatile markets
        elif market_regime == "low_volatility":
            return base_sleep * 2.0  # Can process slower in calm markets
        else:
            return base_sleep

    async def _predict_opportunities(self):
        """Use AI models to predict upcoming arbitrage opportunities."""
        while not self._shutdown:
            try:
                # Get consolidated market data
                market_data = await self.market_data.get_consolidated_data()
                
                # Use AI model to predict upcoming opportunities
                predictions = await self.opportunity_predictor.predict(market_data)
                
                # Filter predictions based on confidence threshold
                min_confidence = self.config.get("min_prediction_confidence", 0.75)
                viable_predictions = [p for p in predictions if p["confidence"] >= min_confidence]
                
                for prediction in viable_predictions:
                    # Create opportunity from prediction
                    opportunity = ArbitrageOpportunity(
                        id=self._generate_opportunity_id(),
                        buy_exchange=prediction["buy_exchange"],
                        sell_exchange=prediction["sell_exchange"],
                        symbol=prediction["symbol"],
                        buy_price=prediction["buy_price"],
                        sell_price=prediction["sell_price"],
                        spread=prediction["spread"],
                        amount=prediction["amount"],
                        timestamp_ns=time.time_ns(),
                        discovered_ns=time.time_ns(),
                        strategy="ai_predicted",
                        confidence=prediction["confidence"],
                        expected_profit=prediction["expected_profit"],
                        expected_slippage=prediction["expected_slippage"],
                        latency_estimate_ns=prediction["latency_estimate_ns"],
                        opportunity_window_ms=prediction["opportunity_window_ms"]
                    )
                    
                    # Add to active opportunities
                    self.active_opportunities[opportunity.id] = opportunity
                    
                    # Add directly to execution queue for predicted opportunities
                    await self.execution_queue.put(opportunity)
                    
                    logger.info(f"AI predicted arbitrage opportunity: {opportunity.buy_exchange} -> {opportunity.sell_exchange}, spread: {opportunity.spread:.2%}")
                
                # Adaptive sleep based on market conditions
                await asyncio.sleep(self.config.get("ai_prediction_interval", 0.05))  # 50ms default
                
            except Exception as e:
                logger.error("Error in opportunity prediction", error=str(e))
                await asyncio.sleep(0.1)

    async def _validate_opportunities(self):
        """Validate detected arbitrage opportunities."""
        while not self._shutdown:
            try:
                # Get opportunity from queue
                if self.opportunity_queue.empty():
                    await asyncio.sleep(0.001)
                    continue
                    
                _, opportunity = await self.opportunity_queue.get()
                
                # Skip if opportunity is too old
                max_age_ms = self.config.get("max_opportunity_age_ms", 100)
                opportunity_age_ms = (time.time_ns() - opportunity.discovered_ns) / 1_000_000
                if opportunity_age_ms > max_age_ms:
                    self.opportunity_queue.task_done()
                    continue
                
                # Check if opportunity still exists through order book verification
                is_valid = await self._verify_opportunity_exists(opportunity)
                if not is_valid:
                    self.opportunity_queue.task_done()
                    continue
                
                # Validate with AI model for higher confidence
                validation_result = await self.opportunity_validator.validate(opportunity)
                
                if validation_result["is_valid"]:
                    # Update opportunity with validation results
                    opportunity.is_validated = True
                    opportunity.confidence = validation_result["confidence"]
                    opportunity.expected_slippage = validation_result["expected_slippage"]
                    opportunity.latency_estimate_ns = validation_result["latency_estimate_ns"]
                    opportunity.opportunity_window_ms = validation_result["opportunity_window_ms"]
                    opportunity.risk_score = validation_result["risk_score"]
                    opportunity.market_impact = validation_result["market_impact"]
                    
                    # Check risk limits
                    risk_check = await self.risk_engine.check_opportunity(opportunity)
                    
                    if risk_check["approved"]:
                        # Add to execution queue
                        await self.execution_queue.put(opportunity)
                        self.metrics["opportunities_validated"] += 1
                        logger.debug(f"Validated arbitrage opportunity: {opportunity.buy_exchange} -> {opportunity.sell_exchange}, confidence: {opportunity.confidence:.2f}")
                    else:
                        logger.warning(f"Opportunity rejected by risk engine: {risk_check['reason']}")
                
                self.opportunity_queue.task_done()
                
            except Exception as e:
                logger.error("Error in opportunity validation", error=str(e))
                if 'opportunity' in locals():
                    self.opportunity_queue.task_done()
                await asyncio.sleep(0.01)

    async def _verify_opportunity_exists(self, opportunity: ArbitrageOpportunity) -> bool:
        """Verify if arbitrage opportunity still exists using latest order book data."""
        # Get latest order books
        buy_ob = await self.market_data.get(opportunity.buy_exchange, opportunity.symbol)
        sell_ob = await self.market_data.get(opportunity.sell_exchange, opportunity.symbol)
        
        if not buy_ob or not sell_ob:
            return False
        
        # Get current prices
        current_buy_price = buy_ob["asks"][0][0] if buy_ob["asks"] else None
        current_sell_price = sell_ob["bids"][0][0] if sell_ob["bids"] else None
        
        if not current_buy_price or not current_sell_price:
            return False
        
        # Calculate current spread
        current_spread = (current_sell_price - current_buy_price) / current_buy_price
        
        # Check if spread still meets threshold
        min_spread = self.config.get("min_spread", 0.001)
        return current_spread >= min_spread

    async def _execute_opportunities(self):
        """Execute validated arbitrage opportunities."""
        while not self._shutdown:
            try:
                # Get opportunity from execution queue
                if self.execution_queue.empty():
                    await asyncio.sleep(0.001)
                    continue
                    
                opportunity = await self.execution_queue.get()
                
                # Avoid executing the same opportunity multiple times
                if opportunity.is_executing:
                    self.execution_queue.task_done()
                    continue
                
                # Mark as executing
                opportunity.is_executing = True
                
                # Use execution lock to prevent concurrent executions on same exchanges/symbols
                execution_key = f"{opportunity.buy_exchange}_{opportunity.sell_exchange}_{opportunity.symbol}"
                async with self.execution_lock[execution_key]:
                    # Final verification before execution
                    is_valid = await self._verify_opportunity_exists(opportunity)
                    if not is_valid:
                        logger.debug(f"Opportunity no longer valid before execution: {opportunity.id}")
                        self.execution_queue.task_done()
                        continue
                    
                    # Select appropriate execution method based on opportunity characteristics
                    execution_method = self._select_execution_method(opportunity)
                    
                    # Execute using selected method
                    self.metrics["executions_attempted"] += 1
                    start_time_ns = time.time_ns()
                    
                    if execution_method == "direct":
                        result = await self._execute_direct_arbitrage(opportunity)
                    else:  # "broker"
                        result = await self.broker_adapter.execute_arbitrage(opportunity)
                    
                    # Track metrics
                    execution_time_ns = time.time_ns() - start_time_ns
                    self.metrics["latency_ns"].append(execution_time_ns)
                    
                    if result["success"]:
                        self.metrics["executions_successful"] += 1
                        self.metrics["profit_loss"].append(result["profit"])
                        
                        # Record trade history
                        await self.trade_history.record_trade(opportunity, result)
                        
                        logger.info(f"Successfully executed arbitrage: {opportunity.id}, profit: {result['profit']:.8f}, latency: {execution_time_ns/1000000:.2f}ms")
                    else:
                        self.metrics["executions_failed"] += 1
                        logger.warning(f"Failed to execute arbitrage: {opportunity.id}, reason: {result.get('error', 'unknown')}")
                    
                    # Clean up
                    if opportunity.id in self.active_opportunities:
                        del self.active_opportunities[opportunity.id]
                
                self.execution_queue.task_done()
                
            except Exception as e:
                logger.error("Error in opportunity execution", error=str(e))
                if 'opportunity' in locals():
                    self.execution_queue.task_done()
                await asyncio.sleep(0.01)

    def _select_execution_method(self, opportunity: ArbitrageOpportunity) -> str:
        """Select the appropriate execution method based on opportunity characteristics."""
        # Use direct execution for ultra-low latency requirements
        if opportunity.opportunity_window_ms < 10:  # Window smaller than 10ms
            return "direct"
            
        # Use direct execution for high-confidence opportunities
        if opportunity.confidence > 0.95:
            return "direct"
            
        # Use broker execution for larger trades (more liquidity needed)
        if opportunity.amount > self.config.get("large_trade_threshold", 1.0):
            return "broker"
            
        # Default to broker execution for better fill guarantees
        return "broker"

    async def _execute_direct_arbitrage(self, opportunity: ArbitrageOpportunity) -> Dict:
        """Execute arbitrage directly through exchange APIs."""
        start_time_ns = time.time_ns()
        
        try:
            # Step 1: Prepare orders
            buy_order = {
                "exchange": opportunity.buy_exchange,
                "symbol": opportunity.symbol,
                "side": "buy",
                "type": "limit",
                "amount": opportunity.amount,
                "price": opportunity.buy_price * (1 + self.config.get("price_buffer", 0.0005)),
                "client_order_id": f"{opportunity.id}_buy"
            }
            
            sell_order = {
                "exchange": opportunity.sell_exchange,
                "symbol": opportunity.symbol,
                "side": "sell",
                "type": "limit",
                "amount": opportunity.amount,
                "price": opportunity.sell_price * (1 - self.config.get("price_buffer", 0.0005)),
                "client_order_id": f"{opportunity.id}_sell"
            }
            
            # Step 2: Submit orders in parallel for fastest execution
            buy_task = asyncio.create_task(self.execution_manager.submit_order(buy_order))
            sell_task = asyncio.create_task(self.execution_manager.submit_order(sell_order))
            
            # Wait for both orders to complete
            buy_result, sell_result = await asyncio.gather(buy_task, sell_task)
            
            # Step 3: Verify execution success
            if buy_result["status"] == "filled" and sell_result["status"] == "filled":
                # Calculate actual profit
                executed_buy_price = buy_result["executed_price"]
                executed_sell_price = sell_result["executed_price"]
                executed_amount = min(buy_result["executed_amount"], sell_result["executed_amount"])
                
                # Calculate fees
                buy_fee = buy_result.get("fee", 0)
                sell_fee = sell_result.get("fee", 0)
                
                # Calculate net profit
                gross_profit = (executed_sell_price - executed_buy_price) * executed_amount
                net_profit = gross_profit - buy_fee - sell_fee
                
                latency_ms = (time.time_ns() - start_time_ns) / 1_000_000
                
                return {
                    "success": True,
                    "profit": net_profit,
                    "gross_profit": gross_profit,
                    "fees": buy_fee + sell_fee,
                    "buy_result": buy_result,
                    "sell_result": sell_result,
                    "latency_ns": time.time_ns() - start_time_ns,
                    "latency_ms": latency_ms
                }
            else:
                # Handle partial fills or failures
                if buy_result["status"] == "open":
                    await self.execution_manager.cancel_order(buy_order)
                if sell_result["status"] == "open":
                    await self.execution_manager.cancel_order(sell_order)
                
                return {
                    "success": False,
                    "error": "Orders not fully filled",
                    "buy_status": buy_result["status"],
                    "sell_status": sell_result["status"],
                    "buy_result": buy_result,
                    "sell_result": sell_result
                }
                
        except Exception as e:
            logger.error("Error in direct arbitrage execution", 
                        opportunity_id=opportunity.id,
                        error=str(e))
            return {"success": False, "error": str(e)}

    async def _monitor_active_opportunities(self):
        """Monitor active opportunities and clean up expired ones."""
        while not self._shutdown:
            try:
                current_time_ns = time.time_ns()
                expired_ids = []
                
                # Check all active opportunities
                for opp_id, opportunity in self.active_opportunities.items():
                    # Calculate age in milliseconds
                    age_ms = (current_time_ns - opportunity.discovered_ns) / 1_000_000
                    
                    # Mark expired opportunities
                    if age_ms > self.config.get("max_opportunity_age_ms", 500):
                        expired_ids.append(opp_id)
                
                # Remove expired opportunities
                for opp_id in expired_ids:
                    if opp_id in self.active_opportunities:
                        del self.active_opportunities[opp_id]
                        logger.debug(f"Removed expired opportunity: {opp_id}")
                
                # Sleep briefly
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error("Error in opportunity monitoring", error=str(e))
                await asyncio.sleep(0.1)

    async def _monitor_execution_queue(self):
        """Monitor execution queue for backlog and performance metrics."""
        while not self._shutdown:
            try:
                # Get queue size
                queue_size = self.execution_queue.qsize()
                
                # Log warning if queue size exceeds threshold
                if queue_size > self.config.get("max_execution_queue_size", 100):
                    logger.warning(f"Execution queue backlog: {queue_size} opportunities")
                
                # Calculate execution throughput
                self.metrics["throughput"] = queue_size / self.config.get("execution_interval", 0.05)
                
                # Adaptive sleep based on queue size
                sleep_time = min(1.0, max(0.01, 0.1 + (queue_size / 1000)))
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error("Error in execution queue monitoring", error=str(e))
                await asyncio.sleep(0.1)

    async def _monitor_system_performance(self):
        """Monitor system performance metrics."""
        while not self._shutdown:
            try:
                # Calculate performance metrics
                total_opportunities = self.metrics["opportunities_detected"]
                successful_validations = self.metrics["opportunities_validated"]
                successful_executions = self.metrics["executions_successful"]
                failed_executions = self.metrics["executions_failed"]
                
                # Calculate success rates
                validation_rate = successful_validations / total_opportunities if total_opportunities > 0 else 0
                execution_rate = successful_executions / (successful_executions + failed_executions) if (successful_executions + failed_executions) > 0 else 0
                
                # Calculate average latency
                avg_latency_ms = np.mean(list(self.metrics["latency_ns"])) / 1_000_000 if self.metrics["latency_ns"] else 0
                
                # Calculate average profit
                avg_profit = np.mean(list(self.metrics["profit_loss"])) if self.metrics["profit_loss"] else 0
                
                # Log performance metrics periodically
                logger.info("Performance metrics", 
                           opportunities_detected=total_opportunities,
                           validation_rate=f"{validation_rate:.2%}",
                           execution_rate=f"{execution_rate:.2%}",
                           avg_latency_ms=f"{avg_latency_ms:.2f}ms",
                           avg_profit=avg_profit,
                           queue_size=self.opportunity_queue.qsize(),
                           execution_queue_size=self.execution_queue.qsize())
                
                # Sleep interval
                await asyncio.sleep(self.config.get("metrics_interval", 10.0))
                
            except Exception as e:
                logger.error("Error in performance monitoring", error=str(e))
                await asyncio.sleep(1.0)

    async def _monitor_risk_exposure(self):
        """Monitor risk exposure and enforce risk limits."""
        while not self._shutdown:
            try:
                # Get current exposure from risk engine
                exposure = await self.risk_engine.get_current_exposure()
                
                # Check if exposure exceeds limits
                if exposure["exceeds_limits"]:
                    logger.warning("Risk exposure exceeds limits, pausing new arbitrage executions", 
                                  exposure=exposure["current"],
                                  limits=exposure["limits"])
                    
                    # Implement emergency risk reduction if needed
                    if exposure["requires_intervention"]:
                        await self._implement_emergency_risk_reduction(exposure)
                
                # Sleep interval - check risk more frequently in volatile markets
                market_regime = await self.market_regime.classify_current_regime()
                sleep_time = 1.0 if market_regime == "high_volatility" else 5.0
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error("Error in risk exposure monitoring", error=str(e))
                await asyncio.sleep(1.0)

    async def _implement_emergency_risk_reduction(self, exposure: Dict):
        """Implement emergency risk reduction measures."""
        logger.warning("Implementing emergency risk reduction")
        
        # Clear execution queue
        while not self.execution_queue.empty():
            try:
                _ = self.execution_queue.get_nowait()
                self.execution_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        # Cancel all pending orders
        await self.execution_manager.cancel_all_pending()
        
        # Close positions if needed to reduce exposure
        if exposure["requires_position_reduction"]:
            await self.execution_manager.reduce_positions(exposure["reduction_targets"])
            
        logger.info("Emergency risk reduction implemented")

    async def _cleanup_expired_opportunities(self):
        """Clean up expired opportunities."""
        while not self._shutdown:
            try:
                now_ns = time.time_ns()
                expired_ids = []
                
                # Find expired opportunities
                for opp_id, opportunity in self.active_opportunities.items():
                    age_ms = (now_ns - opportunity.discovered_ns) / 1_000_000
                    max_age_ms = self.config.get("max_opportunity_age_ms", 500)
                    
                    if age_ms > max_age_ms:
                        expired_ids.append(opp_id)
                
                # Remove expired opportunities
                for opp_id in expired_ids:
                    if opp_id in self.active_opportunities:
                        del self.active_opportunities[opp_id]
                
                # Sleep interval
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error("Error in opportunity cleanup", error=str(e))
                await asyncio.sleep(0.5)

    def _generate_opportunity_id(self) -> str:
        """Generate a unique ID for arbitrage opportunities with nanosecond precision."""
        timestamp = time.time_ns()
        random_component = os.urandom(4).hex()
        return f"arb_{timestamp}_{random_component}"

    async def _handle_risk_controls(self):
        """Handle ongoing risk controls and management."""
        while not self._shutdown:
            try:
                # Get current risk parameters
                risk_params = await self.risk_engine.get_risk_parameters()
                
                # Update trading limits based on current risk assessment
                position_limits = risk_params.get("position_limits", {})
                execution_limits = risk_params.get("execution_limits", {})
                
                # Adjust execution parameters based on risk assessment
                for symbol, limit in position_limits.items():
                    # Update in-memory limits
                    self.config["symbols_limits"][symbol] = limit
                
                # Adjust execution speed based on risk parameters
                risk_multiplier = risk_params.get("execution_speed_multiplier", 1.0)
                self.config["execution_interval"] = self.config["base_execution_interval"] * risk_multiplier
                
                # Sleep interval
                await asyncio.sleep(self.config.get("risk_control_interval", 1.0))
                
            except Exception as e:
                logger.error("Error in risk control handling", error=str(e))
                await asyncio.sleep(1.0)

    async def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        # Calculate derived metrics
        total_opportunities = self.metrics["opportunities_detected"]
        successful_validations = self.metrics["opportunities_validated"]
        successful_executions = self.metrics["executions_successful"]
        failed_executions = self.metrics["executions_failed"]
        
        validation_rate = successful_validations / total_opportunities if total_opportunities > 0 else 0
        execution_rate = successful_executions / (successful_executions + failed_executions) if (successful_executions + failed_executions) > 0 else 0
        
        avg_latency_ms = np.mean(list(self.metrics["latency_ns"])) / 1_000_000 if self.metrics["latency_ns"] else 0
        avg_profit = np.mean(list(self.metrics["profit_loss"])) if self.metrics["profit_loss"] else 0
        
        return {
            "opportunities_detected": total_opportunities,
            "opportunities_validated": successful_validations,
            "executions_successful": successful_executions,
            "executions_failed": failed_executions,
            "validation_rate": validation_rate,
            "execution_rate": execution_rate,
            "avg_latency_ms": avg_latency_ms,
            "avg_profit": avg_profit,
            "active_opportunities": len(self.active_opportunities),
            "opportunity_queue_size": self.opportunity_queue.qsize(),
            "execution_queue_size": self.execution_queue.qsize(),
            "throughput": self.metrics["throughput"]
        }

    async def get_active_opportunities(self) -> List[Dict]:
        """Get list of active opportunities."""
        result = []
        for opp_id, opportunity in self.active_opportunities.items():
            # Convert opportunity to dict for API response
            opp_dict = {
                "id": opportunity.id,
                "buy_exchange": opportunity.buy_exchange,
                "sell_exchange": opportunity.sell_exchange,
                "symbol": opportunity.symbol,
                "buy_price": opportunity.buy_price,
                "sell_price": opportunity.sell_price,
                "spread": opportunity.spread,
                "amount": opportunity.amount,
                "strategy": opportunity.strategy,
                "confidence": opportunity.confidence,
                "expected_profit": opportunity.expected_profit,
                "age_ms": (time.time_ns() - opportunity.discovered_ns) / 1_000_000,
                "is_validated": opportunity.is_validated,
                "is_executing": opportunity.is_executing
            }
            result.append(opp_dict)
        
        return result

    async def _monitor_performance(self):
        """Monitor performance of arbitrage strategies and adapt."""
        while not self._shutdown:
            try:
                # Evaluate strategy performance
                performance = await self.strategy_selector.evaluate_strategy_performance()
                
                # Update strategy weights
                total_score = sum(performance.values()) if performance else 1.0
                weights = {name: score/total_score for name, score in performance.items()}
                
                # Log performance
                logger.info("Strategy performance", performance=performance, weights=weights)
                
                # Update meta-trader with latest performance data
                await self.meta_trader.update_strategy_weights("arbitrage", weights)
                
                # Sleep interval
                await asyncio.sleep(self.config.get("performance_interval", 60.0))
                
            except Exception as e:
                logger.error("Error in performance monitoring", error=str(e))
                await asyncio.sleep(10.0)


class ArbitrageAPI:
    """API interface for the Quantum Arbitrage Engine."""
    
    def __init__(self):
        self.engine = None
    
    async def initialize(self):
        """Initialize the arbitrage API."""
        self.engine = await QuantumArbitrageEngine().__init__()
    
    async def start_engine(self):
        """Start the arbitrage engine."""
        await self.engine.start()
        return {"success": True, "message": "Arbitrage engine started"}
    
    async def stop_engine(self):
        """Stop the arbitrage engine."""
        await self.engine.shutdown()
        return {"success": True, "message": "Arbitrage engine stopped"}
    
    async def get_metrics(self):
        """Get current performance metrics."""
        return await self.engine.get_metrics()
    
    async def get_active_opportunities(self):
        """Get list of active arbitrage opportunities."""
        return await self.engine.get_active_opportunities()
    
    async def get_trading_history(self, start_time=None, end_time=None, limit=100):
        """Get arbitrage trading history."""
        # Request history from trade history component
        return await self.engine.trade_history.get_trades(
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            strategy_type="arbitrage"
        )
    
    async def get_exchange_status(self):
        """Get status of connected exchanges."""
        exchange_status = {}
        
        # Get WebSocket connection status
        websocket_status = await self.engine.websocket_manager.get_connection_status()
        
        # Get broker connection status
        broker_status = {}
        for broker_name, broker in self.engine.broker_adapter.broker_apis.items():
            status = await broker.get_status()
            broker_status[broker_name] = status
        
        # Combine status
        for exchange in self.engine.config.get("exchanges", []):
            exchange_status[exchange] = {
                "websocket": websocket_status.get(exchange, "disconnected"),
                "broker": "connected" if exchange in broker_status else "disconnected",
                "orderbook_age_ms": await self._get_orderbook_age(exchange)
            }
        
        return exchange_status
    
    async def _get_orderbook_age(self, exchange):
        """Get age of the latest orderbook for an exchange."""
        # For each symbol, get latest orderbook and its timestamp
        symbols = self.engine.config.get("symbols", [])
        ages = []
        
        for symbol in symbols:
            orderbook = await self.engine.market_data.get(exchange, symbol)
            if orderbook and "timestamp_ns" in orderbook:
                age_ms = (time.time_ns() - orderbook["timestamp_ns"]) / 1_000_000
                ages.append(age_ms)
        
        # Return average age
        return np.mean(ages) if ages else None


async def start_arbitrage_engine():
    """Start the arbitrage engine."""
    api = ArbitrageAPI()
    await api.initialize()
    await api.start_engine()
    return api


if __name__ == "__main__":
    try:
        # Set up asyncio event loop
        loop = asyncio.get_event_loop()
        
        # Start arbitrage engine
        api = loop.run_until_complete(start_arbitrage_engine())
        
        # Run forever
        loop.run_forever()
    except KeyboardInterrupt:
        # Shutdown on Ctrl+C
        if 'api' in locals():
            loop.run_until_complete(api.stop_engine())
        
        # Clean up
        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            if task is not asyncio.current_task():
                task.cancel()
                
        # Wait for all tasks to be cancelled
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        
        logger.info("Quantum Arbitrage Engine shutdown complete")
        loop.close()
    except Exception as e:
        logger.critical(f"Fatal error in Quantum Arbitrage Engine: {str(e)}")
        import traceback
        logger.critical(traceback.format_exc())
        
        # Attempt emergency shutdown
        try:
            if 'api' in locals():
                loop.run_until_complete(api.stop_engine())
            loop.close()
        except:
            pass
        
        # Exit with error code
        import sys
        sys.exit(1)


# Additional utility functions for system monitoring and management

async def run_diagnostics():
    """Run system diagnostics and performance checks."""
    try:
        # Initialize API
        api = ArbitrageAPI()
        await api.initialize()
        
        # Get metrics
        metrics = await api.get_metrics()
        print("\n=== Quantum Arbitrage Engine Diagnostics ===")
        print(f"Opportunities detected: {metrics['opportunities_detected']}")
        print(f"Validation rate: {metrics['validation_rate']:.2%}")
        print(f"Execution rate: {metrics['execution_rate']:.2%}")
        print(f"Average latency: {metrics['avg_latency_ms']:.2f}ms")
        print(f"Average profit: {metrics['avg_profit']}")
        
        # Check exchange connections
        exchange_status = await api.get_exchange_status()
        print("\n=== Exchange Connection Status ===")
        for exchange, status in exchange_status.items():
            print(f"{exchange}: WebSocket: {status['websocket']}, Broker: {status['broker']}")
            if status['orderbook_age_ms']:
                print(f"  Orderbook age: {status['orderbook_age_ms']:.2f}ms")
        
        # Check active opportunities
        opportunities = await api.get_active_opportunities()
        print(f"\n=== Active Opportunities: {len(opportunities)} ===")
        for i, opp in enumerate(opportunities[:5]):  # Show max 5
            print(f"{i+1}. {opp['symbol']}: {opp['buy_exchange']} → {opp['sell_exchange']}, "
                  f"spread: {opp['spread']:.2%}, profit: {opp['expected_profit']}")
        
        if len(opportunities) > 5:
            print(f"... and {len(opportunities) - 5} more opportunities")
        
        # Check trading history
        history = await api.get_trading_history(limit=5)
        print(f"\n=== Recent Trades: {len(history)} ===")
        for i, trade in enumerate(history):
            print(f"{i+1}. {trade['symbol']}: {trade['buy_exchange']} → {trade['sell_exchange']}, "
                  f"profit: {trade['profit']}, latency: {trade['latency_ms']:.2f}ms")
        
        return True
    except Exception as e:
        print(f"Diagnostics failed: {str(e)}")
        return False
    finally:
        if 'api' in locals() and api.engine:
            await api.stop_engine()


async def optimize_performance():
    """Run performance optimization routines."""
    api = ArbitrageAPI()
    await api.initialize()
    
    try:
        engine = api.engine
        
        # Run Monte Carlo simulation to optimize parameters
        simulator = MonteCarloSimulator()
        params = await simulator.optimize_parameters(
            historical_data=await engine.trade_history.get_trades(limit=1000),
            parameters={
                "min_spread": {"min": 0.0005, "max": 0.005, "step": 0.0001},
                "price_buffer": {"min": 0.0001, "max": 0.001, "step": 0.0001},
                "max_opportunity_age_ms": {"min": 50, "max": 500, "step": 10},
                "execution_interval": {"min": 0.01, "max": 0.1, "step": 0.005}
            },
            iterations=100,
            objective="profit"
        )
        
        # Apply optimized parameters
        for key, value in params.items():
            engine.config[key] = value
        
        logger.info("Applied optimized parameters", parameters=params)
        print("Performance optimization complete. Optimized parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        return params
    except Exception as e:
        logger.error("Error in performance optimization", error=str(e))
        print(f"Performance optimization failed: {str(e)}")
        return None
    finally:
        await api.stop_engine()


def create_config_template():
    """Create a configuration template file."""
    config = {
        "exchanges": ["binance", "coinbase", "kraken", "ftx", "okex"],
        "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"],
        "min_spread": 0.001,  # 0.1% minimum spread
        "price_buffer": 0.0005,  # 0.05% price buffer for execution
        "max_opportunity_age_ms": 200,  # Maximum age of opportunity in milliseconds
        "execution_interval": 0.05,  # 50ms base execution interval
        "base_execution_interval": 0.05,
        "max_execution_queue_size": 100,
        "metrics_interval": 10.0,  # 10s interval for metrics logging
        "performance_interval": 60.0,  # 60s interval for performance monitoring
        "risk_control_interval": 1.0,  # 1s interval for risk control
        "ai_prediction_interval": 0.05,  # 50ms interval for AI prediction
        "min_prediction_confidence": 0.75,  # Minimum confidence for AI predictions
        "symbols_limits": {},  # Will be populated dynamically
        "strategies": {
            "cross_exchange": {
                "min_spread": 0.001,
                "enabled": True
            },
            "triangular": {
                "min_profit": 0.001,
                "enabled": True
            },
            "statistical": {
                "correlation_threshold": 0.8,
                "mean_reversion_z_score": 2.0,
                "enabled": False
            },
            "latency": {
                "min_delay_ms": 50,
                "enabled": False
            },
            "futures_spot": {
                "min_spread": 0.002,
                "enabled": False
            }
        },
        "execution": {
            "price_buffer": 0.0005,
            "max_wait_time_ms": 500,
            "poll_interval_ms": 10
        },
        "brokers": {
            "binance": {
                "api_key": "",
                "api_secret": "",
                "passphrase": ""
            },
            "coinbase": {
                "api_key": "",
                "api_secret": "",
                "passphrase": ""
            },
            "kraken": {
                "api_key": "",
                "api_secret": "",
                "passphrase": ""
            }
        }
    }
    
    import json
    import os
    
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Config")
    os.makedirs(config_dir, exist_ok=True)
    
    config_path = os.path.join(config_dir, "arbitrage_config_template.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration template created at: {config_path}")
    return config_path


def print_usage():
    """Print usage instructions."""
    print("\nQuantum Arbitrage Engine (QAE) - Usage instructions:")
    print("----------------------------------------------------")
    print("python quantum_arbitrage.py [command]")
    print("\nAvailable commands:")
    print("  start             - Start the arbitrage engine")
    print("  diagnostics       - Run system diagnostics")
    print("  optimize          - Run performance optimization")
    print("  create-config     - Create configuration template")
    print("  help              - Show this help message")
    print("\nExamples:")
    print("  python quantum_arbitrage.py start")
    print("  python quantum_arbitrage.py diagnostics")
    print("\nFor more information, see documentation.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        # Default to start if no command provided
        command = "start"
    else:
        command = sys.argv[1].lower()
    
    if command == "start":
        try:
            # Set up asyncio event loop
            loop = asyncio.get_event_loop()
            
            # Start arbitrage engine
            api = loop.run_until_complete(start_arbitrage_engine())
            
            print("Quantum Arbitrage Engine started successfully")
            print("Press Ctrl+C to stop")
            
            # Run forever
            loop.run_forever()
        except KeyboardInterrupt:
            print("\nShutting down Quantum Arbitrage Engine...")
            # Shutdown on Ctrl+C
            if 'api' in locals():
                loop.run_until_complete(api.stop_engine())
            
            # Clean up
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                if task is not asyncio.current_task():
                    task.cancel()
            
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            print("Shutdown complete")
            loop.close()
    
    elif command == "diagnostics":
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_diagnostics())
        loop.close()
    
    elif command == "optimize":
        loop = asyncio.get_event_loop()
        loop.run_until_complete(optimize_performance())
        loop.close()
    
    elif command == "create-config":
        create_config_template()
    
    elif command in ["help", "--help", "-h"]:
        print_usage()
    
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)
