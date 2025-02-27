# Core/trading/execution/market_impact.py
import numpy as np
import asyncio
from typing import Dict
from decimal import Decimal
from utils.helpers.error_handler import APIException, log_security_event
from Core.data.realtime.market_data import get_order_book_depth
from Core.trading.risk.risk_management import validate_position_size
from Core.trading.hft.liquidity_manager import estimate_hidden_liquidity
from utils.logging.structured_logger import log_market_impact

class MarketImpactAnalyzer:
    """Enhanced market impact analysis with real-time liquidity integration and risk checks"""
    
    def __init__(self, order_book_analyzer, liquidity_calculator):
        self.order_book_analyzer = order_book_analyzer
        self.liquidity_calculator = liquidity_calculator
        self._slippage_model = self._load_slippage_model()

    async def analyze_impact(self, trade_size: Decimal, asset: str) -> Dict:
        """Full impact analysis pipeline with security checks"""
        try:
            # Validate input through security layer
            await self._validate_input(trade_size, asset)
            
            # Get real-time market data from centralized source
            market_data = await self._get_market_data(asset)
            
            # Parallel execution of critical components
            liquidity, order_book_depth, institutional_flow = await asyncio.gather(
                self.liquidity_calculator.get_market_liquidity(asset),
                self.order_book_analyzer.get_depth(asset),
                self._get_institutional_order_flow(asset),
                return_exceptions=True
            )

            # Calculate impacts with multiple fallback strategies
            results = {
                'expected_slippage': self._calculate_adaptive_slippage(trade_size, liquidity, market_data),
                'market_impact_cost': self._estimate_impact_cost(trade_size, order_book_depth, institutional_flow),
                'optimal_execution': self._generate_ai_optimized_schedule(trade_size, liquidity, institutional_flow),
                'risk_approved': await validate_position_size(trade_size, asset)
            }
            
            log_market_impact(asset, results)
            return results

        except Exception as e:
            log_security_event("MARKET_IMPACT_ERROR", f"{asset}: {str(e)}")
            raise APIException("Market impact analysis failed") from e

    def _calculate_adaptive_slippage(self, trade_size: Decimal, liquidity: Decimal, market_data: Dict) -> Decimal:
        """Hybrid slippage model combining current liquidity and historical volatility"""
        if liquidity <= 0:
            return Decimal('Infinity')
            
        # Get volatility from historical data module
        hist_volatility = market_data.get('30d_volatility', 0.2)
        
        # Dynamic slippage coefficient based on market regime
        slippage_coeff = max(0.15, hist_volatility ** 0.5) 
        slippage = (float(trade_size) / float(liquidity)) ** slippage_coeff
        return Decimal(str(round(slippage, 6)))

    def _estimate_impact_cost(self, trade_size: Decimal, depth: Decimal, institutional_flow: Decimal) -> Decimal:
        """Impact cost estimation with dark pool liquidity adjustment"""
        hidden_liquidity = estimate_hidden_liquidity()  # From liquidity_manager
        adjusted_depth = depth + hidden_liquidity
        impact = float(trade_size) / (float(adjusted_depth) + float(institutional_flow) + 1e-6)
        return Decimal(str(round(impact, 6)))

    def _generate_ai_optimized_schedule(self, trade_size: Decimal, liquidity: Decimal, institutional_flow: Decimal) -> str:
        """Execution strategy selection integrated with strategy orchestrator"""
        liquidity_ratio = float(liquidity) / float(trade_size)
        
        if liquidity_ratio < 0.5:
            return "TWAP Execution"
        elif institutional_flow > trade_size * Decimal('1.5'):
            return "Dark Pool Aggregation"
        elif liquidity_ratio > 2.0:
            return "Immediate Execution"
        else:
            return "VWAP Execution"

    async def _get_institutional_order_flow(self, asset: str) -> Decimal:
        """Institutional flow detection using multiple data sources"""
        try:
            # Get real institutional data from trade monitor
            flow_data = await self.order_book_analyzer.get_institutional_flow(asset)
            return Decimal(str(flow_data.get('hidden_volume', 0.0)))
        except Exception as e:
            log_security_event("INSTITUTIONAL_FLOW_ERROR", str(e))
            return Decimal('0')

    async def _validate_input(self, trade_size: Decimal, asset: str):
        """Security validation for all inputs"""
        if trade_size <= Decimal('0'):
            raise ValueError("Invalid trade size")
        if not asset.isalnum():
            raise ValueError("Potential injection attempt detected")

    def _load_slippage_model(self):
        """Load pre-trained model from strategy orchestrator"""
        # Integrated with src/Core/trading/strategies/strategy_orchestrator.py
        return get_optimal_slippage_model()  # From strategy orchestrator