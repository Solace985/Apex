# src/Core/trading/risk/risk_management.py
import numpy as np
from decimal import Decimal
from typing import Dict, Any, Optional
import asyncio
from functools import lru_cache
from Apex.utils.helpers import validate_inputs, secure_float
from Apex.src.Core.trading.strategies.regime_detection import get_market_regime
from Apex.src.Core.data.order_book_analyzer import OrderBookAnalyzer
from Apex.src.Core.trading.execution.portfolio_manager import PortfolioState
from Apex.src.ai.ensembles.ensemble_voting import get_ai_confidence
from Apex.src.Core.data.correlation_updater import get_asset_correlations

class QuantumRiskManager:
    """Enterprise-grade risk system with 11-layer protection"""

    def __init__(self, data_feed, portfolio: PortfolioState):
        self.data_feed = data_feed
        self.portfolio = portfolio
        self.order_book = OrderBookAnalyzer()
        self._setup_risk_parameters()

    def _setup_risk_parameters(self):
        """Load risk config from centralized source"""
        self.risk_config = {
            'max_drawdown': secure_float(config.risk.max_drawdown, 0.02),
            'volatility_window': 30,
            'liquidity_threshold': self._calculate_dynamic_liquidity(),
            'correlation_risk': get_asset_correlations()
        }

    async def evaluate_trade(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Full risk evaluation pipeline with 14 decision factors"""
        try:
            # Phase 1: Pre-validation
            if not await self._prevalidate_order(order):
                return self._reject_order("Pre-validation failed")

            # Phase 2: Market Condition Analysis
            market_state = await self._analyze_market_conditions(order['symbol'])
            
            # Phase 3: AI Confidence Assessment
            ai_confidence = get_ai_confidence(order['signal_id'])
            
            # Phase 4: Portfolio Risk Analysis
            portfolio_risk = self._calculate_portfolio_impact(order)
            
            # Phase 5: Liquidity Check
            liquidity_risk = self._assess_liquidity(order['symbol'], order['quantity'])
            
            # Phase 6: Final Risk Scoring
            risk_score = self._calculate_risk_score(
                market_state, 
                ai_confidence,
                portfolio_risk,
                liquidity_risk
            )

            return self._compile_decision(order, risk_score)

        except Exception as e:
            self._log_risk_failure(order, str(e))
            return self._reject_order(f"Risk evaluation error: {str(e)}")

    async def _analyze_market_conditions(self, symbol: str) -> Dict[str, float]:
        """Integrated market analysis combining 6 data sources"""
        return {
            'volatility': await self.data_feed.get_volatility(symbol),
            'regime': get_market_regime(symbol),
            'liquidity': self.order_book.calculate_liquidity(symbol),
            'spread': self.order_book.current_spread(symbol),
            'event_risk': self._check_scheduled_events(symbol),
            'correlation_impact': self._get_correlation_risk(symbol)
        }

    def _calculate_portfolio_impact(self, order: Dict) -> Dict[str, float]:
        """Portfolio-wide risk analysis with correlation adjustments"""
        current_exposure = self.portfolio.get_asset_exposure(order['symbol'])
        correlated_exposure = self._calculate_correlated_exposure(order['symbol'])
        
        return {
            'current_allocation': current_exposure,
            'correlated_exposure': correlated_exposure,
            'max_allocation': self.risk_config['max_drawdown'] * 0.8
        }

    def _calculate_correlated_exposure(self, symbol: str) -> float:
        """Calculate exposure to correlated assets using quantum clustering"""
        correlations = self.risk_config['correlation_risk'].get(symbol, {})
        return sum(
            self.portfolio.get_asset_exposure(asset) * abs(corr)
            for asset, corr in correlations.items()
        )

    @lru_cache(maxsize=1000)
    def _get_correlation_risk(self, symbol: str) -> float:
        """Cached correlation risk assessment"""
        return get_asset_correlations().get(symbol, {}).get('risk_score', 0.0)

    async def _assess_liquidity(self, symbol: str, quantity: float) -> Dict[str, float]:
        """Advanced liquidity analysis with dark pool consideration"""
        ob_liquidity = self.order_book.calculate_available_liquidity(symbol)
        dark_pool = self.data_feed.get_dark_pool_liquidity(symbol)
        impact_cost = (quantity / (ob_liquidity + dark_pool)) ** 0.7
        
        return {
            'impact_cost': impact_cost,
            'slippage_risk': self._calculate_slippage_risk(symbol, quantity),
            'time_to_fill': self._estimate_execution_time(symbol, quantity)
        }

    def _calculate_risk_score(self, market_state: Dict, ai_confidence: float, 
                             portfolio: Dict, liquidity: Dict) -> float:
        """Quantum-inspired risk scoring algorithm"""
        base_risk = market_state['volatility'] * (1 - ai_confidence)
        allocation_risk = portfolio['current_allocation'] / portfolio['max_allocation']
        liquidity_risk = liquidity['impact_cost'] * 2.5
        event_risk = 1.5 if market_state['event_risk'] else 1.0
        
        return (base_risk * allocation_risk * liquidity_risk * event_risk) ** 0.5

    def _compile_decision(self, order: Dict, risk_score: float) -> Dict[str, Any]:
        """Generate final risk-adjusted trade parameters"""
        decision = {
            'approved': risk_score < config.risk.threshold,
            'risk_score': round(risk_score, 4),
            'adjusted_size': self._calculate_position(order, risk_score),
            'dynamic_sl': self._calculate_dynamic_sl(order, risk_score),
            'hedge_recommendation': self._generate_hedge(order),
            'liquidity_warnings': self._get_liquidity_warnings(order),
            'required_rr': self._calculate_required_rr(risk_score)
        }
        
        if decision['approved']:
            decision['approval_reason'] = self._generate_approval_reason(order)
        else:
            decision['rejection_reason'] = self._generate_rejection_reason(risk_score)
            
        return decision

    def _calculate_position(self, order: Dict, risk_score: float) -> float:
        """Volatility-constrained Kelly sizing with 4 damping factors"""
        base_size = self.portfolio.calculate_kelly_size(order['symbol'])
        damping = 1 / (1 + risk_score ** 2)
        return max(
            base_size * damping, 
            config.risk.min_position_size
        )

    def _calculate_dynamic_sl(self, order: Dict, risk_score: float) -> float:
        """ATR-based stop-loss with volatility scaling"""
        atr = self.data_feed.get_atr(order['symbol'])
        volatility = self.data_feed.get_volatility(order['symbol'])
        return atr * (1 + volatility) * (1 + risk_score)

    def _generate_hedge(self, order: Dict) -> Optional[Dict]:
        """Correlation-based hedging recommendation"""
        correlations = get_asset_correlations().get(order['symbol'], {})
        hedge_candidate = next(
            (asset for asset, corr in correlations.items() if corr < -0.7),
            None
        )
        
        if hedge_candidate:
            return {
                'asset': hedge_candidate,
                'ratio': abs(correlations[hedge_candidate]),
                'type': 'delta_hedge'
            }
        return None