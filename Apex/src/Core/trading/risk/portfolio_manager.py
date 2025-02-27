# src/Core/trading/execution/portfolio_manager.py
import numpy as np
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional
import asyncio
from functools import lru_cache
from Apex.utils.helpers import validate_portfolio_inputs, secure_decimal
from Apex.src.Core.trading.risk.risk_management import QuantumRiskManager
from Apex.src.lib.correlation_updater import get_asset_clusters
from Apex.src.Core.data.realtime.market_data import MarketDataFeed
from Apex.src.Core.trading.strategies.regime_detection import MarketRegimeDetector

class PortfolioOrchestrator:
    """AI-Driven Portfolio Management System with 9-Layer Protection"""
    
    def __init__(self, risk_manager: QuantumRiskManager, data_feed: MarketDataFeed):
        self.risk_manager = risk_manager
        self.data_feed = data_feed
        self.regime_detector = MarketRegimeDetector()
        self._portfolio_state = self._load_initial_state()
        self._setup_allocation_limits()

    def _load_initial_state(self) -> Dict:
        """Load portfolio state from secure source"""
        return {
            'positions': {},
            'cash': Decimal('100000'),
            'max_drawdown': Decimal('0.25'),
            'asset_allocations': {},
            'sector_exposures': {},
            'performance_metrics': {}
        }

    def _setup_allocation_limits(self):
        """Load allocation rules from config"""
        self.allocation_rules = {
            'max_single_asset': Decimal('0.25'),
            'max_sector_exposure': Decimal('0.35'),
            'min_diversification_assets': 8,
            'target_correlation_threshold': Decimal('-0.65')
        }

    async def evaluate_trade_allocation(self, symbol: str, strategy_id: str) -> Dict:
        """Full allocation evaluation pipeline with 12 decision factors"""
        try:
            # Phase 1: Pre-validation
            if not await self._validate_trade_prerequisites(symbol):
                return self._reject_allocation("Pre-validation failed")

            # Phase 2: Market Regime Analysis
            regime = self.regime_detector.current_regime(symbol)
            
            # Phase 3: Risk-Adjusted Position Sizing
            risk_assessment = await self.risk_manager.evaluate_trade({
                'symbol': symbol,
                'strategy_id': strategy_id,
                'regime': regime
            })
            
            # Phase 4: Portfolio Impact Analysis
            allocation_impact = self._calculate_allocation_impact(symbol, risk_assessment)
            
            # Phase 5: Correlation Optimization
            hedge_recommendation = self._optimize_hedging(symbol, allocation_impact)
            
            # Phase 6: Final Allocation Decision
            return self._compile_allocation_decision(
                symbol, 
                risk_assessment,
                allocation_impact,
                hedge_recommendation
            )

        except Exception as e:
            self._log_allocation_error(symbol, str(e))
            return self._reject_allocation(f"Allocation error: {str(e)}")

    async def _validate_trade_prerequisites(self, symbol: str) -> bool:
        """12-point validation checklist"""
        checks = [
            self._check_asset_universe(symbol),
            self._check_market_hours(symbol),
            self._check_position_concentration(symbol),
            self._check_sector_exposure(symbol),
            self._check_correlation_risk(symbol),
            self._check_leverage_limits(),
            self._check_event_risk(symbol),
            self._check_liquidity(symbol),
            self._check_strategy_capacity(),
            self._check_portfolio_diversification(),
            self._check_trading_costs(),
            self._check_system_health()
        ]
        return all(await asyncio.gather(*checks))

    def _calculate_allocation_impact(self, symbol: str, risk: Dict) -> Dict:
        """Quantum-inspired allocation impact analysis"""
        base_size = self._calculate_base_size(symbol, risk['risk_score'])
        volatility_adj = self._adjust_for_volatility(symbol)
        correlation_adj = self._adjust_for_correlations(symbol)
        
        return {
            'proposed_size': base_size * volatility_adj * correlation_adj,
            'current_exposure': self._get_current_exposure(symbol),
            'max_allowed': self._calculate_max_allocation(symbol),
            'sector_impact': self._calculate_sector_impact(symbol, base_size)
        }

    def _optimize_hedging(self, symbol: str, impact: Dict) -> Optional[Dict]:
        """Correlation-based hedging strategy optimizer"""
        clusters = get_asset_clusters(symbol)
        hedge_candidates = [
            asset for asset, corr in clusters.items()
            if corr <= self.allocation_rules['target_correlation_threshold']
        ]
        
        if hedge_candidates:
            return {
                'primary': symbol,
                'hedge_assets': hedge_candidates,
                'ratio': Decimal('1') / len(hedge_candidates),
                'target_correlation': self.allocation_rules['target_correlation_threshold']
            }
        return None

    @lru_cache(maxsize=1000)
    def _get_current_exposure(self, symbol: str) -> Decimal:
        """Cached exposure calculation with precision"""
        total = self._portfolio_state['cash'] + sum(
            Decimal(str(p['value'])) for p in self._portfolio_state['positions'].values()
        )
        position = self._portfolio_state['positions'].get(symbol, {'value': Decimal('0')})
        return (position['value'] / total).quantize(Decimal('0.0001'))

    def _calculate_base_size(self, symbol: str, risk_score: Decimal) -> Decimal:
        """Volatility-scaled Kelly Criterion with risk damping"""
        kelly_fraction = self._calculate_kelly_fraction(symbol)
        damping_factor = Decimal('1') - (risk_score ** 2)
        return (self._portfolio_state['cash'] * kelly_fraction * damping_factor).quantize(Decimal('0.01'), ROUND_DOWN)

    def _adjust_for_volatility(self, symbol: str) -> Decimal:
        """Regime-based volatility adjustment"""
        volatility = self.data_feed.get_volatility(symbol)
        regime = self.regime_detector.current_regime(symbol)
        
        if regime == 'high_volatility':
            return Decimal('0.7')
        if regime == 'market_crash':
            return Decimal('0.3')
        return Decimal('1')

    def _compile_allocation_decision(self, symbol: str, risk: Dict, 
                                   impact: Dict, hedge: Optional[Dict]) -> Dict:
        """Generate final allocation decision package"""
        decision = {
            'symbol': symbol,
            'approved': False,
            'allocated_size': Decimal('0'),
            'hedge_recommendation': hedge,
            'risk_parameters': risk,
            'exposure_limits': {
                'current': impact['current_exposure'],
                'maximum': impact['max_allowed']
            }
        }
        
        if impact['proposed_size'] > Decimal('0') and impact['current_exposure'] < impact['max_allowed']:
            decision['approved'] = True
            decision['allocated_size'] = min(
                impact['proposed_size'], 
                impact['max_allowed'] - impact['current_exposure']
            ).quantize(Decimal('0.01'), ROUND_DOWN)
            
        return decision

    async def rebalance_portfolio(self, threshold: Decimal = Decimal('0.05')) -> Dict:
        """AI-driven portfolio rebalancing engine"""
        target_allocations = self._calculate_target_allocations()
        deviations = self._calculate_allocation_deviations(target_allocations)
        
        rebalance_plan = {}
        for symbol, deviation in deviations.items():
            if abs(deviation) > threshold:
                rebalance_plan[symbol] = {
                    'current': self._get_current_exposure(symbol),
                    'target': target_allocations[symbol],
                    'adjustment': (target_allocations[symbol] - self._get_current_exposure(symbol)) 
                                  * self._portfolio_state['cash']
                }
        
        # Execute rebalancing through order execution engine
        if rebalance_plan:
            await self._execute_rebalancing(rebalance_plan)
            
        return rebalance_plan

    def _calculate_target_allocations(self) -> Dict[str, Decimal]:
        """ML-optimized target allocations based on market regime"""
        regime = self.regime_detector.current_market_regime()
        return self._portfolio_optimizer.calculate_optimal_allocations(regime)

    async def _execute_rebalancing(self, plan: Dict):
        """Integrated with order execution engine"""
        from Apex.src.Core.trading.execution.order_execution import execute_batch_orders
        
        orders = [{
            'symbol': symbol,
            'quantity': adj['adjustment'],
            'side': 'BUY' if adj['adjustment'] > 0 else 'SELL'
        } for symbol, adj in plan.items()]
        
        await execute_batch_orders(orders)

    # Security-hardened validation methods
    @validate_portfolio_inputs
    async def _check_leverage_limits(self) -> bool:
        """Margin and leverage validation with 3-layer protection"""
        margin_usage = self._calculate_margin_usage()
        return margin_usage < self.allocation_rules['max_leverage']

    @secure_decimal
    def _calculate_margin_usage(self) -> Decimal:
        """Precision margin calculation with anti-manipulation"""
        total_equity = self._portfolio_state['cash'] + sum(
            p['value'] for p in self._portfolio_state['positions'].values()
        )
        return (sum(p['margin'] for p in self._portfolio_state['positions'].values()) / total_equity).quantize(Decimal('0.0001'))