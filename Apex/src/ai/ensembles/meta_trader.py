# Apex/src/Core/trading/execution/meta_trader.py
import logging
from typing import Dict, Any, Optional
from utils.logging.structured_logger import StructuredLogger
from utils.helpers.error_handler import handle_api_error
from Core.trading.strategies.regime_detection import RegimeDetector
from ai.ensembles.ensemble_voting import EnsembleVoter
from Core.trading.execution.conflict_resolver import ConflictResolver
from Core.data.correlation_optimizer import CorrelationOptimizer

class MetaTrader:
    """Central decision-making layer for AI-driven trade execution"""
    def __init__(self, market_data_bus: Any):
        self.logger = StructuredLogger(__name__)
        self.market_data_bus = market_data_bus
        self.regime_detector = RegimeDetector()
        self.ensemble_voter = EnsembleVoter()
        self.conflict_resolver = ConflictResolver()
        self.correlation_optimizer = CorrelationOptimizer()
        
        # Initialize signal processors
        self.signal_processors = {
            'technical': self._get_technical_signal,
            'fundamental': self._get_fundamental_signal,
            'sentiment': self._get_sentiment_signal,
            'rl': self._get_rl_signal,
            'correlation': self._get_correlation_signal
        }

    @handle_api_error(retries=3)
    def _get_technical_signal(self) -> Dict[str, float]:
        """Get technical analysis signal from Rust engine"""
        from lib.ta_engine import get_technical_signal  # Rust FFI
        return get_technical_signal(self.market_data_bus.get_ohlc())

    @handle_api_error()
    def _get_fundamental_signal(self) -> Dict[str, float]:
        """Get fundamental analysis signal"""
        return self.market_data_bus.get('fundamentals', {})

    def _get_sentiment_signal(self) -> Dict[str, float]:
        """Get processed sentiment analysis"""
        return self.market_data_bus.get('sentiment', {})

    @handle_api_error()
    def _get_rl_signal(self) -> Dict[str, float]:
        """Get reinforcement learning signal"""
        from ai.reinforcement.q_learning import get_rl_prediction  # Rust FFI
        return get_rl_prediction(self.market_data_bus.get_state())

    def _get_correlation_signal(self) -> Dict[str, float]:
        """Get correlation-based signals"""
        return self.market_data_bus.get('correlation_predictions', {})

    def _validate_signals(self, signals: Dict[str, Any]) -> bool:
        """Security-critical signal validation"""
        required_keys = {'technical', 'fundamental', 'sentiment', 'rl', 'correlation'}
        if not all(key in signals for key in required_keys):
            self.logger.error("Missing critical signals")
            return False
            
        for key, value in signals.items():
            if not isinstance(value, dict) or 'confidence' not in value:
                self.logger.warning(f"Invalid signal format for {key}")
                return False
                
        return True

    def _get_dynamic_weights(self) -> Dict[str, float]:
        """Get regime-aware model weightings"""
        current_regime = self.regime_detector.current_regime()
        base_weights = self.ensemble_voter.get_weights(current_regime)
        market_volatility = self.market_data_bus.get_volatility()
        
        # Adjust weights based on volatility
        return {
            'technical': base_weights['technical'] * (1 - market_volatility),
            'fundamental': base_weights['fundamental'],
            'sentiment': base_weights['sentiment'] * market_volatility,
            'rl': base_weights['rl'] * market_volatility,
            'correlation': base_weights['correlation']
        }

    def _calculate_final_signal(self, signals: Dict, weights: Dict) -> Dict:
        """Calculate weighted trade decision"""
        weighted_scores = {
            'buy': 0.0,
            'sell': 0.0,
            'hold': 0.0,
            'confidence': 0.0
        }
        
        for model, weight in weights.items():
            signal = signals[model]
            weighted_scores[signal['action']] += signal['score'] * weight
            weighted_scores['confidence'] += signal['confidence'] * weight
            
        # Normalize scores
        total = sum(weighted_scores.values()) - weighted_scores['confidence']
        if total == 0:
            return {'action': 'hold', 'confidence': 0.0}
            
        return {
            'action': max(weighted_scores, key=weighted_scores.get),
            'confidence': weighted_scores['confidence'] / len(weights)
        }

    async def execute_trade_decision(self):
        # Consolidated decision flow
        signals = await self.collect_signals()  # From TA/FA/SA/RL/Correlation
        regime = self.regime_detector.current_regime()
        weights = self.ensemble_voter.calculate_weights(signals, regime)
        conflict_check = self.conflict_resolver.validate(signals, weights)
        
        if conflict_check["approved"]:
            risk_check = self.risk_manager.evaluate(signals, weights)
            if risk_check["pass"]:
                await self.order_executor.execute(risk_check["size"])
                self.decision_logger.log_full_decision(
                    signals, weights, regime, risk_check
                )

    def generate_decision(self) -> Optional[Dict[str, Any]]:
        """Generate final trade decision with conflict resolution"""
        try:
            # Collect all signals
            signals = {name: processor() for name, processor in self.signal_processors.items()}
            if not self._validate_signals(signals):
                return None
                
            # Get dynamic weightings
            weights = self._get_dynamic_weights()
            
            # Resolve conflicts
            resolved_signals = self.conflict_resolver.resolve(signals, weights)
            
            # Optimize correlation timing
            optimized_decision = self.correlation_optimizer.optimize(
                self._calculate_final_signal(resolved_signals, weights)
            )
            
            # Apply risk management
            from Core.trading.risk.risk_management import validate_decision
            if validate_decision(optimized_decision):
                return optimized_decision
                
            return {'action': 'hold', 'reason': 'risk_check_failed'}

        except Exception as e:
            self.logger.critical("Decision generation failed", error=str(e))
            return None

    def execute_decision_cycle(self):
        """Full decision cycle integrated with market data"""
        if not self.market_data_bus.is_fresh():
            self.logger.warning("Stale market data")
            return {'action': 'hold', 'reason': 'stale_data'}
            
        decision = self.generate_decision()
        if decision and decision['action'] != 'hold':
            from Core.trading.execution.order_execution import execute_order
            execute_order(decision)
            
        return decision