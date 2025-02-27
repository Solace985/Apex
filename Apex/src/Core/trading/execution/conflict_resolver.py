# Apex/src/Core/trading/execution/conflict_resolver.py
import numpy as np
from typing import Dict, Any, Optional, List
from utils.logging.structured_logger import StructuredLogger
from utils.helpers.error_handler import handle_api_error
from Core.trading.strategies.regime_detection import RegimeDetector
from Core.trading.risk.risk_management import RiskManager
from Core.trading.ai.config import load_config
from Core.trading.strategies.strategy_evaluator import StrategyEvaluator
from Core.trading.logging.decision_logger import log_resolution
from src.ai.reinforcement.q_learning.q_learning import QLearningAI

class ConflictResolver:
    """Advanced AI conflict resolution engine with hybrid capabilities"""
    
    def __init__(self, market_data_bus: Any):
        self.logger = StructuredLogger(__name__)
        self.market_data = market_data_bus
        self.regime_detector = RegimeDetector()
        self.risk_manager = RiskManager()
        self.reinforcement_ai = QLearningAI()
        self.evaluator = StrategyEvaluator()
        self.config = load_config()['conflict_resolution']
        
        # Initialize performance metrics
        self.performance_history = {
            'technical': [],
            'fundamental': [],
            'sentiment': [],
            'rl': [],
            'correlation': []
        }

    def _validate_signals(self, signals: Dict) -> bool:
        """Comprehensive signal validation with security checks"""
        required_keys = {'technical', 'fundamental', 'sentiment', 'rl', 'correlation'}
        if not all(k in signals for k in required_keys):
            self.logger.error("Missing critical signals")
            return False
            
        for model, data in signals.items():
            if not (0 <= data['confidence'] <= 1 and data['action'] in ['buy', 'sell', 'hold']):
                self.logger.warning(f"Invalid data from {model}")
                return False
                
        return True

    def _get_regime_adjustments(self) -> Dict[str, float]:
        """Dynamic regime-based weight adjustments"""
        regime = self.regime_detector.current_regime()
        return self.config['regime_weights'].get(regime, self.config['default_weights'])

    def _calculate_consensus(self, signals: Dict, weights: Dict) -> Dict:
        """Multi-factor consensus calculation with normalization"""
        weighted_scores = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        total_weight = 0.0
        
        for model, data in signals.items():
            weight = weights.get(model, 0.2)
            weighted_scores[data['action']] += data['confidence'] * weight
            total_weight += weight
            
        if total_weight == 0:
            return {'action': 'hold', 'confidence': 0.0}
            
        # Apply softmax normalization
        exp_scores = {k: np.exp(v/total_weight) for k,v in weighted_scores.items()}
        total_exp = sum(exp_scores.values())
        
        return {
            'action': max(exp_scores, key=exp_scores.get),
            'confidence': exp_scores[max(exp_scores, key=exp_scores.get)] / total_exp
        }

    def _detect_weight_bias(self, weights: Dict) -> bool:
        """
        Detects bias when one model is significantly more influential than others.
        Args:
            weights: Model confidence weightings.
        Returns:
            True if bias is detected, otherwise False.
        """
        avg_weight = np.mean(list(weights.values()))
        max_weight = max(weights.values())

        if max_weight > avg_weight * self.config['bias_threshold']:
            self.logger.warning(f"⚠️ Potential bias detected! Max weight: {max_weight}, Avg weight: {avg_weight}")
            return True

        return False

    def _correct_weight_bias(self, weights: Dict) -> Dict:
        """Conservative weight redistribution with momentum"""
        avg_weight = np.mean(list(weights.values()))
        return {k: min(v, avg_weight * self.config['bias_correction_factor']) for k,v in weights.items()}

    def _detect_confidence_bias(self, signals: Dict) -> bool:
        """Signal-level confidence bias detection"""
        confidences = [v['confidence'] for v in signals.values()]
        return max(confidences) > np.mean(confidences) * self.config['confidence_bias_threshold']

    def _correct_confidence_bias(self, signals: Dict) -> Dict:
        """Confidence capping with historical performance awareness"""
        avg_confidence = np.mean([v['confidence'] for v in signals.values()])
        for model in signals:
            historical = self.evaluator.get_model_performance(model)
            signals[model]['confidence'] = min(
                signals[model]['confidence'],
                avg_confidence * self.config['max_confidence_ratio'],
                historical * 1.2
            )
        return signals

    def _apply_rl_override(self, decision: Dict, trade_history: List) -> Dict:
        """Reinforcement learning pattern-based override"""
        if len(trade_history) >= self.config['rl_min_samples']:
            rl_suggestion = self.reinforcement_ai.analyze(trade_history)
            if rl_suggestion['confidence'] > self.config['rl_confidence_threshold']:
                self.logger.info(f"RL override: {rl_suggestion['action']}")
                decision.update({
                    'action': rl_suggestion['action'],
                    'confidence': self._adaptive_confidence_merge(decision['confidence'], rl_suggestion['confidence'])
                })
        return decision

    def _adaptive_confidence_merge(self, current_confidence: float, rl_confidence: float) -> float:
        """
        Merges RL confidence with existing confidence using an adaptive weighting scheme.
        Args:
            current_confidence: AI-generated trade confidence.
            rl_confidence: Confidence from Reinforcement Learning.
        Returns:
            Adjusted confidence score.
        """
        return np.mean([current_confidence, rl_confidence])  # Ensures balance between AI and RL models

    def _check_model_drift(self, signals: Dict) -> bool:
        """Comprehensive model drift detection"""
        drift_flags = []
        for model, data in signals.items():
            historical = self.evaluator.get_model_performance(model)
            current = data['confidence']
            if abs(historical - current) > self.config['drift_threshold']:
                drift_flags.append(True)
        return sum(drift_flags) >= self.config['max_drift_models']

    @handle_api_error(retries=3, cooldown=5)
    def resolve(self, signals: Dict, base_weights: Dict, trade_history: List = None) -> Dict:
        """Master resolution process with full feature integration"""
        # Input validation and sanitization
        if not self._validate_signals(signals):
            return {'action': 'hold', 'reason': 'invalid_signals', 'confidence': 0.0}
            
        try:
            # Phase 1: Weight management
            regime_weights = self._get_regime_adjustments()
            combined_weights = {
                k: base_weights[k] * regime_weights.get(k, 1.0)
                for k in base_weights
            }
            
            # Phase 2: Bias detection and correction
            if self._detect_weight_bias(combined_weights):
                combined_weights = self._correct_weight_bias(combined_weights)
                
            if self._detect_confidence_bias(signals):
                signals = self._correct_confidence_bias(signals)
            
            # Phase 3: Consensus calculation
            consensus = self._calculate_consensus(signals, combined_weights)
            self.logger.info(f"Initial consensus: {consensus}")
            
            # Phase 4: Reinforcement learning override
            if trade_history:
                consensus = self._apply_rl_override(consensus, trade_history)
            
            # Phase 5: Risk management
            risk_assessment = self.risk_manager.assess_decision(consensus)
            if not risk_assessment['approved']:
                return {
                    **consensus,
                    'action': 'hold',
                    'reason': 'risk_violation',
                    'risk_data': risk_assessment
                }
                
            # Phase 6: Drift detection
            if self._check_model_drift(signals):
                self.logger.warning("Model drift detected")
                return {**consensus, 'action': 'hold', 'reason': 'model_drift'}
            
            # Phase 7: Final validation
            consensus['confidence'] = np.clip(consensus['confidence'], 0.01, 0.99)
            self._log_decision(consensus, signals, combined_weights)
            
            return consensus
            
        except Exception as e:
            self.logger.critical("Resolution failed", error=str(e))
            return {'action': 'hold', 'reason': 'system_error', 'confidence': 0.0}

    def _log_decision(self, decision: Dict, signals: Dict, weights: Dict):
        """
        Logs the final trade decision with a detailed audit trail.
        Args:
            decision: Final trade action and confidence.
            signals: AI-generated trade signals.
            weights: Model weight adjustments.
        """
        log_data = {
            "final_decision": decision,
            "signals_used": signals,
            "applied_weights": weights,
            "market_regime": self.regime_detector.current_regime()
        }
        self.logger.info(f"✅ Trade Decision Log: {log_data}")
        log_resolution(log_data)

    def update_performance_metrics(self, trade_outcome: Dict):
        """Update model performance tracking"""
        for model in self.performance_history:
            if model in trade_outcome['contributors']:
                self.performance_history[model].append(trade_outcome['success'])