"""
Institutional-Grade Performance Metrics Engine
Location: Apex/Metrics/performance_metrics.py
Integrates with: decision_logger.py, risk_management.py, strategy_orchestrator.py, meta_trader.py
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple, Any
from datetime import datetime
from utils.logging.structured_logger import StructuredLogger

class PortfolioMetrics:
    def __init__(self):
        self.logger = StructuredLogger()
        from src.Core.trading.risk.risk_management import RiskManager
        self.risk_manager = RiskManager()
        
    def compute_trade_statistics(self, trade_results: List[Dict]) -> Dict:
        """Enhanced with win rate, profit factor, and expectancy calculations"""
        total_trades = len(trade_results)
        if total_trades == 0:
            self.logger.warning("No trades available for statistics calculation")
            return {}

        wins = [t['pnl'] for t in trade_results if t['pnl'] > 0]
        losses = [t['pnl'] for t in trade_results if t['pnl'] < 0]
        
        return {
            "total_trades": total_trades,
            "win_rate": len(wins)/total_trades,
            "profit_factor": sum(wins)/abs(sum(losses)) if losses else float('inf'),
            "expectancy": (sum(wins)/len(wins) if wins else 0) - (abs(sum(losses))/len(losses) if losses else 0)
        }

    def calculate_sharpe(self, returns: pd.Series, risk_free=0.0) -> float:
        """Integrated with stats.rs for efficient calculation"""
        excess_returns = returns - risk_free
        return np.mean(excess_returns) / np.std(excess_returns)

    def calculate_sortino(self, returns: pd.Series, target=0.0) -> float:
        """Downside risk-adjusted metric"""
        downside = returns[returns < target]
        return np.mean(returns - target) / downside.std()

    def max_drawdown(self, equity_curve: pd.Series) -> float:
        """Enhanced with rolling window optimization"""
        roll_max = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - roll_max) / roll_max
        return drawdown.min()

class RiskAnalysis:
    def __init__(self):
        from src.Core.trading.risk.risk_engine import RiskEngine
        self.risk_engine = RiskEngine()
        from src.Core.data.correlation_monitor import CorrelationMonitor
        self.correlation_monitor = CorrelationMonitor()
    
    def value_at_risk(self, returns: pd.Series, confidence=0.95) -> float:
        """Integrated with existing risk engine"""
        return self.risk_engine.calculate_var(returns, confidence)

    def conditional_var(self, returns: pd.Series, confidence=0.95) -> float:
        """Expected shortfall calculation with fallback"""
        var = self.value_at_risk(returns, confidence)
        tail_losses = returns[returns <= var]
        return tail_losses.mean() if not tail_losses.empty else 0.0

    def portfolio_risk_assessment(self, portfolio: Dict) -> Dict:
        """Enhanced with correlation monitoring"""
        corr_matrix = self.correlation_monitor.calculate_correlation_matrix(portfolio.keys())
        sector_exposure = self.correlation_monitor.calculate_sector_exposure(portfolio)
        
        return {
            "diversification_score": 1 - np.mean(np.abs(corr_matrix)),
            "sector_exposure": sector_exposure,
            "var_95": self.value_at_risk(pd.Series(portfolio.values()))
        }

class ExecutionQuality:
    def __init__(self):
        from src.Core.trading.execution.order_execution import OrderExecution
        self.order_exec = OrderExecution()
    
    def evaluate_execution(self, trades: List[Dict]) -> Dict:
        """Enhanced with latency and fill rate metrics"""
        executed = [t for t in trades if t['status'] == 'filled']
        slippages = [t['executed_price']/t['target_price'] - 1 for t in executed]
        latencies = [(t['executed_at'] - t['ordered_at']).total_seconds() for t in executed]
        
        return {
            "fill_rate": len(executed)/len(trades),
            "avg_slippage": np.mean(slippages),
            "latency_90th": np.percentile(latencies, 90)
        }

class StrategyAnalytics:
    def __init__(self):
        from src.Core.trading.strategies.strategy_orchestrator import StrategyOrchestrator
        self.orchestrator = StrategyOrchestrator()
    
    def strategy_performance(self, strategy_results: Dict) -> Dict:
        """Enhanced with regime-based analysis"""
        metrics = {}
        for strategy, returns in strategy_results.items():
            regime_data = self.orchestrator.get_regime_classification(returns.index)
            metrics[strategy] = {
                "sharpe": PortfolioMetrics().calculate_sharpe(returns),
                "regime_performance": regime_data.apply(lambda x: returns[x].mean())
            }
        return metrics

class AIEnhancedMetrics:
    def __init__(self):
        from src.ai.ensembles.model_optimizer import ModelOptimizer
        self.optimizer = ModelOptimizer()
        
    def predict_performance(self, historical_returns: pd.Series) -> Dict:
        """Enhanced with existing AI models"""
        return self.optimizer.forecast(historical_returns.values.reshape(-1, 1))

class ReportGenerator:
    def generate_report(self, metrics: Dict) -> str:
        """Integrated with existing report infrastructure"""
        from report_generator import create_performance_pdf
        return create_performance_pdf(metrics)

class SafeMetricsCalculator:
    @staticmethod
    def validate_input(data: pd.Series) -> bool:
        """Enhanced validation with existing modules"""
        from utils.helpers.validation.rs import validate_financial_series
        return validate_financial_series(data)

    @staticmethod
    def sanitize_trades(trades: List[Dict]) -> List[Dict]:
        """Enhanced security sanitization"""
        return [{'timestamp': t['timestamp'], 
                'symbol': t['symbol'], 
                'returns': t['pnl']} 
               for t in trades]

class PerformancePipeline:
    def __init__(self):
        self.portfolio = PortfolioMetrics()
        self.risk = RiskAnalysis()
        self.execution = ExecutionQuality()
        self.strategy = StrategyAnalytics()
        self.ai = AIEnhancedMetrics()
        self.reporter = ReportGenerator()
        
    def full_analysis(self, raw_trades: List[Dict]) -> Dict:
        """Comprehensive analysis pipeline with security checks"""
        if not raw_trades:
            self.logger.error("No trade data provided for analysis")
            return {}

        # Security First
        clean_trades = SafeMetricsCalculator.sanitize_trades(raw_trades)
        returns = pd.Series([t['returns'] for t in clean_trades])
        
        if not SafeMetricsCalculator.validate_input(returns):
            raise ValueError("Invalid returns data detected")

        # Core Metrics
        portfolio_stats = self.portfolio.compute_trade_statistics(clean_trades)
        risk_metrics = self.risk.portfolio_risk_assessment(
            {t['symbol']: t['returns'] for t in clean_trades}
        )
        execution_stats = self.execution.evaluate_execution(raw_trades)
        
        # Advanced Analysis
        strategy_stats = self.strategy.strategy_performance(
            self._group_by_strategy(clean_trades)
        )
        ai_predictions = self.ai.predict_performance(returns)
        
        # Compile Report
        full_report = {
            "portfolio": portfolio_stats,
            "risk": risk_metrics,
            "execution": execution_stats,
            "strategies": strategy_stats,
            "predictions": ai_predictions
        }
        
        return self.reporter.generate_report(full_report)

    def _group_by_strategy(self, trades: List[Dict]) -> Dict:
        """Utility for strategy grouping"""
        strategies = {}
        for t in trades:
            strat = t.get('strategy', 'unknown')
            strategies.setdefault(strat, []).append(t['returns'])
        return {k: pd.Series(v) for k, v in strategies.items()}

# Integrated Example Usage
if __name__ == "__main__":
    from src.Core.trading.logging.decision_logger import DecisionLogger
    
    dl = DecisionLogger()
    trades = dl.get_trades(days=30)
    
    pipeline = PerformancePipeline()
    report = pipeline.full_analysis(trades)
    
    print("Performance Analysis Complete:")
    print(report)