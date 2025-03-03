import numpy as np
import pandas as pd
from scipy.stats import linregress, norm
from typing import List, Dict, Tuple, Optional, Union
from functools import lru_cache
import warnings

# Internal Apex imports
from Tests.backtesting.backtest_runner import BacktestRunner
from Tests.backtesting.report_generator import ReportGenerator
from src.Core.data.historical_data import HistoricalData
from src.Core.trading.risk.risk_management import RiskManager
from src.Core.trading.ai.situational_analysis import MarketRegimeDetector
from src.Core.trading.execution.order_execution import ExecutionAnalyzer
from src.Core.data.correlation_monitor import CorrelationMonitor
from utils.logging.structured_logger import Logger
from utils.analytics.monte_carlo_simulator import MonteCarloSimulator

class PerformanceEvaluator:
    """
    Advanced trading performance evaluation system with comprehensive metrics.
    Designed for institutional-grade performance analysis of trading strategies.
    
    Optimized for:
    - Speed: Using vectorized operations and avoiding redundant calculations
    - Risk assessment: Comprehensive risk-adjusted metrics
    - Strategy-specific analysis: Performance breakdown by strategy and market regime
    - Portfolio-level insights: Diversification and correlation tracking
    """

    def __init__(self, 
                 risk_free_rate: float = 0.02, 
                 confidence_level: float = 0.95,
                 time_windows: List[str] = None):
        """
        Initialize the performance evaluator with configurable parameters.
        
        Args:
            risk_free_rate: Annual risk-free rate used in risk-adjusted calculations
            confidence_level: Confidence level for VaR calculations
            time_windows: List of time periods for rolling window analysis
        """
        self.logger = Logger("performance_evaluator")
        self.market_regime = MarketRegimeDetector()
        self.risk_manager = RiskManager()
        self.monte_carlo = MonteCarloSimulator()
        self.correlation_monitor = CorrelationMonitor()
        self.execution_analyzer = ExecutionAnalyzer()
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.time_windows = time_windows or ["daily", "weekly", "monthly"]
        
        # Pre-calculate constants
        self.z_score = norm.ppf(self.confidence_level)
        self._reset_cache()

    def _reset_cache(self):
        """Reset all cached calculations"""
        self._cached_equity_curves = {}
        self._cached_drawdowns = {}
        self._cached_returns = {}
        self._cached_profit_loss = {}
        self._cached_regime_performance = {}
        self._cached_monte_carlo = {}
    
    # -------- CORE PERFORMANCE METRICS --------
    
    def calculate_returns(self, trades: List[Dict], cache_key: str = None) -> np.ndarray:
        """Calculate returns array efficiently (vectorized)"""
        if cache_key and cache_key in self._cached_returns:
            return self._cached_returns[cache_key]
            
        profits = np.array([trade.get("profit", 0) for trade in trades])
        
        if cache_key:
            self._cached_profit_loss[cache_key] = profits
            
        return profits
    
    def calculate_core_metrics(self, trades: List[Dict], 
                               initial_capital: float, 
                               years: float, 
                               cache_key: str = None) -> Dict[str, float]:
        """Calculate core performance metrics in one pass (vectorized)"""
        if not trades or initial_capital <= 0 or years <= 0:
            return {
                "total_return": 0.0,
                "roi": 0.0,
                "cagr": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0
            }
        
        # Get profit/loss array (vectorized)
        profits = self.calculate_returns(trades, cache_key)
        total_return = np.sum(profits)
        
        # ROI and CAGR
        roi = total_return / initial_capital
        final_equity = initial_capital + total_return
        cagr = (final_equity / initial_capital) ** (1 / years) - 1
        
        # Win/loss metrics (vectorized)
        wins = profits > 0
        losses = profits < 0
        win_count = np.sum(wins)
        loss_count = np.sum(losses)
        total_count = len(profits)
        
        win_rate = win_count / total_count if total_count > 0 else 0.0
        loss_rate = loss_count / total_count if total_count > 0 else 0.0
        
        # Calculate profit factor (vectorized)
        total_profit = np.sum(profits[wins]) if win_count > 0 else 0
        total_loss = np.sum(np.abs(profits[losses])) if loss_count > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")
        
        # Expectancy calculation
        avg_win = np.mean(profits[wins]) if win_count > 0 else 0
        avg_loss = np.mean(np.abs(profits[losses])) if loss_count > 0 else 0
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        
        return {
            "total_return": round(total_return, 2),
            "roi": round(roi, 4),
            "cagr": round(cagr, 4),
            "win_rate": round(win_rate, 4),
            "loss_rate": round(loss_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "expectancy": round(expectancy, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2)
        }
    
    # -------- RISK METRICS --------
    
    def calculate_equity_curve(self, trades: List[Dict], initial_capital: float = 0, 
                               cache_key: str = None) -> np.ndarray:
        """Generate equity curve efficiently with caching"""
        if cache_key and cache_key in self._cached_equity_curves:
            return self._cached_equity_curves[cache_key]
            
        if not trades:
            curve = np.array([initial_capital])
        else:
            profits = self.calculate_returns(trades)
            curve = np.concatenate(([initial_capital], initial_capital + np.cumsum(profits)))
        
        if cache_key:
            self._cached_equity_curves[cache_key] = curve
            
        return curve
    
    def calculate_drawdowns(self, equity_curve: np.ndarray = None, 
                            trades: List[Dict] = None, 
                            initial_capital: float = 0,
                            cache_key: str = None) -> np.ndarray:
        """Calculate drawdowns with optional caching"""
        if cache_key and cache_key in self._cached_drawdowns:
            return self._cached_drawdowns[cache_key]
        
        if equity_curve is None and trades:
            equity_curve = self.calculate_equity_curve(trades, initial_capital, cache_key)
            
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = peak - equity_curve
        
        if cache_key:
            self._cached_drawdowns[cache_key] = drawdowns
            
        return drawdowns
    
    def calculate_risk_metrics(self, trades: List[Dict], 
                              initial_capital: float, 
                              periods: int = 252,
                              cache_key: str = None) -> Dict[str, float]:
        """Calculate risk metrics efficiently in one pass"""
        if not trades or initial_capital <= 0:
            return {
                "max_drawdown": 0.0,
                "recovery_factor": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "value_at_risk": 0.0,
                "conditional_var": 0.0
            }
        
        # Get returns and equity curve (reuse cached if available)
        returns = self.calculate_returns(trades, cache_key)
        equity_curve = self.calculate_equity_curve(trades, initial_capital, cache_key)
        drawdowns = self.calculate_drawdowns(equity_curve=equity_curve, cache_key=cache_key)
        
        # Max drawdown and recovery factor
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        total_return = np.sum(returns)
        recovery_factor = total_return / max_dd if max_dd > 0 else float("inf")
        
        # Sharpe Ratio calculation
        daily_risk_free = self.risk_free_rate / periods
        mean_return = np.mean(returns)
        std_dev = np.std(returns)
        sharpe = (mean_return - daily_risk_free) / std_dev if std_dev > 0 else 0.0
        sharpe_annualized = sharpe * np.sqrt(periods)
        
        # Sortino Ratio calculation
        downside_returns = returns[returns < daily_risk_free]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino = (mean_return - daily_risk_free) / downside_deviation if downside_deviation > 0 else float("inf")
        sortino_annualized = sortino * np.sqrt(periods)
        
        # VaR and CVaR calculations
        pct_returns = returns / initial_capital
        var_pct = np.percentile(pct_returns, 100 * (1 - self.confidence_level))
        var_amount = abs(var_pct * initial_capital)
        
        # CVaR (Expected Shortfall) calculation
        cvar_threshold = np.percentile(pct_returns, 100 * (1 - self.confidence_level))
        tail_returns = pct_returns[pct_returns <= cvar_threshold]
        cvar_pct = np.mean(tail_returns) if len(tail_returns) > 0 else var_pct
        cvar_amount = abs(cvar_pct * initial_capital)
        
        return {
            "max_drawdown": round(max_dd, 2),
            "max_drawdown_pct": round(max_dd / initial_capital, 4) if initial_capital > 0 else 0.0,
            "recovery_factor": round(recovery_factor, 4),
            "sharpe_ratio": round(sharpe_annualized, 4),
            "sortino_ratio": round(sortino_annualized, 4),
            "value_at_risk": round(var_amount, 2),
            "conditional_var": round(cvar_amount, 2)
        }
    
    def calculate_risk_of_ruin(self, trades: List[Dict], initial_capital: float) -> float:
        """Calculate probability of total account loss based on trade history"""
        if not trades or initial_capital <= 0:
            return 1.0
            
        returns = self.calculate_returns(trades)
        win_probability = np.mean(returns > 0)
        
        if win_probability == 0:
            return 1.0
            
        # Calculate Kelly-based risk of ruin
        wins = returns > 0
        losses = returns < 0
        avg_win = np.mean(returns[wins]) if np.any(wins) else 0
        avg_loss = np.mean(np.abs(returns[losses])) if np.any(losses) else 0
        
        if avg_loss == 0:
            return 0.0
            
        # Risk of ruin calculation
        R = avg_win / avg_loss
        if R <= 1:
            return 1.0
            
        q = (1 - win_probability) / win_probability
        if q >= 1:
            return 1.0
            
        risk = min(max(q ** (initial_capital / avg_loss), 0.0), 1.0)
        return round(risk, 4)
    
    # -------- PORTFOLIO METRICS --------
    
    def calculate_diversification_metrics(self, strategy_returns: Dict[str, np.ndarray],
                                         portfolio_returns: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio diversification metrics"""
        if not strategy_returns or len(strategy_returns) < 2:
            return {
                "diversification_score": 0.0,
                "avg_correlation": 0.0,
                "strategy_contribution": {}
            }
        
        # Get correlation matrix from the correlation monitor
        correlation_matrix = self.correlation_monitor.get_correlation_matrix(strategy_returns)
        
        # Calculate average correlation (measure of diversification)
        # Exclude diagonal elements (self-correlations)
        n_strategies = len(strategy_returns)
        avg_correlation = (np.sum(correlation_matrix) - n_strategies) / (n_strategies * (n_strategies - 1))
        
        # Calculate diversification score (1 - avg_correlation)
        diversification_score = 1 - avg_correlation
        
        # Calculate strategy contribution to portfolio returns
        strategy_contribution = {}
        portfolio_variance = np.var(portfolio_returns) if len(portfolio_returns) > 0 else 0
        
        if portfolio_variance > 0:
            for strategy, returns in strategy_returns.items():
                cov_with_portfolio = np.cov(returns, portfolio_returns)[0, 1]
                contribution = cov_with_portfolio / portfolio_variance
                strategy_contribution[strategy] = round(contribution, 4)
        
        return {
            "diversification_score": round(diversification_score, 4),
            "avg_correlation": round(avg_correlation, 4),
            "strategy_contribution": strategy_contribution
        }
    
    # -------- STRATEGY ANALYTICS --------
    
    def analyze_regime_performance(self, trades: List[Dict], 
                                  initial_capital: float,
                                  cache_key: str = None) -> Dict[str, Dict[str, float]]:
        """Analyze strategy performance across different market regimes"""
        if not trades or cache_key in self._cached_regime_performance:
            return self._cached_regime_performance.get(cache_key, {})
            
        # Get market regime for each trade
        trade_dates = [t.get("timestamp", None) for t in trades]
        trade_regimes = [self.market_regime.detect_regime(date) for date in trade_dates if date]
        
        if len(trade_regimes) != len(trades):
            return {}
            
        # Group trades by regime
        regime_indices = {}
        for i, regime in enumerate(trade_regimes):
            if regime not in regime_indices:
                regime_indices[regime] = []
            regime_indices[regime].append(i)
        
        # Calculate performance metrics per regime
        results = {}
        for regime, indices in regime_indices.items():
            regime_trades = [trades[i] for i in indices]
            n_trades = len(regime_trades)
            
            if n_trades < 2:
                continue
                
            # Calculate core metrics for this regime subset
            regime_profits = np.array([t.get("profit", 0) for t in regime_trades])
            
            results[regime] = {
                "trade_count": n_trades,
                "win_rate": round(np.mean(regime_profits > 0), 4),
                "avg_return": round(np.mean(regime_profits), 2),
                "total_return": round(np.sum(regime_profits), 2),
                "sharpe": round(np.mean(regime_profits) / np.std(regime_profits) if np.std(regime_profits) > 0 else 0, 4)
            }
        
        if cache_key:
            self._cached_regime_performance[cache_key] = results
            
        return results
    
    def optimize_position_sizing(self, trades: List[Dict], 
                                initial_capital: float) -> Dict[str, float]:
        """Calculate optimal position sizing using Kelly Criterion"""
        if not trades or initial_capital <= 0:
            return {"kelly_fraction": 0.0, "optimal_trade_size_pct": 0.0}
            
        # Get win probability and average win/loss
        returns = self.calculate_returns(trades)
        win_probability = np.mean(returns > 0)
        
        wins = returns > 0
        losses = returns < 0
        avg_win = np.mean(returns[wins]) if np.any(wins) else 0
        avg_loss = np.mean(np.abs(returns[losses])) if np.any(losses) else 0
        
        if avg_loss == 0 or win_probability == 0:
            return {"kelly_fraction": 0.0, "optimal_trade_size_pct": 0.0}
        
        # Calculate Kelly fraction (optimal percentage of capital to risk per trade)
        # Kelly = W - (1-W)/R where W = win probability, R = win/loss ratio
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = win_probability - ((1 - win_probability) / win_loss_ratio)
        
        # The Half-Kelly is considered safer in practice (more conservative)
        half_kelly = max(0, kelly_fraction / 2)
        
        # Calculate optimal position size as percentage of capital
        avg_trade_size_pct = np.mean([t.get("size", 0) / initial_capital * 100 for t in trades if "size" in t])
        
        return {
            "kelly_fraction": round(kelly_fraction, 4),
            "half_kelly": round(half_kelly, 4),
            "optimal_trade_size_pct": round(half_kelly * 100, 2),
            "current_avg_size_pct": round(avg_trade_size_pct, 2)
        }
    
    # -------- PREDICTIVE ANALYTICS --------
    
    def monte_carlo_risk_projection(self, trades: List[Dict], 
                                   initial_capital: float,
                                   simulation_count: int = 1000,
                                   cache_key: str = None) -> Dict[str, float]:
        """Perform Monte Carlo simulation to project future risk"""
        if not trades or initial_capital <= 0:
            return {}
            
        # Check cache first
        if cache_key and cache_key in self._cached_monte_carlo:
            return self._cached_monte_carlo[cache_key]
            
        # Get returns for simulation
        returns = self.calculate_returns(trades)
        
        # Run Monte Carlo simulation
        projections = self.monte_carlo.simulate(returns.tolist(), 
                                                initial_capital=initial_capital,
                                                simulations=simulation_count)
        
        result = {
            "worst_case_loss": round(projections["5th_percentile"], 2),
            "expected_return": round(projections["median"], 2),
            "best_case_return": round(projections["95th_percentile"], 2),
            "max_drawdown_95pct": round(projections.get("max_drawdown_95pct", 0.0), 2)
        }
        
        if cache_key:
            self._cached_monte_carlo[cache_key] = result
            
        return result
    
    # -------- COMPREHENSIVE REPORTS --------
    
    def generate_performance_report(self, trades: List[Dict], 
                                   market_returns: List[float],
                                   initial_capital: float,
                                   trading_period_years: float,
                                   strategy_id: str = None) -> Dict[str, Union[float, Dict]]:
        """
        Generate a comprehensive performance report combining all metrics.
        Optimized for speed and memory efficiency.
        """
        if not trades:
            self.logger.warn("No trades provided for performance analysis")
            return {}
        
        # Create a cache key for this analysis
        cache_key = f"{strategy_id}_{len(trades)}_{initial_capital}"
        
        # 1. Get execution quality metrics (from execution_analyzer)
        execution_metrics = self.execution_analyzer.get_summary_metrics(trades)
        
        # 2. Calculate core performance metrics (vectorized)
        core_metrics = self.calculate_core_metrics(trades, initial_capital, trading_period_years, cache_key)
        
        # 3. Calculate risk metrics (vectorized)
        risk_metrics = self.calculate_risk_metrics(trades, initial_capital, cache_key=cache_key)
        risk_metrics["risk_of_ruin"] = self.calculate_risk_of_ruin(trades, initial_capital)
        
        # 4. Calculate alpha/beta against benchmark
        alpha_beta = self.calculate_alpha_beta(trades, market_returns)
        
        # 5. Regime analysis 
        regime_performance = self.analyze_regime_performance(trades, initial_capital, cache_key)
        
        # 6. Position sizing optimization
        position_sizing = self.optimize_position_sizing(trades, initial_capital)
        
        # 7. Monte Carlo projection
        monte_carlo = self.monte_carlo_risk_projection(trades, initial_capital, cache_key=cache_key)
        
        # 8. Strategy diversification (if strategy_id provided)
        diversification = {}
        if strategy_id and hasattr(self, '_strategy_returns'):
            portfolio_returns = self.calculate_returns(trades)
            diversification = self.calculate_diversification_metrics(self._strategy_returns, portfolio_returns)
        
        # Compile full report
        report = {
            **core_metrics,
            **risk_metrics,
            "execution_summary": execution_metrics,
            "alpha_beta": alpha_beta,
            "regime_performance": regime_performance,
            "position_sizing": position_sizing,
            "monte_carlo_projection": monte_carlo,
            "diversification": diversification
        }
        
        return report
    
    def calculate_alpha_beta(self, trades: List[Dict], market_returns: List[float]) -> Dict[str, float]:
        """Calculate Alpha and Beta against a benchmark"""
        if not trades or len(trades) != len(market_returns):
            return {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0}
            
        strategy_returns = self.calculate_returns(trades)
        market_returns_array = np.array(market_returns)
        
        # Calculate beta (slope) and alpha (intercept)
        slope, intercept, r_value, _, _ = linregress(market_returns_array, strategy_returns)
        
        return {
            "alpha": round(intercept, 4),
            "beta": round(slope, 4),
            "r_squared": round(r_value ** 2, 4)
        }
    
    # -------- STRATEGY COMPARISON --------
    
    def register_strategy_returns(self, strategy_id: str, returns: np.ndarray):
        """Register a strategy's returns for portfolio-level analysis"""
        if not hasattr(self, '_strategy_returns'):
            self._strategy_returns = {}
        
        self._strategy_returns[strategy_id] = returns
    
    def compare_strategies(self, strategy_metrics: Dict[str, Dict]) -> Dict[str, Dict]:
        """Compare multiple strategies to identify best performers"""
        if not strategy_metrics:
            return {}
            
        # Create rankings
        rankings = {}
        
        # Metrics to compare (higher is better)
        metrics_higher_better = ["sharpe_ratio", "sortino_ratio", "win_rate", "profit_factor", "expectancy", "cagr"]
        
        # Metrics to compare (lower is better)
        metrics_lower_better = ["max_drawdown_pct", "risk_of_ruin"]
        
        # Generate rankings for each metric
        for metric in metrics_higher_better + metrics_lower_better:
            # Extract values from each strategy
            values = {}
            for strategy_id, metrics in strategy_metrics.items():
                if metric in metrics:
                    values[strategy_id] = metrics[metric]
            
            if not values:
                continue
                
            # Sort strategies
            if metric in metrics_higher_better:
                sorted_strategies = sorted(values.items(), key=lambda x: x[1], reverse=True)
            else:
                sorted_strategies = sorted(values.items(), key=lambda x: x[1])
                
            # Store rankings
            rankings[metric] = {s[0]: i+1 for i, s in enumerate(sorted_strategies)}
        
        # Calculate overall ranking
        overall_ranking = {}
        for strategy_id in strategy_metrics:
            ranks = [rankings.get(metric, {}).get(strategy_id, len(strategy_metrics)) 
                    for metric in rankings.keys()]
            overall_ranking[strategy_id] = sum(ranks) / len(ranks) if ranks else 0
        
        # Return ranking summary
        return {
            "metric_rankings": rankings,
            "overall_ranking": dict(sorted(overall_ranking.items(), key=lambda x: x[1]))
        }