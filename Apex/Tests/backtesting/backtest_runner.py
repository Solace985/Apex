"""
Institutional-Grade Backtesting Engine - Core Component
Integrates with: market_data.py, meta_trader.py, risk_management.py, decision_logger.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from pathlib import Path

# Integrated with existing project modules
from src.Core.data.market_data import MarketDataAPI
from src.Core.trading.ai.meta_trader import MetaTrader
from src.Core.trading.logging.decision_logger import DecisionLogger
from src.Core.trading.risk.risk_management import RiskEngine
from src.Metrics.performance_metrics import SharpeRatio, MaxDrawdownCalculator

class BacktestOrchestrator:
    def __init__(self):
        self.data_loader = MarketDataAPI()
        self.simulator = BacktestEngine()
        self.risk_engine = RiskEngine()
        self.logger = DecisionLogger()
        self.performance = BacktestMetrics()

    def run_full_backtest(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = '1D'
    ) -> Dict:
        """Main entry point for institutional-grade backtesting"""
        results = {}
        
        for symbol in symbols:
            # Integrated with market_data.py
            hist_data = self.data_loader.get_historical_data(
                symbol, start, end, timeframe
            )
            
            # Walk-forward optimization
            for train_data, test_data in self._walk_forward_split(hist_data):
                # Simulate AI model retraining
                self._retrain_ai_models(train_data)
                
                # Execute backtest
                trades = self.simulator.run(test_data)
                
                # Analyze results
                results[symbol] = self.performance.calculate(trades)
                
                # Integrated with decision_logger.py
                self._log_backtest_results(trades, symbol)
        
        return self._generate_final_report(results)

    def _walk_forward_split(self, data: pd.DataFrame, 
                          train_ratio: float = 0.7,
                          window: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Walk-forward optimization using existing data"""
        train_size = int(len(data) * train_ratio)
        for i in range(train_size, len(data), window):
            yield data.iloc[:i], data.iloc[i:i+window]

    def _retrain_ai_models(self, train_data: pd.DataFrame):
        """Integrated with reinforcement_learning.py"""
        from src.ai.reinforcement.learning import update_models
        update_models(train_data)

    def _log_backtest_results(self, trades: List[Dict], symbol: str):
        """Integrated with decision_logger.py"""
        for trade in trades:
            self.logger.log_decision({
                'symbol': symbol,
                'timestamp': trade['entry_time'],
                'action': trade['action'],
                'size': trade['size'],
                'model_weights': trade['model_weights'],
                'risk_params': trade['risk_params']
            })

    def _generate_final_report(self, results: Dict) -> Dict:
        """Integrated with report_generator.py"""
        from report_generator import create_performance_report
        return create_performance_report(results)


class BacktestEngine:
    def __init__(self):
        self.meta_trader = MetaTrader(simulated=True)
        self.current_positions = {}

    def run(self, data: pd.DataFrame) -> List[Dict]:
        """Execute backtest using existing trading infrastructure"""
        trades = []
        
        for idx, row in data.iterrows():
            # Get AI decision from meta_trader.py
            decision = self.meta_trader.generate_signal(row.to_dict())
            
            # Apply risk management from risk_management.py
            if not self.risk_engine.approve_trade(decision):
                continue
                
            # Execute simulated trade
            trade_result = self.meta_trader.execute_trade(
                symbol=decision['symbol'],
                action=decision['action'],
                size=decision['size'],
                price=row['close']
            )
            
            # Track performance
            trades.append({
                **trade_result,
                'model_weights': decision['model_weights'],
                'risk_params': decision['risk_params']
            })
            
            # Update positions
            self._update_positions(trade_result)
        
        return trades

    def _update_positions(self, trade: Dict):
        """Position management using existing execution engine"""
        symbol = trade['symbol']
        if trade['action'] == 'EXIT':
            del self.current_positions[symbol]
        else:
            self.current_positions[symbol] = trade


class BacktestMetrics:
    def __init__(self):
        self.metrics = {
            'sharpe': SharpeRatio(),
            'drawdown': MaxDrawdownCalculator(),
            'win_rate': lambda x: np.mean(x)
        }

    def calculate(self, trades: List[Dict]) -> Dict:
        """Comprehensive performance analysis"""
        returns = [t['pnl_pct'] for t in trades if t['pnl_pct'] is not None]
        winning = [t for t in trades if t['pnl'] > 0]
        
        return {
            'sharpe_ratio': self.metrics['sharpe'](returns),
            'max_drawdown': self.metrics['drawdown'](returns),
            'win_rate': self.metrics['win_rate'](len(winning)/len(trades)),
            'profit_factor': sum(t['pnl'] for t in winning) / 
                            abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        }


# Security Validation
class BacktestValidator:
    @staticmethod
    def validate_inputs(data: pd.DataFrame):
        """Integrated with validation.rs"""
        from utils.helpers.validation.rs import validate_ohlc
        if not validate_ohlc(data):
            raise ValueError("Invalid OHLC data structure")
            
    @staticmethod
    def sanitize_trade(trade: Dict) -> Dict:
        """Prevent injection attacks"""
        return {k: v for k, v in trade.items() if k in [
            'symbol', 'action', 'size', 'price', 'timestamp'
        ]}


# Integration Tests
if __name__ == "__main__":
    # Initialize with existing config
    from src.Config.config_loader import load_backtest_config
    
    config = load_backtest_config()
    orchestrator = BacktestOrchestrator()
    
    # Run full backtest using existing asset universe
    results = orchestrator.run_full_backtest(
        symbols=config['assets'],
        start=datetime(2023, 1, 1),
        end=datetime(2024, 1, 1),
        timeframe=config['timeframe']
    )
    
    print("\n=== Backtest Results ===")
    print(json.dumps(results, indent=2))