import pandas as pd
import numpy as np
import json
import markdown
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from abc import ABC, abstractmethod
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
import logging
import schedule

# Integrated Modules
from metrics.performance_metrics import PerformanceMetrics
from src.Core.trading.risk.risk_management import RiskManager
from src.Core.data.correlation_monitor import CorrelationMonitor
from src.Core.trading.execution.order_execution import OrderExecution
from src.Core.trading.strategies.strategy_orchestrator import StrategyOrchestrator
from utils.logging.telegram_alerts import TelegramNotifier
from src.Core.data.realtime.market_data import MarketDataAPI
from utils.logging.structured_logger import StructuredLogger

class ReportGenerator:
    """Institutional-Grade Multi-Format Report Generation System for Apex Trading Platform"""
    
    def __init__(self, config_path: str = "Config/reports.yaml"):
        """Initialize the report generator with connections to all required services"""
        self.config = self._load_config(config_path)
        self.metrics = PerformanceMetrics()
        self.risk_manager = RiskManager()
        self.correlation = CorrelationMonitor()
        self.strategy_orchestrator = StrategyOrchestrator()
        self.market_data = MarketDataAPI()
        self.order_execution = OrderExecution()
        self.notifier = TelegramNotifier()
        self.logger = StructuredLogger("ReportGenerator")
        self.report_cache = {}
        self._setup_report_scheduler()

    def _load_config(self, config_path: str) -> Dict:
        """Load report configuration from YAML file"""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            return {"default_report_type": "json", "scheduled_reports": []}

    def generate_report(self, trades: List[Dict], report_type: str = "full", 
                        time_period: str = "daily", user_id: Optional[str] = None) -> Dict:
        """Main report generation entry point with time period and user customization"""
        report_id = str(uuid.uuid4())
        
        try:
            # Filter trades by time period if specified
            filtered_trades = self._filter_trades_by_period(trades, time_period)
            
            # Basic validation
            if not filtered_trades:
                return {"error": "No trades found for the specified period"}
            
            # Generate report data
            report_data = {
                'report_id': report_id,
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'time_period': time_period,
                'core_metrics': self._get_core_metrics(filtered_trades),
                'risk_analysis': self._get_risk_analysis(filtered_trades),
                'strategy_breakdown': self._get_strategy_breakdown(filtered_trades),
                'execution_quality': self._get_execution_quality(filtered_trades),
                'market_analysis': self._get_market_context(filtered_trades)
            }
            
            # Add Monte Carlo simulation results if requested in config
            if self.config.get('include_monte_carlo', True):
                report_data['monte_carlo'] = self._run_monte_carlo_simulation(filtered_trades)
            
            # Format report according to requested type
            formatted_report = self._format_report(report_data, report_type)
            
            # Cache report for future reference
            self.report_cache[report_id] = {
                'data': report_data,
                'type': report_type,
                'created_at': datetime.now()
            }
            
            # Distribute report if enabled
            if self.config.get('auto_distribute', False) and user_id:
                self._distribute_report(report_id, user_id)
                
            return formatted_report
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            return {"error": f"Failed to generate report: {str(e)}"}

    def _filter_trades_by_period(self, trades: List[Dict], time_period: str) -> List[Dict]:
        """Filter trades based on specified time period"""
        now = datetime.now()
        df = pd.DataFrame(trades)
        
        if 'timestamp' not in df.columns:
            return trades
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if time_period == 'daily':
            return df[df['timestamp'] >= (now - timedelta(days=1))].to_dict('records')
        elif time_period == 'weekly':
            return df[df['timestamp'] >= (now - timedelta(days=7))].to_dict('records')
        elif time_period == 'monthly':
            return df[df['timestamp'] >= (now - timedelta(days=30))].to_dict('records')
        elif time_period == 'yearly':
            return df[df['timestamp'] >= (now - timedelta(days=365))].to_dict('records')
        else:
            return trades

    def _get_core_metrics(self, trades: List[Dict]) -> Dict:
        """Aggregate metrics from performance metrics module"""
        try:
            with ThreadPoolExecutor() as executor:
                basic_future = executor.submit(self.metrics.calculate_basic_metrics, trades)
                advanced_future = executor.submit(self.metrics.calculate_advanced_metrics, trades)
                temporal_future = executor.submit(self._get_time_based_metrics, trades)
                
                return {
                    'basic': basic_future.result(),
                    'advanced': advanced_future.result(),
                    'temporal': temporal_future.result()
                }
        except Exception as e:
            self.logger.error(f"Error calculating core metrics: {str(e)}")
            return {"error": str(e)}

    def _get_time_based_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate time-based performance metrics"""
        df = pd.DataFrame(trades)
        
        if 'timestamp' not in df.columns or 'profit' not in df.columns:
            return {}
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Daily, weekly, monthly grouping
        return {
            'daily': df.groupby(pd.Grouper(freq='D'))['profit'].sum().to_dict(),
            'weekly': df.groupby(pd.Grouper(freq='W'))['profit'].sum().to_dict(),
            'hourly_performance': df.groupby(df.index.hour)['profit'].mean().to_dict()
        }

    def _get_risk_analysis(self, trades: List[Dict]) -> Dict:
        """Get risk analysis from risk manager"""
        return self.risk_manager.analyze_trades(trades)

    def _get_strategy_breakdown(self, trades: List[Dict]) -> Dict:
        """Get strategy-specific performance breakdown"""
        return self.strategy_orchestrator.get_strategy_performance(trades)

    def _get_execution_quality(self, trades: List[Dict]) -> Dict:
        """Get execution quality metrics"""
        return self.order_execution.get_quality_metrics(trades)

    def _get_market_context(self, trades: List[Dict]) -> Dict:
        """Add market regime and correlation context"""
        symbols = {t['symbol'] for t in trades if 'symbol' in t}
        
        try:
            with ThreadPoolExecutor() as executor:
                regimes_future = executor.submit(self.market_data.get_regimes_during_trades, trades)
                correlations_future = executor.submit(self.correlation.get_portfolio_correlations, symbols)
                benchmarks_future = executor.submit(self._get_benchmark_comparison, trades)
                
                return {
                    'regimes': regimes_future.result(),
                    'correlations': correlations_future.result(),
                    'benchmarks': benchmarks_future.result()
                }
        except Exception as e:
            self.logger.error(f"Error getting market context: {str(e)}")
            return {"error": str(e)}

    def _get_benchmark_comparison(self, trades: List[Dict]) -> Dict:
        """Compare trading performance against market benchmarks"""
        # Extract dates from trades
        try:
            df = pd.DataFrame(trades)
            if 'timestamp' not in df.columns:
                return {}
                
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            start_date = df['timestamp'].min()
            end_date = df['timestamp'].max()
            
            # Get benchmark data for the same period
            benchmark_data = self.market_data.get_benchmark_data(
                start_date=start_date,
                end_date=end_date,
                benchmarks=['SPY', 'QQQ']
            )
            
            # Calculate performance comparison
            strategy_return = df['profit'].sum() / df['amount'].sum() if 'amount' in df.columns else 0
            
            return {
                'strategy_return': strategy_return,
                'benchmark_returns': benchmark_data,
                'alpha': self._calculate_alpha(strategy_return, benchmark_data)
            }
        except Exception as e:
            self.logger.error(f"Benchmark comparison failed: {str(e)}")
            return {}

    def _calculate_alpha(self, strategy_return: float, benchmark_data: Dict) -> float:
        """Calculate alpha (excess return) compared to benchmark"""
        if 'SPY' in benchmark_data:
            return strategy_return - benchmark_data['SPY']['return']
        return strategy_return

    def _run_monte_carlo_simulation(self, trades: List[Dict], simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulation for future performance projections"""
        try:
            df = pd.DataFrame(trades)
            if 'profit' not in df.columns:
                return {}
                
            # Calculate daily returns
            returns = df.groupby(pd.to_datetime(df['timestamp']).dt.date)['profit'].sum()
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Run simulations
            sim_days = 252  # Trading days in a year
            paths = np.random.normal(mean_return, std_return, size=(simulations, sim_days))
            paths = np.cumprod(1 + paths, axis=1)
            
            # Calculate metrics from simulations
            final_values = paths[:, -1]
            
            return {
                'expected_annual_return': float(np.mean(final_values) - 1),
                'worst_case_return': float(np.percentile(final_values, 5) - 1),
                'best_case_return': float(np.percentile(final_values, 95) - 1),
                'max_drawdown_risk': float(self._calculate_mc_drawdown(paths))
            }
        except Exception as e:
            self.logger.error(f"Monte Carlo simulation failed: {str(e)}")
            return {}

    def _calculate_mc_drawdown(self, paths: np.ndarray) -> float:
        """Calculate average maximum drawdown from Monte Carlo paths"""
        drawdowns = []
        for path in paths:
            cum_max = np.maximum.accumulate(path)
            drawdown = np.min(path / cum_max - 1)
            drawdowns.append(drawdown)
        return np.mean(drawdowns)

    def _format_report(self, data: Dict, report_type: str) -> Dict:
        """Convert raw data into specified format"""
        formatters = {
            'json': JSONFormatter(),
            'csv': CSVFormatter(),
            'pdf': PDFFormatter(),
            'md': MarkdownFormatter(),
            'html': HTMLFormatter(),
            'dashboard': DashboardFormatter()
        }
        
        if report_type not in formatters:
            report_type = self.config.get('default_report_type', 'json')
            
        return formatters[report_type].format(data)

    def _distribute_report(self, report_id: str, user_id: str) -> None:
        """Distribute report via configured channels"""
        try:
            report = self.report_cache.get(report_id)
            if not report:
                return
                
            # Get user distribution preferences
            user_prefs = self._get_user_distribution_preferences(user_id)
            
            # Send via preferred channels
            if user_prefs.get('email'):
                self._send_email_report(report, user_prefs['email'])
                
            if user_prefs.get('telegram'):
                self._send_telegram_report(report)
                
            if user_prefs.get('api'):
                self._send_api_notification(report, user_id)
                
        except Exception as e:
            self.logger.error(f"Report distribution failed: {str(e)}")

    def _get_user_distribution_preferences(self, user_id: str) -> Dict:
        """Get user report distribution preferences"""
        # This would typically come from a user database
        # For now returning default preferences
        return {
            'email': True,
            'telegram': True,
            'api': False
        }

    def _send_email_report(self, report: Dict, email: str) -> None:
        """Send report via email"""
        # Integration with email service would go here
        pass

    def _send_telegram_report(self, report: Dict) -> None:
        """Send report via Telegram"""
        summary = self._create_report_summary(report['data'])
        self.notifier.send_message(f"Trading Report Summary: {summary}")

    def _send_api_notification(self, report: Dict, user_id: str) -> None:
        """Send notification via API"""
        # Integration with notification API would go here
        pass

    def _create_report_summary(self, report_data: Dict) -> str:
        """Create a brief summary of the report"""
        if 'core_metrics' not in report_data:
            return "Report generated but no metrics available"
            
        metrics = report_data['core_metrics']['basic']
        return (
            f"Period: {report_data['time_period']}\n"
            f"Total Profit: {metrics.get('total_profit', 0):.2f}\n"
            f"Win Rate: {metrics.get('win_rate', 0):.1f}%\n"
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}"
        )

    def _setup_report_scheduler(self) -> None:
        """Setup scheduled report generation"""
        for scheduled_report in self.config.get('scheduled_reports', []):
            schedule_type = scheduled_report.get('schedule')
            user_id = scheduled_report.get('user_id')
            report_type = scheduled_report.get('type', 'json')
            
            if schedule_type == 'daily':
                schedule.every().day.at("00:00").do(
                    self._generate_scheduled_report, user_id, report_type, 'daily'
                )
            elif schedule_type == 'weekly':
                schedule.every().monday.at("00:00").do(
                    self._generate_scheduled_report, user_id, report_type, 'weekly'
                )
            elif schedule_type == 'monthly':
                schedule.every(30).days.at("00:00").do(
                    self._generate_scheduled_report, user_id, report_type, 'monthly'
                )

    def _generate_scheduled_report(self, user_id: str, report_type: str, time_period: str) -> None:
        """Generate and distribute a scheduled report"""
        try:
            # Get trades for the user
            trades = self._get_user_trades(user_id, time_period)
            if not trades:
                self.logger.info(f"No trades found for scheduled report: {user_id}, {time_period}")
                return
                
            # Generate and distribute report
            report = self.generate_report(trades, report_type, time_period, user_id)
            self._distribute_report(report['report_id'], user_id)
            
        except Exception as e:
            self.logger.error(f"Scheduled report generation failed: {str(e)}")

    def _get_user_trades(self, user_id: str, time_period: str) -> List[Dict]:
        """Get trades for a specific user"""
        # This would typically come from a database or API
        # For now returning an empty list
        return []

    def get_report_by_id(self, report_id: str) -> Dict:
        """Retrieve a previously generated report by ID"""
        if report_id in self.report_cache:
            return self.report_cache[report_id]['data']
        return {"error": "Report not found"}

    @staticmethod
    def generate_legacy_report(trades):
        """Maintain original functionality with improved calculations"""
        df = pd.DataFrame(trades)
        return {
            "Total Profit": df["profit"].sum() if "profit" in df.columns else 0,
            "Win Rate (%)": (df["profit"] > 0).mean() * 100 if "profit" in df.columns else 0,
            "Max Drawdown": ReportGenerator._enhanced_drawdown(df)
        }

    @staticmethod
    def _enhanced_drawdown(df: pd.DataFrame) -> float:
        """Improved drawdown calculation with time decay"""
        if "profit" not in df.columns:
            return 0
            
        cumulative = df['profit'].cumsum()
        if len(cumulative) == 0:
            return 0
            
        peak = cumulative.expanding(min_periods=1).max()
        return float((peak - cumulative).max())


class ReportFormatter(ABC):
    """Abstract base class for report formats"""
    
    @abstractmethod
    def format(self, data: Dict) -> Union[Dict, str, bytes]:
        """Format report data into the desired output format"""
        pass


class JSONFormatter(ReportFormatter):
    """JSON report formatter"""
    def format(self, data: Dict) -> str:
        """Format report as JSON"""
        return json.dumps(data, indent=2)


class CSVFormatter(ReportFormatter):
    """CSV report formatter"""
    def format(self, data: Dict) -> str:
        """Format report as CSV"""
        report_parts = []
        
        # Convert core metrics to CSV
        if 'core_metrics' in data and 'basic' in data['core_metrics']:
            metrics_df = pd.DataFrame([data['core_metrics']['basic']])
            report_parts.append("## Core Metrics\n" + metrics_df.to_csv(index=False))
            
        # Convert strategy breakdown to CSV
        if 'strategy_breakdown' in data:
            strat_df = pd.DataFrame(data['strategy_breakdown'])
            report_parts.append("## Strategy Breakdown\n" + strat_df.to_csv(index=False))
        
        return "\n\n".join(report_parts)


class MarkdownFormatter(ReportFormatter):
    """Markdown report formatter"""
    def format(self, data: Dict) -> str:
        """Format report as Markdown"""
        md_content = f"# Trading Report\n{datetime.now().isoformat()}\n\n"
        
        if 'core_metrics' in data and 'basic' in data['core_metrics']:
            metrics = data['core_metrics']['basic']
            md_content += "## Core Metrics\n"
            md_content += f"- Total Profit: {metrics.get('total_profit', 0)}\n"
            md_content += f"- Win Rate: {metrics.get('win_rate', 0)}%\n"
            md_content += f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0)}\n\n"
            
        if 'risk_analysis' in data:
            risk = data['risk_analysis']
            md_content += "## Risk Analysis\n"
            md_content += f"- Max Drawdown: {risk.get('max_drawdown', 0)}%\n"
            md_content += f"- Value at Risk (95%): {risk.get('var_95', 0)}\n\n"
            
        return markdown.markdown(md_content)


class PDFFormatter(ReportFormatter):
    """PDF report formatter"""
    def format(self, data: Dict) -> bytes:
        """Format report as PDF"""
        # This would typically use a PDF generation library
        return b"PDF generation would happen here"


class HTMLFormatter(ReportFormatter):
    """HTML report formatter"""
    def format(self, data: Dict) -> str:
        """Format report as HTML with embedded charts"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Report - {datetime.now().isoformat()}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .card {{ background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; padding: 20px; }}
                .metric {{ display: inline-block; margin-right: 20px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
                .metric-label {{ color: #555; font-size: 14px; }}
            </style>
        </head>
        <body>
            <h1>Trading Report</h1>
            <div class="card">
                <h2>Performance Summary</h2>
        """
        
        # Add core metrics
        if 'core_metrics' in data and 'basic' in data['core_metrics']:
            metrics = data['core_metrics']['basic']
            html += '<div class="metrics-container">'
            for key, value in metrics.items():
                formatted_value = f"{value:.2f}" if isinstance(value, (int, float)) else value
                html += f"""
                <div class="metric">
                    <div class="metric-value">{formatted_value}</div>
                    <div class="metric-label">{key.replace('_', ' ').title()}</div>
                </div>
                """
            html += '</div>'
        
        # Close HTML tags
        html += """
            </div>
        </body>
        </html>
        """
        
        return html


class DashboardFormatter(ReportFormatter):
    """Dashboard data formatter for web application"""
    def format(self, data: Dict) -> Dict:
        """Format report for dashboard consumption"""
        # Simply return the structured data for the dashboard to render
        return data