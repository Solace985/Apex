import yaml
import os
from pydantic import BaseModel
from typing import List, Dict

# 🔹 Define Config Classes
class ExecutionConfig(BaseModel):
    slippage_tolerance: float
    latency_budget_ms: int
    retry_attempts: int

class RiskConfig(BaseModel):
    max_drawdown: float
    daily_loss_limit: float
    volatility_threshold: float
    risk_threshold: float
    stop_loss: float
    take_profit: float
    position_sizing_strategy: str

class AIConfig(BaseModel):
    enable_live_adaptation: bool
    strategy_selection: str
    model_path: str

class BacktestingConfig(BaseModel):
    use_backtest: bool
    historical_data_path: str
    start_date: str
    end_date: str
    capital: float
    commission: float

class WebsocketConfig(BaseModel):
    enable_real_time_data: bool
    polygon_key: str
    symbols: List[str]

class LoggingConfig(BaseModel):
    level: str
    log_to_file: bool
    log_file_path: str

class DatabaseConfig(BaseModel):
    path: str
    backup_frequency: str
    log_trades: bool

class NotificationConfig(BaseModel):
    enable_alerts: bool
    email_alerts: bool
    telegram_alerts: bool
    telegram_api_key: str
    telegram_chat_id: str

class StrategyConfig(BaseModel):
    enabled: List[str]

class TradingModeConfig(BaseModel):
    live_trading: bool
    testnet: bool

class RetailConfig(BaseModel):
    mode: str
    execution: ExecutionConfig
    risk: RiskConfig
    ai_trading: AIConfig
    backtesting: BacktestingConfig
    websocket: WebsocketConfig
    logging: LoggingConfig
    database: DatabaseConfig
    notifications: NotificationConfig
    strategies: StrategyConfig
    trading_mode: TradingModeConfig

# 🔹 Load Configuration Function
def load_config():
    """Loads settings from both config.yaml & settings.yaml and merges them."""
    config_path = "Retail/Config/config.yaml"
    settings_path = "Retail/Config/settings.yaml"
    
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        with open(settings_path, "r") as f:
            settings_data = yaml.safe_load(f)
        
        # Merge settings.yaml into config.yaml
        merged_config = {**config_data, **settings_data}
        return RetailConfig(**merged_config)
    except FileNotFoundError as e:
        raise Exception(f"❌ Configuration file missing: {e}")
    except yaml.YAMLError as e:
        raise Exception(f"❌ YAML Parsing Error: {e}")

# Example Usage
config = load_config()
print(f"Trading Mode: {config.mode}")  # Prints: "testing" or "live"
print(f"Enabled Strategies: {config.strategies.enabled}")  # Prints: ['TrendFollowing', 'MeanReversion']
