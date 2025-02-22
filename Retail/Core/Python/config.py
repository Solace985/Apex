import yaml
import os
import json
import logging
import threading
import keyring
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Load environment variables
load_dotenv()
# Fetch encryption key securely and store in memory
CACHED_SECRET_KEY = keyring.get_password("ApexBot", "CONFIG_SECRET_KEY")
if not CACHED_SECRET_KEY:
    CACHED_SECRET_KEY = Fernet.generate_key().decode()
    keyring.set_password("ApexBot", "CONFIG_SECRET_KEY", CACHED_SECRET_KEY)

cipher = Fernet(CACHED_SECRET_KEY.encode())

def encrypt_api_key(api_key: str) -> str:
    return cipher.encrypt(api_key.encode()).decode()

def decrypt_api_key(encrypted_key: str) -> str:
    return cipher.decrypt(encrypted_key.encode()).decode()

# 🔹 Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 🔹 Define Config Classes with Defaults
class ExecutionConfig(BaseModel):
    slippage_tolerance: float = 0.01
    latency_budget_ms: int = 50
    retry_attempts: int = 3

class RiskConfig(BaseModel):
    max_drawdown: float = 0.1
    daily_loss_limit: float = 0.05
    volatility_threshold: float = 0.2
    risk_threshold: float = 0.15
    stop_loss: float = 0.02
    take_profit: float = 0.05
    position_sizing_strategy: str = "Kelly Criterion"

class AIConfig(BaseModel):
    enable_live_adaptation: bool = True
    strategy_selection: str = "dynamic"
    model_path: str = "models/best_model.pkl"

class BacktestingConfig(BaseModel):
    use_backtest: bool = False
    historical_data_path: str = "data/historical.csv"
    start_date: str = "2023-01-01"
    end_date: str = "2024-01-01"
    capital: float = 100000.0
    commission: float = 0.001

class WebsocketConfig(BaseModel):
    enable_real_time_data: bool = True
    polygon_key: Optional[str] = Field(default=None, description="Encrypted API Key")
    symbols: List[str] = ["AAPL", "TSLA", "BTC/USD"]

    def get_polygon_key(self, use_decrypted=False) -> str:
        """ Only decrypt API key when explicitly requested. """
        if use_decrypted:
            return decrypt_api_key(self.polygon_key) if self.polygon_key else None
        return self.polygon_key  # Return encrypted by default

class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "logs/trading.log"

class DatabaseConfig(BaseModel):
    path: str = "db/trading.db"
    backup_frequency: str = "daily"
    log_trades: bool = True

class NotificationConfig(BaseModel):
    enable_alerts: bool = True
    email_alerts: bool = False
    telegram_alerts: bool = True
    telegram_api_key: Optional[str] = Field(default=None, description="Encrypted API Key")
    telegram_chat_id: str = ""

    def get_telegram_key(self, use_decrypted=False) -> str:
        """ Only decrypt API key when explicitly requested. """
        if use_decrypted:
            return decrypt_api_key(self.telegram_api_key) if self.telegram_api_key else None
        return self.telegram_api_key  # Return encrypted by default

class StrategyConfig(BaseModel):
    enabled: List[str] = ["TrendFollowing", "MeanReversion"]

class TradingModeConfig(BaseModel):
    live_trading: bool = False
    testnet: bool = True

class RetailConfig(BaseModel):
    mode: str = "paper"
    execution: ExecutionConfig = ExecutionConfig()
    risk: RiskConfig = RiskConfig()
    ai_trading: AIConfig = AIConfig()
    backtesting: BacktestingConfig = BacktestingConfig()
    websocket: WebsocketConfig = WebsocketConfig()
    logging: LoggingConfig = LoggingConfig()
    database: DatabaseConfig = DatabaseConfig()
    notifications: NotificationConfig = NotificationConfig()
    strategies: StrategyConfig = StrategyConfig()
    trading_mode: TradingModeConfig = TradingModeConfig()

    class Config:
        extra = "ignore"  # Ignores unknown fields instead of crashing

# 🔹 Load Configuration Function with Live Reloading
def load_config():
    """Loads and validates settings from config.yaml and settings.yaml, with encryption support."""
    config_path = "Retail/Config/config.yaml"
    settings_path = "Retail/Config/settings.yaml"

    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}

        with open(settings_path, "r") as f:
            settings_data = yaml.safe_load(f) or {}

        # Merge settings.yaml into config.yaml
        merged_config = {**config_data, **settings_data}

        return RetailConfig(**merged_config)

    except FileNotFoundError as e:
        logger.error(f"❌ Configuration file missing: {e}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"❌ YAML Parsing Error: {e}")
        return None
    except ValidationError as e:
        logger.error(f"❌ Configuration Validation Error: {e}")
        return None

# 🔹 Live Configuration Reloading (Uses Watchdog)
class ConfigFileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path in ["Retail/Config/config.yaml", "Retail/Config/settings.yaml"]:
            logger.info("🔄 Config file updated. Reloading...")
            load_config()

def watch_config():
    """ Monitors config.yaml for changes and reloads automatically. """
    event_handler = ConfigFileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path="Retail/Config", recursive=False)
    observer.start()

    def stop_observer():
        observer.stop()
        observer.join()
    
    threading.Thread(target=stop_observer, daemon=True).start()

# Start watching for config changes in the background
watch_config()

# Load initial config
config = load_config()
logger.info(f"✅ Trading Mode: {config.mode}")
logger.info(f"✅ Enabled Strategies: {config.strategies.enabled}")

# -------- Key Management System --------
class SecureVault:
    _VAULT_KEY = os.environ.get('VAULT_KEY', Fernet.generate_key())
    _cipher_suite = Fernet(_VAULT_KEY)
    
    @classmethod
    def encrypt(cls, plaintext: str) -> bytes:
        return cls._cipher_suite.encrypt(plaintext.encode())
    
    @classmethod
    def decrypt(cls, ciphertext: bytes) -> str:
        return cls._cipher_suite.decrypt(ciphertext).decode()

# -------- Environment Validation --------
class EnvValidator:
    REQUIRED_ENV = ['VAULT_KEY', 'EXCHANGE_ENV']
    
    @classmethod
    def validate(cls):
        missing = [var for var in cls.REQUIRED_ENV if not os.environ.get(var)]
        if missing:
            raise EnvironmentError(f"Critical env vars missing: {missing}")

# -------- Configuration Loader --------
class AppConfig:
    def __init__(self):
        EnvValidator.validate()
        
        self.API_KEY = SecureVault.decrypt(
            os.environ['ENCRYPTED_API_KEY']
        )
        self.SECRET_KEY = SecureVault.decrypt(
            os.environ['ENCRYPTED_SECRET_KEY']
        )
        self.STRATEGY = os.getenv('TRADING_STRATEGY', 'SAFE_MODE')
        
        # Runtime validation
        if not self._validate_strategy():
            logging.critical("Invalid strategy configuration")
            self.STRATEGY = 'FAILSAFE'
    
    def _validate_strategy(self) -> bool:
        VALID_STRATEGIES = ['LSTM_V1', 'SAFE_MODE', 'FAILSAFE']
        return self.STRATEGY in VALID_STRATEGIES

# -------- Singleton Initialization --------
config = AppConfig()
