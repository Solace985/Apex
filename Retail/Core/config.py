import yaml
from pydantic import BaseModel

class ExecutionConfig(BaseModel):
    slippage_tolerance: float = 0.001
    latency_budget_ms: int = 50

class RiskConfig(BaseModel):
    max_drawdown: float = 5.0
    daily_loss_limit: float = 2.0
    volatility_threshold: float = 35.0

class RetailConfig(BaseModel):
    execution: ExecutionConfig
    risk: RiskConfig

def load_config():
    with open("Retail/config/settings.yaml") as f:
        raw_config = yaml.safe_load(f)
    return RetailConfig(**raw_config)
