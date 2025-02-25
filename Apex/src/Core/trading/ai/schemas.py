from pydantic import BaseModel, Field, confloat, validator
from datetime import datetime
from typing import Optional, Literal
import re

class OrderSchema(BaseModel):
    """Validated trading order schema"""
    symbol: str = Field(..., min_length=2, max_length=10, 
                      description="Asset ticker symbol in uppercase")
    amount: confloat(gt=0, le=1000000) = Field(..., 
                      description="Positive quantity with max 1M limit")
    price: confloat(gt=0) = Field(..., 
                      description="Price per unit must be positive")
    order_type: Literal["LIMIT", "MARKET", "STOP", "STOP_LIMIT"] = "LIMIT"
    strategy_id: Optional[str] = Field(None, min_length=4,
                      description="Optional strategy identifier")
    tif: Literal["GTC", "IOC", "FOK"] = "GTC"
    expiration: Optional[datetime] = None
    client_order_id: Optional[str] = Field(None, regex=r"^[A-Z0-9-]{8,20}$")

    @validator('symbol')
    def validate_symbol(cls, value):
        pattern = r"^[A-Z0-9/.]+$"
        if not re.match(pattern, value):
            raise ValueError("Invalid symbol format")
        return value.upper()

class ExecutionResult(BaseModel):
    """Standardized execution outcome schema"""
    status: Literal["SUCCESS", "FAILED", "PARTIALLY_FILLED"]
    order_id: Optional[str] = Field(None, regex=r"^[A-Z0-9-]+$")
    execution_time: datetime
    broker: str
    filled_price: Optional[confloat(gt=0)] = None
    requested_price: confloat(gt=0)
    fees: confloat(ge=0) = 0.0
    filled_qty: confloat(ge=0) = 0.0
    remaining_qty: confloat(ge=0) = 0.0
    slippage: Optional[float] = None
    impact_cost: Optional[float] = None
    market_impact: Optional[float] = None

    @validator('remaining_qty')
    def validate_qty_consistency(cls, v, values):
        if 'filled_qty' in values and 'amount' in values:
            if not (0 <= v <= values['amount'] - values['filled_qty']):
                raise ValueError("Invalid remaining quantity")
        return v