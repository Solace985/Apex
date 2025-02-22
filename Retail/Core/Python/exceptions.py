class ExecutionError(Exception):
    """Base class for all execution errors"""
    def __init__(self, message, original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception
        self.code = "EX000"

class RetryableError(ExecutionError):
    """Transient errors that warrant retry"""
    def __init__(self, message, retry_after=60):
        super().__init__(f"Retryable: {message}")
        self.code = "EX429"
        self.retry_after = retry_after

class FatalExecutionError(ExecutionError):
    """Permanent failures requiring intervention"""
    def __init__(self, message):
        super().__init__(f"Fatal: {message}")
        self.code = "EX500"

class OrderValidationError(FatalExecutionError):
    """Invalid order structure or values"""
    def __init__(self, field, reason):
        super().__init__(f"Validation failed for {field}: {reason}")
        self.code = "EX400"

class BrokerError(RetryableError):
    """Broker-specific transient errors"""
    def __init__(self, broker, message):
        super().__init__(f"{broker} error: {message}")
        self.code = "EX503"

class CircuitBreakerError(FatalExecutionError):
    """Circuit breaker has tripped"""
    def __init__(self, instrument):
        super().__init__(f"Circuit tripped for {instrument}")
        self.code = "EX503"

class SlippageExceededError(ExecutionError):
    """Order rejected due to excessive slippage"""
    def __init__(self, allowed, actual):
        super().__init__(f"Slippage {actual}% > {allowed}%")
        self.code = "EX406"

class OrderTooLargeError(OrderValidationError):
    """Order size exceeds market liquidity"""
    def __init__(self, symbol, attempted, available):
        super().__init__("amount", 
            f"{attempted} exceeds {available} for {symbol}")
        self.code = "EX413"

class RiskCheckFailedError(FatalExecutionError):
    """Order blocked by risk management system"""
    def __init__(self, check_name):
        super().__init__(f"Risk check failed: {check_name}")
        self.code = "EX451"