from abc import ABC, abstractmethod
from typing import Dict, Any

class BrokerAPI(ABC):
    """Abstract class for broker integration."""

    @abstractmethod
    def place_order(self, order_details: Dict[str, Any]) -> Dict[str, Any]:
        """Places an order with the broker."""
        pass

    @abstractmethod
    def fetch_account_balance(self) -> Dict[str, Any]:
        """Fetches the account balance from the broker."""
        pass

    @abstractmethod
    def fetch_open_orders(self) -> Dict[str, Any]:
        """Fetches all open orders from the broker."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancels an order using the broker API."""
        pass
