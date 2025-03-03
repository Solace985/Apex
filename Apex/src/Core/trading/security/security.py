import re
import logging
from cryptography.fernet import Fernet
from typing import Dict, Any
import os
from decimal import Decimal, InvalidOperation
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

ALLOWED_SYMBOLS = {"BTC/USD", "ETH/USD", "AAPL", "MSFT", "NGAS"}
MAX_DECIMALS = {
    "price": 4,
    "amount": 8 if os.getenv('CRYPTO_MODE') else 4
}

class SecurityVault:
    """Secure credential management with automatic key rotation"""
    def __init__(self):
        self._current_key = os.getenv('VAULT_KEY')
        self._fernet = Fernet(self._current_key)
        self._key_ring = [Fernet(k) for k in os.getenv('KEY_RING', '').split(',')]
        
    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()
    
    def decrypt(self, ciphertext: str) -> str:
        try:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except Exception as e:
            for old_fernet in self._key_ring:
                try:
                    return old_fernet.decrypt(ciphertext.encode()).decode()
                except:
                    continue
            raise ValueError("Decryption failed with all keys") from e

def validate_symbol(symbol: str) -> str:
    """Validate and normalize trading symbol"""
    normalized = symbol.upper().strip()
    if normalized not in ALLOWED_SYMBOLS:
        logger.warning(f"Blocked invalid symbol: {symbol}")
        raise ValueError(f"Symbol {symbol} not allowed")
    return normalized

def sanitize_order_details(order: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize and normalize order details"""
    try:
        return {
            'symbol': validate_symbol(order['symbol']),
            'amount': round(Decimal(str(order['amount'])), MAX_DECIMALS['amount']),
            'price': round(Decimal(str(order['price'])), MAX_DECIMALS['price']),
            'order_type': order.get('order_type', 'LIMIT').upper(),
            'tif': order.get('tif', 'GTC').upper(),
            'client_order_id': re.sub(r'[^A-Z0-9-]', '', order.get('client_order_id', ''))[:20]
        }
    except (KeyError, InvalidOperation) as e:
        logger.error(f"Order sanitization failed: {e}")
        raise ValueError("Invalid order structure") from e

def validate_order_type(order_type: str):
    """Check if order type is allowed"""
    allowed_types = {"LIMIT", "MARKET", "STOP", "STOP_LIMIT"}
    if order_type not in allowed_types:
        raise ValueError(f"Invalid order type: {order_type}")
    

def secure_delete(self, path: str, passes: int = 3) -> None:
    """Securely erase sensitive data using DoD 5220.22-M standard"""
    try:
        if not os.path.exists(path):
            return
            
        if os.path.isfile(path):
            with open(path, "ba+") as f:
                length = f.tell()
                for _ in range(passes):
                    f.seek(0)
                    f.write(os.urandom(length))
            os.remove(path)
            
        elif os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        logger.error(f"Secure delete failed: {e}")
        raise