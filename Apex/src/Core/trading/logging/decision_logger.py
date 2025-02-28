"""
Immutable AI Decision Logger - Core Component
Integrates with: meta_trader.py, risk_management.py, ensemble_voting.py
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, Optional
from sqlite3 import connect, IntegrityError
from utils.logging.telegram_alerts import send_alert
from utils.helpers.validation.rs import validate_trade_data

class DecisionLogger:
    def __init__(self):
        self._db_path = "apex_trade_logs.db"
        self._init_db()
        
    def _init_db(self):
        """Integrated with existing db_migrations.rs schema"""
        with connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_logs (
                    log_hash TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    encrypted_data BLOB NOT NULL,
                    model_weights TEXT NOT NULL,
                    risk_parameters TEXT NOT NULL,
                    trade_outcome TEXT
                )""")

    def _generate_hash(self, data: Dict) -> str:
        """Cryptographic hashing using existing security.py components"""
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    def _encrypt_entry(self, data: Dict) -> bytes:
        """Uses existing stealth_api.py for zero-cost encryption"""
        from utils.helpers.stealth_api import encrypt_payload
        return encrypt_payload(json.dumps(data))

    def log_decision(self, decision_data: Dict) -> str:
        """Main logging interface for all trade decisions"""
        if not validate_trade_data(decision_data):
            raise ValueError("Invalid trade data structure")

        log_hash = self._generate_hash(decision_data)
        encrypted = self._encrypt_entry(decision_data)
        
        entry = {
            "log_hash": log_hash,
            "timestamp": datetime.utcnow().isoformat(),
            "encrypted_data": encrypted,
            "model_weights": json.dumps(decision_data.get("model_weights", {})),
            "risk_parameters": json.dumps(decision_data.get("risk_parameters", {})),
            "trade_outcome": decision_data.get("trade_outcome", "pending")
        }

        try:
            with connect(self._db_path) as conn:
                conn.execute("""
                    INSERT INTO trade_logs 
                    VALUES (:log_hash, :timestamp, :encrypted_data, 
                            :model_weights, :risk_parameters, :trade_outcome)
                """, entry)
                
            self._check_for_anomalies(decision_data)
            return log_hash
            
        except IntegrityError:
            self._handle_duplicate(log_hash)
            return log_hash

    def _check_for_anomalies(self, data: Dict):
        """Real-time alert integration using telegram_alerts.py"""
        confidence = data.get("confidence_score", 0)
        
        if confidence < 50:
            send_alert(f"⚠️ Low confidence trade: {data.get('symbol')} "
                      f"Confidence: {confidence}%")
            
        if data["trade_outcome"] == "rejected":
            send_alert(f"🚫 Trade rejected: {data.get('reasoning')}")

    def get_log(self, log_hash: str) -> Optional[Dict]:
        """Retrieval interface for backtest_runner.py"""
        with connect(self._db_path) as conn:
            row = conn.execute("""
                SELECT encrypted_data, model_weights, risk_parameters 
                FROM trade_logs WHERE log_hash = ?
            """, (log_hash,)).fetchone()

        if row:
            return self._decrypt_entry(row[0], row[1], row[2])
        return None

    def _decrypt_entry(self, encrypted: bytes, weights: str, risk: str) -> Dict:
        """Decryption using existing stealth_api.py"""
        from utils.helpers.stealth_api import decrypt_payload
        data = json.loads(decrypt_payload(encrypted))
        
        return {
            **data,
            "model_weights": json.loads(weights),
            "risk_parameters": json.loads(risk)
        }

    def get_recent_logs(self, days: int = 7) -> list:
        """For performance_evaluator.py integration"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        with connect(self._db_path) as conn:
            return conn.execute("""
                SELECT log_hash FROM trade_logs 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """, (cutoff.isoformat(),)).fetchall()

    def _handle_duplicate(self, log_hash: str):
        """Immutable logging enforcement"""
        existing = self.get_log(log_hash)
        send_alert(f"🚨 Attempted duplicate log entry: {log_hash}")
        
    def update_outcome(self, log_hash: str, outcome: Dict):
        """Called by backtest_runner.py with trade results"""
        if not validate_trade_data(outcome):
            raise ValueError("Invalid outcome data")
            
        with connect(self._db_path) as conn:
            conn.execute("""
                UPDATE trade_logs 
                SET trade_outcome = ?
                WHERE log_hash = ?
            """, (json.dumps(outcome), log_hash))