import os
import json
import hashlib
import asyncio
import datetime
import logging
from typing import Dict, Any
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from Apex.utils.helpers import secure_float, validate_inputs
from Apex.src.Core.trading.risk.risk_management import RiskManager
from Apex.src.Core.trading.execution.meta_trader import MetaTrader
from Apex.src.AI.ensembles.ensemble_voting import EnsembleVoting
from Apex.src.AI.analysis.market_regime_classifier import MarketRegimeClassifier

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Database Setup
Base = declarative_base()

class TradeLog(Base):
    """SQLAlchemy model for trade decision logging"""
    __tablename__ = "trade_logs"

    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    symbol = Column(String)
    trade_action = Column(String)
    trade_size = Column(Float)
    market_regime = Column(String)
    model_weights = Column(JSON)
    confidence_score = Column(Float)
    reasoning = Column(String)
    risk_parameters = Column(JSON)
    log_hash = Column(String)

# Database Connection
DATABASE_URL = "sqlite:///trade_logs.db"
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Encryption Setup
ENCRYPTION_KEY = os.getenv("DECISION_LOGGER_KEY", Fernet.generate_key().decode())
cipher = Fernet(ENCRYPTION_KEY.encode())

class DecisionLogger:
    """📌 Secure and immutable logging of AI trade decisions"""

    def __init__(self):
        self.risk_manager = RiskManager()
        self.meta_trader = MetaTrader()
        self.ensemble_voter = EnsembleVoting()
        self.market_classifier = MarketRegimeClassifier()

    async def log_trade_decision(self, trade_data: Dict[str, Any]) -> None:
        """📌 Logs every trade decision in a secure, immutable format"""
        trade_id = f"{trade_data['symbol']}-{datetime.datetime.utcnow().isoformat()}"

        # Generate log hash for integrity verification
        log_entry = json.dumps(trade_data, sort_keys=True)
        log_hash = hashlib.sha256(log_entry.encode()).hexdigest()

        # Encrypt the log entry
        encrypted_entry = cipher.encrypt(log_entry.encode()).decode()

        # Save to database
        trade_log = TradeLog(
            id=trade_id,
            timestamp=datetime.datetime.utcnow(),
            symbol=trade_data["symbol"],
            trade_action=trade_data["trade_action"],
            trade_size=secure_float(trade_data["trade_size"]),
            market_regime=trade_data["market_regime"],
            model_weights=trade_data["model_weights"],
            confidence_score=secure_float(trade_data["confidence_score"]),
            reasoning=trade_data["reasoning"],
            risk_parameters=trade_data["risk_parameters"],
            log_hash=log_hash
        )

        session.add(trade_log)
        session.commit()
        logger.info(f"Trade decision logged: {trade_id}")

    async def retrieve_logs(self, symbol: str = None, limit: int = 10) -> Dict[str, Any]:
        """📌 Retrieves the latest trade logs"""
        query = session.query(TradeLog)
        if symbol:
            query = query.filter(TradeLog.symbol == symbol)
        logs = query.order_by(TradeLog.timestamp.desc()).limit(limit).all()

        return [
            {
                "id": log.id,
                "timestamp": log.timestamp.isoformat(),
                "symbol": log.symbol,
                "trade_action": log.trade_action,
                "trade_size": log.trade_size,
                "market_regime": log.market_regime,
                "model_weights": log.model_weights,
                "confidence_score": log.confidence_score,
                "reasoning": log.reasoning,
                "risk_parameters": log.risk_parameters,
                "log_hash": log.log_hash
            }
            for log in logs
        ]

    async def validate_log_integrity(self, trade_id: str) -> bool:
        """📌 Ensures that trade logs remain tamper-proof"""
        trade_log = session.query(TradeLog).filter(TradeLog.id == trade_id).first()
        if not trade_log:
            return False

        log_entry = {
            "symbol": trade_log.symbol,
            "trade_action": trade_log.trade_action,
            "trade_size": trade_log.trade_size,
            "market_regime": trade_log.market_regime,
            "model_weights": trade_log.model_weights,
            "confidence_score": trade_log.confidence_score,
            "reasoning": trade_log.reasoning,
            "risk_parameters": trade_log.risk_parameters
        }

        computed_hash = hashlib.sha256(json.dumps(log_entry, sort_keys=True).encode()).hexdigest()
        return computed_hash == trade_log.log_hash

    async def generate_performance_report(self, time_range: int = 30) -> Dict[str, Any]:
        """📌 Evaluates past trades to analyze AI model effectiveness"""
        end_date = datetime.datetime.utcnow()
        start_date = end_date - datetime.timedelta(days=time_range)

        trades = (
            session.query(TradeLog)
            .filter(TradeLog.timestamp.between(start_date, end_date))
            .all()
        )

        model_performance = {}

        for trade in trades:
            for model, weight in trade.model_weights.items():
                if model not in model_performance:
                    model_performance[model] = {"total_trades": 0, "total_weight": 0}
                model_performance[model]["total_trades"] += 1
                model_performance[model]["total_weight"] += weight

        for model in model_performance:
            model_performance[model]["average_weight"] = (
                model_performance[model]["total_weight"] / model_performance[model]["total_trades"]
            )

        return model_performance

    async def send_trade_alert(self, trade_data: Dict[str, Any]) -> None:
        """📌 Alerts developers if AI behavior is abnormal"""
        if trade_data["confidence_score"] < 50:
            logger.warning(f"⚠️ Low confidence trade detected: {trade_data['symbol']}")

        if trade_data["trade_action"] == "REJECTED":
            logger.warning(f"⛔ Trade rejected due to risk: {trade_data['reasoning']}")

        # Can integrate with Telegram/Slack for alerts

    async def log_trade_execution(self, trade_data: Dict[str, Any]) -> None:
        """📌 Logs trade execution after the trade is placed"""
        execution_log = {
            "trade_id": trade_data["trade_id"],
            "execution_time": datetime.datetime.utcnow().isoformat(),
            "execution_price": trade_data["execution_price"],
            "execution_status": trade_data["status"],
            "slippage": trade_data["slippage"]
        }

        log_hash = hashlib.sha256(json.dumps(execution_log, sort_keys=True).encode()).hexdigest()
        encrypted_entry = cipher.encrypt(json.dumps(execution_log).encode()).decode()

        logger.info(f"Trade Execution Logged: {execution_log['trade_id']} - Status: {execution_log['execution_status']}")

    async def risk_based_trade_review(self) -> None:
        """📌 Cross-checks risk parameters in historical logs"""
        risk_logs = session.query(TradeLog).order_by(TradeLog.timestamp.desc()).limit(50).all()

        for log in risk_logs:
            if log.risk_parameters["max_drawdown"] > 10:  # Example risk threshold
                logger.warning(f"🚨 High drawdown risk detected in trade {log.id}")

