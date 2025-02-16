import logging
from logging.handlers import RotatingFileHandler

class Logger:
    def __init__(self, name="RetailBot"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Setup log rotation (Max 5MB per file, keeps last 5 logs)
        log_handler = RotatingFileHandler(f"logs/{name}.log", maxBytes=5*1024*1024, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_handler.setFormatter(formatter)
        self.logger.addHandler(log_handler)

        # Also log to console (optional but recommended)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_info(self, message):
        self.logger.info(message)

    def log_error(self, message):
        self.logger.error(message)

def get_logger(name="RetailBot"):
    """Sets up and returns a logger with rotating file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if not logger.hasHandlers():
        handler = RotatingFileHandler(f"logs/{name}.log", maxBytes=5*1024*1024, backupCount=5)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

# Usage in different modules:
logger = Logger()
logger.log_info("This is a test log")
