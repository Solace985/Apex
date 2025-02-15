import logging
from logging.handlers import RotatingFileHandler

def setup_logger():
    logger = logging.getLogger('RetailBot')
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler('retailbot.log', maxBytes=2000, backupCount=5)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger 