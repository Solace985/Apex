# Logger System:

# Please include this in both skeletons

import logging
from logging import getLogger

# Configure global logger
logger = getLogger('AITradingBot')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter('%(asctime)s - %(levelname)s - %(message)s')
logger.addHandler(handler)

# Use logger throughout the code
logger.info('Starting AggressorBot...') #this line was added by VS code recommendations so evaulate whether it is of use or not.



# Data Handling:

# Common data handling functions

def DataFeedManager():
    pass

def MarketDataAnalyzer():
    pass

def HistoricalDataAccessor():
    pass

