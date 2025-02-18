import ccxt
import pandas as pd

exchange = ccxt.binance()
symbol = "BTC/USDT"
timeframe = "1m"
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)

df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
df.to_csv("Data/market_data.csv", index=False)
print("Market data saved successfully!")
