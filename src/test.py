import yfinance as yf
import time

tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]

for symbol in tickers:
    print(f"Fetching {symbol}")
    df = yf.download(symbol, start="2024-05-01", end="2024-06-01")
    print(df.head())
    time.sleep(5)  # wait 5 seconds between requests
