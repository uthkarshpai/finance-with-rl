import yfinance as yf
import pandas as pd
import numpy as np

def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']
    data = data.ffill().bfill()

    # Add technical indicators
    for ticker in data.columns:
        df = pd.DataFrame(data[ticker])
        df['return'] = df[ticker].pct_change()
        df['ma10'] = df[ticker].rolling(window=10).mean()
        df['ma50'] = df[ticker].rolling(window=50).mean()
        df['volatility'] = df['return'].rolling(window=10).std()
        data[ticker + '_ma10'] = df['ma10']
        data[ticker + '_ma50'] = df['ma50']
        data[ticker + '_vol'] = df['volatility']

    data = data.ffill().bfill()
    print("Final columns:", data.columns.tolist())
    return data
