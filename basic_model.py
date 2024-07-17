import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from time import sleep
import os

# Define the list of SPDR sector ETFs
spdr_etfs = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']

# Fetch historical data with retry mechanism
start_date = '2004-01-01'
end_date = '2024-05-31'
cache_file = 'spdr_data.csv'

def download_data(tickers, start, end, retries=3):
    # Check if cached data exists
    if os.path.exists(cache_file):
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print("Data loaded from cache.")
        return data

    data = None
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    while data.empty:
        print(f"Download failed: {e}. Retrying...")
        sleep(1)
        data = yf.download(tickers, start=start, end=end)['Adj Close']


    data.to_csv(cache_file)  # Save to cache
    print("Data downloaded and cached.")
    return data

data = download_data(spdr_etfs, start_date, end_date)

if data is None:
    print("Failed to download data after multiple attempts.")
else:
    # Calculate daily returns
    returns = data.pct_change().dropna()

    # Define look-back period and number of top sectors to invest in
    look_back_period = 20
    top_n = 3

    def get_top_sectors(returns, look_back_period, top_n):
        momentum = returns.rolling(window=look_back_period).mean().dropna()
        return momentum.apply(lambda x: x.nlargest(top_n).index, axis=1)

    top_sectors = get_top_sectors(returns, look_back_period, top_n)

    # Initialize portfolio returns
    portfolio_returns = pd.Series(index=returns.index, data=0.0)
    hold_returns = pd.Series(index=returns.index, data=0.0)

    # Simulate strategy
    for date in top_sectors.index:
        selected_sectors = top_sectors.loc[date]
        portfolio_returns[date] = returns.loc[date, selected_sectors].mean()
        hold_returns[date] = returns.loc[date].mean()

    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    hold_cumulative_returns = (1 + hold_returns).cumprod() - 1
    # Plot cumulative returns
    plt.figure(figsize=(12, 8))
    cumulative_returns.plot()
    hold_cumulative_returns.plot()
    plt.title('Daily Rebalance Strategy Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.show()
