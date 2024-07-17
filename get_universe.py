import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from time import sleep
import os

# Define the list of SPDR sector ETFs
spdr_etfs = ['XLE', 'XLU', 'CNRG']

# Fetch historical data with retry mechanism
start_date = '2018-10-23'
end_date = '2024-05-31'
cache_file = 'universe.csv'

def download_data(tickers, start, end):
    data = None
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    while data.empty:
        print(f"Download failed. Retrying...")
        sleep(1)
        data = yf.download(tickers, start=start, end=end)['Adj Close']


    data.to_csv(cache_file)  # Save to cache
    print("Data downloaded and cached.")
    return data

data = download_data(spdr_etfs, start_date, end_date)
