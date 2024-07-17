import pandas as pd
import yfinance as yf
from datetime import datetime
from time import sleep

# Define the file paths and ETF names
files = {
    'XLE': 'holdings-daily-us-en-xle.xlsx',
    'XLU': 'holdings-daily-us-en-xlu.xlsx',
    'CNRG': 'holdings-daily-us-en-cnrg.xlsx'
}

start_date = '2018-11-05'
end_date = '2024-05-31'
cache_file = 'universe_individual.csv'

def get_tickers_from_file(file):
    df = pd.read_excel(file, header=4)
    tickers = df['Ticker'].dropna().unique().tolist()
    return tickers

all_tickers = []
for etf, file in files.items():
    tickers = get_tickers_from_file(file)
    all_tickers.extend(tickers)

while '-' in all_tickers:
    all_tickers.remove('-')

# Remove duplicates
all_tickers = list(set(all_tickers))

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

data = download_data(all_tickers, start_date, end_date)
