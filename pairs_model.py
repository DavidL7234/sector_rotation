import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import coint

# Load the data from CSV
data_file = 'universe_individual.csv'
data = pd.read_csv(data_file, index_col=0, parse_dates=True).dropna(axis=1)

# Calculate daily returns
returns = data.pct_change().dropna()

# Function to find cointegrated pairs
def find_cointegrated_pairs(data, significance=0.05):
    n = data.shape[1]
    pvalue_matrix = np.ones((n, n))
    keys = data.columns
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            stock1 = data[keys[i]]
            stock2 = data[keys[j]]
            result = coint(stock1, stock2)
            pvalue = result[1]
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance:
                pairs.append((keys[i], keys[j]))
    return pairs, pvalue_matrix

# Find cointegrated pairs
pairs, pvalue_matrix = find_cointegrated_pairs(data)

print("Cointegrated pairs:", pairs)

# Function to simulate pairs trading strategy
def pairs_trading_strategy(data, pairs, entry_threshold=2, exit_threshold=0.5):
    returns = pd.DataFrame(index=data.index)
    for pair in pairs:
        stock1, stock2 = pair
        spread = data[stock1] - data[stock2]
        zscore = (spread - spread.mean()) / spread.std()
        
        long_signal = zscore < -entry_threshold
        short_signal = zscore > entry_threshold
        exit_signal = np.abs(zscore) < exit_threshold

        position = np.zeros_like(zscore)
        position[long_signal] = 1
        position[short_signal] = -1
        position[exit_signal] = 0
        
        # Forward fill positions
        position = pd.Series(position, index=data.index).ffill().shift()

        # Calculate daily returns
        stock1_returns = data[stock1].pct_change().dropna()
        stock2_returns = data[stock2].pct_change().dropna()

        pair_returns = position * (stock1_returns - stock2_returns)
        returns[pair] = pair_returns

    cumulative_returns = (1 + returns.mean(axis=1)).cumprod() - 1
    return cumulative_returns

# Simulate pairs trading strategy
cumulative_returns = pairs_trading_strategy(data, pairs)

# Plot cumulative returns
plt.figure(figsize=(12, 8))
plt.plot(cumulative_returns, label='Pairs Trading Strategy')
plt.title('Pairs Trading Strategy Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()

