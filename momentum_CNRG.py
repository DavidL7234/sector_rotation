import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load the data from CSV
data_file = 'universe.csv'
data = pd.read_csv(data_file, index_col=0, parse_dates=True)

def get_top_sectors_lag_hold(returns, look_back_period, hold_period, top_n, strategy_type='high'):
    momentum = returns.rolling(window=look_back_period).mean().dropna()
    if strategy_type == 'high':
        top_sectors = momentum.apply(lambda x: x.nlargest(top_n).index, axis=1)
    elif strategy_type == 'low':
        top_sectors = momentum.apply(lambda x: x.nsmallest(top_n).index, axis=1)
    elif strategy_type == 'high_minus_low':
        top_sectors_high = momentum.apply(lambda x: x.nlargest(top_n).index, axis=1)
        top_sectors_low = momentum.apply(lambda x: x.nsmallest(top_n).index, axis=1)
        top_sectors = pd.DataFrame({'high': top_sectors_high, 'low': top_sectors_low})
    else:
        raise ValueError("Invalid strategy type. Choose from 'high', 'low', 'high_minus_low'.")

    lag_hold_sectors = pd.Series(index=returns.index)
    cnrg_selection = []
    for i in range(look_back_period, len(top_sectors), hold_period):
        if strategy_type == 'high_minus_low':
            current_sectors_high = top_sectors['high'].iloc[i]
            current_sectors_low = top_sectors['low'].iloc[i]
        else:
            current_sectors = top_sectors.iloc[i]
        
        if i + hold_period > len(lag_hold_sectors):
            hold_period = len(lag_hold_sectors) - i  # Adjust hold period for the last segment
        
        for j in range(hold_period):
            if strategy_type == 'high_minus_low':
                lag_hold_sectors.iloc[i + j] = (current_sectors_high, current_sectors_low)
                if 'CNRG' in current_sectors_high or 'CNRG' in current_sectors_low:
                    cnrg_selection.append(i+j)
            else:
                lag_hold_sectors.iloc[i + j] = current_sectors
                if 'CNRG' in current_sectors:
                    cnrg_selection.append(i+j)

    return lag_hold_sectors.dropna(), cnrg_selection

def simulate_strategy(returns, top_sectors, strategy_type='high'):
    # Initialize portfolio returns
    portfolio_returns = pd.Series(index=returns.index, data=0.0)

    # Simulate strategy
    if strategy_type == 'high_minus_low':
        for date in top_sectors.index:
            selected_sectors_high, selected_sectors_low = top_sectors.loc[date]
            long_returns = returns.loc[date, selected_sectors_high].mean()
            short_returns = returns.loc[date, selected_sectors_low].mean()
            portfolio_returns[date] = long_returns - short_returns
    else:
        for date in top_sectors.index:
            selected_sectors = top_sectors.loc[date]
            if strategy_type == 'high':
                portfolio_returns[date] = returns.loc[date, selected_sectors].mean()
            elif strategy_type == 'low':
                portfolio_returns[date] = -returns.loc[date, selected_sectors].mean()
    
    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    return cumulative_returns

def test_multiple_lag_hold(returns, lag_values, hold_values, top_n=1, strategy_type='high'):
    results = {}
    total_cnrg_selections = []
    for lag in lag_values:
        for hold in hold_values:
            top_sectors, cnrg_selection = get_top_sectors_lag_hold(returns, lag, hold, top_n, strategy_type)
            cumulative_returns = simulate_strategy(returns, top_sectors, strategy_type)
            results[(lag, hold)] = cumulative_returns
            total_cnrg_selections += cnrg_selection
    return results, total_cnrg_selections

def highlight_cnrg_selections(ax, cnrg_selections, dates):
    for idx in cnrg_selections:
        ax.axvspan(dates[idx], dates[idx+1], color='yellow', alpha=0.3)

# Calculate daily returns
returns = data.pct_change().dropna()

# Define lag and hold values to test
lag_values = [10, 20, 30]
hold_values = [1]

# Test for 'high' strategy
results_high, total_cnrg_selections_high = test_multiple_lag_hold(returns, lag_values, hold_values, top_n=1, strategy_type='high')

# Test for 'low' strategy
results_low, total_cnrg_selections_low = test_multiple_lag_hold(returns, lag_values, hold_values, top_n=1, strategy_type='low')

# Test for 'high_minus_low' strategy
results_high_minus_low, total_cnrg_selections_high_minus_low = test_multiple_lag_hold(returns, lag_values, hold_values, top_n=1, strategy_type='high_minus_low')

# Plot cumulative returns for each combination of lag and hold for 'high' strategy
fig, ax = plt.subplots(figsize=(12, 8))
for (lag, hold), cumulative_returns in results_high.items():
    ax.plot(cumulative_returns, label=f'High Lag: {lag}, Hold: {hold}')
highlight_cnrg_selections(ax, total_cnrg_selections_high, cumulative_returns.index)
plt.title('Lag-Hold Model Strategy Cumulative Returns (High)')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()

# Plot cumulative returns for each combination of lag and hold for 'low' strategy
fig, ax = plt.subplots(figsize=(12, 8))
for (lag, hold), cumulative_returns in results_low.items():
    ax.plot(cumulative_returns, label=f'Low Lag: {lag}, Hold: {hold}')
highlight_cnrg_selections(ax, total_cnrg_selections_low, cumulative_returns.index)
plt.title('Lag-Hold Model Strategy Cumulative Returns (Low)')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()

# Plot cumulative returns for each combination of lag and hold for 'high_minus_low' strategy
fig, ax = plt.subplots(figsize=(12, 8))
for (lag, hold), cumulative_returns in results_high_minus_low.items():
    ax.plot(cumulative_returns, label=f'High minus Low Lag: {lag}, Hold: {hold}')
highlight_cnrg_selections(ax, total_cnrg_selections_high_minus_low, cumulative_returns.index)
plt.title('Lag-Hold Model Strategy Cumulative Returns (High minus Low)')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()

# Print frequency of CNRG selection
print(f"Total CNRG selections for 'high' strategy: {len(total_cnrg_selections_high)}")
print(f"Total CNRG selections for 'low' strategy: {len(total_cnrg_selections_low)}")
print(f"Total CNRG selections for 'high_minus_low' strategy: {len(total_cnrg_selections_high_minus_low)}")
