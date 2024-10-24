import numpy as np
import pandas as pd

# Set display options
pd.set_option('display.max_rows', 1000)  # Set the maximum number of rows to display
pd.set_option('display.max_columns', 100)  # Set the maximum number of columns to display
pd.set_option('display.width', 1000) 



def main(signal, signal_last, prices, portfolio_size=100, long_only=True, min_size=60):
    """
    Args:
        signal (pd.Series): signal for each stock at prediction time
        signal_last (pd.Series): signal for each stock from the last period
        prices (pd.DataFrame): full prices data (used for checking sufficient data)
        portfolio_size (int): number of stocks to select
        long_only (bool): whether to only long the stocks
        min_size (int): minimum number of non-NA prices for a stock to be selected

    Returns:
        list: list of selected stocks based on the turnover constraint
    """

    # Set signals to 0 for stocks with less than 60 non-NA price values
    sufficient_data = prices[-min_size:].count() >= min_size
    signal = signal.where(sufficient_data, 0)
    signal_last = signal_last.where(sufficient_data, 0)
    
    # 

    # Ensure the signals are absolute values (if long_only, or can adjust later for short/long)
    signal = signal.abs()
    signal_last = signal_last.abs()
    
    # print signals, where column is sorted by absolute values
    # print(f"Signal: {signal.sort_values(ascending=False)}")
    # print prices, sorted according to the signal values
    # print(f"Prices: {prices[signal.sort_values(ascending=False).index]}")
    

    # Determine the number of stocks that can be replaced (25% of portfolio size)
    n_replace = int(0.25 * portfolio_size)
    # n_replace = 25

    # Get the top stocks from the last period's signals
    sort_past_signals = signal_last.sort_values(ascending=False)
    # print(f"Past signals: {sort_past_signals}")

    # Keep 75% of the stocks from the old portfolio
    n_keep = portfolio_size - n_replace
    keep_stocks = sort_past_signals.index[:n_keep].tolist()

    # Select top stocks from the current signal, but exclude those already in keep_stocks
    sort_signals = signal.sort_values(ascending=False)
    new_candidates = [stock for stock in sort_signals.index if stock not in keep_stocks]

    # Select 25% new stocks from the current signal
    new_stocks = new_candidates[:n_replace]

    # Combine the kept stocks from the last period and the newly selected stocks
    selected_stocks = keep_stocks + new_stocks
    # print(f"Selected stocks: {selected_stocks}")
    # print(f"Number of selected stocks: {len(selected_stocks)}")
    
    # print('prices of selected stocks')
    # print(prices[selected_stocks])
    
    # print('signal of selected stocks')
    # print(signal[selected_stocks])
    
    # print the top 10 stocks

    return selected_stocks
