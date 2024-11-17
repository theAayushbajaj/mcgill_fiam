import numpy as np
import pandas as pd

def main(signal, signal_last, prices, portfolio_size, long_only=True, min_size=60):
    """
    Args:
        signal (pd.Series): signal for each stock at prediction time
        prices (pd.DataFrame): prices dataframe, full data
        portfolio_size (int): number of stocks to select
        long_only (bool): whether to only long the stocks
        min_size (int): minimum number of non-NA prices for a stock to be selected
        
    Returns:
        list: list of selected stocks
    """

    # Set signals to 0 for stocks with less than 60 non-NA price values
    sufficient_data = prices.count() >= min_size
    
    # long top portfolio_size//2, short bottom portfolio_size//2
    sort_signals = signal.sort_values(ascending=False)
    long_stocks = sort_signals.index[:portfolio_size//2].tolist()
    short_stocks = sort_signals.index[-portfolio_size//2:].tolist()
    return (long_stocks, short_stocks)
