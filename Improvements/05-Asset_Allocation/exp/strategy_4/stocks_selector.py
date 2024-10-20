import numpy as np
import pandas as pd

import pypfopt

def main(signal, prices, portfolio_size, long_only=True, min_size=60):
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
    # Select top portfolio_size stocks according to the market cap
    selected_stocks = signal.sort_values(ascending=False).index[:portfolio_size]
    return selected_stocks