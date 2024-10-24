import numpy as np
import pandas as pd


def main(signal, prices, portfolio_size=100, long_only=True, min_size=60):
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
    # check = False
    # if not check:
    #     print('Check for stocks_selector')
    #     print(f'portfolio size is {portfolio_size}')
    #     print(f'long only is {long_only}')
    #     print(f'min size is {min_size}')
    #     check = True

    # Set signals to 0 for stocks with less than 60 non-NA price values
    sufficient_data = prices.count() >= min_size
    signal = signal.where(sufficient_data, 0)

    portfolio_size = min(portfolio_size, (signal > 0).sum())
    if not long_only:
        # Select top 100 stocks based on absolute signal value
        sort_signals = signal.abs()
        sort_signals = sort_signals.sort_values(ascending=False)
        selected_stocks = sort_signals.index[:portfolio_size].tolist()
        signal = signal[selected_stocks]
    else:
        # Select top 100 stocks based on signal value
        sort_signals = signal.abs()
        sort_signals = sort_signals.sort_values(ascending=False)
        selected_stocks = sort_signals.index[:portfolio_size].tolist()
        signal = signal[selected_stocks]

    return selected_stocks
