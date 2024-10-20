"""
Module for signal-based stock selection with data sufficiency filtering.

Key Features:
- Selects stocks based on signal values.
- Filters stocks with insufficient price history.
- Supports long-only and long-short portfolios.

Functions:
- main(signal, prices, portfolio_size, long_only=True,
  min_size=60): Stock selection based on signals.
"""

def select_stocks(signal, prices, portfolio_size, long_only=True, min_size=60):
    """
    Selects stocks for a portfolio based on signal values, with optional long-short selection.

    Parameters:
    -----------
    - signal : pd.Series
        Signal values for stock selection (higher values indicate stronger preference).
    - prices : pd.DataFrame
        Historical price data for stocks.
    - portfolio_size : int
        Number of stocks to include in the portfolio.
    - long_only : bool, optional (default=True)
        If True, selects stocks with positive signals only;
        If False, selects based on absolute signal values.
    - min_size : int, optional (default=60)
        Minimum number of non-NA price entries required to include a stock.

    Returns:
    --------
    - selected_stocks : list
        List of stocks selected for the portfolio.
    """

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
        sort_signals = signal
        sort_signals = sort_signals.sort_values(ascending=False)
        selected_stocks = sort_signals.index[:portfolio_size].tolist()
        signal = signal[selected_stocks]

    return selected_stocks
