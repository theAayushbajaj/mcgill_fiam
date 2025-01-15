# %%
"""
This script takes objects created in useful_objects.py and allocates a portion
of the available capital to top N stocks.
"""

import numpy as np
import pandas as pd


import os
import sys

# Add the directory containing hrp.py to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import mean_cov_computer
import stocks_selector
import weight_optimizer


def asset_allocator(
    start_date, end_date, prices, signals, market_caps_df, portfolio_size=100,
):
    """
    _summary_

    Args:
        start_date (int): index from which start computing covariance matrix
        end_date (int): index at which to make predictions
        prices (pd.DataFrame): prices dataframe, full data
        signals (pd.DataFrame): signals dataframe, full data
    """
    # =============== INITIALIZE WEIGHT ALLOCATOR ===============
    # Initialize the weights to return
    weights = pd.DataFrame(index=prices.columns)
    weights["Weight"] = 0.0

    # Step 0) Adjust the prices to the start and end date
    # Index in terms of months
    prices = prices[start_date : end_date + 1]

    # =========================================================================

    # Step 1) Stock Selection
    signal_end = signals.iloc[end_date]
    selected_stocks = stocks_selector.main(
        signal_end, prices, portfolio_size, min_size=60, **stockSelectorKwargs
    )
    # Filter the data to only include the selected stocks
    prices = prices[selected_stocks]
    returns = prices.pct_change().dropna()
    market_caps_df = market_caps_df[selected_stocks]
    market_caps = market_caps_df.iloc[end_date]

    # Step 2) Compute the Covariance matrix and Expected Returns vector
    u_vector, cov_matrix = mean_cov_computer.main(
        returns, signal_end, market_caps, selected_stocks, **meanCovKwargs
    )

    # Step 3) Compute the optimal weights
    optimized_weights = weight_optimizer.main(
        weights, cov_matrix, u_vector, selected_stocks, **weightOptimizerKwargs
    )

    return optimized_weights


# %%


if __name__ == "__main__":
    # Set the current working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    prices = pd.read_pickle("../../objects/prices.pkl")
    signals = pd.read_pickle("../../objects/signals.pkl")
    market_caps_df = pd.read_pickle("../../objects/market_caps.pkl")
    excess_returns = pd.read_pickle("../../objects/stockexret.pkl")
    benchmark_df = pd.read_csv("../../objects/mkt_ind.csv")
    benchmark_df["t1"] = pd.to_datetime(benchmark_df["t1"])
    benchmark_df["t1_index"] = pd.to_datetime(benchmark_df["t1_index"])
    kwargs = {
        "pred_vol_scale": 1.00,
        "tau": 1.00,  # the higher tau, the more weight is given to predictions
        "prices": prices,
        "signals": signals,
        "market_caps_df": market_caps_df,
        "portfolio_size": 100,
        "long_only": True,
    }
    
    start_date = 0
    end_date = 140

    # Run the asset allocator
    weights = asset_allocator(start_date=start_date, end_date=end_date, **kwargs)

    print("Market exposure (sum of weights): ", weights.sum())
    print("Sum of absolute values of weights: ", weights.abs().sum())
    print(weights)

# %%
