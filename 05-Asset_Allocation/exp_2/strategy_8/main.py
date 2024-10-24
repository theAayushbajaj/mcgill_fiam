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
    start_date,
    end_date,
    prices,
    signals,
    market_caps_df,
    benchmark_df,
    portfolio_size=100,
    previous_weight=None,
    **kwargs
):
    """
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
    if previous_weight is None:
        previous_weight = weights.copy()

    # Benchmark returns
    benchmark_df = benchmark_df.iloc[start_date:end_date]

    # Step 0) Adjust the prices to the start and end date
    # Index in terms of months
    prices = prices[start_date : end_date + 1]

    # =========================================================================

    # Step 1) Stock Selection
    signal_end = signals.iloc[end_date]
    stock_selector_kwargs = {
        key: kwargs[key] for key in ["min_size", "long_only"] if key in kwargs
    }
    selected_stocks = stocks_selector.main(
        signal_end, prices, portfolio_size, **stock_selector_kwargs
    )
    # Filter the data to only include the selected stocks
    prices = prices[selected_stocks]
    signal_end = signal_end[selected_stocks]
    returns = prices.pct_change().dropna()
    market_caps_df = market_caps_df[selected_stocks]
    market_caps = market_caps_df.iloc[end_date]

    # Step 2) Compute the Covariance matrix and Expected Returns vector
    mean_cov_kwargs = {
        key: kwargs[key]
        for key in ["pred_vol_scale", "tau", "lambda_"]
        if key in kwargs
    }
    u_vector, cov_matrix = mean_cov_computer.main(
        returns, signal_end, market_caps, selected_stocks, **mean_cov_kwargs
    )

    # Step 3) Compute the optimal weights
    weight_optimizer_kwargs = {
        key: kwargs[key] for key in ["long_only"] if key in kwargs
    }
    optimized_weights = weight_optimizer.main(
        weights,
        cov_matrix,
        u_vector,
        selected_stocks,
        benchmark_df,
        **weight_optimizer_kwargs
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
        "min_size": 60,
        "lambda_": 2.5,
        "benchmark_df": benchmark_df,
    }
    start_date = 0
    end_date = 140
    previous_weights = pd.DataFrame(data={"Weight": 0.0}, index=prices.columns)
    # Run the asset allocator
    weights = asset_allocator(start_date=start_date, end_date=end_date, previous_weight = None, **kwargs)

    print("Market exposure (sum of weights): ", weights.sum())
    print("Sum of absolute values of weights: ", weights.abs().sum())
    print(weights)

# %%
