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
import avici

model = avici.load_pretrained(download="scm-v0")




def asset_allocator(
    start_date,
    end_date,
    prices,
    signals,
    market_caps_df,
    benchmark_df,
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
        "min_size": kwargs.get("min_size", 60),
        "long_only": kwargs.get("long_only", True),
        "portfolio_size": kwargs.get("portfolio_size", 100),
    }
    signal_stockSelector = market_caps_df.iloc[end_date]
    selected_stocks = stocks_selector.main(
        signal_stockSelector, prices, **stock_selector_kwargs
    )
    # Filter the data to only include the selected stocks
    prices = prices[selected_stocks]
    signal_end = signal_end[selected_stocks]
    returns = prices.pct_change().dropna()
    g_prob = model(x=returns.to_numpy())
    normalized_matrix = g_prob / np.max(g_prob)
    # print('returns', returns)
    market_caps_df = market_caps_df[selected_stocks]
    market_caps = market_caps_df.iloc[end_date]

    # Step 2) Compute the Covariance matrix and Expected Returns vector
    mean_cov_kwargs = {
        "tau": kwargs.get("tau", 1.0),
        "lambda_": kwargs.get("lambda_", 3.07),
    }
    u_vector, cov_matrix = mean_cov_computer.main(
        returns, signal_end, market_caps, selected_stocks, **mean_cov_kwargs
    )

    # Step 3) Compute the optimal weights
    weight_optimizer_kwargs = {
        "lambda_": kwargs.get("lambda_", 3.07),
        "soft_risk": kwargs.get("soft_risk", 0.01),
        
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
