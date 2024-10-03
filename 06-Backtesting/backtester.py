# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import yfinance as yf

import sys
import os

sys.path.append("../05-Asset_Allocation")
import strategy as strat

from backtest_stats import *


def compute_weights_for_period(i, strategy, **kwargs):
    # Call the strategy to get the weights for the current period
    weights = strategy(
        start_date=0,
        end_date=i,
        **kwargs,
    )
    return i, weights["Weight"].values


def backtest(
    excess_returns,
    strategy,
    rebalance_period=1,
    start_month_pred=100,
    **kwargs,
):
    """
    Inputs :
    - Start Date
    - End Date
    - STRATEGY which outputs weights
        - The strategy's arguments

    Outputs :
    A dataframe where the row index are the same as prices
    Divided in two parts :
    - First half: Stockexret
    - Second half : Weights
    - example columns : [appl exret, msft exret, appl weight, msft weight]

    Reason : for each row, dot product exret and weights = trade return

    It should take the strategy, roll it forward, and compute the weights,
    trade by trade, from start date to end date


    """

    # Set backtest_df as excess_returns, add '_excess' to the columns
    # backtest_df = excess_returns.copy()
    # backtest_df.columns = [col + '_excess' for col in excess_returns.columns]

    # Add columns for the weights (all zeros, column with _weight suffix)
    weights_df = pd.DataFrame(
        columns=excess_returns.columns, index=excess_returns.index
    )
    # weights_df = weights_df.fillna(0.0)
    # weights_df

    # Get the maximum number of workers (equal to the number of available CPU cores)
    max_workers = os.cpu_count()
    print(f"Using {max_workers} workers.")

    # Set up the parallel execution using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the pool
        futures = [
            executor.submit(
                compute_weights_for_period,
                i,
                strategy,
                **kwargs,
            )
            for i in range(start_month_pred, len(excess_returns), rebalance_period)
        ]

        # As each future completes, update the weights dataframe with a progress bar
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Backtesting"
        ):
            i, weights = future.result()
            weights_df.iloc[i] = weights

    # ffil NAs
    weights_df = weights_df.ffill()
    # fill the remaing NAs with 0
    weights_df = weights_df.fillna(0.0)

    return weights_df


def stats(weights_df, excess_returns_df, benchmark, start_month_pred=100):
    """
    Compute the backtest statistics, compare with S&P 500, and plot cumulative return over time.
    """
    trading_log = get_trading_log(excess_returns_df, weights_df)

    TradingLog_Stats = get_TL_Stats(trading_log, weights_df)

    Trading_Stats = Performance_Benchmark(trading_log, benchmark, weights_df)

    return Trading_Stats, TradingLog_Stats


# %%


if __name__ == "__main__":
    # %%
    prices = pd.read_pickle("../objects/prices.pkl")
    signals = pd.read_pickle("../objects/signals.pkl")
    market_caps_df = pd.read_pickle("../objects/market_caps.pkl")
    excess_returns = pd.read_pickle("../objects/stockexret.pkl")
    benchmark_df = pd.read_csv("../objects/mkt_ind.csv")
    benchmark_df["t1"] = pd.to_datetime(benchmark_df["t1"])
    benchmark_df["t1_index"] = pd.to_datetime(benchmark_df["t1_index"])
    kwargs = {
        "pred_vol_scale": 1.00,
        "tau": 1.00,  # the higher tau, the more weight is given to predictions
        "prices": prices,
        "signals": signals,
        "market_caps_df": market_caps_df,
        "BL": True,
        "LW": True,
        "N_Stocks": 50,
    }
    rebalance_period = 1
    strategy = strat.asset_allocator
    start_month_pred = 120
    # %%

    weights = backtest(
        excess_returns,
        strategy,
        rebalance_period,
        start_month_pred,
        **kwargs,
    )
    # Number of non 0 columns per row in weights
    print(weights.astype(bool).sum(axis=1).value_counts())
    # %%
    # weights
    # %%
    weights = weights.iloc[start_month_pred:]
    excess_returns = excess_returns.iloc[start_month_pred:]
    Trading_Stats, TradingLog_Stats = stats(
        weights, excess_returns, benchmark_df
    )
    
    #%%
    # Present your top 10 holdings on average over OOS testing period, 
    # 01/2010 to 12/2023
    print()
    print("Top 10 Holdings on Average Over OOS Testing Period")
    # print(TradingLog_Stats['Stock']['Total'].sort_values(ascending = False).iloc[:10])
    weight_stock = weights.sum(axis=0)
    weight_stock = weight_stock / weights.shape[0]
    print(weight_stock.sort_values(ascending=False).iloc[:10])
    #%%
    
    print()
    print("Overall Stats :")
    TradingLog_Stats['Overall']
    print(TradingLog_Stats['Overall'])
    
    print()
    print("Long vs Short Stats :")
    print(TradingLog_Stats['Long_Short'])
    
    # save in objects
    # Save Trading_Stats dictionary
    with open("../objects/Trading_Stats.pkl", "wb") as f:
        pickle.dump(Trading_Stats, f)

    # Save TradingLog_Stats dictionary
    with open("../objects/TradingLog_Stats.pkl", "wb") as f:
        pickle.dump(TradingLog_Stats, f)

# %%
