"""
This script runs the backtest on the results of the strategy.
"""

import warnings
import pickle
import sys
import os
from backtest_stats import get_tl_stats, get_trading_log, performance_benchmark
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Set the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.append("../05-Asset_Allocation")
import strategy_20.main as strat
path_to_strategy = "../05-Asset_Allocation/strategy_2"


def compute_weights_for_period(args):
    """
    Inputs:
    - args: Tuple containing (i, strategy, kwargs)
    Outputs:
    - i: End Date
    - Weights calculated by the strategy for the current period
    """
    i, strategy, kwargs = args
    # Call the strategy to get the weights for the current period
    weights = strategy(
        # start_date= i - 60,
        start_date= 0,
        end_date=i,
        **kwargs,
    )
    total_allocation = weights["Weight"].sum()
    # Uncomment the following line if you want to print the total allocation
    # print(f'At date {i}, the total allocation in the portfolio is {total_allocation}')
    return i, weights["Weight"]


def backtest(
    excess_returns,
    strategy,
    rebalance_period=1,
    start_month_pred=100,
    **kwargs,
):
    """
    Inputs :
    - excess_returns: DataFrame of excess returns
    - strategy: Function that outputs weights
    - rebalance_period: Frequency of rebalancing
    - start_month_pred: Start month index for predictions

    Outputs :
    - weights_df: DataFrame containing the weights over time
    """

    # Initialize the weights DataFrame with NaNs
    weights_df = pd.DataFrame(
        data=np.nan, columns=excess_returns.columns, index=excess_returns.index
    )

    # Create a list of arguments for each period
    args_list = []
    for i in range(start_month_pred, len(excess_returns), rebalance_period):
        args_list.append((i, strategy, kwargs))

    # Determine the number of processes to use
    num_processes = min(cpu_count(), len(args_list))

    # Use multiprocessing Pool to compute weights in parallel
    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(compute_weights_for_period, args_list),
                total=len(args_list),
                desc="Backtesting",
            )
        )

    # Collect the results into the weights DataFrame
    for i, weights in results:
        weights_df.iloc[i] = weights

    # Forward fill NAs to propagate weights to future periods
    weights_df = weights_df.ffill()
    # Fill any remaining NAs (at the beginning) with 0 to avoid NaNs in calculations
    weights_df = weights_df.fillna(0.0)

    return weights_df



def stats(weights_df, excess_returns_df, benchmark, start_month_pred=100):
    """
    Compute the backtest statistics, compare with S&P 500, and plot cumulative return over time.
    """
    trading_log = get_trading_log(excess_returns_df, weights_df)

    trading_log_stats = get_tl_stats(trading_log, weights_df)

    trading_stats = performance_benchmark(trading_log, benchmark, weights_df)

    return trading_stats, trading_log_stats


if __name__ == "__main__":
    if input("Do you want to run useful objects? (y/n) (Should do 'y' if first time training signals): ") == "y":
        import useful_objects
    prices = pd.read_pickle("../objects/prices.pkl")
    signals = pd.read_pickle("../objects/signals.pkl")
    factor_signals = pd.read_pickle("../objects/factor_signals.pkl")
    market_caps_df = pd.read_pickle("../objects/market_caps.pkl")
    excess_returns = pd.read_pickle("../objects/stockexret.pkl")
    benchmark_df = pd.read_csv("../objects/mkt_ind.csv")
    benchmark_df["t1"] = pd.to_datetime(benchmark_df["t1"])
    benchmark_df["t1_index"] = pd.to_datetime(benchmark_df["t1_index"])
    WINDOW_SIZE = 60
    kwargs = {
        # Stock Selection
        "min_size": WINDOW_SIZE,
        "long_only": True,
        "portfolio_size": 80,
        # Covariance Estimation, Black Litterman
        "tau": 1.0,
        "lambda_": 2,
        "use_ema": True,
        "window": WINDOW_SIZE,
        "span": WINDOW_SIZE,
        # Weight Optimization
        'soft_risk': 0.01,
        "num_scenarios": 20,
        "uncertainty_level": 0.05,
        "total_allocation": 1.0,
        "n_clusters": 6,
        # OBJECTS
        "prices": prices,
        "signals": signals,
        "market_caps_df": market_caps_df,
        "benchmark_df": benchmark_df,
    }
    REBALANCE_PERIOD = 1
    strategy = strat.asset_allocator
    START_MONTH_PRED = 121

    weights = backtest(
        excess_returns,
        strategy,
        REBALANCE_PERIOD,
        START_MONTH_PRED,
        **kwargs,
    )

    # Number of non-zero columns per row in weights
    print(weights.astype(bool).sum(axis=1).value_counts())

    # Trim the weights and excess returns DataFrames
    weights = weights.iloc[START_MONTH_PRED:]
    excess_returns = excess_returns.iloc[START_MONTH_PRED:]
    Trading_Stats, TradingLog_Stats = stats(weights, excess_returns, benchmark_df)

    # Present your top 10 holdings on average over OOS testing period,
    # 01/2010 to 12/2023
    print()
    print("Top 10 Holdings on Average Over OOS Testing Period")
    weight_stock = weights.mean(axis=0)
    print(weight_stock.sort_values(ascending=False).iloc[:10])

    # Save Trading_Stats dictionary
    with open("../objects/Trading_Stats.pkl", "wb") as f:
        pickle.dump(Trading_Stats, f)

    # Save TradingLog_Stats dictionary
    with open("../objects/TradingLog_Stats.pkl", "wb") as f:
        pickle.dump(TradingLog_Stats, f)

    # Save weights
    weights.to_pickle("../objects/weights.pkl")
    print("Objects saved successfully.")
