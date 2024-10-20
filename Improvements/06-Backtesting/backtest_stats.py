"""
This script computes the backtest statistics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from scipy.stats import norm

# ================== PARAMETERS ==================
# Period yearly
TIME_DELTA = 1 / 12


def get_trading_log(excess_returns, weights):
    """
    Inputs :
    - excess_returns : pd.DataFrame : excess returns for each stock FOR THE
    FOLLOWING MONTH
    - weights : pd.DataFrame : weights for each stock

    Outputs :
    - trading_log : pd.DataFrame : trade returns for each stock
    """
    # Simply a point to point multiplication
    trading_log = excess_returns * weights
    trading_log.fillna(0, inplace=True)
    return trading_log


# ================== Trading Log Stats ==================


def get_tl_stats(trading_log, weights):
    """
    Inputs :
    - trading_log : pd.DataFrame : trade returns for each stock

    Outputs :
    - trading_log_stats : pd.DataFrame : trading log stats
    """
    # Dataframe of number of hit and miss per trade (per index)
    hit_miss_trade = pd.DataFrame()
    hit_miss_trade["Hit"] = (trading_log > 0).sum(axis=1)
    hit_miss_trade["Miss"] = (trading_log < 0).sum(axis=1)
    hit_miss_trade["Total"] = hit_miss_trade["Hit"] + hit_miss_trade["Miss"]

    # Averages
    # Average return per hit
    hit_miss_trade["Avg Hit Ret"] = trading_log[trading_log > 0].mean(axis=1)
    # Average return per miss
    hit_miss_trade["Avg Miss Ret"] = trading_log[trading_log < 0].mean(axis=1)
    # Average return per trade
    hit_miss_trade["Avg Trade Ret"] = trading_log[trading_log != 0].mean(axis=1)

    # Hits minus Misses
    hit_miss_trade["Hits - Misses"] = hit_miss_trade["Hit"] - hit_miss_trade["Miss"]
    # Hit ratio
    hit_miss_trade["Hit Ratio"] = hit_miss_trade["Hit"] / hit_miss_trade["Total"]
    # Cumulative sum of hits - misses
    hit_miss_trade["Cumul Hits - Misses"] = hit_miss_trade["Hits - Misses"].cumsum()

    # Now, per stock
    # Average return per hit
    hit_miss_stock = pd.DataFrame()
    hit_miss_stock["Hit"] = (trading_log > 0).sum(axis=0)
    hit_miss_stock["Miss"] = (trading_log < 0).sum(axis=0)
    hit_miss_stock["Total"] = hit_miss_stock["Hit"] + hit_miss_stock["Miss"]

    # Averages
    # Average return per hit
    hit_miss_stock["Avg Hit Ret"] = trading_log[trading_log > 0].mean(axis=0)
    # Average return per miss
    hit_miss_stock["Avg Miss Ret"] = trading_log[trading_log < 0].mean(axis=0)
    # Average return per trade
    hit_miss_stock["Avg Trade Ret"] = trading_log[trading_log != 0].mean(axis=0)

    # Hits minus Misses
    hit_miss_stock["Hits - Misses"] = hit_miss_stock["Hit"] - hit_miss_stock["Miss"]
    # Hit ratio
    hit_miss_stock["Hit Ratio"] = hit_miss_stock["Hit"] / hit_miss_stock["Total"]

    # Dataframe analysing hits/misses per long and short trades
    hit_miss_long_short = pd.DataFrame()
    hit_miss_long_short["Long Return"] = (trading_log[weights > 0]).sum(axis=1)
    hit_miss_long_short["Short Return"] = (trading_log[weights < 0]).sum(axis=1)
    hit_miss_long_short["Num Long"] = (weights > 0).sum(axis=1)
    hit_miss_long_short["Num Short"] = (weights < 0).sum(axis=1)
    hit_miss_long_short["Long Avg Return"] = (trading_log[weights > 0]).mean(axis=1)
    hit_miss_long_short["Short Avg Return"] = (trading_log[weights < 0]).mean(axis=1)

    # cumulative return
    hit_miss_long_short["Cumul Long Return"] = hit_miss_long_short[
        "Long Return"
    ].cumsum()
    hit_miss_long_short["Cumul Short Return"] = hit_miss_long_short[
        "Short Return"
    ].cumsum()

    # Overview
    hit_miss_overall = {}
    hit_miss_overall["Hit"] = hit_miss_stock["Hit"].sum()
    hit_miss_overall["Miss"] = hit_miss_stock["Miss"].sum()
    hit_miss_overall["Total"] = hit_miss_overall["Hit"] + hit_miss_overall["Miss"]

    # Averages
    # Average return per hit
    hit_miss_overall["Avg Hit Ret"] = (
        trading_log[trading_log > 0].sum().sum()
    ) / hit_miss_overall["Hit"]

    # Average return per miss
    hit_miss_overall["Avg Miss Ret"] = (
        trading_log[trading_log < 0].sum().sum()
    ) / hit_miss_overall["Miss"]

    # Average return per trade
    hit_miss_overall["Avg Trade Ret"] = (
        trading_log[trading_log != 0].sum().sum()
    ) / hit_miss_overall["Total"]

    # Hits minus Misses
    hit_miss_overall["Hits - Misses"] = (
        hit_miss_overall["Hit"] - hit_miss_overall["Miss"]
    )

    # Hit ratio
    hit_miss_overall["Hit Ratio"] = hit_miss_overall["Hit"] / hit_miss_overall["Total"]
    hit_miss_overall = pd.DataFrame(hit_miss_overall, index=["Overall"])

    hit_miss_stats = {
        "Trade": hit_miss_trade,
        "Stock": hit_miss_stock,
        "Long_Short": hit_miss_long_short,
        "Overall": hit_miss_overall,
    }

    return hit_miss_stats


def performance_benchmark(trading_log, benchmark, weights_df):
    """
    Inputs :
    - trading_log : pd.DataFrame : trade returns for each stock

    Outputs :
    - trading_stats : pd.DataFrame : trading stats
    """
    trading_stats = {}
    portfolio_rets = trading_log.sum(axis=1)
    portfolio_rets = pd.DataFrame(portfolio_rets)
    portfolio_rets.reset_index(inplace=True)

    benchmark["exc_return"] = benchmark["sp_ret"] - benchmark["rf"]
    benchmark = benchmark[["t1", "exc_return"]]
    # create df as merge of portfolio_rets and benchmark with t1, t1 as index
    df = pd.merge(portfolio_rets, benchmark, on="t1")
    df.columns = ["t1", "Portfolio", "Benchmark"]
    # set t1 as index
    df.set_index("t1", inplace=True)

    # Now we have returns of each trade and benchmark, we can compute the stats
    trading_stats["Portfolio"] = compute_stats(df["Portfolio"])
    trading_stats["Benchmark"] = compute_stats(df["Benchmark"])
    trading_stats["Correlation"] = df["Portfolio"].corr(df["Benchmark"])

    trading_stats["Portfolio"]["PSR"] = compute_psr(df["Portfolio"])
    trading_stats["Portfolio"]["Information Ratio"] = compute_ir(
        df["Portfolio"], df["Benchmark"]
    )

    # Compute Portfolio Turnover
    portfolio_turnover = compute_portfolio_turnover(weights_df)
    trading_stats["Portfolio"]["Portfolio Turnover"] = portfolio_turnover

    # Compute Portfolio Alpha
    alpha = compute_portfolio_alpha(df["Portfolio"], df["Benchmark"])
    trading_stats["Portfolio"]["Alpha Annualized"] = alpha

    # Compute Portfolio Annualized Tracking Error
    tracking_error = (
        trading_stats["Portfolio"]["Annualized Volatility"]
        - trading_stats["Benchmark"]["Annualized Volatility"]
    )
    trading_stats["Portfolio"]["Annualized Tracking Error"] = tracking_error
    trading_stats["Portfolio"]["Per trade Tracking Error"] = compute_tracking_error(
        df["Portfolio"], df["Benchmark"]
    )

    plot_cumulative(
        trading_stats["Portfolio"]["Cumulative Return"],
        trading_stats["Benchmark"]["Cumulative Return"],
    )
    
    plot_weights(weights_df)

    # Print Benchmark Stats
    print("Benchmark Stats :")
    benchmark_stats = {
        k: v for k, v in trading_stats["Benchmark"].items() if k != "Cumulative Return"
    }
    for k, v in benchmark_stats.items():
        print(f"{k}: {v:.4f}")
    print()

    # Print Portfolio Stats
    print("Portfolio Stats :")
    portfolio_stats = {
        k: v for k, v in trading_stats["Portfolio"].items() if k != "Cumulative Return"
    }
    for k, v in portfolio_stats.items():
        print(f"{k}: {v:.4f}")

    # Print Correlation
    print(
        f"Correlation between Portfolio and Benchmark: {trading_stats['Correlation']:.4f}"
    )

    return trading_stats


def compute_stats(returns):
    """
    Inputs :
    - returns : pd.Series with index as datetime64[ns], one column of returns

    Outputs :
    - stats : dict : dictionary of statistics

    """
    stats = {}
    returns = returns.dropna()

    # Cumulative return
    stats["Cumulative Return"] = (1 + returns).cumprod()
    # Total return
    stats["Total Return"] = stats["Cumulative Return"].iloc[-1] - 1
    # Total periods
    total_periods = len(returns)
    # Total time in years
    total_years = TIME_DELTA * total_periods
    # Annualized Return
    stats["Annualized Return"] = (
        stats["Cumulative Return"].iloc[-1] ** (1 / total_years) - 1
    )
    # Annualized Volatility
    stats["Annualized Volatility"] = returns.std() / np.sqrt(TIME_DELTA)
    # Sharpe Ratio
    # Assuming risk-free rate is zero (since we are working with excess returns)
    stats["Sharpe Ratio"] = stats["Annualized Return"] / stats["Annualized Volatility"]
    # Max Drawdown
    # Compute drawdowns
    cumulative = stats["Cumulative Return"]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    stats["Max Drawdown"] = drawdown.min()
    # Calmar Ratio
    stats["Calmar Ratio"] = stats["Annualized Return"] / abs(stats["Max Drawdown"])
    # Time Under Water
    stats["Time Under Water"] = (drawdown < 0).sum() * TIME_DELTA
    # Maximum One-Month Loss
    stats["Maximum One-Month Loss"] = returns.min()
    return stats


def compute_ir(returns, benchmark):
    """
    Computes the Information Ratio between returns and benchmark.
    Inputs:
    - returns : pd.Series : portfolio returns
    - benchmark : pd.Series : benchmark returns
    Output:
    - Information Ratio : float
    """
    # Active return
    active_return = returns - benchmark
    # Mean of active return
    mean_active_return = active_return.mean()
    # Tracking error
    tracking_error = active_return.std()
    # Number of periods per year
    periods_per_year = 1 / TIME_DELTA
    # Annualized active return and tracking error
    mean_active_return_annualized = mean_active_return * periods_per_year
    tracking_error_annualized = tracking_error * np.sqrt(periods_per_year)
    # Information Ratio
    information_ratio = mean_active_return_annualized / tracking_error_annualized
    return information_ratio


def compute_psr(returns, benchmark=None):
    """
    Computes the Probabilistic Sharpe Ratio for the returns.
    Inputs:
    - returns : pd.Series : portfolio returns
    - benchmark : not used
    Output:
    - PSR : float : Probabilistic Sharpe Ratio
    """

    returns = returns.dropna()
    n = len(returns)
    # Observed Sharpe Ratio
    sr_obs = returns.mean() / returns.std()
    # PSR assuming the reference Sharpe Ratio is zero
    psr = norm.cdf(sr_obs * np.sqrt(n))
    return psr


def compute_portfolio_turnover(weights_df):
    """
    Computes the Portfolio Turnover.
    Inputs:
    - weights_df : pd.DataFrame : portfolio weights over time
    Output:
    - turnover : float : annualized portfolio turnover
    """
    delta_weights = weights_df.diff()
    # For each period, compute turnover as sum of absolute value of weight changes
    turnover_per_period = delta_weights.abs().sum(axis=1)
    # Average turnover per period
    average_turnover = turnover_per_period.mean()
    # Average trade turnover
    return average_turnover


def plot_cumulative(portfolio_cumulative, benchmark_cumulative):
    """
    Plots the cumulative returns of portfolio and benchmark.
    Inputs:
    - portfolio_cumulative : pd.Series : cumulative returns of the portfolio
    - benchmark_cumulative : pd.Series : cumulative returns of the benchmark
    """
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_cumulative.index, portfolio_cumulative.values, label="Portfolio")
    plt.plot(benchmark_cumulative.index, benchmark_cumulative.values, label="Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Cumulative Return Comparison")
    plt.legend()
    plt.show()
    
import matplotlib.pyplot as plt

def plot_weights(weights_df):
    """
    To track how the weights are being allocated, plot the sum of the weights
    and the sum of the absolute weights.
    """
    # Calculate sum of weights and sum of absolute weights
    weights = weights_df.sum(axis=1)
    abs_weights = weights_df.abs().sum(axis=1)
    
    # Round the sums to avoid floating-point errors
    weights_rounded = weights.round(4)
    abs_weights_rounded = abs_weights.round(4)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot the sum of weights (solid line)
    plt.plot(weights_rounded.index, weights_rounded.values, label="Sum of Weights")
    
    # Plot the sum of absolute weights (dotted line)
    plt.plot(abs_weights_rounded.index, abs_weights_rounded.values, label="Sum of Absolute Weights", linestyle=':')
    
    # Set plot labels and title
    plt.xlabel("Date")
    plt.ylabel("Weights")
    plt.title("Weights Allocation")
    
    # Add a legend
    plt.legend()
    
    # Set y-axis limits to focus on small variations around 1.0
    plt.ylim(0.99, 1.01)
    
    # Add grid lines for better readability
    plt.grid(True)
    
    # Show the plot
    plt.show()




def compute_portfolio_alpha(returns, benchmark):
    """
    Computes the Portfolio Alpha.
    Inputs:
    - returns : pd.Series : portfolio excess returns
    - benchmark : pd.Series : benchmark excess returns
    Output:
    - alpha : float : portfolio alpha
    """
    # Regression
    model = sm.ols(
        formula="returns ~ benchmark",
        data=pd.DataFrame({"returns": returns, "benchmark": benchmark}),
    )
    results = model.fit()
    # Alpha
    alpha = results.params["Intercept"]
    # Annualized alpha
    alpha_annualized = alpha * (1 / TIME_DELTA)
    return alpha_annualized


def compute_tracking_error(returns, benchmark):
    """
    Computes the Tracking Error.
    Inputs:
    - returns : pd.Series : portfolio excess returns
    - benchmark : pd.Series : benchmark excess returns
    Output:
    - tracking_error : float : tracking error
    """
    # Portfolio volatility
    portfolio_volatility = returns.std()
    # Benchmark volatility
    benchmark_volatility = benchmark.std()
    # Period tracking error
    tracking_error = portfolio_volatility - benchmark_volatility
    return tracking_error
