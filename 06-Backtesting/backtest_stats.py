import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm

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

def get_TL_Stats(trading_log, weights):
    """
    Inputs :
    - trading_log : pd.DataFrame : trade returns for each stock

    Outputs :
    - trading_log_stats : pd.DataFrame : trading log stats
    """
    # Dataframe of number of hit and miss per trade (per index)
    hit_miss_trade = pd.DataFrame()
    hit_miss_trade['Hit'] = (trading_log > 0).sum(axis=1)
    hit_miss_trade['Miss'] = (trading_log < 0).sum(axis=1)
    hit_miss_trade['Total'] = hit_miss_trade['Hit'] + hit_miss_trade['Miss']
    
    # Averages
    # Average return per hit
    hit_miss_trade['Avg Hit Ret'] = trading_log[trading_log > 0].mean(axis=1)
    # Average return per miss
    hit_miss_trade['Avg Miss Ret'] = trading_log[trading_log < 0].mean(axis=1)
    # Average return per trade
    hit_miss_trade['Avg Trade Ret'] = trading_log[trading_log!=0].mean(axis=1)
    
    # Hits minus Misses
    hit_miss_trade['Hits - Misses'] = hit_miss_trade['Hit'] - hit_miss_trade['Miss']
    # Hit ratio
    hit_miss_trade['Hit Ratio'] = hit_miss_trade['Hit'] / hit_miss_trade['Total']
    # Cumulative sum of hits - misses
    hit_miss_trade['Cumul Hits - Misses'] = hit_miss_trade['Hits - Misses'].cumsum()
    
    # Now, per stock
    # Average return per hit
    hit_miss_stock = pd.DataFrame()
    hit_miss_stock['Hit'] = (trading_log > 0).sum(axis=0)
    hit_miss_stock['Miss'] = (trading_log < 0).sum(axis=0)
    hit_miss_stock['Total'] = hit_miss_stock['Hit'] + hit_miss_stock['Miss']
    
    # Averages
    # Average return per hit
    hit_miss_stock['Avg Hit Ret'] = trading_log[trading_log > 0].mean(axis=0)
    # Average return per miss
    hit_miss_stock['Avg Miss Ret'] = trading_log[trading_log < 0].mean(axis=0)
    # Average return per trade
    hit_miss_stock['Avg Trade Ret'] = trading_log[trading_log!=0].mean(axis=0)
    
    # Hits minus Misses
    hit_miss_stock['Hits - Misses'] = hit_miss_stock['Hit'] - hit_miss_stock['Miss']
    # Hit ratio
    hit_miss_stock['Hit Ratio'] = hit_miss_stock['Hit'] / hit_miss_stock['Total']
    
    # Dataframe analysing hits/misses per long and short trades
    hit_miss_long_short = pd.DataFrame()
    hit_miss_long_short['Long Return'] = (trading_log[weights > 0]).sum(axis=1)
    hit_miss_long_short['Short Return'] = (trading_log[weights < 0]).sum(axis=1)
    hit_miss_long_short['Num Long'] = (weights > 0).sum(axis=1)
    hit_miss_long_short['Num Short'] = (weights < 0).sum(axis=1)
    hit_miss_long_short['Long Avg Return'] = (trading_log[weights > 0]).mean(axis=1)
    hit_miss_long_short['Short Avg Return'] = (trading_log[weights < 0]).mean(axis=1)
    
    # cumulative return
    hit_miss_long_short['Cumul Long Return'] = hit_miss_long_short['Long Return'].cumsum()
    hit_miss_long_short['Cumul Short Return'] = hit_miss_long_short['Short Return'].cumsum()
    

    # Overview
    hit_miss_overall = {}
    hit_miss_overall['Hit'] = hit_miss_stock['Hit'].sum()
    hit_miss_overall['Miss'] = hit_miss_stock['Miss'].sum()
    hit_miss_overall['Total'] = hit_miss_overall['Hit'] + hit_miss_overall['Miss']
    # Averages
    # Average return per hit
    hit_miss_overall['Avg Hit Ret'] = (trading_log[trading_log > 0].sum().sum()) / hit_miss_overall['Hit']
    # Average return per miss
    hit_miss_overall['Avg Miss Ret'] = (trading_log[trading_log < 0].sum().sum()) / hit_miss_overall['Miss']
    # Average return per trade
    hit_miss_overall['Avg Trade Ret'] = (trading_log[trading_log!=0].sum().sum()) / hit_miss_overall['Total']
    # Hits minus Misses
    hit_miss_overall['Hits - Misses'] = hit_miss_overall['Hit'] - hit_miss_overall['Miss']
    # Hit ratio
    hit_miss_overall['Hit Ratio'] = hit_miss_overall['Hit'] / hit_miss_overall['Total']
    hit_miss_overall = pd.DataFrame(hit_miss_overall, index=['Overall'])
    
    Hit_Miss_Stats = {  
        'Trade' : hit_miss_trade,
        'Stock' : hit_miss_stock,
        'Long_Short' : hit_miss_long_short,
        'Overall' : hit_miss_overall
    }
    
    return Hit_Miss_Stats
    
    
def Performance_Benchmark(trading_log, benchmark, weights_df):
    Trading_Stats = {}
    portfolio_rets = trading_log.sum(axis=1)
    portfolio_rets = pd.DataFrame(portfolio_rets)
    portfolio_rets.reset_index(inplace=True)

    benchmark['exc_return'] = benchmark['sp_ret'] - benchmark['rf']
    benchmark = benchmark[['t1', 'exc_return']]
    # create df as merge of portfolio_rets and benchmark with t1, t1 as index
    df = pd.merge(portfolio_rets, benchmark, on='t1')
    df.columns = ['t1', 'Portfolio', 'Benchmark']
    # set t1 as index
    df.set_index('t1', inplace=True)

    # Now we have returns of each trade and benchmark, we can compute the stats
    Trading_Stats['Portfolio'] = Compute_Stats(df['Portfolio'])
    Trading_Stats['Benchmark'] = Compute_Stats(df['Benchmark'])
    Trading_Stats['Correlation'] = df['Portfolio'].corr(df['Benchmark'])

    Trading_Stats['Portfolio']['PSR'] = Compute_PSR(df['Portfolio'])
    Trading_Stats['Portfolio']['Information Ratio'] = Compute_IR(df['Portfolio'], df['Benchmark'])

    # Compute Portfolio Turnover
    Portfolio_Turnover = Compute_Portfolio_Turnover(weights_df)
    Trading_Stats['Portfolio']['Portfolio Turnover'] = Portfolio_Turnover
    
    # Compute Portfolio Alpha
    Alpha = Compute_Portfolio_Alpha(df['Portfolio'], df['Benchmark'])
    Trading_Stats['Portfolio']['Alpha'] = Alpha

    Plot_Cumulative(Trading_Stats['Portfolio']['Cumulative Return'], Trading_Stats['Benchmark']['Cumulative Return'])

    # Print Benchmark Stats
    print('Benchmark Stats :')
    benchmark_stats = {k: v for k, v in Trading_Stats['Benchmark'].items() if k != 'Cumulative Return'}
    for k, v in benchmark_stats.items():
        print(f"{k}: {v:.4f}")
    print()

    # Print Portfolio Stats
    print('Portfolio Stats :')
    portfolio_stats = {k: v for k, v in Trading_Stats['Portfolio'].items() if k != 'Cumulative Return'}
    for k, v in portfolio_stats.items():
        print(f"{k}: {v:.4f}")
    
    # Print Correlation
    print(f"Correlation between Portfolio and Benchmark: {Trading_Stats['Correlation']:.4f}")

    return Trading_Stats
    
    
def Compute_Stats(returns):
    """
    Inputs : 
    - returns : pd.Series with index as datetime64[ns], one column of returns

    Outputs :
    - stats : dict : dictionary of statistics

    """
    stats = {}
    returns = returns.dropna()
    time_delta = 1/12  # Assuming monthly data

    # Cumulative return
    stats['Cumulative Return'] = (1 + returns).cumprod()
    # Total return
    stats['Total Return'] = stats['Cumulative Return'].iloc[-1] - 1
    # Total periods
    total_periods = len(returns)
    # Total time in years
    total_time = time_delta * total_periods
    # Annualized Return
    stats['Annualized Return'] = stats['Cumulative Return'].iloc[-1] ** (1 / total_time) - 1
    # Annualized Volatility
    # Number of periods per year
    periods_per_year = 1 / time_delta  # = 12
    stats['Annualized Volatility'] = returns.std() * np.sqrt(periods_per_year)
    # Sharpe Ratio
    # Assuming risk-free rate is zero (since we are working with excess returns)
    stats['Sharpe Ratio'] = stats['Annualized Return'] / stats['Annualized Volatility']
    # Max Drawdown
    # Compute drawdowns
    cumulative = stats['Cumulative Return']
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    stats['Max Drawdown'] = drawdown.min()
    # Calmar Ratio
    stats['Calmar Ratio'] = stats['Annualized Return'] / abs(stats['Max Drawdown'])
    # Time Under Water
    stats['Time Under Water'] = (drawdown < 0).sum() / len(drawdown)
    # Maximum One-Month Loss
    stats['Maximum One-Month Loss'] = returns.min()
    return stats

def Compute_IR(returns, benchmark):
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
    time_delta = 1/12  # Assuming monthly data
    periods_per_year = 1 / time_delta
    # Annualized active return and tracking error
    mean_active_return_annualized = mean_active_return * periods_per_year
    tracking_error_annualized = tracking_error * np.sqrt(periods_per_year)
    # Information Ratio
    information_ratio = mean_active_return_annualized / tracking_error_annualized
    return information_ratio

def Compute_PSR(returns, benchmark=None):
    """
    Computes the Probabilistic Sharpe Ratio for the returns.
    Inputs:
    - returns : pd.Series : portfolio returns
    - benchmark : not used
    Output:
    - PSR : float : Probabilistic Sharpe Ratio
    """
    from scipy.stats import norm

    returns = returns.dropna()
    N = len(returns)
    # Observed Sharpe Ratio
    SR_obs = returns.mean() / returns.std()
    # PSR assuming the reference Sharpe Ratio is zero
    PSR = norm.cdf(SR_obs * np.sqrt(N))
    return PSR

def Compute_Portfolio_Turnover(weights_df):
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
    # Assuming periods are monthly
    periods_per_year = 12
    # Annualize turnover
    annualized_turnover = average_turnover * periods_per_year
    return annualized_turnover

def Plot_Cumulative(portfolio_cumulative, benchmark_cumulative):
    """
    Plots the cumulative returns of portfolio and benchmark.
    Inputs:
    - portfolio_cumulative : pd.Series : cumulative returns of the portfolio
    - benchmark_cumulative : pd.Series : cumulative returns of the benchmark
    """
    plt.figure(figsize=(12,6))
    plt.plot(portfolio_cumulative.index, portfolio_cumulative.values, label='Portfolio')
    plt.plot(benchmark_cumulative.index, benchmark_cumulative.values, label='Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Return Comparison')
    plt.legend()
    plt.show()

def Compute_Portfolio_Alpha(returns, benchmark):
    """
    Computes the Portfolio Alpha.
    Inputs:
    - returns : pd.Series : portfolio excess returns
    - benchmark : pd.Series : benchmark excess returns
    Output:
    - alpha : float : portfolio alpha
    """
    # Regression
    model = sm.ols(formula='returns ~ benchmark', data=pd.DataFrame({'returns': returns, 'benchmark': benchmark}))
    results = model.fit()
    # Alpha
    alpha = results.params['Intercept']
    return alpha