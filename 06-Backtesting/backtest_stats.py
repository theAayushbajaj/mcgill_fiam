import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    
    
def Performance_Benchmark(trading_log, benchmark):
    Trading_Stats = {}
    portfolio_rets = trading_log.sum(axis=1)
    portfolio_rets = pd.DataFrame(portfolio_rets)
    portfolio_rets.reset_index(inplace=True)
    
    benchmark['exc_return'] = benchmark['sp_ret']-benchmark['rf']
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
    
    Trading_Stats['Portfolio']['PSR'] = Compute_PSR(df['Portfolio'], df['Portfolio'])
    Trading_Stats['Portfolio']['Information Ratio'] = Compute_IR(df['Portfolio'], df['Benchmark'])
    
    Plot_Cumulative(Trading_Stats['Portfolio']['Cumulative Return'], Trading_Stats['Benchmark']['Cumulative Return'])
    
    print('Benchmark Stats :')
    print(Trading_Stats['Benchmark'])
    print()
    print('Portfolio Stats :')
    print(Trading_Stats['Portfolio'])
    
    return Trading_Stats
    
    
def Compute_Stats(returns):
    """
    Inputs : 
    - pd.DataFrame with index as datetime64[ns], one column of returns
    
    Outputs :
    - stats : dict : dictionary of statistics
    
    """
    stats = {}
    time_delta = returns.index[1] - returns.index[0]
    # time_delta in year
    time_delta = time_delta.days / 365.25
    
    # Cumulative return
    stats['Cumulative Return'] = (1 + returns).cumprod()
    
    

def Compute_IR(returns, benchmark):
    pass

def Compute_PSR(returns, benchmark):
    pass

def Plot_Cumulative(portfolio, benchmark):
    pass
    