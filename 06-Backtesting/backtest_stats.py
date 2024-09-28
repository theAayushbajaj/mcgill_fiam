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
    return trading_log

# ================== Trading Log Stats ==================

def get_TL_Stats(trading_log):
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
    hit_miss_trade['Avg Hit'] = trading_log[trading_log > 0].mean(axis=1)
    # Average return per miss
    hit_miss_trade['Avg Miss'] = trading_log[trading_log < 0].mean(axis=1)
    # Average return per trade
    hit_miss_trade['Avg Trade'] = trading_log[trading_log!=0].mean(axis=1)
    
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
    hit_miss_stock['Avg Hit'] = trading_log[trading_log > 0].mean(axis=0)
    # Average return per miss
    hit_miss_stock['Avg Miss'] = trading_log[trading_log < 0].mean(axis=0)
    # Average return per trade
    hit_miss_stock['Avg Trade'] = trading_log[trading_log!=0].mean(axis=0)
    
    # Hits minus Misses
    hit_miss_stock['Hits - Misses'] = hit_miss_stock['Hit'] - hit_miss_stock['Miss']
    # Hit ratio
    hit_miss_stock['Hit Ratio'] = hit_miss_stock['Hit'] / hit_miss_stock['Total']
    
    # Overview
    hit_miss_overall = pd.DataFrame()
    hit_miss_overall['Hit'] = hit_miss_stock['Hit'].sum()
    hit_miss_overall['Miss'] = hit_miss_stock['Miss'].sum()
    hit_miss_overall['Total'] = hit_miss_overall['Hit'] + hit_miss_overall['Miss']
    # Averages
    # Average return per hit
    hit_miss_overall['Avg Hit'] = (trading_log[trading_log > 0].sum().sum()) / hit_miss_overall['Hit']
    # Average return per miss
    hit_miss_overall['Avg Miss'] = (trading_log[trading_log < 0].sum().sum()) / hit_miss_overall['Miss']
    # Average return per trade
    hit_miss_overall['Avg Trade'] = (trading_log[trading_log!=0].sum().sum()) / hit_miss_overall['Total']
    # Hits minus Misses
    hit_miss_overall['Hits - Misses'] = hit_miss_overall['Hit'] - hit_miss_overall['Miss']
    # Hit ratio
    hit_miss_overall['Hit Ratio'] = hit_miss_overall['Hit'] / hit_miss_overall['Total']
    
    Hit_Miss_Stats = {  
        'Trade' : hit_miss_trade,
        'Stock' : hit_miss_stock,
        'Overall' : hit_miss_overall
    }
    
    return Hit_Miss_Stats
    
    

    