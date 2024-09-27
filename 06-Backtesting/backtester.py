import numpy as np
import pandas as pd

def backtest(*args, **kwargs):
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
    pass

def stats():
    """
    Compute stats of the strategy according to the performance column
    (dot product)
    """
    pass

log(rt) - log(r_t-1)
log(rt) - (alpha)log(r_t-1)