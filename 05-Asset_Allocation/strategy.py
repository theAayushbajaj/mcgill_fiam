import numpy as np
import pandas as pd

def asset_allocator(*args, **kwargs):
    """
    Inputs :
    - Start Date
    - End Date
    - prices : pd.DataFrame : to compute returns and then covariance matrix
    - signals : A pandas datafram of signals (predictions and probabilities)
    Should have the same columns as prices
    We'll need a function to gather the signals (simply the predictions and 
    probabilities columns at row End Date)
    
    Outputs :
    Dataframe of weights, with the same columns as prices
    """
    pass