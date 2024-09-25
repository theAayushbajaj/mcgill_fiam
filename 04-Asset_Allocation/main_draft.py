#%%
import numpy as np
import pandas as pd

import pickle
#%%

def asset_allocator():
    """
    1) Create a dataframe of returns from STARTDATE to ENDDATE
    2) Compute the covariance matrix of the returns
    3) Gather the signals from columns 'prediction' and 'probability'
    4) Mix Up together
    """
    pass


#%%

# LOAD /Users/paulkelendji/Desktop/GitHub_paul/mcgill_fiam/objects/prices.pkl
path = '../objects/prices.pkl'

prices = pd.read_pickle(path)
prices
# %%

Start_Date = '2015-01-01'
End_Date = '2019-12-01'

training = prices.loc[Start_Date:End_Date].dropna(axis=1)
training
# %%
training_rets = training.pct_change().dropna()
training_rets

# %%
cov_matrix = training_rets.cov()
cov_matrix
# %%
