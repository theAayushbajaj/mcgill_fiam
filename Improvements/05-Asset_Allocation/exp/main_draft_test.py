#%%
import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf

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

#%%
# step 0
# IF THERE ARE NO PREDICTIONS, RUN THIS, ELSE SKIPPPPPP
if input('Do you want to add predictions to the stock data? (y/n)') == 'y':
    import tmp_simualate_pred


# %%

# Step 1) Gather the returns
Start_Date = '2015-01-01'
End_Date = '2019-12-01'

training = prices.loc[Start_Date:End_Date].dropna(axis=1)
returns = training.pct_change().dropna()
returns
# %%
# Step 2) 
# Estimate the Ledoit-Wolf Shrunk Covariance Matrix
lw = LedoitWolf()
shrunk_cov_matrix = lw.fit(returns).covariance_
shrunk_cov_matrix.shape

# %%

# %%
