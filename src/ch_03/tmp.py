#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# %%
import os

# Run the setup script
%run ../config/setup_project.py

# Call the function to set up the project path
setup_project_path()

# Now you can import your modules
from src.utils import helper as h_
from ch_02 import code_ch_02 as ch2

# %%
df = pd.read_parquet("../Data/IVE_kibot.parq")
df
# %%
# load ../data/variables_ch2.pkl
%run ../ch_02/code_ch_02.py

path = '../Data/variables_ch2.pkl'
import pickle
with open(path, 'rb') as f:
    bars = pickle.load(f)
    bar_time = pickle.load(f)

# %% [markdown]
### SNIPPET 3.1 DAILY VOLATILITY ESTIMATES
# %%
def getDailyVol(close, span0=100):
    # daily vol, reindexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily returns
    df0 = df0.ewm(span=span0).std()
    return df0

# %%
bar_time.df_OLHC
# %%
# set index of bar_time.df_OLHC as 'time_open'
bar_time.df_OLHC.set_index('time_open', inplace=True)
bar_time.df_OLHC
# %%
getDailyVol(bar_time.df_OLHC['close'])