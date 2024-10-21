    #%%
import numpy as np
import pandas as pd
import pickle

#%%

# load objects/weights.pkl
with open("signals.pkl", "rb") as f:
    signals = pickle.load(f)
    

# %%

w = weights.iloc[-5]

# %%

weights
# %%
