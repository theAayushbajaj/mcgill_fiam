"""Code to solve exercises in Chapter 4 of Advances in Financial Machine 
Learning by Marcos Lopez de Prado.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sys
import os

# Import project setup script
from config.setup_project import setup_project_path, add_src_path
# Setup the project path
setup_project_path()
# Add src path
add_src_path()
from src.ch_03 import code_ch_03 as f_ch03

# **SNIPPET 4.1** **ESTIMATING THE UNIQUENESS OF A LABEL**

def mpNumCoEvents(closeIdx, t1, molecule):
    '''
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed
    Any event that starts before t1[molecule].max() impacts the count.
    '''
    #1) find events that span the period [molecule[0], molecule[-1]]
    t1 = t1.fillna(closeIdx[-1])  # unclosed events still must impact other weights
    t1 = t1[t1 >= molecule[0]]  # events that end at or after molecule[0]
    t1 = t1.loc[:t1[molecule].max()]  # events that start at or before t1[molecule].max()
    
    #2) count events spanning a bar
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=closeIdx[iloc[0]:iloc[1] + 1])
    for tIn, tOut in t1.items():  # Changed from iteritems() to items()
        count.loc[tIn:tOut] += 1
    return count.loc[molecule[0]:t1[molecule].max()]

# **SNIPPET 4.2** **ESTIMATING THE AVERAGE UNIQUENESS OF A LABEL**

def mpSampleTW(t1, numCoEvents, molecule):
    # Derive average uniqueness over the event's lifespan
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].items():  # Changed from .iteritems() to .items()
        wght.loc[tIn] = (1. / numCoEvents.loc[tIn:tOut]).mean()
    return wght

# SNIPPET 4.3 BUILD AN INDICATOR MATRIX
def getIndMatrix(barIx, t1):
    # Get indicator matrix
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
    for i, (t0, t1) in enumerate(t1.items()):  # Changed from .iteritems() to .items()
        indM.loc[t0:t1, i] = 1.
    return indM

# SNIPPET 4.4 COMPUTE AVERAGE UNIQUENESS
def getAvgUniqueness(indM):
    # Average uniqueness from indicator matrix
    c = indM.sum(axis=1)  # concurrency
    u = indM.div(c, axis=0)  # uniqueness
    avgU = u[u > 0].mean()  # average uniqueness
    return avgU

# SNIPPET 4.5 RETURN SAMPLE FROM SEQUENTIAL BOOTSTRAP
def seqBootstrap(indM, me=None):
    # Generate a sample via sequential bootstrap
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series()
        for i in indM:
            indM_ = indM[phi + [i]]  # reduce indM
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob = avgU / avgU.sum()  # draw prob
        phi += [np.random.choice(indM.columns, p=prob)]
    return phi

from tqdm import tqdm
# SNIPPET 4.5 RETURN SAMPLE FROM SEQUENTIAL BOOTSTRAP
def seqBootstrap(indM, sLength=None, uniqueObj = 0.8):
    # Generate a sample via sequential bootstrap
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    
    # Use tqdm to add a progress bar
    for _ in tqdm(range(sLength), desc="Sequential Bootstrap Progress"):
        avgU = pd.Series()
        for i in indM:
            indM_ = indM[phi + [i]]  # reduce indM
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob = avgU / avgU.sum()  # draw prob
        phi += [np.random.choice(indM.columns, p=prob)]
        if len(phi) % 50 == 0:
            print(f"Sample length: {len(phi)}")
            print('Sequential uniqueness:',getAvgUniqueness(indM[phi]).mean())
            print()
        if getAvgUniqueness(indM[phi]).mean() < uniqueObj:
            break
    
    return phi

# SNIPPET 4.6 EXAMPLE OF SEQUENTIAL BOOTSTRAP

def main(t1, **kwargs):
    # t1=pd.Series([2,3,5],index=[0,2,4]) # t0,t1 for each feature obs
    # barIx=range(t1.max()+1) # index of bars
    barIx = t1.index
    indM=getIndMatrix(barIx,t1)
    # phi=np.random.choice(indM.columns,size=indM.shape[1])
    phi=np.random.choice(indM.columns,size=kwargs['sLength'])
    # print(phi)
    print('Standard uniqueness:',getAvgUniqueness(indM[phi]).mean())
    phi=seqBootstrap(indM, **kwargs)
    # print(phi)
    print('Sequential uniqueness:',getAvgUniqueness(indM[phi]).mean())
    return phi

# **SNIPPET 4.10** **DETERMINATION OF SAMPLE WEIGHT BY ABSOLUTE RETURN ATTRIBUTION**
def mpSampleW(t1, numCoEvents, close, molecule):
    # Derive sample weight by return attribution
    ret = np.log(close).diff()  # log-returns, so that they are additive
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].items():  # Changed from iteritems() to items()
        wght.loc[tIn] = (ret.loc[tIn:tOut] / numCoEvents.loc[tIn:tOut]).sum()
    return wght.abs()

# **SNIPPET 4.11** **IMPLEMENTATION OF TIME-DECAY FACTORS**


def getTimeDecay(tW, clfLastW=1.0):
    # apply piecewise-linear decay to observed uniqueness (tW)
    # newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW = tW.sort_index().cumsum()
    if clfLastW >= 0:
        slope = (1.0 - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1.0 / ((clfLastW + 1) * clfW.iloc[-1])
    const = 1.0 - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW < 0] = 0
    print(const, slope)
    return clfW

# Add technical features (exo 4.3)
def add_techs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df must have 'close' column, in index as datetime.
    """
    # 1. Rolling Volatility (Standard Deviation)
    df['rolling_vol_10'] = df['close'].rolling(window=10).std()
    df['rolling_vol_20'] = df['close'].rolling(window=20).std()

    # 2. Moving Averages
    df['MA_10'] = df['close'].rolling(window=10).mean()
    df['MA_20'] = df['close'].rolling(window=20).mean()  # Define MA_20 for Bollinger Bands
    df['MA_50'] = df['close'].rolling(window=50).mean()

    # 3. Exponential Moving Averages (EMA)
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # 4. Relative Strength Index (RSI)
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 5. Bollinger Bands
    df['BB_upper'] = df['MA_20'] + (df['rolling_vol_20'] * 2)
    df['BB_lower'] = df['MA_20'] - (df['rolling_vol_20'] * 2)

    # 6. MACD (Moving Average Convergence Divergence)
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 7. Bar duration as time_close (index) - time_open
    df['bar_duration'] = df.index.to_series().diff().dt.total_seconds()
    
    # 8. Daily Volatility
    daily_vol = f_ch03.getDailyVol(close=df['close'], span0=100).dropna()
    df['daily_vol'] = daily_vol
    

    # Drop any rows with NaN values (due to rolling calculations)
    df = df.dropna()

    return df