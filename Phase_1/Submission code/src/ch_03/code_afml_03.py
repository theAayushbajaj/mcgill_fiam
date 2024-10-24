# import standard libs
import datetime
import json
import os
import pickle
import re
import sys
import time
from collections import OrderedDict as od
from pathlib import Path, PurePath

# import python scientific stack
import pandas as pd
import pandas_datareader.data as web
from IPython.core.debugger import set_trace as bp
from IPython.display import display
from tqdm import tqdm

pd.set_option("display.max_rows", 100)
from multiprocessing import cpu_count

from dask import dataframe as dd
from dask.diagnostics import ProgressBar

pbar = ProgressBar()
pbar.register()
import math

import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from numba import jit

# import ffn


# %% [markdown]
# ### Symmetric CUSUM Filter [2.4]
#


# %%
def getTEvents(gRaw, h):
    tEvents, sPos, sNeg = [], 0, 0
    diff = np.log(gRaw).diff().dropna()
    for i in tqdm(diff.index[1:]):
        try:
            pos, neg = float(sPos + diff.loc[i]), float(sNeg + diff.loc[i])
        except Exception as e:
            print(e)
            print(sPos + diff.loc[i], type(sPos + diff.loc[i]))
            print(sNeg + diff.loc[i], type(sNeg + diff.loc[i]))
            break
        sPos, sNeg = max(0.0, pos), min(0.0, neg)
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


# %% [markdown]
# ### Daily Volatility Estimator [3.1]
#


# %%
def getDailyVol(close, span0=100):
    # daily vol reindexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(
        close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0] :]
    )
    try:
        df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily rets
    except Exception as e:
        print(f"error: {e}\nplease confirm no duplicate indices")
    df0 = df0.ewm(span=span0).std().rename("dailyVol")
    return df0


# %% [markdown]
# ### Triple-Barrier Labeling Method [3.2]
#


# %%
def applyPtSlOnT1(close, events, ptSl, molecule):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[["t1"]].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events_["trgt"]
    else:
        pt = pd.Series(index=events.index)  # NaNs
    if ptSl[1] > 0:
        sl = -ptSl[1] * events_["trgt"]
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in events_["t1"].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1]  # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, "side"]  # path returns
        out.loc[loc, "sl"] = df0[df0 < sl[loc]].index.min()  # earliest stop loss
        out.loc[loc, "pt"] = df0[df0 > pt[loc]].index.min()  # earliest profit taking
    return out


# %% [markdown]
# ### Gettting Time of First Touch (getEvents) [3.3], [3.6]
#


# %%
def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    # 1) get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]  # minRet
    # 2) get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    # 3) form events object, apply stop loss on t1
    if side is None:
        side_, ptSl_ = pd.Series(1.0, index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]
    events = pd.concat({"t1": t1, "trgt": trgt, "side": side_}, axis=1).dropna(
        subset=["trgt"]
    )
    df0 = mpPandasObj(
        func=applyPtSlOnT1,
        pdObj=("molecule", events.index),
        numThreads=numThreads,
        close=close,
        events=events,
        ptSl=ptSl_,
    )
    events["t1"] = df0.dropna(how="all").min(axis=1)  # pd.min ignores nan
    if side is None:
        events = events.drop("side", axis=1)
    return events


# %% [markdown]
# ### Adding Vertical Barrier [3.4]
#


# %%
def addVerticalBarrier(tEvents, close, numDays=1):
    t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=tEvents[: t1.shape[0]])
    return t1


# %% [markdown]
# ### Labeling for side and size [3.5]
#

# %%


def getBinsOld(events, close):
    # 1) prices aligned with events
    events_ = events.dropna(subset=["t1"])
    px = events_.index.union(events_["t1"].values).drop_duplicates()
    px = close.reindex(px, method="bfill")
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out["ret"] = px.loc[events_["t1"].values].values / px.loc[events_.index] - 1
    out["bin"] = np.sign(out["ret"])
    # where out index and t1 (vertical barrier) intersect label 0
    try:
        locs = out.query("index in @t1").index
        out.loc[locs, "bin"] = 0
    except:
        pass
    return out


# %% [markdown]
# ### Expanding getBins to Incorporate Meta-Labeling [3.7]
#


# %%
def getBins(events, close):
    """
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    """
    # 1) prices aligned with events
    events_ = events.dropna(subset=["t1"])
    px = events_.index.union(events_["t1"].values).drop_duplicates()
    px = close.reindex(px, method="bfill")
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out["ret"] = px.loc[events_["t1"].values].values / px.loc[events_.index] - 1
    if "side" in events_:
        out["ret"] *= events_["side"]  # meta-labeling
    out["bin"] = np.sign(out["ret"])
    if "side" in events_:
        out.loc[out["ret"] <= 0, "bin"] = 0  # meta-labeling
    return out


# %% [markdown]
# ### Dropping Unnecessary Labels [3.8]
#


# %%
def dropLabels(events, minPct=0.05):
    # apply weights, drop labels with insufficient examples
    while True:
        df0 = events["bin"].value_counts(normalize=True)
        if df0.min() > minPct or df0.shape[0] < 3:
            break
        print("dropped label: ", df0.argmin(), df0.min())
        events = events[events["bin"] != df0.argmin()]
    return events


# %% [markdown]
# ### Linear Partitions [20.4.1]
#


# %%
def linParts(numAtoms, numThreads):
    # partition of atoms with a single loop
    parts = np.linspace(0, numAtoms, min(numThreads, numAtoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


# %%
def nestedParts(numAtoms, numThreads, upperTriang=False):
    # partition of atoms with an inner loop
    parts, numThreads_ = [0], min(numThreads, numAtoms)
    for num in range(numThreads_):
        part = 1 + 4 * (
            parts[-1] ** 2 + parts[-1] + numAtoms * (numAtoms + 1.0) / numThreads_
        )
        part = (-1 + part**0.5) / 2.0
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upperTriang:  # the first rows are heaviest
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts


# %% [markdown]
# ### multiprocessing snippet [20.7]
#


# %%
def mpPandasObj(func, pdObj, numThreads=24, mpBatches=1, linMols=True, **kargs):
    """
    Parallelize jobs, return a dataframe or series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kwds: any other argument needed by func

    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kwds)
    """
    import pandas as pd

    # if linMols:parts=linParts(len(argList[1]),numThreads*mpBatches)
    # else:parts=nestedParts(len(argList[1]),numThreads*mpBatches)
    if linMols:
        parts = linParts(len(pdObj[1]), numThreads * mpBatches)
    else:
        parts = nestedParts(len(pdObj[1]), numThreads * mpBatches)

    jobs = []
    for i in range(1, len(parts)):
        job = {pdObj[0]: pdObj[1][parts[i - 1] : parts[i]], "func": func}
        job.update(kargs)
        jobs.append(job)
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads=numThreads)
    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out
    for i in out:
        df0 = df0.append(i)
    df0 = df0.sort_index()
    return df0


# %% [markdown]
# ### single-thread execution for debugging [20.8]
#


# %%
def processJobs_(jobs):
    # Run jobs sequentially, for debugging
    out = []
    for job in jobs:
        out_ = expandCall(job)
        out.append(out_)
    return out


# %% [markdown]
# ### Example of async call to multiprocessing lib [20.9]
#

import datetime as dt

# %%
import multiprocessing as mp


# ________________________________
def reportProgress(jobNum, numJobs, time0, task):
    # Report progress as asynch jobs are completed
    msg = [float(jobNum) / numJobs, (time.time() - time0) / 60.0]
    msg.append(msg[1] * (1 / msg[0] - 1))
    timeStamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = (
        timeStamp
        + " "
        + str(round(msg[0] * 100, 2))
        + "% "
        + task
        + " done after "
        + str(round(msg[1], 2))
        + " minutes. Remaining "
        + str(round(msg[2], 2))
        + " minutes."
    )
    if jobNum < numJobs:
        sys.stderr.write(msg + "\r")
    else:
        sys.stderr.write(msg + "\n")
    return


# ________________________________
def processJobs(jobs, task=None, numThreads=24):
    # Run in parallel.
    # jobs must contain a 'func' callback, for expandCall
    if task is None:
        task = jobs[0]["func"].__name__
    pool = mp.Pool(processes=numThreads)
    outputs, out, time0 = pool.imap_unordered(expandCall, jobs), [], time.time()
    # Process asyn output, report progress
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        reportProgress(i, len(jobs), time0, task)
    pool.close()
    pool.join()  # this is needed to prevent memory leaks
    return out


# %% [markdown]
# ### Unwrapping the Callback [20.10]
#


# %%
def expandCall(kargs):
    # Expand the arguments of a callback function, kargs['func']
    func = kargs["func"]
    del kargs["func"]
    out = func(**kargs)
    return out


# %% [markdown]
# ### Pickle Unpickling Objects [20.11]
#


# %%
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)


# ________________________________
def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


# ________________________________
