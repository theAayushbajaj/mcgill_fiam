"""Code to solve exercises in Chapter 3 of Advances in Financial Machine 
Learning by Marcos Lopez de Prado.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .code_afml_03 import *


# SNIPPET 3.1 DAILY VOLATILITY ESTIMATES
def getDailyVol(close, span0=100):
    # daily vol, reindexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(
        close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0] :]
    )
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily returns
    df0 = df0.ewm(span=span0).std()
    return df0


# Adapt getTEvents(gRaw, h) so h is a Series of thresholds


# SNIPPET 2.4 THE SYMMETRIC CUSUM FILTER ADDAPTED FOR 3.1
def getTEvents(gRaw, h):
    # inner join to have only the common dates
    # if h is not an a flot or int
    if not isinstance(h, (float, int)):
        tEvents, sPos, sNeg = [], 0, 0
        diff = gRaw.diff().dropna()
        common_dates = diff.index.intersection(h.index)
        diff = diff.loc[common_dates]
        h = h.loc[common_dates]
        for i in diff.index:
            try:
                sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
                if sNeg < -h.loc[i]:
                    sNeg = 0
                    tEvents.append(i)
                elif sPos > h.loc[i]:
                    sPos = 0
                    tEvents.append(i)
            # return pd.DatetimeIndex(tEvents)
            except Exception as e:
                print(e)
        return tEvents
    else:
        tEvents, sPos, sNeg = [], 0, 0
        diff = gRaw.diff()
        for i in diff.index[1:]:
            sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
            if sNeg < -h:
                sNeg = 0; tEvents.append(i)
            elif sPos > h:
                sPos = 0; tEvents.append(i)
        # return pd.DatetimeIndex(tEvents)
        return tEvents


# SNIPPET 3.4 ADDING A VERTICAL BARRIER

"""The error occurs because `tEvents` is a list, and 
`pd.Timedelta(days=numDays)` cannot be directly added to a list. You need to 
convert `tEvents` to a pandas `DatetimeIndex` or a `Series` before performing 
the addition.

Here's how you can modify the `addVerticalBarrier` function to fix this issue:
"""


def addVerticalBarrier(tEvents, close, numDays=1):
    # Convert tEvents to a DatetimeIndex
    tEvents = pd.DatetimeIndex(tEvents)

    # Add the time delta to tEvents
    t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))

    # Ensure t1 does not go beyond the available close data
    t1 = t1[t1 < close.shape[0]]

    # Create a Series with t1 as the index and tEvents as values
    t1 = pd.Series(close.index[t1], index=tEvents[: t1.shape[0]])

    return t1


# # SNIPPET 3.6 EXPANDING getEvents TO INCORPORATE META-LABELING


# def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
#     # 1) get target
#     trgt = trgt.loc[tEvents]
#     trgt = trgt[trgt > minRet]  # minRet
#     # 2) get t1 (max holding period)
#     if t1 is False:
#         t1 = pd.Series(pd.NaT, index=tEvents)
#     # 3) form events object, apply stop loss on t1
#     if side is None:
#         side_, ptSl_ = pd.Series(1.0, index=trgt.index), [ptSl[0], ptSl[0]]
#     else:
#         side_, ptSl_ = side.loc[trgt.index], ptSl[:2]
#     events = pd.concat({"t1": t1, "trgt": trgt, "side": side_}, axis=1).dropna(
#         subset=["trgt"]
#     )
#     df0 = mpPandasObj(
#         func=applyPtSlOnT1,
#         pdObj=("molecule", events.index),
#         numThreads=numThreads,
#         close=close,
#         events=events,
#         ptSl=ptSl_,
#     )
#     events["t1"] = df0.dropna(how="all").min(axis=1)  # pd.min ignores nan
#     if side is None:
#         events = events.drop("side", axis=1)
#     return events


def applyPtSlOnT1(close, events, ptSl, molecule):
    # Apply stop loss/profit taking, if it takes place before t1 (end of event)
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

    for loc, t1 in events_["t1"].fillna(close.index[-1]).items():
        df0 = close[loc:t1]  # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, "side"]  # path returns
        out.loc[loc, "sl"] = df0[df0 < sl[loc]].index.min()  # earliest stop loss
        out.loc[loc, "pt"] = df0[df0 > pt[loc]].index.min()  # earliest profit taking

    return out


def mpPandasObj(func, pdObj, numThreads=24, mpBatches=1, linMols=True, **kargs):
    import pandas as pd

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
        df0 = pd.concat([df0, i])

    df0 = df0.sort_index()
    return df0


# # **SNIPPET 3.5** **LABELING FOR SIDE AND SIZE**


# def getBins(events, close):
#     # 1) prices aligned with events
#     events_ = events.dropna(subset=["t1"])
#     px = events_.index.union(events_["t1"].values).drop_duplicates()
#     px = close.reindex(px, method="bfill")

#     # 2) create out object
#     out = pd.DataFrame(index=events_.index)
#     out["ret"] = px.loc[events_["t1"].values].values / px.loc[events_.index] - 1
#     out["bin"] = np.sign(out["ret"])
#     return out


# **SNIPPET 3.8** **DROPPING UNDER-POPULATED LABELS**


def dropLabels(events, minPct=0.05):
    # apply weights, drop labels with insufficient examples
    while True:
        df0 = events["bin"].value_counts(normalize=True)
        if df0.min() > minPct or df0.shape[0] < 3:
            break
        print("dropped label", df0.argmin(), df0.min())
        events = events[events["bin"] != df0.argmin()]
    return events


# 3.3 Adjust the getBins function (Snippet 3.5) to return a 0 whenever the vertical barrier is the one touched first.


def getBinsNew(events, close, t1=None):
    """
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    -t1 is original vertical barrier series
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

    if "side" not in events_:
        # only applies when not meta-labeling
        # to update bin to 0 when vertical barrier is touched, we need the original
        # vertical barrier series since the events['t1'] is the time of first
        # touch of any barrier and not the vertical barrier specifically.
        # The index of the intersection of the vertical barrier values and the
        # events['t1'] values indicate which bin labels needs to be turned to 0
        vtouch_first_idx = events[events["t1"].isin(t1.values)].index
        out.loc[vtouch_first_idx, "bin"] = 0.0

    if "side" in events_:
        out.loc[out["ret"] <= 0, "bin"] = 0  # meta-labeling
    return out


# **SNIPPET 3.7** **EXPANDING `getBins` TO INCORPORATE META-LABELING**


def getBins(events, close):
    """
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    - events.index is event's starttime
    - events['t1'] is event's endtime
    - events['trgt'] is event's target
    - events['side'] (optional) implies the algo's position side
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


# **SNIPPET 3.6** **EXPANDING `getEvents` TO INCORPORATE META-LABELING**


def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    # 1) get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]  # minRet

    # 2) get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)

    # 3) form events object, apply stop loss on t1
    if side is None:
        side_ = pd.Series(1.0, index=trgt.index)
        ptSl_ = [ptSl[0], ptSl[0]]
    else:
        side_ = side.loc[trgt.index]
        ptSl_ = [ptSl[0], ptSl[1]]  # Use the list directly

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
