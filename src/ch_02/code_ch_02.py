"""Code to solve exercises in Chapter 2 of Advances in Financial Machine 
Learning by Marcos Lopez de Prado.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def compute_vwap(
    df: pd.DataFrame, price_column: str, volume_column: str
) -> pd.DataFrame:
    """
    Computes the volume weighted average price. Returns a DataFrame with a new
    column 'vwap'.
    To use on a DataFrame group by bar key.

    Example:
    For time bars, group by 15Min freq
    data.groupby(pd.Grouper(key='timestamp', freq='15Min'))

    For tick bars, group by the desired number of ticks per bar
    data_tick_grp = data.reset_index().assign(grpId=lambda row:
    row.index // num_ticks_per_bar) # Assign a group ID to each tick
    data_tick_vwap =  data_tick_grp.groupby('grpId')
    """
    v = df[volume_column]
    p = df[price_column]
    vwap = np.sum(p * v) / np.sum(v)
    df["vwap"] = vwap
    return df


def make_OLHC_bars(df: pd.DataFrame):
    """
    Inputs a dataframe (GROUP BY OBJECT), and returns a
    dataframe with columns:
    - open (first price in the bar)
    - low (min price in the bar)
    - high (max price in the bar)
    - close (last price in the bar)
    - vwap (volume weighted average price in the bar)
    - volume (total volume in the bar)
    - time (timestamp of the last price in the bar)
    """
    # df = df.copy()

    def _agg(df):
        """Helper function to apply to each group"""
        return pd.Series(
            {
                "time_open": df["dates"].iloc[0],
                "open": df["price"].iloc[0],
                "low": df["price"].min(),
                "high": df["price"].max(),
                "close": df["price"].iloc[-1],
                "time_close": df["dates"].iloc[-1],
                "vwap": (df["price"] * df["v"]).sum() / df["v"].sum(),
                "volume": df["v"].sum(),
            }
        )

    return df.apply(_agg).reset_index(drop=True)

# SNIPPET 2.4 THE SYMMETRIC CUSUM FILTER
def getTEvents(gRaw, h):
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
# Example Usage
# yt = bar.df_OLHC['close'].pct_change().dropna()
# tEvents = getTEvents(yt, h=0.05)
# bar.df_OLHC['cusum'] = 0
# bar.df_OLHC['cusum'].loc[tEvents] = 1
# bar.df_OLHC['cusum'].sum()


class Bar:
    def __init__(self, grouped_df: pd.DataFrame, Bar_Type: str = None):
        """
        df must be a GroupBy object
        """
        self.Bar_Type = Bar_Type
        self.num_of_bars = len(grouped_df)

        df_vwap = grouped_df.apply(
            lambda x: compute_vwap(x, price_column="price", volume_column="v")
        )

        df_vwap.reset_index(drop=True, inplace=True)
        self.df = df_vwap

        self.df_OLHC = make_OLHC_bars(df_vwap)
