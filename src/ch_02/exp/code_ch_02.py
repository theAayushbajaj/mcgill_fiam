import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class BarType(Enum):
    TICK = "tick"
    VOLUME = "volume"
    DOLLAR = "dollar"


class Sampling(ABC):
    @property
    @abstractmethod
    def units(self):
        """Returns:
        {'column': col name, 'Accumulate': boolean indicating accumulation,
        default_threshold: int or callable}
        """
        pass

    @abstractmethod
    def sample(self, df: pd.DataFrame, threshold: int = None, **kwargs):
        pass


# ============================2.1==============================================


class BarSampling(Sampling):
    @property
    @abstractmethod
    def units(self):
        """Returns:
        {'column': col name, 'Accumulate': boolean indicating accumulation,
        default_threshold: int or callable}
        """
        pass

    def sample(self, df: pd.DataFrame, threshold: int = None, thresh_mult: float = 1.0):
        """df with columns such as 'price', 'v', 'dv'
        Returns a list of indices of sampled bars"""
        if threshold is None:
            default_threshold = self.units["default_threshold"]
            threshold = (
                default_threshold(df[self.units["column"]])
                if callable(default_threshold)
                else default_threshold
            )

        t = df[self.units["column"]]
        ts = 0
        idx = []
        for i, x in enumerate(t):
            ts += x if self.units["Accumulate"] else 1
            if ts >= threshold * thresh_mult:
                idx.append(i)
                ts = 0
        return idx


class TickBar_Sampling(BarSampling):
    @property
    def units(self):
        return {"column": "price", "Accumulate": False, "default_threshold": 1_000}


class VolumeBar_Sampling(BarSampling):
    @property
    def units(self):
        return {
            "column": "v",
            "Accumulate": True,
            "default_threshold": lambda s: s.mean(),
        }


class DollarBar_Sampling(BarSampling):
    @property
    def units(self):
        return {
            "column": "dv",
            "Accumulate": True,
            "default_threshold": lambda s: s.mean(),
        }


# ============================2.2==============================================
# Tick Imbalance bars


class ImbalancedBarSampling(Sampling):
    @property
    @abstractmethod
    def units(self):
        """Returns:
        {'column': col name, 'Accumulate': boolean indicating accumulation,
        default_threshold: int or callable}
        """
        pass

    def bt_sequence(self, df: pd.DataFrame) -> pd.Series:
        """df with columns such as 'price', 'v', 'dv'
        Returns a series of b_t values, same length as df"""
        serie = df["price"].diff().apply(lambda x: 0 if x == 0 else abs(x) / x)
        # replace NaN with 0
        serie = serie.fillna(0)
        return serie

    def exp_ma(
        self,
        serie: pd.Series,
        cache: dict,
        start: int = 0,
        end: int = None,
        alpha: float = 0.95,
    ) -> float:
        """Exponential moving average on series 'series', from index 'start' to 'end'"""
        if end is None:
            end = len(serie)

        if start == end:
            return serie[start]

        if (start, end) in cache:
            return cache[(start, end)]

        ma = (
            alpha * self.exp_ma(serie, cache, start=start, end=end - 1, alpha=alpha)
            + (1 - alpha) * serie[end - 1]
        )
        cache[(start, end)] = ma
        return ma

    def sample(self, df: pd.DataFrame, t0: int = 10):
        """df with columns such as 'price', 'v', 'dv'
        Returns a list of indices of sampled bars

        t0 is the initial threshold for imbalance bars
        """
        bt = self.bt_sequence(df)
        # btvt
        if self.units["column"] is None:
            st = bt.copy()
        else:
            st = bt * df[self.units["column"]]

        theta_t = 0
        idx = [0, t0]
        for i, x in enumerate(
            tqdm(st, desc=f"Sampling Imbalance Bars {self.units['column']}")
        ):
            theta_t += x

            if (
                abs(theta_t)
                >= (
                    pd.Series(idx)
                    .diff()
                    .dropna()
                    .ewm(span=len(idx) - 1)
                    .mean()
                    .iloc[-1]
                    * abs(st[idx].ewm(span=len(idx)).mean().iloc[-1])
                )
                and i > t0
            ):
                idx.append(i)
                print(f"{self.units['column']} Ids : {idx}")
                print(f"theta_t : {theta_t}")
                print(f"E_0(T) : {pd.Series(idx).diff().dropna().ewm(span=len(idx) - 1).mean().iloc[-1]}")
                print(f"P(2b-1) : {abs(st[idx].ewm(span=len(idx)).mean().iloc[-1])}")
                print()
                theta_t = 0
                

        return idx


class Tick_ImbalanceBar(ImbalancedBarSampling):
    @property
    def units(self):
        return {"column": None}


class Volume_ImbalanceBar(ImbalancedBarSampling):
    @property
    def units(self):
        return {"column": "v"}


class Dollar_ImbalanceBar(ImbalancedBarSampling):
    @property
    def units(self):
        return {"column": "dv"}


# ============================================================================


class BarSampler:
    def __init__(self, df: pd.DataFrame, ImbalanceBar: bool = True):
        self.df = df
        self.ImbalanceBar = ImbalanceBar

    def sample(self, bar_type: BarType, *args, **kwargs):
        """
        args:
        bar_type: {'tick', 'volume', 'dollar'}

        kwargs:
        threshold: int, default None (if None, the default threshold is used)

        thresh_mult: int, default 1 (multiplier for threshold)

        scale: bool, default False (if True, the data is scaled)

        scale_type: {'minmax', 'standard'}, default 'standard'

        Returns a list of indices of sampled bars
        """
        if not self.ImbalanceBar:
            if bar_type == BarType.TICK:
                sampler = TickBar_Sampling()
            elif bar_type == BarType.VOLUME:
                sampler = VolumeBar_Sampling()
            elif bar_type == BarType.DOLLAR:
                sampler = DollarBar_Sampling()
            else:
                raise ValueError(f"Bar type {bar_type} not supported")
        else:
            if bar_type == BarType.TICK:
                sampler = Tick_ImbalanceBar()
            elif bar_type == BarType.VOLUME:
                sampler = Volume_ImbalanceBar()
            elif bar_type == BarType.DOLLAR:
                sampler = Dollar_ImbalanceBar()
            else:
                raise ValueError(f"Bar type {bar_type} not supported")

        indices = sampler.sample(self.df, *args, **kwargs)
        return indices

    def count_bars_weekly(self, bar_type: BarType, *args, **kwargs):
        """
        Count the number of bars produced by tick, volume, and dollar bars on a
        weekly basis. Plot a time series of that bar count.
        """
        indices = self.sample(bar_type, *args, **kwargs)
        sampled_df = self.df.iloc[indices]
        weekly_counts = sampled_df.resample("W").size()

        return weekly_counts

    def plot_weekly_counts(self, threshold: int = None, thresh_mult=1):
        """
        Plot the number of bars produced on a weekly basis for tick, volume,
        and dollar bars.
        """
        tick_counts = self.count_bars_weekly(
            BarType.TICK, threshold=threshold, thresh_mult=1
        )
        volume_counts = self.count_bars_weekly(
            BarType.VOLUME, threshold=None, thresh_mult=thresh_mult
        )
        dollar_counts = self.count_bars_weekly(
            BarType.DOLLAR, threshold=None, thresh_mult=thresh_mult
        )

        plt.figure(figsize=(14, 7))
        plt.plot(tick_counts, label="Tick Bars", marker="o")
        plt.plot(volume_counts, label="Volume Bars", marker="s")
        plt.plot(dollar_counts, label="Dollar Bars", marker="^")

        plt.title("Weekly Bar Counts")
        plt.xlabel("Date")
        plt.ylabel("Number of Bars")
        plt.legend()
        plt.grid(True)
        plt.show()
        return tick_counts, volume_counts, dollar_counts

    def parallel_sample(self, t0: int = 10):
        """
        Sample tick, volume, and dollar imbalance bars in parallel.
        """
        #bar_types = [BarType.TICK, BarType.VOLUME, BarType.DOLLAR]
        # bar_types without hardcoded values
        bar_types = [bar_type for bar_type in BarType]

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.sample, bar_type, t0=t0): bar_type
                for bar_type in bar_types
            }

            results = {}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Sampling Bars in Parallel",
            ):
                bar_type = futures[future]
                results[bar_type] = future.result()

        return results
