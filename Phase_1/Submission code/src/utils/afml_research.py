import numpy as np
import pandas as pd
import datetime as dt

from sklearn.datasets import make_classification

def create_price_data(start_price: float = 1000.00, mu: float = .0, var: float = 1.0, n_samples: int = 1000000):
    i = np.random.normal(mu, var, n_samples)
    df0 = pd.date_range(periods=n_samples, freq=pd.tseries.offsets.Minute(), end=dt.datetime.today())
    X = pd.Series(i, index=df0, name='close').to_frame()
    X.at[df0[0], 'close'] = start_price  # Corrected to use .at instead of .iat
    X.cumsum().plot.line()
    return X.cumsum()


def make_randomt1_data(n_samples: int =10000, max_days: float = 5., Bdate: bool = True):
    # generate a random dataset for a classification problem
    if Bdate:
        _freq = pd.tseries.offsets.BDay()
    else:
        _freq = 'D'
    _today = dt.datetime.today()
    df0 = pd.date_range(periods=n_samples, freq=_freq, end=_today)
    rand_days = np.random.uniform(1, max_days, n_samples)
    rand_days = pd.Series([dt.timedelta(days = d) for d in rand_days], index = df0)
    df1 = df0 + pd.to_timedelta(rand_days, unit='d')
    df1.sort_values(inplace=True)
    X = pd.Series(df1, index = df0, name='t1').to_frame()
    return X

def make_classification_data(n_features=40, n_informative=10, n_redundant=10, n_samples=10000, days: int = 1):
    # generate a random dataset for a classification problem
    _today = dt.datetime.today()    
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, random_state=0, shuffle=False)
    df0 = pd.date_range(periods=n_samples, freq=pd.tseries.offsets.BDay(), end=_today)
    X = pd.DataFrame(X, index=df0)
    y = pd.Series(y, index=df0).to_frame('bin')
    df0 = ['I_%s' % i for i in range(n_informative)] + ['R_%s' % i for i in range(n_redundant)]
    df0 += ['N_%s' % i for i in range(n_features - len(df0))]
    X.columns = df0
    y['w'] = 1.0 / y.shape[0]
    y['t1'] = pd.Series(y.index, index=y.index - dt.timedelta(days = days))
    y.at[-1:, 't1'] = _today
    y.t1.fillna(method ='bfill', inplace = True)
    return X, y

def create_portfolio(price: list = [95, 1000], position: list = [1000, 10000], n_sample: int = 10000, imbalance: bool = True):
    df = make_randomt1_data(n_samples = n_sample, max_days = 10., Bdate = True)
    df['open'] = np.random.uniform(price[0], price[1], n_sample)
    df['close'] = df['open'].apply(lambda x: x * np.random.uniform(0.8, 1.2))
    df['open_pos'] = np.random.uniform(position[0], position[1], n_sample).round()
    df['close_pos'] = np.random.uniform(position[0], position[1], n_sample).round()

    df['yield'] = np.random.uniform(0.012, 0.12, n_sample)
    df['expense'] = df['open'].apply(lambda x: x * np.random.uniform(0.005, 0.05))
    df['label'] = np.nan
    if imbalance:
        for idx in df[df['open']<200].index:df.loc[idx,'label'] = 1
    else:
        for idx in df[df['open']<df['open'].quantile(0.4)].index:df.loc[idx,'label'] = 1
    df['label'].fillna(0, inplace= True)
    
    df['clean_open'] = df['open'].apply(lambda x: x * np.random.uniform(0.92, 0.95))
    df['clean_close'] = df['yield']
    df['clean_close'] = df['clean_close'].apply(lambda x: 1-x).mul(df['close'])
    df['yield'] = df['yield'].mul(df['close'])
        

    p_str = "Sample Portfolio Construct:\n{0}\nEquity label: {1}\nBond label: {2}\nEquity to Debt Ratio: {3:.4f}"
    p(p_str.format("=" * 55,
                   df['label'].value_counts()[0],
                   df['label'].value_counts()[1], 
                   df['label'].value_counts()[0]/df['label'].value_counts()[1]))
    
    junk_bond = df[(df['yield'] >= 0.1) & (df['label'] == 1)].count()[0]
    div_equity = df[(df['yield'] >= 0.1) & (df['label'] != 1)].count()[0]
    p("\nJunk bond (Below BBB-grade): {0} %\nDividend equity: {1} %".format(100 * junk_bond/ n_sample, 100 * div_equity/n_sample))
    return df

def create_rtndf(n_samples: int = 1000, max_days: int = 5, rtn:list = [-.5, .5], Bdate: bool = True):
    if rtn[0] >= rtn[1] or rtn[1] < 0:
        rtn = [-.5,.5]
    df = make_randomt1_data(n_samples = n_samples//2, max_days = max_days, Bdate = Bdate)
    idx = pd.to_datetime(df.index.union(df.t1).sort_values())
    df = pd.DataFrame(1., index = idx, columns = np.arange(n_samples))
    df = df.applymap(lambda x: np.random.uniform(rtn[0], rtn[1]))
    return df