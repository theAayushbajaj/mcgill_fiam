from sklearn.datasets import make_classiﬁcation
import numpy as np
import pandas as pd

import sys
import os



current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root_dir)

from src.ch_08 import code_ch_08 as f_ch08



def read_kibot_ticks(fp):
    # read tick data from http://www.kibot.com/support.aspx#data_format
    cols = list(map(str.lower,['Date','Time','Price','Bid','Ask','Size']))
    df = (pd.read_csv(fp, header=None)
          .rename(columns=dict(zip(range(len(cols)),cols)))
          .assign(dates=lambda df: (pd.to_datetime(df['date']+df['time'],
                                                  format='%m/%d/%Y%H:%M:%S')))
          .assign(v=lambda df: df['size']) # volume
          .assign(dv=lambda df: df['price']*df['size']) # dollar volume
          .drop(['date','time'],axis=1)
          .set_index('dates')
          .drop_duplicates())
    return df

def  prices_features_sim(trnsX, n_feats=2, b0=100, drift=0.01, sigma=0.2, freq='D'):
    n_samples = len(trnsX)
    
    # Select informative features 
    informative_feats = trnsX.iloc[:, :n_feats]
    
    # Standardized features
    informative_feats = (informative_feats - informative_feats.mean()) / informative_feats.std()
    
    # influece series
    influence_series = informative_feats.sum(axis=1)
    if n_feats > 1:
        influence_series /= n_feats
    
    prices = np.zeros(n_samples)
    prices[0] = b0
    
    for t in range(1, n_samples):
        shock = np.random.normal(0, 1)
        delta_time = trnsX.index[t] - trnsX.index[t-1]
        delta_time = delta_time.total_seconds() / (60*60*24*365)
        log_ret = (drift - 0.5 * sigma**2) * delta_time + sigma * np.sqrt(delta_time) * shock
        log_ret += influence_series[t]/1000
        prices[t] = prices[t-1] * np.exp(log_ret)
        
    prices_df = pd.DataFrame(prices, index=trnsX.index, columns=['Simulated Price'])
    
    influence_series = pd.DataFrame(influence_series, index=trnsX.index, columns=['Influence'])
    
    return pd.concat([prices_df, influence_series, trnsX], axis=1)
    
# Example usage:
# # Generate synthetic data
# X, cont = getTestData(n_features=20, n_informative=5, n_redundant=5, n_samples=1000, time_unit='H')

# # Simulate prices influenced by the first 2 informative features
# X_with_prices = prices_features_sim(X, n_feats=2, b0=100, drift=0.01, sigma=0.2, freq='H')

# print(X_with_prices.head())

if __name__ == '__main__':
    # Simulate data
    X, cont = f_ch08.getTestData(n_features=20, n_informative=5, n_redundant=5, n_samples=1000, time_unit='H')
    
    # Simulate prices influenced by the first 2 informative features
    X_with_prices = prices_features_sim(X, n_feats=2, b0=100, drift=0.01, sigma=0.2, freq='H')
    
    print(X_with_prices)