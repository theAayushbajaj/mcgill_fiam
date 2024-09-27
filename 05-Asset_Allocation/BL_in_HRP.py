#%%
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
import scipy.cluster.hierarchy as sch
from sklearn.covariance import LedoitWolf
#%%

# HRP Functions
#----------------------------------------------

def getIVP(cov, **kargs):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def getClusterRiskReturn(cov, expected_returns, cluster_items):
    # Compute variance and expected return per cluster
    cov_cluster = cov.loc[cluster_items, cluster_items]  # Covariance matrix slice
    returns_cluster = expected_returns.loc[cluster_items]  # Expected returns of cluster assets
    weights = getIVP(cov_cluster).reshape(-1, 1)  # Inverse variance weights

    cluster_variance = np.dot(np.dot(weights.T, cov_cluster), weights)[0, 0]
    cluster_return = np.dot(weights.T, returns_cluster.values.reshape(-1, 1))[0, 0]
    return cluster_variance, cluster_return

def getQuasiDiag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0]).sort_index()  # concatenate and sort
        sortIx.index = range(sortIx.shape[0])  # re-index
    return sortIx.tolist()

def getRecBipart(cov, expected_returns, sortIx):
    # Compute HRP allocation with expected returns
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # Initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i)//2), (len(i)//2, len(i))) if len(i) > 1]  # Bi-section
        for i in range(0, len(cItems), 2):  # Parse in pairs
            cItems0 = cItems[i]  # Cluster 1
            cItems1 = cItems[i + 1]  # Cluster 2

            # Compute cluster variance and expected return
            cVar0, cRet0 = getClusterRiskReturn(cov, expected_returns, cItems0)
            cVar1, cRet1 = getClusterRiskReturn(cov, expected_returns, cItems1)

            alpha = ((cRet0 + cVar1) / (cVar0 + cVar1 + cRet0 + cRet1))#  - (cRet1 / (cRet0 + cRet1))
            alpha = 2 * alpha - 1

            # Update weights
            w[cItems0] *= alpha
            w[cItems1] *= 1 - alpha if alpha >= 0 else -1 - alpha  # weight 2
    return w

def correlDist(corr):
    # A distance matrix based on correlation, where 0 <= d[i,j] <= 1
    dist = ((1 - corr) / 2.) ** .5  # distance matrix
    return dist

def plotCorrMatrix(path, corr, labels=None):
    # Heatmap of the correlation matrix
    if labels is None:
        labels = []
    mpl.pcolor(corr)
    mpl.colorbar()
    mpl.yticks(np.arange(.5, corr.shape[0] + .5), labels)
    mpl.xticks(np.arange(.5, corr.shape[0] + .5), labels)
    mpl.savefig(path)
    mpl.clf()
    mpl.close()  # reset pylab
#%%

#----------------------------------------------

def main():
    # Load your data
    #%%
    path_prices = '../objects/prices.pkl'
    path_market_caps = '../objects/market_caps.pkl'
    path_signals = '../objects/signals.pkl'

    prices = pd.read_pickle(path_prices)
    market_caps_df = pd.read_pickle(path_market_caps)
    signals = pd.read_pickle(path_signals)
    #%%

    # Initialize the weights DataFrame
    weights = pd.DataFrame(index=prices.columns)
    weights['Weight'] = 0.0
    #%%

    # Parameters
    Start_Date = 0
    End_Date = 100  # Index for which we will predict
    prices = prices[Start_Date:End_Date + 1]
    #%%

    # Stock Selection
    signals_end = signals.iloc[End_Date]
    signals_end = signals_end.where(prices.count() > 60, 0)
    abs_signals = abs(signals_end).sort_values(ascending=False)
    selected_stocks = abs_signals.index[:100].tolist()
    signals_end = signals_end[selected_stocks]
    prices = prices[selected_stocks]
    market_caps_df = market_caps_df[selected_stocks]
    #%%

    # Covariance Matrix using Ledoit-Wolf shrinkage
    returns = prices.pct_change().dropna()
    lw = LedoitWolf()
    shrunk_cov_matrix = lw.fit(returns).covariance_
    cov = pd.DataFrame(shrunk_cov_matrix, index=selected_stocks, columns=selected_stocks)
    #%%
    volatility = returns.std()
    #%%
    # Compute Expected Returns from Classifier Signals
    # scaling_factor = 0.05  # Adjust based on your preference
    Expected_Returns = signals_end * volatility
    Expected_Returns = Expected_Returns[selected_stocks]  # Ensure alignment
    #%%

    # Hierarchical Clustering and Sorting
    corr = returns.corr()
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()

    # Apply Modified HRP Algorithm with Expected Returns
    hrp_weights = getRecBipart(cov, Expected_Returns, sortIx)
    print(f'Market exposure: {hrp_weights.sum()}')
    print(f'Sum of absolute values of weights: {hrp_weights.abs().sum()}')
    #%%
    hrp_weights /= hrp_weights.sum()  # Normalize weights to sum to 1

    # Assign the weights to the output DataFrame
    weights.loc[hrp_weights.index, 'Weight'] = hrp_weights
    print(weights)
    #%%

if __name__ == '__main__':
    main()

# %%
