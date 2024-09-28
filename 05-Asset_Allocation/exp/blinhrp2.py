# %%
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
import scipy.cluster.hierarchy as sch
from sklearn.covariance import LedoitWolf
import cvxpy as cp  # Import cvxpy for optimization

# %%

# HRP Functions
# ----------------------------------------------


def getIVP(cov, **kargs):
    # Compute the inverse-variance portfolio
    ivp = 1.0 / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def getClusterRiskReturn(cov, expected_returns, cluster_items):
    # Compute variance and expected return per cluster
    cov_cluster = cov.loc[cluster_items, cluster_items]  # Covariance matrix slice
    returns_cluster = expected_returns.loc[
        cluster_items
    ]  # Expected returns of cluster assets
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


def optimize_cluster_allocation(cov, mu, total_abs_weight):
    """
    Optimize the allocation within a cluster while respecting the constraints:
    - Sum of absolute weights equals total_abs_weight
    - Weights between -1 and +1
    """
    assets = cov.index.tolist()
    N = len(assets)
    
    # Variables
    w = cp.Variable(N)
    u = cp.Variable(N)
    
    # Objective Function: Minimize variance minus expected return
    objective = cp.Minimize(cp.quad_form(w, cov.values) - mu.values.T @ w)
    
    # Constraints
    constraints = [
        cp.sum(u) == total_abs_weight,
        w - u <= 0,
        -w - u <= 0,
        w >= -1,
        w <= 1,
        u >= 0
    ]
    
    # Problem Definition
    prob = cp.Problem(objective, constraints)
    
    # Solve the Problem
    prob.solve(solver=cp.OSQP)
    
    # Check if the problem was solved successfully
    if prob.status != cp.OPTIMAL:
        raise ValueError(f"Optimization failed with status {prob.status}")
    
    # Return the optimal weights as a Pandas Series
    optimal_weights = pd.Series(w.value, index=assets)
    return optimal_weights


def getRecBipart(cov, expected_returns, sortIx):
    # Compute HRP allocation with expected returns and constraints
    # Initialize weights to zero
    w = pd.Series(0.0, index=sortIx)
    # Total absolute weight to allocate
    total_abs_weight = 1.0
    # Start with the full list of items
    cItems = [sortIx]
    while len(cItems) > 0:
        # Bi-section
        cItems = [
            cluster[j:k]
            for cluster in cItems
            for j, k in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
            if len(cluster) > 0
        ]
        # Process clusters in pairs
        for i in range(0, len(cItems), 2):
            if i + 1 >= len(cItems):
                continue  # Skip if there is an odd number of clusters
            cItems0 = cItems[i]
            cItems1 = cItems[i + 1]

            # Combined assets
            cluster_assets = cItems0 + cItems1

            # Covariance and returns for the combined cluster
            cov_cluster = cov.loc[cluster_assets, cluster_assets]
            mu_cluster = expected_returns.loc[cluster_assets]

            # Allocate weights within the combined cluster
            cluster_weights = optimize_cluster_allocation(
                cov_cluster, mu_cluster, total_abs_weight
            )

            # Assign weights to the assets
            w.update(cluster_weights)

            # Update the total absolute weight (should remain 1)
            total_abs_weight = w.abs().sum()
    return w


def correlDist(corr):
    # A distance matrix based on correlation, where 0 <= d[i,j] <= 1
    dist = ((1 - corr) / 2.0) ** 0.5  # distance matrix
    return dist


def plotCorrMatrix(path, corr, labels=None):
    # Heatmap of the correlation matrix
    if labels is None:
        labels = []
    mpl.pcolor(corr)
    mpl.colorbar()
    mpl.yticks(np.arange(0.5, corr.shape[0] + 0.5), labels)
    mpl.xticks(np.arange(0.5, corr.shape[0] + 0.5), labels)
    mpl.savefig(path)
    mpl.clf()
    mpl.close()  # reset pylab


# %%

# ----------------------------------------------


def main():
    # Load your data
    # %%
    path_prices = "../objects/prices.pkl"
    path_market_caps = "../objects/market_caps.pkl"
    path_signals = "../objects/signals.pkl"

    prices = pd.read_pickle(path_prices)
    market_caps_df = pd.read_pickle(path_market_caps)
    signals = pd.read_pickle(path_signals)
    # %%

    # Initialize the weights DataFrame
    weights = pd.DataFrame(index=prices.columns)
    weights["Weight"] = 0.0
    # %%

    # Parameters
    Start_Date = 0
    End_Date = 100  # Index for which we will predict
    prices = prices[Start_Date : End_Date + 1]
    # %%

    # Stock Selection
    signals_end = signals.iloc[End_Date]
    signals_end = signals_end.where(prices.count() > 60, 0)
    abs_signals = abs(signals_end).sort_values(ascending=False)
    selected_stocks = abs_signals.index[:100].tolist()
    signals_end = signals_end[selected_stocks]
    prices = prices[selected_stocks]
    market_caps_df = market_caps_df[selected_stocks]
    # %%

    # Covariance Matrix using Ledoit-Wolf shrinkage
    returns = prices.pct_change().dropna()
    lw = LedoitWolf()
    shrunk_cov_matrix = lw.fit(returns).covariance_
    cov = pd.DataFrame(
        shrunk_cov_matrix, index=selected_stocks, columns=selected_stocks
    )
    # %%
    volatility = returns.std()
    # %%
    # Compute Expected Returns from Classifier Signals
    scaling_factor = 0.50  # Adjust based on your preference
    Expected_Returns = scaling_factor * signals_end * volatility
    Expected_Returns = Expected_Returns[selected_stocks]  # Ensure alignment
    # %%

    # Hierarchical Clustering and Sorting
    corr = returns.corr()
    dist = correlDist(corr)
    link = sch.linkage(dist, "single")
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()

    # Apply Modified HRP Algorithm with Expected Returns and Constraints
    hrp_weights = getRecBipart(cov, Expected_Returns, sortIx)
    print(f"Market exposure (sum of weights): {hrp_weights.sum()}")
    print(f"Sum of absolute values of weights: {hrp_weights.abs().sum()}")
    print(f"Minimum weight: {hrp_weights.min()}, Maximum weight: {hrp_weights.max()}")
    # %%
    # Ensure weights sum to desired total (adjust if necessary)
    hrp_weights /= hrp_weights.abs().sum()  # Normalize weights to sum of absolute values equals 1

    # Clip weights to be within bounds
    hrp_weights = hrp_weights.clip(lower=-1, upper=1)

    # Assign the weights to the output DataFrame
    weights.loc[hrp_weights.index, "Weight"] = hrp_weights
    print(weights)
    # %%


if __name__ == "__main__":
    main()

# %%
