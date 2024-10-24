"""
This script implements the HRP algorithm to obtain the optimal portfolio weights.
"""

# SNIPPET 16.4 FULL IMPLEMENTATION OF THE HRP ALGORITHM
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd

import scipy.cluster.hierarchy as sch


def get_ivp(cov):
    """
    Compute the inverse-variance portfolio (IVP) based on the covariance matrix.

    The inverse-variance portfolio is a method of portfolio allocation that
    assigns weights to assets inversely proportional to their variances,
    leading to a portfolio that minimizes risk.

    Parameters:
    cov (numpy.ndarray): A square covariance matrix of asset returns,
                         where the diagonal elements represent the
                         variances of the individual assets.

    Returns:
    numpy.ndarray: An array representing the weights of the assets in the
                   inverse-variance portfolio. The weights sum to 1.
    """
    ivp = 1.0 / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def get_cluster_var(cov, c_items):
    """
    Compute the variance of a specified cluster based on its covariance matrix.

    This function calculates the variance of a cluster of assets by
    extracting the relevant submatrix from the overall covariance matrix
    and applying the inverse-variance portfolio method.

    Parameters:
    cov (pandas.DataFrame or numpy.ndarray): The covariance matrix of asset returns,
                                             where rows and columns represent assets.
    c_items (list or array-like): A list or array of indices or names that specify
                                   the assets belonging to the cluster.

    Returns:
    float: The variance of the specified cluster of assets.
    """
    cov_ = cov.loc[c_items, c_items]
    w_ = get_ivp(cov_).reshape(-1, 1)
    c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return c_var


def get_cluster_mean(mu, c_items):
    """
    Compute the expected return of a specified cluster based on asset returns.

    This function calculates the expected return for a cluster of assets by
    averaging the returns of the assets in the cluster using equal weights.

    Parameters:
    mu (pandas.Series or numpy.ndarray): A series or array representing the expected returns
                                         of individual assets.
    c_items (list or array-like): A list or array of indices or names that specify
                                   the assets belonging to the cluster.

    Returns:
    float: The expected return of the specified cluster of assets.
    """
    mu_ = mu.loc[c_items]
    w_ = np.ones(len(mu_)) / len(mu_)  # Equal weights within the cluster
    c_mean = np.dot(w_, mu_)
    return c_mean


def get_quasi_diag(link):
    """
    Sort clustered items based on hierarchical clustering linkage.

    This function processes the linkage matrix obtained from hierarchical
    clustering and returns a quasi-diagonal list of indices representing
    the order of clustered items sorted by distance. It follows the
    clustering structure to reflect the hierarchy of clusters.

    Parameters:
    link (numpy.ndarray): A linkage matrix from hierarchical clustering,
                          where each row represents a merge step, and the
                          last column contains the number of original items.

    Returns:
    list: A list of indices representing the order of clustered items sorted
          by distance, reflecting their hierarchical structure.
    """
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]  # number of original items

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # make space
        df0 = sort_ix[sort_ix >= num_items]  # find clusters
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()  # concatenate and sort
        sort_ix.index = range(sort_ix.shape[0])  # re-index

    return sort_ix.tolist()


def get_rec_bipart(cov, sort_ix, mu, long_only=True):
    """
    Compute asset allocation using the Hierarchical Risk Parity (HRP) approach.

    This function allocates weights to assets based on their expected returns
    and variances using a recursive bipartitioning strategy. It calculates
    cluster variances and expected returns, then allocates weights
    proportionally to the Sharpe ratios of the clusters.

    Parameters:
    cov (pd.DataFrame): A covariance matrix of asset returns.
    sort_ix (list): A list of sorted indices representing the order of
                    assets/clusters.
    mu (pd.Series): A series of expected returns for the assets.
    risk_averse (float, optional): A risk aversion parameter affecting
                                    weight allocation. Default is 2.

    Returns:
    pd.Series: A series representing the optimal asset weights based on HRP
                allocation.
    """
    w = pd.Series(1, index=sort_ix)
    c_items = [sort_ix]  # initialize all items in one cluster
    while len(c_items) > 0:
        c_items = [
            i[j:k]
            for i in c_items
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]  # bi-section
        for i in range(0, len(c_items), 2):  # parse in pairs
            c_items_0 = c_items[i]  # cluster 1
            c_items_1 = c_items[i + 1]  # cluster 2
            # Calculate cluster variances and expected returns
            c_var_0 = get_cluster_var(cov, c_items_0)
            c_var_1 = get_cluster_var(cov, c_items_1)
            c_mean_0 = get_cluster_mean(mu, c_items_0)
            c_mean_1 = get_cluster_mean(mu, c_items_1)
            # Calculate Sharpe ratios for clusters
            sr0 = c_mean_0 / (np.sqrt(c_var_0)) if c_var_0 > 0 else 0
            sr1 = c_mean_1 / (np.sqrt(c_var_1)) if c_var_1 > 0 else 0

            # Long Only
            if long_only:
                sr0 = max(sr0, 0)
                sr1 = max(sr1, 0)

            denom = abs(sr0) + abs(sr1) + 1e-6
            # Allocate weights proportional to Sharpe ratios
            alpha = sr1 / denom if denom != 0 else 0.5
            w[c_items_0] *= sr0 / denom
            w[c_items_1] *= sr1 / denom
    return w


def correl_dist(corr):
    """
    Compute a distance matrix based on the correlation matrix.

    This function converts a correlation matrix into a distance matrix
    suitable for clustering. The resulting distance values are in the
    range [0, 1], where 0 indicates perfect correlation and 1 indicates
    no correlation.

    Parameters:
    corr (pd.DataFrame or np.ndarray): A correlation matrix with values in
                                        the range [-1, 1].

    Returns:
    pd.DataFrame or np.ndarray: A distance matrix derived from the
                                 correlation matrix, where distances are
                                 in the range [0, 1].
    """
    dist = ((1 - corr) / 2.0) ** 0.5  # distance matrix
    return dist


def plot_corr_matrix(path, corr, labels=None):
    """
    Plot and save a heatmap of the correlation matrix.

    This function generates a heatmap visualization of the provided
    correlation matrix and saves it to the specified file path. If
    labels are provided, they will be used for the axes; otherwise,
    default labels will be used.

    Parameters:
    path (str): The file path where the heatmap image will be saved.
    corr (np.ndarray or pd.DataFrame): The correlation matrix to visualize.
    labels (list, optional): A list of labels for the x and y axes. If None,
                             no labels will be applied.

    Returns:
    None: This function saves the heatmap to the specified file path and
          does not return any value.
    """
    if labels is None:
        labels = []
    mpl.pcolor(corr)
    mpl.colorbar()
    mpl.yticks(np.arange(0.5, corr.shape[0] + 0.5), labels)
    mpl.xticks(np.arange(0.5, corr.shape[0] + 0.5), labels)
    mpl.saveﬁg(path)
    mpl.clf()
    mpl.close()  # reset pylab


def main(
    weights,
    posterior_cov,
    posterior_mean,
    selected_stocks,
    benchmark_df,
    long_only=True,
    link_method="single",
):
    """
    _summary_

    Args:
        weights (pd.DataFrame): DataFrame containing the weights of the asset,
                                All possible stocks (not just selected ones)
        posterior_cov (pd.DataFrame): Posterior covariance matrix of the selected stocks
        posterior_mean (pd.Series): Posterior mean of the selected stocks
        selected_stocks (list): List of selected stocks

    Returns:
        pd.DataFrame: DataFrame containing the weights of all the assets
        (not selected stocks will have 0 weight)
    """
    # Reconstruct the correlation matrix from the posterior covariance matrix
    std_devs = np.sqrt(np.diag(posterior_cov))
    # Avoid division by zero
    std_devs[std_devs == 0] = 1e-6
    corr = posterior_cov / np.outer(std_devs, std_devs)
    corr = np.clip(corr, -1, 1)
    corr.values[range(corr.shape[0]), range(corr.shape[1])] = 1.0
    corr = pd.DataFrame(corr, index=selected_stocks, columns=selected_stocks)

    # Now compute the distance matrix
    dist = correl_dist(corr)

    # Check for NaNs in the distance matrix
    if np.isnan(dist.to_numpy()).any():
        # print("NaNs detected in distance matrix.")
        dist = np.nan_to_num(dist, nan=1e6)

    dist = pd.DataFrame(dist, index=selected_stocks, columns=selected_stocks)

    # Plot correlation matrix
    # plotCorrMatrix('HRP_BL_corr0.png', corr, labels=corr.columns)

    # Cluster using hierarchical clustering
    link = sch.linkage(dist, method=link_method)
    sort_ix = get_quasi_diag(link)
    sort_ix = corr.index[sort_ix].tolist()

    # Reorder covariance matrix for clustered stocks
    cov_reordered = posterior_cov.loc[sort_ix, sort_ix]
    # corr_reordered = corr.loc[sort_ix, sort_ix]

    # Plot reordered correlation matrix
    # plotCorrMatrix('HRP_BL_corr1.png', corr_reordered, labels=corr_reordered.columns)

    # Apply HRP with Black-Litterman posterior covariance
    hrp_weights = get_rec_bipart(
        cov_reordered, sort_ix, posterior_mean, long_only=long_only
    )


    # Assign the weights to the output DataFrame
    weights.loc[hrp_weights.index, "Weight"] = hrp_weights

    return weights
