"""
This script implements the HRP algorithm to obtain the optimal portfolio weights.
"""

# SNIPPET 16.4 FULL IMPLEMENTATION OF THE HRP ALGORITHM
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd

import scipy.cluster.hierarchy as sch
from scipy.optimize import minimize


def get_quasi_diag(link):
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


# def get_rec_bipart(cov, mu, c_items, lambda_, benchmark, soft_risk, w_c):
#     # Base case: if cluster has one asset, assign the cluster weight to the asset
#     if len(c_items) == 1:
#         return pd.Series({c_items[0]: w_c})
#     else:
#         # Split the cluster into two sub-clusters
#         split = len(c_items) // 2
#         c_items_0 = c_items[:split]
#         c_items_1 = c_items[split:]

#         # Optimize weights within each sub-cluster
#         w1_star, c1_score = optimize_cluster(cov, mu, c_items_0, lambda_, benchmark, soft_risk)
#         w2_star, c2_score = optimize_cluster(cov, mu, c_items_1, lambda_, benchmark, soft_risk)
        
#         # c1_score, c2_score = -c1_score, -c2_score # Minimize negative objective function
#         c1_score, c2_score = max(c1_score, 0 + 1e-6), max(c2_score, 0 + 1e-6)
#         # print(f"Cluster 1 score: {c1_score}, Cluster 2 score: {c2_score}")

#         # Avoid division by zero
#         total_score = c1_score + c2_score + 1e-6

#         # Allocate cluster weights based on scores
#         if total_score < 1e-6:
#             alloc_c1 = alloc_c2 = w_c * 0.5  # Equal allocation
#         else:
#             alloc_c1 = w_c * (c1_score / total_score)
#             alloc_c2 = w_c * (c2_score / total_score)
#         # Scale the weights within each sub-cluster by the allocated cluster weight
#         w1 = pd.Series(w1_star, index=c_items_0) * alloc_c1
#         w2 = pd.Series(w2_star, index=c_items_1) * alloc_c2

#         # Recursively compute weights for sub-clusters
#         w_sub1 = get_rec_bipart(cov, mu, c_items_0, lambda_, benchmark, soft_risk, alloc_c1)
#         w_sub2 = get_rec_bipart(cov, mu, c_items_1, lambda_, benchmark, soft_risk, alloc_c2)

#         # Combine weights
#         w = pd.concat([w_sub1, w_sub2])
#         w = w / w.sum()


#         return w
    
    
def get_rec_bipart(cov, mu, sort_ix, lambda_, benchmark, soft_risk, long_only=True):
    w = pd.Series(1, index=sort_ix)
    c_items = [sort_ix]

    while len(c_items) > 0:
        c_items_new = []
        for cluster in c_items:
            if len(cluster) > 1:
                split = len(cluster) // 2
                c_items_0 = cluster[:split]
                c_items_1 = cluster[split:]
                # Append sub-clusters
                c_items_new.extend([c_items_0, c_items_1])

                # Get optimal weights per cluster
                w1_star, c1_score = optimize_cluster(
                    cov, mu, c_items_0, lambda_, benchmark, soft_risk
                )
                w2_star, c2_score = optimize_cluster(
                    cov, mu, c_items_1, lambda_, benchmark, soft_risk
                )
            
                c1_score = np.maximum(c1_score, 0) + 1e-6
                c2_score = np.maximum(c2_score, 0) + 1e-6
                
                # print(f"Cluster 1: {c_items_0}")
                # print(f"Total weight: {np.sum(w1_star)}")
                # print(f"Scores: {c1_score/(c1_score + c2_score)}")
                # print(f"Cluster 2: {c_items_1}")
                # print(f"Total weight: {np.sum(w2_star)}")
                # print(f"Scores: {c2_score/(c1_score + c2_score)}")

                # Allocate cluster weight according to score
                w[c_items_0] *= w1_star/(w1_star + w2_star + 1e-6) * (c1_score / (c1_score + c2_score))
                w[c_items_1] *= w2_star/(w1_star + w2_star + 1e-6) * (c2_score / (c1_score + c2_score))
                
                # w[c_items_0] *= (c1_score / (c1_score + c2_score))
                # w[c_items_1] *= (c2_score / (c1_score + c2_score))
                
                
                # print(f'Sum of current weights: {np.sum(w)}')

    return w




def optimize_cluster(cov, mu, c_items, lambda_, benchmark, soft_risk=0.01):
    benchmark_std = benchmark["sp_ret"].std()
    cov_cluster = cov.loc[c_items, c_items]
    mu_cluster = mu.loc[c_items]

    def objective(w):
        return mu_cluster @ w - lambda_ * 0.5 * w @ cov_cluster @ w

    def constraint(w):
        eq_cons = []
        inequality_cons = []

        # Sum of weights = 1
        eq_cons.append(np.sum(w) - 1)

        # Risk lower than benchmark
        inequality_cons.append(
            w @ cov_cluster @ w
            - (
                benchmark_std**2 + soft_risk
            )  # w @ posterior_cov @ w <= benchmark_std**2 + soft_risk
        )

        # w <= 0.10 for each stock
        # inequality_cons.extend(w - 0.10)

        return np.array(eq_cons), np.array(inequality_cons)

    weights_init = np.ones(len(c_items)) / len(c_items)

    # bounds 0 <= w <= 1 for all w
    bounds = [(0, 1)] * len(c_items)

    result = minimize(
        fun=lambda w: -objective(w),
        x0=weights_init,
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {"type": "eq", "fun": lambda w: constraint(w)[0]},
            {"type": "ineq", "fun": lambda w: -constraint(w)[1]},
        ],
        options={"disp": False, "maxiter": 1000, "ftol": 1e-6},
    )
    
    # Sharpe ratio
    

    return result.x, objective(result.x)


def correl_dist(corr):
    dist = ((1 - corr) / 2.0) ** 0.5  # distance matrix
    return dist


def main(
    weights,
    posterior_cov,
    posterior_mean,
    selected_stocks,
    benchmark_df,
    lambda_=3.07,
    soft_risk=0.01,
    long_only=True,
    link_method="single",
):
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
        dist = pd.DataFrame(np.nan_to_num(dist, nan=dist.mean()), index=selected_stocks, columns=selected_stocks)

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
        cov_reordered,
        posterior_mean,
        sort_ix,
        lambda_,
        benchmark_df,
        soft_risk,
        # w_c=1.0
    )

    # Assign the weights to the output DataFrame
    weights.loc[hrp_weights.index, "Weight"] = hrp_weights

    return weights
