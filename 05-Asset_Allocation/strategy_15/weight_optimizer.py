import numpy as np
import pandas as pd
from scipy.optimize import minimize

from hmmlearn.hmm import GaussianHMM

import scipy.cluster.hierarchy as sch

import matplotlib.pyplot as mpl
from scipy.cluster.hierarchy import linkage, fcluster


def generate_scenarios(
    posterior_mean, posterior_cov, num_scenarios=10, uncertainty_level=0.05
):
    """
    Generate different perturbations for mean and covariance based on uncertainty level.
    Args:
        posterior_mean (pd.Series): Mean returns.
        posterior_cov (pd.DataFrame): Covariance matrix.
        num_scenarios (int): Number of scenarios to generate.
        uncertainty_level (float): The degree of uncertainty applied to the perturbations.
    Returns:
        scenarios_mu (list): List of perturbed mean return vectors.
        scenarios_cov (list): List of perturbed covariance matrices.
    """
    np.random.seed(42)  # for reproducibility
    scenarios_mu = []
    scenarios_cov = []

    for _ in range(num_scenarios):
        # Apply random perturbation to mean
        perturbed_mean = posterior_mean + uncertainty_level * np.random.randn(
            len(posterior_mean)
        )

        # Apply random perturbation to covariance (keeping symmetry)
        perturbation = uncertainty_level * np.random.randn(*posterior_cov.shape)
        perturbed_cov = posterior_cov + perturbation @ perturbation.T

        # Ensure positive-definiteness of covariance
        perturbed_cov = (perturbed_cov + perturbed_cov.T) / 2
        min_eigenvalue = np.min(np.linalg.eigvals(perturbed_cov))
        if min_eigenvalue < 0:
            perturbed_cov += np.eye(posterior_cov.shape[0]) * (-min_eigenvalue + 1e-6)

        scenarios_mu.append(perturbed_mean)
        scenarios_cov.append(perturbed_cov)

    return scenarios_mu, scenarios_cov


def predict_volatility_state(benchmark_df):

    # Preprocess the volatility data to remove NaNs and infs
    returns_or_volatility = benchmark_df["sp_ret"].ewm(span=12).std()
    returns_or_volatility = returns_or_volatility.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    # Ensure that there are no NaN values left
    returns_or_volatility = returns_or_volatility.fillna(method="ffill").fillna(
        method="bfill"
    )

    # Initialize and train HMM with adjusted number of components
    hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000)

    # Fit the HMM and get the current hidden state
    train_data = returns_or_volatility.values.reshape(-1, 1)
    hmm.fit(train_data)
    current_state = hmm.predict(train_data)[-1]  # Get the latest state

    return current_state


# Compute the inverse-variance portfolio
def getIVP(cov, **kargs):
    ivp = 1.0 / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


# Compute variance per cluster
def getClusterVar(cov, cItems):
    cov_ = cov.loc[cItems, cItems]  # matrix slice
    w = getIVP(cov_).reshape(-1, 1)
    cVar = np.dot(np.dot(w.T, cov_), w)[0, 0]
    return cVar

def getClusterMeanVar():
    pass


# Sort clustered items by distance
def getQuasiDiag(link):
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index
        df0 = df0.values - numItems
        sortIx[i] = link[df0, 0]  # item 1
        df0 = pd.Series(link[df0, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0])  # item 2 (replacing append with pd.concat)
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = range(sortIx.shape[0])  # re-index
    return sortIx.tolist()


# Compute HRP allocation
def getRecBipart(cov, sortIx):
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [
            i[j:k]
            for i in cItems
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]  # bi-section
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= 1 - alpha  # weight 2
    return w


# A distance matrix based on correlation
def correlDist(corr):
    dist = ((1 - corr) / 2.0) ** 0.5  # distance matrix
    return dist


def robust_optimizer(
    weights,
    posterior_cov,
    posterior_mean,
    selected_stocks,
    returns,
    benchmark_df,
    lambda_=1.0,
    soft_risk=0.01,
    num_scenarios=10,
    uncertainty_level=0.05,
    total_allocation=1.0,
    vol_state=1,
):
    """
    _summary_

    Args:
        weights (pd.DataFrame): DataFrame containing the weights of the asset,
                                All possible stocks (not just selected ones)
        posterior_cov (pd.DataFrame): Posterior covariance matrix of the selected stocks
        posterior_mean (pd.Series): Posterior mean of the selected stocks
        selected_stocks (list): List of selected stocks
        returns (pd.DataFrame): DataFrame containing the returns of the selected stocks
        benchmark_df(pd.DataFrame): DataFrame containing the benchmark returns as 'sp_ret'

    Returns:
        pd.DataFrame: DataFrame containing the weights of all the assets
        (not selected stocks will have 0 weight)
    """
    # check = False
    # if not check:
    #     print("Check for weight_optimizer")
    #     print(f"lambda is {lambda_}")
    #     print(f"soft risk is {soft_risk}")
    #     check = True

    n = len(selected_stocks)

    # Generate multiple scenarios for posterior_mean and posterior_cov
    scenarios_mu, scenarios_cov = generate_scenarios(
        posterior_mean, posterior_cov, num_scenarios, uncertainty_level
    )

    # Objective function: maximize the worst-case scenario's performance
    def objective(w):
        worst_case_return = np.inf
        for mu_scenario, cov_scenario in zip(scenarios_mu, scenarios_cov):
            # Compute the return and risk for each scenario
            scenario_return = mu_scenario @ w
            scenario_risk = w @ cov_scenario @ w

            # Combine return and risk in a penalized objective
            penalized_value = (
                -scenario_return + lambda_**vol_state * 0.5 * scenario_risk
            )
            worst_case_return = min(worst_case_return, penalized_value)

        return worst_case_return

    # Define the constraint functions
    def constraint_eq(w):
        # Sum of weights should equal 1
        return (
            np.sum(w) - total_allocation + 0.1 * vol_state
        )  # np.sum(w)= (1 - vol_state)

    # Bounds for each weight: 0 <= w <= 0.10
    bounds = [(0, 0.10)] * n

    # Initial guess for weights (equally distributed)
    w0 = np.ones(n) / n

    # Solve the optimization problem using scipy.optimize.minimize
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {"type": "eq", "fun": constraint_eq},
        ],
        options={"disp": False, "maxiter": 1000, "ftol": 1e-5},  # Adjust ftol
    )

    if not result.success:
        raise ValueError("Optimization failed:", result.message)

    weights_opt = result.x

    # Update the weights dataframe
    weights.loc[selected_stocks, "Weight"] = weights_opt
    
    # mean and variance
    mean = posterior_mean @ weights_opt
    variance = weights_opt @ posterior_cov @ weights_opt

    return mean, variance


def main(
    weights,
    posterior_cov,
    posterior_mean,
    selected_stocks,
    returns,
    benchmark_df,
    lambda_=1.0,
    soft_risk=0.01,
    num_scenarios=10,
    uncertainty_level=0.05,
    total_allocation=1.0,
    n_clusters=4,
):
    """
    _summary_

    Args:
        weights (pd.DataFrame): DataFrame containing the weights of the asset,
                                All possible stocks (not just selected ones)
        posterior_cov (pd.DataFrame): Posterior covariance matrix of the selected stocks
        posterior_mean (pd.Series): Posterior mean of the selected stocks
        selected_stocks (list): List of selected stocks
        returns (pd.DataFrame): DataFrame containing the returns of the selected stocks
        benchmark_df(pd.DataFrame): DataFrame containing the benchmark returns as 'sp_ret'

    Returns:
        pd.DataFrame: DataFrame containing the weights of all the assets
        (not selected stocks will have 0 weight)
    """
    std_devs = np.sqrt(np.diag(posterior_cov))
    std_devs[std_devs == 0] = 1e-6
    corr = posterior_cov / np.outer(std_devs, std_devs)
    corr = np.clip(corr, -1, 1)
    corr.values[range(corr.shape[0]), range(corr.shape[1])] = 1.0
    corr = pd.DataFrame(corr, index=selected_stocks, columns=selected_stocks)
    
    # Now compute the distance matrix
    dist = correlDist(corr)

    # Check for NaNs in the distance matrix
    if np.isnan(dist.to_numpy()).any():
        dist = np.nan_to_num(dist, nan=1e6)

    dist = pd.DataFrame(dist, index=selected_stocks, columns=selected_stocks)

    # Cluster using hierarchical clustering
    link = sch.linkage(dist, method="single")
    sort_ix = getQuasiDiag(link)
    sort_ix = corr.index[sort_ix].tolist()

    # Reorder covariance matrix for clustered stocks
    cov_reordered = posterior_cov.loc[sort_ix, sort_ix]