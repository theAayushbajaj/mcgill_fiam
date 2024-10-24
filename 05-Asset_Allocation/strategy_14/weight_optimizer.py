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


# Heatmap of the correlation matrix
def plotCorrMatrix(path, corr, labels=None):
    if labels is None:
        labels = []
    mpl.pcolor(corr)
    mpl.colorbar()
    mpl.yticks(np.arange(0.5, corr.shape[0] + 0.5), labels)
    mpl.xticks(np.arange(0.5, corr.shape[0] + 0.5), labels)
    mpl.savefig(path)
    mpl.clf()
    mpl.close()  # reset pylab
    return


# Generate correlated data
def generateData(nObs, size0, size1, sigma1):
    # Time series of correlated variables
    # 1) generating some uncorrelated data
    np.random.seed(seed=12345)
    random.seed(12345)
    x = np.random.normal(0, 1, size=(nObs, size0))  # each row is a variable
    # 2) creating correlation between the variables
    cols = [random.randint(0, size0 - 1) for i in range(size1)]
    y = x[:, cols] + np.random.normal(0, sigma1, size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    x = pd.DataFrame(x, columns=range(1, x.shape[1] + 1))
    return x, cols


def main_HRP():
    # 1) Generate correlated data
    nObs, size0, size1, sigma1 = 10000, 5, 5, 0.25
    x, cols = generateData(nObs, size0, size1, sigma1)
    print([(j + 1, size0 + i) for i, j in enumerate(cols)])
    cov, corr = x.cov(), x.corr()

    # 2) compute and plot correlation matrix
    plotCorrMatrix("HRP3_corr0.png", corr, labels=corr.columns)

    # 3) cluster
    dist = correlDist(corr)
    link = sch.linkage(dist, "single")
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()  # recover labels
    df0 = corr.loc[sortIx, sortIx]  # reorder
    plotCorrMatrix("HRP3_corr1.png", df0, labels=df0.columns)

    # 4) Capital allocation
    hrp = getRecBipart(cov, sortIx)
    print(hrp)
    return


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
    vol_state=1
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
    
    sharpe_ratio = -result.fun

    return weights, sharpe_ratio


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
    n_clusters=4
):
    # Step 0 : Get the volatility state
    benchmark_std = benchmark_df["sp_ret"].std()
    # Predict Volatility State
    try:
        vol_state = predict_volatility_state(benchmark_df)
    except:
        vol_state = 1

    # benchmark_std as the min of the
    n = len(selected_stocks)
    

    # Step 1: Hierarchical clustering using the HRP framework
    corr = returns.corr()
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)  # Use your HRP-style clustering
    sortIx = corr.index[sortIx].tolist()  # Reorder stocks based on clusters

    # Step 2: Split into clusters using HRP method
    clusters = {}
    for i in range(1, n_clusters+1):
        clusters[i] = sortIx[(i-1)*len(sortIx)//n_clusters : i*len(sortIx)//n_clusters]
        # debugging
        # print(f"Cluster {i}: {clusters[i]}")

    # Step 3: Apply robust optimization within each cluster
    cluster_weights = {}
    cluster_variances = {}
    cluster_score = {}
    for i in clusters:
        cluster_stocks = clusters[i]
        cluster_returns = returns[cluster_stocks]
        cluster_posterior_mean = posterior_mean[cluster_stocks]
        cluster_posterior_cov = posterior_cov.loc[cluster_stocks, cluster_stocks]

        cluster_weights_df = pd.DataFrame(index=cluster_stocks)
        cluster_weights_df['Weight'] = 0.0

        # Apply robust optimization to get weights within the cluster
        cluster_weights_opt, score = robust_optimizer(
            cluster_weights_df,
            cluster_posterior_cov,
            cluster_posterior_mean,
            cluster_stocks,
            cluster_returns,
            benchmark_df,
            lambda_=lambda_,
            soft_risk=soft_risk,
            num_scenarios=num_scenarios,
            uncertainty_level=uncertainty_level,
            total_allocation=1.0,
            vol_state=vol_state
        )
        cluster_weights[i] = cluster_weights_opt['Weight']
        cluster_score[i] = score
        # debugging
        # print(f"Cluster {i} weights: {cluster_weights[i]}")
        # print(f"Total allocation in cluster {i}: {cluster_weights[i].sum()}")

        # Calculate cluster variance
        w = cluster_weights[i].values
        cov = cluster_posterior_cov.values
        cluster_variance = np.dot(w.T, np.dot(cov, w))
        cluster_variances[i] = cluster_variance

    # Step 4: Allocate between clusters based on HRP logic (inverse variance)
    inv_cluster_vars = {i: 1.0 / cluster_variances[i] for i in cluster_variances}
    total_inv_var = sum(inv_cluster_vars.values())
    cluster_allocations = {i: inv_cluster_vars[i] / total_inv_var for i in inv_cluster_vars}
    
    # score allocation
    # total_score = sum(cluster_score.values())
    # cluster_allocations = {i: cluster_score[i] / total_score for i in cluster_score}
    
    # equal allocation
    # cluster_allocations = {i: 1/len(cluster_variances) for i in cluster_variances}
    
    # debugging
    print(f"Cluster allocations: {cluster_allocations}")
    # print(f"Total allocation across clusters: {sum(cluster_allocations.values())}")

    # Step 5: Combine cluster weights to form final portfolio
    final_weights_series = pd.Series(0.0, index=selected_stocks)
    for i in clusters:
        cluster_stocks = clusters[i]
        w_cluster = cluster_weights[i] * cluster_allocations[i]
        final_weights_series[cluster_stocks] = w_cluster
        
    # debuging
    # total sum
    # print(f"Total allocation in final portfolio: {final_weights_series.sum()}")
    # # how many stocks > 0
    # print(f"Number of stocks with non-zero allocation: {final_weights_series[final_weights_series > 0].shape[0]}")

    # Update weights DataFrame
    weights.loc[selected_stocks, "Weight"] = final_weights_series
    # debugging
    # print(f"Final portfolio weights: {final_weights_series}")
    # print(f"Total allocation in final portfolio: {final_weights_series.sum()}")
    
    # if final_weights_series.sum > 1.0001, raise an error
    if final_weights_series.sum() > 1.0001:
        raise ValueError("Portfolio allocation exceeds 100%")

    return weights
