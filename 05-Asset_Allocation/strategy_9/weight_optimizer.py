import numpy as np
import pandas as pd
from scipy.optimize import minimize


def main(
    weights,
    posterior_cov,
    posterior_mean,
    selected_stocks,
    benchmark_df,
    lambda_=1.0,
    soft_risk=0.01
):
    """
    Maximizes mu^T w - (1/2) * risk_aversion * w^T Sigma w subject to w >= 0
    and 0<= sum(w) <= 1 and w^T Sigma w <= vol(benchmark)^2

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
    benchmark_std = benchmark_df["sp_ret"].std()

    def objective(w):
        return posterior_mean @ w - lambda_ * 0.5 * w @ posterior_cov @ w

    def constraint(w):
        eq_cons = []
        inequality_cons = []

        inequality_cons.append(np.sum(w) - 1.0) # sum(w) <= 1.0
        inequality_cons.append(0.90 - np.sum(w)) # sum(w) >= 0.90

        # Risk lower than benchmark
        inequality_cons.append(
            w @ posterior_cov @ w - (benchmark_std**2 + soft_risk**2) # w @ posterior_cov @ w <= benchmark_std**2 + soft_risk
        )
        
        # w <= 0.10 for each stock
        inequality_cons.extend(w - 0.10)
        

        return np.array(eq_cons), np.array(inequality_cons)

    weights_init = np.ones(len(selected_stocks)) / len(selected_stocks)

    # bounds 0 <= w <= 1 for all w
    bounds = [(0, 1)] * len(selected_stocks)

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

    weights_opt = result.x

    weights.loc[selected_stocks, "Weight"] = weights_opt


    return weights
