import numpy as np
import pandas as pd
from scipy.optimize import minimize

# PyPortfolioOpt Imports
from pypfopt import risk_models, expected_returns, EfficientFrontier


def main(
    weights,
    posterior_cov,
    posterior_mean,
    selected_stocks,
    benchmark_df,
    risk_aversion=1.0,
    long_only=True,
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

    # You don't have to provide expected returns in this case
    ef = EfficientFrontier(None, posterior_cov, weight_bounds=(0, None))
    ef.min_volatility()
    ef_weights = ef.clean_weights()
    ef_weights = pd.Series(ef_weights)
    weights.loc[ef_weights.index, "Weight"] = ef_weights
    return weights
