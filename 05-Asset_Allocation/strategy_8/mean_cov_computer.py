"""
This script computes the expected returns and covariance matrix of the assets
using the Black-Litterman model.
"""

import numpy as np
import pandas as pd

# Black-Litterman Imports
from scipy.linalg import inv

# Ledoit-Wolf Imports
from sklearn.covariance import LedoitWolf

# PyPortfolioOpt Imports
from pypfopt import risk_models, expected_returns, EfficientFrontier


def get_market_implied_returns(cov, market_weights, lambda_=2.5):
    """Compute market-implied returns."""
    # pi = delta * cov * market_weights
    pi = lambda_ * cov @ market_weights
    return pi


def main(
    returns,
    signals,
    market_caps,
    selected_stocks,
    lambda_=2.5,
):
    """
    Notes
    tau is inversely proportional to the relative weight given to the prior
    
    Args:
        returns (pd.DataFrame): returns dataframe, only selected stocks
        signals (pd.Series): signal for each selected stock at prediction time
        market_caps (pd.Series): market capitalizations for each selected stock
        selected_stocks (list): list of selected stocks
        pred_vol_scale (float): scaling factor for the predicted volatility
        tau (float): tau parameter for the Black-Litterman model
        lambda_ (float): lambda parameter for the Black-Litterman model
        
    Returns:
        pd.Series: posterior mean vector of expected returns for the selected stocks
        pd.DataFrame: posterior covariance matrix of the selected stocks
    """
    # Compute Prior Covariance Matrix
    l_wolf = LedoitWolf()
    shrunk_cov_matrix = l_wolf.fit(returns).covariance_
    cov = pd.DataFrame(
        shrunk_cov_matrix, index=selected_stocks, columns=selected_stocks
    )

    # Get market capitalizations vector (N x 1) column vector
    market_weights = market_caps / market_caps.sum()

    # Implied Excess Equilibrium Return Vector (N x 1 column vector)
    pi = get_market_implied_returns(cov, market_weights, lambda_)
    
    return pi, cov
