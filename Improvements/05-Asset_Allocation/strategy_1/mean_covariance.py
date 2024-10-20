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

# Black-Litterman Functions


def black_litterman(cov, market_implied_returns, p, q, omega, tau=0.05):
    """Black-Litterman model."""
    pi = market_implied_returns
    tau_cov = tau * cov
    m_inverse = inv(tau_cov)
    omega_inv = inv(omega)

    # Compute posterior covariance
    posterior_cov = inv(m_inverse + p.T @ omega_inv @ p)

    # Compute posterior mean
    posterior_mean = posterior_cov @ (m_inverse @ pi + p.T @ omega_inv @ q)

    return posterior_mean, posterior_cov


def get_market_implied_returns(cov, market_weights, lambda_=2.5):
    """Compute market-implied returns."""

    # pi = delta * cov * market_weights
    pi = lambda_ * cov @ market_weights
    return pi


def compute_mean_covariance(
    returns,
    signals,
    market_caps,
    selected_stocks,
    pred_vol_scale=1.0,
    tau=1.0,
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

    # Scale the signals by the predicted volatility
    # UNCERTAINTY ON THE SIGNALS
    returns_vol = returns.std()
    q = signals * returns_vol * pred_vol_scale

    # Define P (Identity matrix for individual asset views)
    p = np.eye(len(signals))

    # Covariance on views (diagonal matrix)
    class_probs = signals.abs()
    omega_values = class_probs * (1 - class_probs)
    omega = np.diag(omega_values) * pred_vol_scale**2

    posterior_mean, posterior_cov = black_litterman(cov, pi, p, q, omega, tau=tau)
    posterior_mean = pd.Series(posterior_mean, index=selected_stocks)
    posterior_cov = pd.DataFrame(
        posterior_cov, index=selected_stocks, columns=selected_stocks
    )

    # Ensure posterior covariance matrix is positive definite
    # Add a small value to the diagonal if necessary
    min_eigenvalue = np.min(np.linalg.eigvals(posterior_cov))
    if min_eigenvalue < 0:
        # print("Adding small value to diagonal of posterior covariance matrix.")
        posterior_cov += np.eye(len(posterior_cov)) * (-min_eigenvalue + 1e-6)

    return posterior_mean, posterior_cov
