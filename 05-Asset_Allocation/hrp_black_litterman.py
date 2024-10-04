"""
This script combines Black-Litterman along with the HRP algorithm.
"""

import numpy as np

# Black-Litterman Imports
from scipy.linalg import inv

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


# Main function integrating Black-Litterman with HRP


def black_litterman_pipeline(
    cov, signals, market_caps, returns_vol, pred_vol_scale=1.0, tau=1.0, lambda_=2.5
):
    """
    Notes
    tau is inversely proportional to the relative weight given to the prior
    """
    # Get market capitalizations vector (N x 1) column vector
    market_weights = market_caps / market_caps.sum()

    # Implied Excess Equilibrium Return Vector (N x 1 column vector)
    pi = get_market_implied_returns(cov, market_weights, lambda_)

    # Scale the signals by the predicted volatility
    # UNCERTAINTY ON THE SIGNALS
    q = signals * returns_vol * pred_vol_scale

    # Define P (Identity matrix for individual asset views)
    p = np.eye(len(signals))

    # Covariance on views (diagonal matrix)
    class_probs = signals.abs()
    omega_values = class_probs * (1 - class_probs)
    omega = np.diag(omega_values) * pred_vol_scale**2

    posterior_mean, posterior_cov = black_litterman(cov, pi, p, q, omega, tau=tau)
    return posterior_mean, posterior_cov
