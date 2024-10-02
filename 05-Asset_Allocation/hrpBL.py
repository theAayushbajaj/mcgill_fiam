# %%
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
import scipy.cluster.hierarchy as sch
from sklearn.covariance import LedoitWolf

# Black-Litterman Imports
from scipy.linalg import inv

from hrp import *

# %%

# HRP Functions


# Black-Litterman Functions


def black_litterman(cov, market_implied_returns, P, Q, Omega, tau=0.05):
    """Black-Litterman model."""
    pi = market_implied_returns
    tau_cov = tau * cov
    M_inverse = inv(tau_cov)
    Omega_inv = inv(Omega)
    # Compute posterior covariance
    posterior_cov = inv(M_inverse + P.T @ Omega_inv @ P)
    # Compute posterior mean
    posterior_mean = posterior_cov @ (M_inverse @ pi + P.T @ Omega_inv @ Q)
    return posterior_mean, posterior_cov


def get_market_implied_returns(cov, market_weights, lambda_=2.5):
    """Compute market-implied returns."""
    # pi = delta * cov * market_weights
    pi = lambda_ * cov @ market_weights
    return pi


# Main function integrating Black-Litterman with HRP


def BL_pipeline(
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
    Q = signals * returns_vol * pred_vol_scale

    # Define P (Identity matrix for individual asset views)
    P = np.eye(len(signals))

    # Covariance on views (diagonal matrix)
    Class_probs = signals.abs()
    Omega_values = Class_probs * (1 - Class_probs)
    Omega = np.diag(Omega_values) * pred_vol_scale**2

    posterior_mean, posterior_cov = black_litterman(cov, pi, P, Q, Omega, tau=tau)
    return posterior_mean, posterior_cov


# %%

if __name__ == "__main__":
    main()


# %%
