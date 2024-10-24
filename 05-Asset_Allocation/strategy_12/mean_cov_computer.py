import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
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

def make_positive_definite(cov_matrix):
    min_eigenvalue = np.min(np.linalg.eigvals(cov_matrix))
    if min_eigenvalue < 0:
        cov_matrix += np.eye(cov_matrix.shape[0]) * (-min_eigenvalue + 1e-6)
    return cov_matrix


def main(
    returns,
    signals,
    market_caps,
    selected_stocks,
    tau=1.0,
    lambda_=2.5,
    use_ema=False,  # Option to use EMA
    window=60,  # Rolling window size for moving average
    span=60,    # Span for EMA
):
    # check = False
    # if not check:
    #     # print('Check for mean_cov_computer')
    #     # print(f'tau is {tau}')
    #     # print(f'lambda is {lambda_}')
    #     check = True
    
    # print("returns:")
    # print(returns)
    # print("returns.shape:")
    # print(returns.shape)

    
    # Compute Prior Covariance Matrix and Expected Returns
    if use_ema:
        # Use EMA for both covariance and returns
        returns_ewm = returns.ewm(span=span)
        cov_ewm = returns_ewm.cov().dropna().iloc[-len(selected_stocks):, -len(selected_stocks):]
        cov_ewm = make_positive_definite(cov_ewm)
        mean_returns = returns_ewm.mean().dropna().iloc[-1, :]
        
        # print("returns_ewm.mean():")
        # print(returns_ewm.mean())
        # print("returns_ewm.mean().dropna():")
        # print(returns_ewm.mean().dropna())
        # print("returns_ewm.mean().dropna().shape:")
        # print(returns_ewm.mean().dropna().shape)

        
        # Apply Ledoit-Wolf shrinkage to EMA covariance matrix
        l_wolf = LedoitWolf()
        shrunk_cov_matrix = l_wolf.fit(cov_ewm).covariance_
        cov = pd.DataFrame(shrunk_cov_matrix, index=selected_stocks, columns=selected_stocks)
    
    else:
        # Use rolling window for covariance and mean
        returns_rolling = returns.rolling(window=window)
        cov_rolling = returns_rolling.cov().dropna().iloc[-len(selected_stocks):, -len(selected_stocks):]
        mean_returns = returns_rolling.mean().dropna().iloc[-1, :]
        
        # Apply Ledoit-Wolf shrinkage to rolling window covariance matrix
        l_wolf = LedoitWolf()
        shrunk_cov_matrix = l_wolf.fit(cov_rolling).covariance_
        cov = pd.DataFrame(shrunk_cov_matrix, index=selected_stocks, columns=selected_stocks)

    # Get market capitalizations vector
    market_weights = market_caps / market_caps.sum()

    # Implied Excess Equilibrium Return Vector
    pi = get_market_implied_returns(cov, market_weights, lambda_)

    # Scale the signals by predicted volatility
    q = signals.abs() * mean_returns

    # Define P (Identity matrix for individual asset views)
    p = np.eye(len(signals))

    # Covariance on views (diagonal matrix)
    class_probs = signals.abs()
    returns_vol = returns.std(axis=0) + 1e-6
    omega_values = class_probs * (1 - class_probs) * returns_vol ** 2
    omega = np.diag(omega_values)

    posterior_mean, posterior_cov = black_litterman(cov, pi, p, q, omega, tau=tau)
    posterior_mean = pd.Series(posterior_mean, index=selected_stocks)
    posterior_cov = pd.DataFrame(posterior_cov, index=selected_stocks, columns=selected_stocks)

    # Ensure posterior covariance matrix is positive definite
    min_eigenvalue = np.min(np.linalg.eigvals(posterior_cov))
    if min_eigenvalue < 0:
        posterior_cov += np.eye(len(posterior_cov)) * (-min_eigenvalue + 1e-6)

    return posterior_mean, posterior_cov
