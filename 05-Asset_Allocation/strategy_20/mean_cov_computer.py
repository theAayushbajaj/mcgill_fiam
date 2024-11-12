import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from scipy.linalg import inv

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

    
    # Compute Prior Covariance Matrix and Expected Returns
    if use_ema:
        # Use EMA for both covariance and returns
        returns_ewm = returns.ewm(span=span)
        cov_ewm = returns_ewm.cov().dropna().iloc[-len(selected_stocks):, -len(selected_stocks):]
        cov_ewm = make_positive_definite(cov_ewm)
        mean_returns = returns_ewm.mean().dropna().iloc[-1, :]
        
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

    posterior_mean = pd.Series(mean_returns, index=selected_stocks)
    posterior_cov = pd.DataFrame(cov, index=selected_stocks, columns=selected_stocks)

    # Ensure posterior covariance matrix is positive definite
    min_eigenvalue = np.min(np.linalg.eigvals(posterior_cov))
    if min_eigenvalue < 0:
        posterior_cov += np.eye(len(posterior_cov)) * (-min_eigenvalue + 1e-6)

    return posterior_mean, posterior_cov
