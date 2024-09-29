#%%
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
#%%

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

def BL_pipeline(cov, signals, market_caps, returns_vol,
                pred_vol_scale=1.0, tau=1.0, lambda_=2.5):
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
    Omega_values = Class_probs*(1-Class_probs)
    Omega = np.diag(Omega_values) * pred_vol_scale**2
    
    posterior_mean, posterior_cov = black_litterman(cov, pi, P, Q, Omega, tau=tau)
    return posterior_mean, posterior_cov
    

#%%
def main():
    # Load your data
    #%%
    path = '../objects/prices.pkl'
    prices = pd.read_pickle(path)
    path = '../objects/market_caps.pkl'
    market_caps_df = pd.read_pickle(path)
    path = '../objects/signals.pkl'
    signals = pd.read_pickle(path)
    
    # Initalize the weights to return
    weights = pd.DataFrame(index=prices.columns)
    weights['Weight'] = 0.0
    weights
    
    #%%
    
    # Step 0) Parameters
    # Index in terms of months
    Start_Date = 0
    End_Date = 100 # index for which we will predict
    prices = prices[Start_Date:End_Date+1]
    
    # Step 1) Stock Selection
    # Select the top 100 stocks based on the absolute value of the signal
    # These are the top 100 stocks we have the most confidence in
    signals_end = signals.iloc[End_Date]
    signals_end
    #%%
    # if there are less than 60 non-NA values for a stock, set the signal to 0
    signals_end = signals_end.where(prices.count() > 60, 0)
    signals_end
    #%%
    
    abs_signals = abs(signals_end)
    abs_signals = abs_signals.sort_values(ascending=False)
    selected_stocks = abs_signals.index[:100]
    selected_stocks = selected_stocks.tolist()
    signals_end = signals_end[selected_stocks]
    # Filter the objects to only include the selected stocks
    prices = prices[selected_stocks]
    market_caps_df = market_caps_df[selected_stocks]
    #%%
    
    # Step 2) Compute the convariance matrix
    # Covariance matrix using Ledoit-Wolf shrinkage
    returns = prices.pct_change().dropna()
    returns
    #%%
    lw = LedoitWolf()
    shrunk_cov_matrix = lw.fit(returns).covariance_
    cov = pd.DataFrame(shrunk_cov_matrix, index=selected_stocks, columns=selected_stocks)
    #%%
    
    # Step 3) Market-implied returns
    # Assuming 'market_equity' is the market capitalization for each stock
    market_caps = market_caps_df.iloc[End_Date]
    
    # Normalize to get market weights
    market_weights = market_caps / market_caps.sum()
    
    # Use these weights to compute market-implied returns
    pi = np.dot(cov, market_weights)
    pi = pd.Series(pi, index=selected_stocks)
    pi
    #%%
    # Step 3) Classifier outputs and rolling volatility for lambda_
    volatility = returns.std()
    volatility
    #%%
    
    # Use rolling volatility as a scaling factor for lambda
    lambda_ = volatility
    lambda_
    #%%
    
    # Convert classifier signals to views
    Q = signals_end * lambda_
    
    
    # Define P (Identity matric for individual stock views)
    P = np.eye(len(selected_stocks))
    
    Omega = np.diag([0.1]*len(Q))
    # Extract classifier probabilities from the signals
    classifier_probs = signals_end.abs()  # Probabilities are typically positive values
    Omega = np.diag(1 - classifier_probs)  # Use 1 - probability to reflect uncertainty

    # Ensure that we avoid zeros in the diagonal for computational stability
    Omega = np.where(Omega == 0, 1e-6, Omega)

    
    #%%
    # Step 4) Black-Litterman
    posterior_mean, posterior_cov = black_litterman(cov, pi, P, Q, Omega)
    posterior_mean = pd.Series(posterior_mean, index=selected_stocks)
    posterior_cov = pd.DataFrame(posterior_cov, index=selected_stocks, columns=selected_stocks)
    
    #%%
    
    # Step 5) Hierarchical Risk Parity (HRP)
    # Correlation matrix
    corr_1 = returns.corr()
    corr_1
    dist_1 = correlDist(corr_1)
    dist_1
    #%%
    # OR SHOULD WE DO
    # CORR = POSTERIOR_COV / (STD * STD)
    # Step 5) Hierarchical Risk Parity (HRP)
    # Calculate standard deviations from the posterior covariance matrix
    std_devs = np.sqrt(np.diag(posterior_cov))
    std_devs = np.diag(std_devs + 1e-6)  # Adding a small regularization term
    # inverse of the standard deviations
    S = np.linalg.inv(std_devs)
    #%%

    # Calculate the posterior correlation matrix from posterior covariance
    # correlation = S-1 * Cov * S-1
    corr = np.dot(S, np.dot(posterior_cov, S))
    # adjust diag to 1
    diag_idx = np.diag_indices(corr.shape[0])
    corr[diag_idx] = 1
    corr = pd.DataFrame(corr, index=selected_stocks, columns=selected_stocks)
    corr
    #%%

    # Now compute the distance matrix
    dist = correlDist(corr)
    # print total of NAs
    print(np.sum(np.isnan(dist)).sum())
    
    # fill NAs in dist with correspind values from dist_1
    dist = np.where(np.isnan(dist), dist_1, dist)
    dist = pd.DataFrame(dist, index=selected_stocks, columns=selected_stocks)
    dist
    
    # Inpect for NAs
        

#%%


    
    # Plot correlation matrix
    plotCorrMatrix('HRP_BL_corr0.png', corr, labels=corr.columns)
    #%%
    
    # Cluster using hierarchical clustering
    # dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()
    
    # Reorder covariance matrix for clustered stocks
    cov_reordered = cov.loc[sortIx, sortIx]
    corr_reordered = corr.loc[sortIx, sortIx]
    
    # Plot reordered correlation matrix
    plotCorrMatrix('HRP_BL_corr1.png', corr_reordered, labels=corr_reordered.columns)
    
    # Apply HRP with Black-Litterman posterior covariance
    hrp_weights = getRecBipart(pd.DataFrame(posterior_cov, index=returns.columns, columns=returns.columns), sortIx)
    print(hrp_weights)
    print('Market exposure : ', hrp_weights.sum())
    print('Sum of absolute values of weights: ', hrp_weights.abs().sum())
    #%%
    # assign the weights to the output
    weights.loc[hrp_weights.index, 'Weight'] = hrp_weights
    weights
    #%%
    
    
    
    
#%%

def main_2():
    weights = pd.DataFrame(index=prices.columns)
    weights['Weight'] = 0.0
    
    # Step 0) Adjust the prices to the start and end date
    # Index in terms of months
    prices = prices[start_date:end_date+1]
    
    # Step 1) Stock Selection
    # Select the top 100 stocks based on the absolute value of the signal
    signals_end = signals.iloc[end_date]
    # Set signals to 0 for stocks with less than 60 non-NA price values
    sufficient_data = prices.count() > 60
    signals_end = signals_end.where(sufficient_data, 0)

    # Select top 100 stocks based on absolute signal value
    abs_signals = signals_end.abs()
    abs_signals = abs_signals.sort_values(ascending=False)
    selected_stocks = abs_signals.index[:100].tolist()
    signals_end = signals_end[selected_stocks]

    # Filter the data to only include the selected stocks
    prices = prices[selected_stocks]
    market_caps_df = market_caps_df[selected_stocks]
    
    # Step 2) Compute the covariance matrix
    # Covariance matrix using Ledoit-Wolf shrinkage
    returns = prices.pct_change().dropna()
    lw = LedoitWolf()
    shrunk_cov_matrix = lw.fit(returns).covariance_
    cov = pd.DataFrame(shrunk_cov_matrix, index=selected_stocks, columns=selected_stocks)
    
    # Step 3) Market-implied returns
    # Extract market capitalizations
    market_caps = market_caps_df.iloc[end_date]
    # Normalize to get market weights
    market_weights = market_caps / market_caps.sum()

    # Use these weights to compute market-implied returns
    delta = 2.5  # Risk aversion coefficient (common default)
    pi = get_market_implied_returns(cov, market_weights, delta)
    pi = pd.Series(pi, index=selected_stocks)
    
    # Step 4) Incorporate views from signals
    volatility = returns.std()
    Q = signals_end * volatility * lambda_

    # Define P (Identity matrix for individual asset views)
    P = np.eye(len(selected_stocks))

    # Construct Omega, the diagonal covariance matrix of the error terms in the views
    # Confidence levels are between 0 and 1
    confidence_levels = signals_end.abs()
    # Ensure confidence levels are not exactly 0 to avoid division by zero
    confidence_levels = confidence_levels.replace(0, 1e-6)
    # Calculate the variance of each view
    tau = 0.05  # Scaling factor for uncertainty in the prior estimate
    diag_cov = np.diag(cov)
    Omega_values = ((1 - confidence_levels) / confidence_levels) * tau * diag_cov
    # Handle infinite or NaN values
    Omega_values = np.where(np.isfinite(Omega_values), Omega_values, 1e6)
    Omega = np.diag(Omega_values)
    
    # Step 5) Black-Litterman
    posterior_mean, posterior_cov = black_litterman(cov, pi, P, Q, Omega, tau=tau)
    posterior_mean = pd.Series(posterior_mean, index=selected_stocks)
    posterior_cov = pd.DataFrame(posterior_cov, index=selected_stocks, columns=selected_stocks)
    
    # Ensure posterior covariance matrix is positive definite
    # Add a small value to the diagonal if necessary
    min_eigenvalue = np.min(np.linalg.eigvals(posterior_cov))
    if min_eigenvalue < 0:
        posterior_cov += np.eye(len(posterior_cov)) * (-min_eigenvalue + 1e-6)
        
    # Step 6) Hierarchical Risk Parity (HRP)
    # Reconstruct the correlation matrix from the posterior covariance matrix
    std_devs = np.sqrt(np.diag(posterior_cov))
    # Avoid division by zero
    std_devs[std_devs == 0] = 1e-6
    # Compute correlation matrix
    corr = posterior_cov / np.outer(std_devs, std_devs)
    # Ensure values are within [-1, 1]
    corr = np.clip(corr, -1, 1)
    # Set diagonal elements to 1
    np.fill_diagonal(corr, 1.0)
    corr = pd.DataFrame(corr, index=selected_stocks, columns=selected_stocks)

    # Now compute the distance matrix
    dist = correlDist(corr)

    # Check for NaNs in the distance matrix
    if np.isnan(dist).any():
        print("NaNs detected in distance matrix.")
        # Investigate and handle NaNs
        # Replace NaNs with large distances to prevent clustering them together
        dist = np.nan_to_num(dist, nan=1e6)
    dist = pd.DataFrame(dist, index=selected_stocks, columns=selected_stocks)

    # Plot correlation matrix
    plotCorrMatrix('HRP_BL_corr0.png', corr, labels=corr.columns)

    # Cluster using hierarchical clustering
    link = sch.linkage(dist, method='single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()

    # Reorder covariance matrix for clustered stocks
    cov_reordered = posterior_cov.loc[sortIx, sortIx]
    corr_reordered = corr.loc[sortIx, sortIx]

    # Plot reordered correlation matrix
    plotCorrMatrix('HRP_BL_corr1.png', corr_reordered, labels=corr_reordered.columns)

    # Apply HRP with Black-Litterman posterior covariance
    hrp_weights = getRecBipart(cov_reordered, sortIx)
    print("HRP Weights:")
    print(hrp_weights)
    print('Market exposure (sum of weights): ', hrp_weights.sum())
    print('Sum of absolute values of weights: ', hrp_weights.abs().sum())

    # Assign the weights to the output DataFrame
    weights.loc[hrp_weights.index, 'Weight'] = hrp_weights
    print("Final Weights:")
    print(weights)
    return weights
    
    
if __name__ == '__main__':
    main()
    
    
    

# %%
