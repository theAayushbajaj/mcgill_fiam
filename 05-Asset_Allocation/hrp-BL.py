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
    M_inverse = inv(tau * cov)
    P_T_Omega_P = np.dot(np.dot(P.T, inv(Omega)), P)
    
    posterior_cov = inv(M_inverse + P_T_Omega_P)
    part2 = np.dot(np.dot(P.T, inv(Omega)), Q)
    posterior_mean = np.dot(posterior_cov, (np.dot(M_inverse, pi) + part2))
    
    return posterior_mean, posterior_cov

def get_market_implied_returns(cov, market_caps):
    """Compute market-implied returns."""
    market_weights = market_caps / market_caps.sum()
    market_returns = np.dot(cov, market_weights)
    return market_returns

# Main function integrating Black-Litterman with HRP

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
    
    # Omega = np.diag([0.1]*len(Q))
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
    corr = returns.corr()
    # OR SHOULD WE DO
    # CORR = POSTERIOR_COV / (STD * STD)
    # Step 5) Hierarchical Risk Parity (HRP)
    # Calculate standard deviations from the posterior covariance matrix
    std_devs = np.sqrt(np.diag(posterior_cov))
    std_devs

    # Calculate the posterior correlation matrix from posterior covariance
    corr = posterior_cov / np.outer(std_devs, std_devs)
   # Check for NaN or inf values in the correlation matrix
    if np.any(np.isnan(corr)) or np.any(np.isinf(corr)) or not np.all(np.isfinite(corr)):
        print("NaN or infinite values found in the correlation matrix.")
        return None
    
# Check for NaN or infinite values in the correlation matrix
    if np.any(np.isnan(corr)) or np.any(np.isinf(corr)) or not np.all(np.isfinite(corr)):
        print("NaN or infinite values found in the correlation matrix.")
        return None

    # Now compute the distance matrix
    dist = correlDist(corr)

    # Ensure the distance matrix contains only finite values
    if np.any(np.isnan(dist)) or np.any(np.isinf(dist)) or not np.all(np.isfinite(dist)):
        print("NaN or infinite values found in the distance matrix.")
        return None
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
    #%%
    # assign the weights to the output
    weights.loc[hrp_weights.index, 'Weight'] = hrp_weights
    weights
    #%%
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    

# %%
