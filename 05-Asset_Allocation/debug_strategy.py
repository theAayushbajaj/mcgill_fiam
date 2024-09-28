#%%
import numpy as np
import pandas as pd

from hrp import *
from hrpBL import *
#%%
def asset_allocator(start_date, end_date, prices, signals, market_caps_df, lambda_=0.5, tau = 1.0):
    #%%
    lambda_=0.5
    tau = 1.0
    prices = pd.read_pickle('../objects/prices.pkl')
    signals = pd.read_pickle('../objects/signals.pkl')
    market_caps_df = pd.read_pickle('../objects/market_caps.pkl')
    start_date = 0
    end_date = 100
    #%%
    """
    Inputs :
    - Start Date
    - End Date
    - prices : pd.DataFrame : to compute returns and then covariance matrix
    - signals : A pandas datafram of signals (predictions and probabilities)
    Should have the same columns as prices
    We'll need a function to gather the signals (simply the predictions and 
    probabilities columns at row End Date)
    
    Outputs :
    Dataframe of weights, with the same columns as prices
    """
    # Initialize the weights to return
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

    # Extract probabilities and directions from signals
    # Assuming signals are probabilities times direction (-1 or 1)
    probabilities = signals_end.abs()
    # Ensure probabilities are within (0, 1)
    probabilities = probabilities.clip(1e-6, 1 - 1e-6)

    # Compute Omega as variance of Bernoulli distribution
    Omega_values = tau * probabilities * (1 - probabilities)
    # Set a minimum variance to avoid zeros
    min_variance = 1e-6
    Omega_values = np.maximum(Omega_values, min_variance)
    Omega = np.diag(Omega_values)
    
    # Step 5) Black-Litterman
    posterior_mean, posterior_cov = black_litterman(cov, pi, P, Q, Omega, tau=tau)
    posterior_mean = pd.Series(posterior_mean, index=selected_stocks)
    posterior_cov = pd.DataFrame(posterior_cov, index=selected_stocks, columns=selected_stocks)
    
    # Ensure posterior covariance matrix is positive definite
    # Add a small value to the diagonal if necessary
    min_eigenvalue = np.min(np.linalg.eigvals(posterior_cov))
    if min_eigenvalue < 0:
        print("Adding small value to diagonal of posterior covariance matrix.")
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
    corr.values[range(corr.shape[0]), range(corr.shape[1])] = 1.0
    corr = pd.DataFrame(corr, index=selected_stocks, columns=selected_stocks)

    # Now compute the distance matrix
    dist = correlDist(corr)

    # Check for NaNs in the distance matrix
    if np.isnan(dist.to_numpy()).any():
        print("NaNs detected in distance matrix.")
        # Investigate and handle NaNs
        # Replace NaNs with large distances to prevent clustering them together
        dist = np.nan_to_num(dist, nan=1e6)
        
    dist = pd.DataFrame(dist, index=selected_stocks, columns=selected_stocks)

    # Plot correlation matrix
    # plotCorrMatrix('HRP_BL_corr0.png', corr, labels=corr.columns)

    # Cluster using hierarchical clustering
    #%%
    link = sch.linkage(dist, method='single')
    #%%
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()

    # Reorder covariance matrix for clustered stocks
    cov_reordered = posterior_cov.loc[sortIx, sortIx]
    corr_reordered = corr.loc[sortIx, sortIx]

    # Plot reordered correlation matrix
    # plotCorrMatrix('HRP_BL_corr1.png', corr_reordered, labels=corr_reordered.columns)

    # Apply HRP with Black-Litterman posterior covariance
    hrp_weights = getRecBipart(cov_reordered, sortIx)
    # print("HRP Weights:")
    # print(hrp_weights)
    # print('Market exposure (sum of weights): ', hrp_weights.sum())
    # print('Sum of absolute values of weights: ', hrp_weights.abs().sum())

    # Assign the weights to the output DataFrame
    weights.loc[hrp_weights.index, 'Weight'] = hrp_weights
    # print("Final Weights:")
    # print(weights)
    #%%
    return weights
# %%
