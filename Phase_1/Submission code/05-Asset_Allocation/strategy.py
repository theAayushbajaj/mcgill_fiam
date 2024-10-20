"""
This script takes objects created in useful_objects.py and allocates a portion
of the available capital to top N stocks.
"""

import numpy as np
import pandas as pd

from hrp import correl_dist, get_quasi_diag, get_rec_bipart
from hrp_black_litterman import black_litterman_pipeline
from sklearn.covariance import LedoitWolf
import scipy.cluster.hierarchy as sch

def asset_allocator(
    start_date=0,
    end_date=None,
    prices=None,
    signals=None,
    market_caps_df=None,
    pred_vol_scale=0.5,
    tau=1.0,
    bl = True,
    lw = True,
    n_stocks = 75
):
    """
    Inputs :
    - Start Date
    - End Date # THIS IS THE DATE WE ARE MAKING THE PREDICTIONS FOR
    - prices : pd.DataFrame : to compute returns and then covariance matrix
    - signals : A pandas datafram of signals (predictions and probabilities)
    Should have the same columns as prices
    We'll need a function to gather the signals (simply the predictions and
    probabilities columns at row End Date)

    Outputs :
    Dataframe of weights, with the same columns as prices
    """

    # For debugging
    # print("Lambda: ", lambda_)
    # print("Tau: ", tau)

    # Initialize the weights to return
    weights = pd.DataFrame(index=prices.columns)
    weights["Weight"] = 0.0

    # Step 0) Adjust the prices to the start and end date
    # Index in terms of months
    prices = prices[start_date : end_date + 1]

    # Step 1) Stock Selection
    # Select the top 100 stocks based on the absolute value of the signal
    signals_end = signals.iloc[end_date]
    # Set signals to 0 for stocks with less than 60 non-NA price values
    sufficient_data = prices.count() >= 60
    signals_end = signals_end.where(sufficient_data, 0)

    # Select top 100 stocks based on absolute signal value
    abs_signals = signals_end.abs()
    abs_signals = abs_signals.sort_values(ascending=False)
    selected_stocks = abs_signals.index[:n_stocks].tolist()
    signals_end = signals_end[selected_stocks]

    # Filter the data to only include the selected stocks
    prices = prices[selected_stocks]
    market_caps_df = market_caps_df[selected_stocks]

    # Step 2) Compute the covariance matrix
    # Covariance matrix using Ledoit-Wolf shrinkage
    returns = prices.pct_change().dropna()
    if lw:

        l_wolf = LedoitWolf()
        shrunk_cov_matrix = l_wolf.fit(returns).covariance_
        cov = pd.DataFrame(
            shrunk_cov_matrix, index=selected_stocks, columns=selected_stocks
        )
    else:
        cov = returns.cov()
        cov = pd.DataFrame(cov, index=selected_stocks, columns=selected_stocks)

    # Step 3) Market-implied returns
    # Extract market capitalizations
    market_caps = market_caps_df.iloc[end_date]

        # Step 4) Incorporate views from signals
    if bl:
        # Step 5) Black-Litterman
        volatility = returns.std()
        posterior_mean, posterior_cov = black_litterman_pipeline(cov,
                                                    signals_end,
                                                    market_caps,
                                                    volatility,
                                                    pred_vol_scale,
                                                    tau,
                                                    )

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

    else:
        pi = pd.Series(pi, index=selected_stocks)
        # equally weighted returns
        pi = pd.Series(1/len(selected_stocks), index=selected_stocks)
        posterior_mean = pi
        posterior_cov = cov


    # Step 6) Hierarchical Risk Parity (HRP)
    # Reconstruct the correlation matrix from the posterior covariance matrix
    std_devs = np.sqrt(np.diag(posterior_cov))
    # Avoid division by zero
    std_devs[std_devs == 0] = 1e-6
    corr = posterior_cov / np.outer(std_devs, std_devs)
    corr = np.clip(corr, -1, 1)
    corr.values[range(corr.shape[0]), range(corr.shape[1])] = 1.0
    corr = pd.DataFrame(corr, index=selected_stocks, columns=selected_stocks)

    # Now compute the distance matrix
    dist = correl_dist(corr)

    # Check for NaNs in the distance matrix
    if np.isnan(dist.to_numpy()).any():
        # print("NaNs detected in distance matrix.")
        dist = np.nan_to_num(dist, nan=1e6)

    dist = pd.DataFrame(dist, index=selected_stocks, columns=selected_stocks)

    # Plot correlation matrix
    # plotCorrMatrix('HRP_BL_corr0.png', corr, labels=corr.columns)

    # Cluster using hierarchical clustering
    link = sch.linkage(dist, method="single")
    sort_ix = get_quasi_diag(link)
    sort_ix = corr.index[sort_ix].tolist()

    # Reorder covariance matrix for clustered stocks
    cov_reordered = posterior_cov.loc[sort_ix, sort_ix]
    # corr_reordered = corr.loc[sort_ix, sort_ix]

    # Plot reordered correlation matrix
    # plotCorrMatrix('HRP_BL_corr1.png', corr_reordered, labels=corr_reordered.columns)

    # Apply HRP with Black-Litterman posterior covariance
    hrp_weights = get_rec_bipart(cov_reordered, sort_ix, posterior_mean)

    # print("HRP Weights:")
    # print(hrp_weights)
    # print('Market exposure (sum of weights): ', hrp_weights.sum())
    # print('Sum of absolute values of weights: ', hrp_weights.abs().sum())

    # Assign the weights to the output DataFrame
    weights.loc[hrp_weights.index, "Weight"] = hrp_weights
    # print("Final Weights Sum:")
    # print(weights.abs().sum())
    # print(weights.sum())

    return weights


if __name__ == "__main__":
    # Load data
    prices_df = pd.read_pickle("objects/prices.pkl")
    signals_df = pd.read_pickle("objects/signals.pkl")
    market_caps_df = pd.read_pickle("objects/market_caps.pkl")
    START_DATE = 0
    END_DATE = 100

    # Run the asset allocator
    weights = asset_allocator(START_DATE, END_DATE, prices_df, signals_df, market_caps_df)
    print("Market exposure (sum of weights): ", weights.sum())
    print("Sum of absolute values of weights: ", weights.abs().sum())
    print(weights)
