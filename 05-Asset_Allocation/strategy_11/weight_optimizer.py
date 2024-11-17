import numpy as np
import pandas as pd
from scipy.optimize import minimize


def main(
    weights,
    posterior_cov,
    posterior_mean,
    long_stocks,
    short_stocks,
    benchmark_df,
    lambda_=1.0,
    soft_risk=0.01
):

    
    # equally weighted
    selected_stocks = long_stocks + short_stocks
    # weight +1/len(selected_stocks) for long stocks, -1/len(selected_stocks) for short stocks
    weight_long = 1 / len(long_stocks)
    weight_short = -1 / len(short_stocks)
    
    weights.loc[long_stocks, "Weight"] = weight_long
    weights.loc[short_stocks, "Weight"] = weight_short
    
    return weights