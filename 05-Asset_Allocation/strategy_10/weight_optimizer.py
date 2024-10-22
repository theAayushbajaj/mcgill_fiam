import numpy as np
import pandas as pd
from scipy.optimize import minimize


def main(
    weights,
    posterior_cov,
    posterior_mean,
    selected_stocks,
    benchmark_df,
    lambda_=1.0,
    soft_risk=0.01
):

    
    # equally weighted
    weight_opt = 1 / len(selected_stocks)

    weights.loc[selected_stocks, "Weight"] = weight_opt


    return weights
