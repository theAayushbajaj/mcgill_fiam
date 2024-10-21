import numpy as np
import pandas as pd
import cvxpy as cp

def robust_optimize_portfolio(cov, mu, lambda_, benchmark_std, soft_risk=0.01):
    n = len(mu)
    w = cp.Variable(n)

    # Define the uncertainty set for expected returns
    # For simplicity, we'll assume the estimation error covariance is proportional to the asset covariances
    Sigma_mu = cov.values * 0.1  # 10% of asset covariances
    delta = 0.1  # Uncertainty level

    # Robust objective: maximize worst-case expected return
    robust_obj = mu.values @ w - lambda_ * 0.5 * cp.quad_form(w, cov.values)
    robust_obj -= delta * cp.norm(Sigma_mu @ w, 2)

    # Constraints
    constraints = [
        cp.sum(w) <= 1.0,
        w >= 0,
        w <= 0.10,
        cp.quad_form(w, cov.values) <= (benchmark_std**2 + soft_risk**2)
    ]

    # Define and solve the problem
    prob = cp.Problem(cp.Maximize(robust_obj), constraints)
    prob.solve(solver=cp.SCS)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError("Optimization failed.")

    optimized_weights = w.value
    optimized_weights = pd.Series(optimized_weights, index=mu.index)
    # optimized_weights /= optimized_weights.sum()  # Normalize weights

    # Calculate the worst-case expected return
    worst_case_return = mu.values @ optimized_weights.values - delta * np.linalg.norm(Sigma_mu @ optimized_weights.values)

    return optimized_weights, worst_case_return

# Usage example
def main(
    weights,
    posterior_cov,
    posterior_mean,
    selected_stocks,
    benchmark_df,
    lambda_=3.07,
    soft_risk=0.01,
):
    # Prepare data
    cov = posterior_cov.loc[selected_stocks, selected_stocks]
    mu = posterior_mean.loc[selected_stocks]
    benchmark_std = benchmark_df["sp_ret"].std()

    # Optimize portfolio
    optimized_weights, worst_case_return = robust_optimize_portfolio(
        cov, mu, lambda_, benchmark_std, soft_risk
    )

    # Assign weights to the output DataFrame
    weights.loc[optimized_weights.index, "Weight"] = optimized_weights

    return weights
