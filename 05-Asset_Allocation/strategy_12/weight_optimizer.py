import numpy as np
import pandas as pd
from scipy.optimize import minimize

from hmmlearn.hmm import GaussianHMM




def generate_scenarios(posterior_mean, posterior_cov, num_scenarios=10, uncertainty_level=0.05):
    """
    Generate different perturbations for mean and covariance based on uncertainty level.
    Args:
        posterior_mean (pd.Series): Mean returns.
        posterior_cov (pd.DataFrame): Covariance matrix.
        num_scenarios (int): Number of scenarios to generate.
        uncertainty_level (float): The degree of uncertainty applied to the perturbations.
    Returns:
        scenarios_mu (list): List of perturbed mean return vectors.
        scenarios_cov (list): List of perturbed covariance matrices.
    """
    np.random.seed(42)  # for reproducibility
    scenarios_mu = []
    scenarios_cov = []

    for _ in range(num_scenarios):
        # Apply random perturbation to mean
        perturbed_mean = posterior_mean + uncertainty_level * np.random.randn(len(posterior_mean))

        # Apply random perturbation to covariance (keeping symmetry)
        perturbation = uncertainty_level * np.random.randn(*posterior_cov.shape)
        perturbed_cov = posterior_cov + perturbation @ perturbation.T

        # Ensure positive-definiteness of covariance
        perturbed_cov = (perturbed_cov + perturbed_cov.T) / 2
        min_eigenvalue = np.min(np.linalg.eigvals(perturbed_cov))
        if min_eigenvalue < 0:
            perturbed_cov += np.eye(posterior_cov.shape[0]) * (-min_eigenvalue + 1e-6)

        scenarios_mu.append(perturbed_mean)
        scenarios_cov.append(perturbed_cov)

    return scenarios_mu, scenarios_cov

def predict_volatility_state(benchmark_df):
    
    # Preprocess the volatility data to remove NaNs and infs
    returns_or_volatility = benchmark_df["sp_ret"].ewm(span=12).std()
    returns_or_volatility = returns_or_volatility.replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure that there are no NaN values left
    returns_or_volatility = returns_or_volatility.fillna(method='ffill').fillna(method='bfill')

    # Initialize and train HMM with adjusted number of components
    hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000)

    # Fit the HMM and get the current hidden state
    train_data = returns_or_volatility.values.reshape(-1, 1)
    hmm.fit(train_data)
    current_state = hmm.predict(train_data)[-1]  # Get the latest state
    
    return current_state
    


def main(
    weights,
    posterior_cov,
    posterior_mean,
    selected_stocks,
    benchmark_df,
    lambda_=1.0,
    soft_risk=0.01,
    num_scenarios=10,
    uncertainty_level=0.05,
    total_allocation = 1.0
):
    """
    Robust portfolio optimization that accounts for multiple scenarios of the covariance matrix
    and mean returns.
    """
    # check = False
    # if not check:
    #     print("Check for weight_optimizer")
    #     print(f"lambda is {lambda_}")
    #     print(f"soft risk is {soft_risk}")
    #     check = True

    benchmark_std = benchmark_df["sp_ret"].std()
    # Predict Volatility State
    try:
        current_state = predict_volatility_state(benchmark_df)
    except:
        current_state=1
    
    # benchmark_std as the min of the 
    n = len(selected_stocks)

    # Generate multiple scenarios for posterior_mean and posterior_cov
    scenarios_mu, scenarios_cov = generate_scenarios(posterior_mean, posterior_cov, num_scenarios, uncertainty_level)

    # Objective function: maximize the worst-case scenario's performance
    def objective(w):
        worst_case_return = np.inf
        for mu_scenario, cov_scenario in zip(scenarios_mu, scenarios_cov):
            # Compute the return and risk for each scenario
            scenario_return = mu_scenario @ w
            scenario_risk = w @ cov_scenario @ w

            # Combine return and risk in a penalized objective
            penalized_value = -scenario_return + lambda_**current_state * 0.5 * scenario_risk
            worst_case_return = min(worst_case_return, penalized_value)

        return worst_case_return

    # Define the constraint functions
    def constraint_eq(w):
        # Sum of weights should equal 1
        return np.sum(w) - total_allocation + 0.1*current_state # np.sum(w)= (1 - current_state)

    def constraint_ineq(w):
        # Risk constraint: ensure portfolio variance is less than benchmark
        return benchmark_std**2 + soft_risk - w @ posterior_cov.values @ w

    # Bounds for each weight: 0 <= w <= 0.10
    bounds = [(0, 0.1)] * n

    # Initial guess for weights (equally distributed)
    w0 = np.ones(n) / n

    # Solve the optimization problem using scipy.optimize.minimize
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {"type": "eq", "fun": constraint_eq},    # Equality constraint
            # {"type": "ineq", "fun": constraint_ineq}  # Inequality (risk) constraint
        ],
        options={"disp": False, "maxiter": 1000}
    )

    if not result.success:
        raise ValueError("Optimization failed:", result.message)

    weights_opt = result.x

    # Update the weights dataframe
    weights.loc[selected_stocks, "Weight"] = weights_opt

    return weights
