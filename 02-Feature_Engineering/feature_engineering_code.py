"""
This script adds the following features to the dataset:
    1. 'target' --->>> 1 if excess return >= 0. Otherwise -1.
    2. 't1_index' --->>> First business day of the current month.
    3. 't1' --->>> Last business day of the current month.
    4. 'total_return' --->>> excess return + risk free rate.
    5. 'adj_price' --->>> Adjusted price that takes into account stock splits, dividends etc.
    6. 'log_price' --->>> Logarithm of 'adj_price'.
    7. 'log_diff' --->>> Difference between consecutive values of log price.
    8. 'weight_attr' --->>> Absolute value of excess return.
    9. 'frac_diff' --->>> Fractionally differentiated price as feature.
"""

import os
import warnings

import json

import pandas as pd
from pandas.tseries.offsets import BMonthBegin
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

# set the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set the directory containing your stock CSV files
STOCKS_DATA_DIR = "../stocks_data"

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(STOCKS_DATA_DIR) if f.endswith(".csv")]

# Fractionally Differentiated Price as a Feature


# SNIPPET 5.3 THE NEW FIXED-WIDTH WINDOW FRACDIFF METHOD
def frac_diff_ffd(series, d, thres=1e-5):
    """
    Compute fractional differencing using a constant width window.

    Parameters:
    series (DataFrame): Input time series data.
    d (float): Fractional differencing order (positive value).
    thres (float): Weight cut-off threshold for the window (default is 1e-5).

    Returns:
    DataFrame: Fractionally differenced series.
    """
    # 1) Compute weights for the longest series
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    # 2) Apply weights to values
    result_df = {}
    for name in series.columns:
        # Corrected the fill method to 'ffill'
        series_f, df_ = series[[name]].fillna(method="ffill").dropna(), pd.Series()
        for iloc1 in range(width, series_f.shape[0]):
            loc0, loc1 = series_f.index[iloc1 - width], series_f.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue  # exclude NAs
            df_[loc1] = np.dot(w.T, series_f.loc[loc0:loc1])[0, 0]
        result_df[name] = df_.copy(deep=True)
    result_df = pd.concat(result_df, axis=1)
    return result_df


def get_weights_ffd(d, thres):
    """
    Calculate fractional differencing weights.

    Generates weights based on the fractional order `d` until the last weight
    is below the specified threshold `thres`.

    Parameters:
    d (float): The fractional differencing order (positive value).
    thres (float): Cut-off threshold for weights.

    Returns:
    numpy.ndarray: Column vector of calculated weights in reverse order.
    """
    w = [1.0]
    while abs(w[-1]) > thres:
        w_ = -w[-1] / len(w) * (d - len(w) + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def find_min_ffd(input_df, col_name="adj_price"):
    """
    Find the minimum fractional differencing order for stationarity.

    Parameters:
    input_df (DataFrame): Input DataFrame containing time series data.
    col_name (str): Column name to analyze (default is 'adj_price').

    Returns:
    float: Minimum differencing order (d) for which the series is stationary; If none, returns 1 .
    """
    d_found = False

    out = pd.DataFrame(columns=["adfStat", "pVal", "lags", "nObs", "95% conf", "corr"])

    # Assuming 'Close' is the column name in df
    for d in np.linspace(0, 1, 21):
        df1 = np.log(
            input_df[[col_name]]
        )  # .resample('1D').last()  # Downcast to daily observations
        df2 = frac_diff_ffd(df1, d, thres=0.01)
        corr = np.corrcoef(df1.loc[df2.index, col_name], df2[col_name])[0, 1]
        adf_result = adfuller(df2[col_name], maxlag=1, regression="c", autolag=None)
        out.loc[d] = (
            list(adf_result[:4]) + [adf_result[4]["5%"]] + [corr]
        )  # With critical value

        if not d_found and adf_result[1] < 0.05:
            d_found = True
            d_req = d
            return d_req
    return 1


# Structural Breaks


# Snippet 17.1: SADF's inner loop
def get_bsadf(log_p, min_sl, constant, lags):
    """
    Calculate the bootstrapped ADF statistic (bsadf) for a given log price series.

    Parameters:
    log_p (DataFrame): Logarithm of the price series to analyze.
    min_sl (int): Minimum sample length for ADF computation.
    constant (bool): Indicates if a constant term should be included in the regression.
    lags (int): Number of lags to include in the regression.

    Returns:
    dict: A dictionary containing the date of the last observation and
    the maximum bootstrapped ADF statistic ('bsadf').
    """
    y, x = get_y_x(log_p, constant=constant, lags=lags)
    start_points = range(0, y.shape[0] + lags - min_sl + 1)
    bsadf = None
    all_adf = []

    # Loop through the time series and compute ADF statistics
    for start in start_points:
        y_, x_ = y[start:], x[start:]
        b_mean_, b_std_ = get_betas(y_, x_)
        b_mean_, b_std_ = b_mean_[0, 0], b_std_[0, 0] ** 0.5
        all_adf.append(b_mean_ / b_std_)  # ADF statistic

        # Update bsadf with the maximum ADF value
        if bsadf is None or all_adf[-1] > bsadf:
            bsadf = all_adf[-1]

    # Return result as a dictionary with the final bsadf
    out = {"date": log_p.index[-1], "bsadf": bsadf}
    return out


# Snippet 17.2: Preparing the datasets
def get_y_x(series, constant, lags):
    """
    Prepare the dependent and independent variables for regression
    analysis by applying differencing and lags.

    Parameters:
    series (Series): The input time series to process.
    constant (str): Specifies the inclusion of constant or trend terms
                    ('nc' for no constant, 'c' for constant, 'ct' for constant
                    and linear trend, 'ctt' for constant and quadratic trend).
    lags (int): The number of lags to apply to the differenced series.

    Returns:
    tuple: A tuple containing:
        - y (ndarray): The dependent variable (differenced original series).
        - x (ndarray): The independent variable matrix (lagged differenced series
                       with constant/trend if specified).
    """
    # Compute differences and apply lags
    series_ = series.diff().dropna()
    x = lag_df(series_, lags).dropna()

    seriesdf = series.to_frame()
    series_df = series_.to_frame()

    # Set the lagged level of the original series as the first column
    x.iloc[:, 0] = seriesdf.values[-x.shape[0] - 1 : -1, 0]
    y = series_df.iloc[-x.shape[0] :].values

    # Add constant or time trend components if needed
    if constant != "nc":  # Add a constant if necessary
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
    if constant[:2] == "ct":  # Add a linear time trend if necessary
        trend = np.arange(x.shape[0]).reshape(-1, 1)
        x = np.append(x, trend, axis=1)
    if constant == "ctt":  # Add a second-degree polynomial time trend if necessary
        x = np.append(x, trend**2, axis=1)

    return y, x


# Snippet 17.3: Apply lags to a dataframe
def lag_df(df0, lags):
    """
    Create a DataFrame with lagged versions of the input DataFrame or Series.

    Parameters:
    df0 (DataFrame or Series): The input data for which lags are to be computed.
    lags (int or list): The number of lags to apply. If an integer is provided,
                        lags from 0 to the specified number are created.
                        If a list is provided, specific lags will be
                        generated based on the list values.

    Returns:
    DataFrame: A DataFrame with lagged versions of the input data,
    where each column is named using the original column name followed by the lag value.
    """
    df0_ = df0.to_frame()
    df1 = pd.DataFrame()

    if isinstance(lags, int):
        lags = range(lags + 1)
    else:
        lags = [int(lag) for lag in lags]

    # Apply lags to the dataframe
    for lag in lags:
        df_ = df0_.shift(lag).copy(deep=True)

        # If df_ is a Series, convert it to a DataFrame
        if isinstance(df_, pd.Series):
            df_ = df_.to_frame()

        df_.columns = [str(i) + "_" + str(lag) for i in df_.columns]
        df1 = df1.join(df_, how="outer")

    return df1


# Snippet 17.4: Fitting the ADF specification (regression)
def get_betas(y, x):
    """
    Calculate the regression coefficients and their variance using the normal equations.

    Parameters:
    y (ndarray): The dependent variable (response) values, shape (n_samples,).
    x (ndarray): The independent variable (predictor) values, shape (n_samples, n_features).

    Returns:
    tuple: A tuple containing:
        - bMean (ndarray): The estimated regression coefficients.
        - bVar (ndarray): The variance-covariance matrix of the estimated coefficients.
    """
    # Perform the regression using the normal equations
    xy = np.dot(x.T, y)
    xx = np.dot(x.T, x)
    xxinv = np.linalg.inv(xx)
    b_mean = np.dot(xxinv, xy)

    # Calculate the variance of the betas
    err = y - np.dot(x, b_mean)
    b_var = np.dot(err.T, err) / (x.shape[0] - x.shape[1]) * xxinv

    return b_mean, b_var


# Outer loop for SADF computation
def get_sadf(log_p, min_sl, constant, lags):
    """
    Compute the Supremum Augmented Dickey-Fuller (SADF) statistic for a given time series.

    Parameters:
    log_p (DataFrame): A DataFrame containing the log-transformed price series.
    min_sl (int): The minimum sample length required for computing the SADF.
    constant (str): Specifies the inclusion of a constant or
                    trend in the regression ('nc', 'c', 'ct', 'ctt').
    lags (int or list): The number of lags to include in the regression model.

    Returns:
    DataFrame: A DataFrame with the SADF values and corresponding dates,
               indexed from min_sl onwards.
    """
    gsadf_values = []

    # Iterate over the time series and compute bsadf for each window
    for window_end in range(min_sl, len(log_p)):
        window_log_p = log_p.iloc[: window_end + 1]  # Define the current window
        bsadf_result = get_bsadf(window_log_p, min_sl, constant, lags)
        gsadf_values.append(bsadf_result["bsadf"])

    # Return a DataFrame with sadf values and corresponding time index
    return pd.DataFrame({"date": log_p.index[min_sl:], "sadf": gsadf_values})


def getTimeDecay(tW, clfLastW=1.0):
    # Apply piecewise-linear decay to observed uniqueness (tW)
    # Newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW = pd.Series(tW).sort_index().cumsum()
    # len(clfW)
    if clfLastW >= 0:
        slope = (1.0 - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1.0 / ((clfLastW + 1) * clfW.iloc[-1])

    const = 1.0 - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW < 0] = 0

    # print(const, slope)
    return clfW


total = len(csv_files)


# Add 't1_index' which is the first business day of the current month
# and 't1' which is the last business day of the current month.

print("Adding 't1' and 't1_index', 'year', 'month' columns to each file...\n")
with tqdm(total=total) as pbar:
    for file_name in csv_files:
        file_path = os.path.join(STOCKS_DATA_DIR, file_name)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, index_col="t1", parse_dates=True)

        # df['t1']=df.index

        # 't1_index' which is the first business day of the current month
        df["t1_index"] = df.index - BMonthBegin(1)

        df.to_csv(file_path)

        pbar.update(1)
print("\n")


# Adjusted Price
# adj_price(t+1) = adj_price(t) * (1 + stock_exret(t) + rf(t))

print("Adding 'adj_price' column to each file...\n")
with tqdm(total=total) as pbar:
    for file_name in csv_files:
        file_path = os.path.join(STOCKS_DATA_DIR, file_name)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, index_col="t1", parse_dates=True)

        # Calculate the total return
        df["total_return"] = df["stock_exret"] + df["rf"]
        # lag the total return
        df["total_return"] = df["total_return"].shift(1)
        df["total_return"].iloc[0] = 0

        # Start the adjusted price at the initial 'prc' value
        # initial_price = df['prc'].iloc[0]
        initial_price = 100

        # Compute the adjusted price
        df["adj_price"] = initial_price * (1 + df["total_return"]).cumprod()

        # Save the updated DataFrame back to the CSV file
        df.to_csv(file_path)

        pbar.update(1)
print("\n")


# Add log price and log-diff columns to each stock CSV file.
# For the first row, the log-diff is set to 0.
# Return attribution weight as the absolute value of the stock_exret.

print("Adding 'log_price', 'log_diff' and 'weight_attr' columns to each file...\n")
with tqdm(total=total) as pbar:
    for file_name in csv_files:
        file_path = os.path.join(STOCKS_DATA_DIR, file_name)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, index_col="t1", parse_dates=True)
        df["log_price"] = df["adj_price"].apply(lambda x: 0 if x == 0 else np.log(x))

        # Calculate the log-diff column
        df["log_diff"] = df["log_price"].diff()
        df["log_diff"].iloc[0] = 0

        # Before training, needs to be scaled with
        # *= X_train.shape[0]/X_train['weight_attr'].sum()
        # df['weight_attr'] = df['stock_exret'].abs()
        df["weight_attr"] = df["alpha"].abs()
        # fill NaN values with 0
        df["weight_attr"] = df["weight_attr"].fillna(0)

        # Save the updated DataFrame back to the CSV file (or to a new file)
        df.to_csv(file_path)

        pbar.update(1)
print("\n")


# Add fractionally differentiated price as a feature
print("Adding 'frac_diff' column to each file...\n")
with tqdm(total=total) as pbar:
    for file_name in csv_files:
        file_path = os.path.join(STOCKS_DATA_DIR, file_name)

        df = pd.read_csv(file_path, index_col="t1", parse_dates=True)
        d_frac = find_min_ffd(df)

        frac_diff = frac_diff_ffd(df[["log_price"]], d=d_frac, thres=0.01)
        df["frac_diff"] = frac_diff
        df["frac_diff"] = df["frac_diff"].fillna(0)
        df.to_csv(file_path)

        pbar.update(1)
print("\n")


# Add structural breaks
print("Adding 'sadf' column to each file...\n")
with tqdm(total=total) as pbar:
    for file_name in csv_files:
        file_path = os.path.join(STOCKS_DATA_DIR, file_name)

        df = pd.read_csv(file_path, index_col="t1", parse_dates=True)
        sadf = get_sadf(df["log_price"], 20, "ct", 1)

        # Merge SADF values with the original DataFrame
        # Assuming 'sadf_result' is a DataFrame with 'Date' as the index
        df_with_sadf = df.join(sadf.set_index("date"), on="t1")
        df_with_sadf["sadf"] = df_with_sadf["sadf"].fillna(0)
        df_with_sadf.to_csv(file_path)

        pbar.update(1)
print("\n")


# Fill missing values with the previous value
# and remaining NaNs with 1e6.
print("Forward-fill NaN values and replacing any remaining NaN values with 1e6...\n")
with tqdm(total=total) as pbar:
    for file_name in csv_files:
        file_path = os.path.join(STOCKS_DATA_DIR, file_name)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, index_col="t1", parse_dates=True)

        # Fill missing values with the previous value
        df.fillna(method="ffill", axis=0, inplace=True)
        # backfill
        # df.fillna(method='bfill', axis=0, inplace=True)

        # Fill the remaining NaNs with 1_000_000
        # df.fillna(1_000_000, inplace=True)

        # Save the updated DataFrame back to the CSV file
        df.to_csv(file_path)

        pbar.update(1)
print("\n")

# Generate final datasets with all added features
# Stack all the CSV files into one DataFrame

print("Adding 'random column of range 1 to 100' column to each file...\n")
with tqdm(total=total) as pbar:
    for file_name in csv_files:
        file_path = os.path.join(STOCKS_DATA_DIR, file_name)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, index_col="t1", parse_dates=True)

        df["random"] = np.random.randint(1, 101, size=len(df))

        # Save the updated DataFrame back to the CSV file
        df.to_csv(file_path)

        pbar.update(1)
print("\n")


print("Adding time decay to the ...\n")
with tqdm(total=total) as pbar:
    for file_name in csv_files:
        file_path = os.path.join(STOCKS_DATA_DIR, file_name)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, index_col="t1", parse_dates=True)

        start_date = df.index.min()

        # df['days_since_start'] = (df.index - start_date).days
        time_window = (df.index - start_date).days
        # print('Time window : ', time_window.tolist())
        y = getTimeDecay(time_window.tolist(), clfLastW=0)

        y.index = df.index

        df["clfw"] = y

        # print(df['clfw'])
        # Save the updated DataFrame back to the CSV file
        df.to_csv(file_path)

        pbar.update(1)

dfs = []
print("Saving datasets...\n")
with tqdm(total=total) as pbar:
    for file_name in csv_files:
        file_path = os.path.join(STOCKS_DATA_DIR, file_name)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Append the DataFrame to the list
        dfs.append(df)

        pbar.update(1)
print("\n")

# Concatenate all the DataFrames in the list
# and sort by values in 't1'.
FULL_stacked_data = pd.concat(dfs, ignore_index=True)
FULL_stacked_data = FULL_stacked_data.sort_values(by="t1")

# Load relevant feature list
FEATURES_PATH = "../raw_data/factor_char_list.csv"
features = pd.read_csv(FEATURES_PATH)
features_list = features.values.ravel().tolist()

OBJECTS_DIR = "../objects"
# load factor list from object : objects/factors_list.json
with open(f"{OBJECTS_DIR}/factors_list.json", "r") as f:
    factors_list = json.load(f)
# Added features
added_features = ["log_diff", "frac_diff", "sadf"]


# Dataset creation for Causal Inference.
causal_dataset = FULL_stacked_data[features_list + ["target"]]
causal_dataset.to_csv(f"{OBJECTS_DIR}/causal_dataset.csv")

# Save to json added features, factors and features list for future use
with open(f"{OBJECTS_DIR}/added_features.json", "w") as f:
    json.dump(added_features, f)

with open(f"{OBJECTS_DIR}/features_list.json", "w") as f:
    json.dump(features_list, f)


# Scaling the new features (olf features are already scaled)
print("Scaling the new features...\n")
for date, group in FULL_stacked_data.groupby("t1"):
    # Standardize each column within the group, skipping NaNs
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(group[added_features])

    # Create a DataFrame with standardized values and maintain the original index
    standardized_df = pd.DataFrame(
        standardized_features, columns=added_features, index=group.index
    )

    # Fill NaNs with the median of each column post-standardization
    standardized_df = standardized_df.apply(lambda x: x.fillna(x.median()))

    # Assign the standardized values back to the original DataFrame
    FULL_stacked_data.loc[group.index, added_features] = standardized_df

print("\n")

X_DATASET_transformed = FULL_stacked_data[
    features_list + added_features + factors_list + ["random"]
]
# Save the standardized X_DATASET
X_DATASET_transformed.to_pickle(f"{OBJECTS_DIR}/X_DATASET.pkl")


# Y_DATASET contains all the target variables.
relevant_targets = [
    "stock_ticker",
    "stock_exret",
    "target",
    "t1",
    "t1_index",
    "weight_attr",
    "alpha",
    "market_exret",
]
Y_DATASET = FULL_stacked_data[relevant_targets]
Y_DATASET.to_pickle(f"{OBJECTS_DIR}/Y_DATASET.pkl")

# WEIGHT_SAMPLING contains all the sample weights.
WEIGHT_SAMPLING = FULL_stacked_data["weight_attr"]
WEIGHT_SAMPLING.to_pickle(f"{OBJECTS_DIR}/WEIGHT_SAMPLING.pkl")

# FULL_stacked_data contains all possible features and targets.
FULL_stacked_data.to_pickle(f"{OBJECTS_DIR}/FULL_stacked_data.pkl")
