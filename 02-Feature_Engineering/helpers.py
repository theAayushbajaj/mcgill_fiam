import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
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