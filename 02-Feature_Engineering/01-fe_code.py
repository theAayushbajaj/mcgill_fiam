#%%
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import pickle



#%%
# Add the 'target' column to each stock CSV file

# set the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set the directory containing your stock CSV files
stocks_data_dir = '../stocks_data'

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(stocks_data_dir) if f.endswith('.csv')]

#%%
# ADDING TARGET COLUMN TO EACH CSV FILE

# Loop through each CSV file
for file_name in csv_files:
    file_path = os.path.join(stocks_data_dir, file_name)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Check if 'stock_exret' column exists
    if 'stock_exret' in df.columns:
        # Create the 'target' column as the sign of 'stock_exret'
        df['target'] = df['stock_exret'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        # Save the updated DataFrame back to the CSV file (or to a new file)
        df.to_csv(file_path, index=False)
    else:
        print(f"'stock_exret' column not found in {file_name}")

# %%

# Add Weight Sampling
# NOO, WILL BE IN THE PIPELINE OF THE MODEL, SINCE IT IS TESTED

# %%

# Create t1 object, save it in \objects folder as t1.pkl

# APPL stock has all the datetime rows
path = '../stocks_data/AAPL_03783310_14593.csv'
df = pd.read_csv(path)
t1 = pd.Series()
tmp = df['datetime'].shift(-1).dropna()
tmp = pd.to_datetime(tmp)
# last date
result = tmp.iloc[-1] + pd.DateOffset(days=5) + pd.tseries.offsets.BMonthEnd(1)
tmp = pd.concat([tmp, pd.Series([result])], ignore_index=True)
t1 = tmp
# index as first business day of the following month
t1.index = pd.to_datetime(df['datetime']) + pd.DateOffset(days=5) - pd.tseries.offsets.BMonthBegin(1)
# save t1
with open('../objects/t1.pkl', 'wb') as f:
    pickle.dump(t1, f)
    
#%%

# Adjusted Price
# Start the adjusted price at 100
# adj_price(t+1) = adj_price(t) * (1 + stock_exret(t) + rf(t))

# Loop through each CSV file
for file_name in csv_files:
    file_path = os.path.join(stocks_data_dir, file_name)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    
    # Calculate the total return
    df['total_return'] = df['stock_exret'] + df['rf']
    # lag the total return
    df['total_return'] = df['total_return'].shift(1)
    df['total_return'].iloc[0] = 0
    
    # Start the adjusted price at the initial 'prc' value
    initial_price = df['prc'].iloc[0]
    
    # Compute the adjusted price
    df['adj_price'] = initial_price * (1 + df['total_return']).cumprod()
    
    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path)


# %%

# Add log price and log-diff columns to each stock CSV file
# For the first row, the log-diff is set to 0 (TO BE CHECKED) TODO

# return attribution weight as the absolute value of the stock_exret

# Loop through each CSV file
for file_name in csv_files:
    file_path = os.path.join(stocks_data_dir, file_name)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    df['log_price'] = df['adj_price'].apply(lambda x: 0 if x == 0 else np.log(x))
    
    # Calculate the log-diff column
    df['log_diff'] = df['log_price'].diff()
    df['log_diff'].iloc[0] = 0
    
    # Before training, needs to be scaled with 
    # *= X_train.shape[0]/X_train['weight_attr'].sum()
    df['weight_attr'] = df['stock_exret'].abs()
    
    # Save the updated DataFrame back to the CSV file (or to a new file)
    df.to_csv(file_path)
# %%

# Fractionally Differentiated Price as a Feature

# SNIPPET 5.3 THE NEW FIXED-WIDTH WINDOW FRACDIFF METHOD
def fracDiff_FFD(series, d, thres=1e-5):
    '''
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    #1) Compute weights for the longest series
    w = getWeights_FFD(d, thres)
    width = len(w) - 1
    #2) Apply weights to values
    df = {}
    for name in series.columns:
        # Corrected the fill method to 'ffill'
        seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1-width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue  # exclude NAs
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def getWeights_FFD(d, thres):
    # Appends while the last weight is above the threshold
    w = [1.]
    while abs(w[-1]) > thres:
        w_ = -w[-1] / len(w) * (d - len(w) + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def findMinFFD(df, col_name = 'adj_price'):
    from statsmodels.tsa.stattools import adfuller
    import matplotlib.pyplot as plt
    d_found = False
    
    out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    
    # Assuming 'Close' is the column name in df
    for d in np.linspace(0, 1, 21):
        df1 = np.log(df[[col_name]])# .resample('1D').last()  # Downcast to daily observations
        df2 = fracDiff_FFD(df1, d, thres=.01)
        corr = np.corrcoef(df1.loc[df2.index, col_name], df2[col_name])[0, 1]
        adf_result = adfuller(df2[col_name], maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(adf_result[:4]) + [adf_result[4]['5%']] + [corr]  # With critical value
        # print(f'Fractional differentiation order: {d}, ADF Statistic: {adf_result[0]}, Correlation: {corr}')
        if not d_found and adf_result[1] < 0.05:
            d_found = True
            d_req = d
            return d_req
    return 1

# Loop through each CSV file
for file_name in tqdm(csv_files):
    file_path = os.path.join(stocks_data_dir, file_name)
    
    df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    d_frac = findMinFFD(df)
    col = 'log_price'
    frac_diff = fracDiff_FFD(df[[col]], d=d_frac, thres=0.01)
    df['frac_diff'] = frac_diff
    df['frac_diff'] = df['frac_diff'].fillna(0)
    df.to_csv(file_path)
    
# %%

# Missing Values
# Fill missing values with the previous value
# Loop through each CSV file

for file_name in csv_files:
    file_path = os.path.join(stocks_data_dir, file_name)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    
    # Fill missing values with the previous value
    df.fillna(method='ffill', axis=0, inplace=True)
    
    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path)
# %%

# Structural Breaks

# Snippet 17.1: SADF's inner loop
def get_bsadf(logP, minSL, constant, lags):
    y, x = getYX(logP, constant=constant, lags=lags)
    startPoints = range(0, y.shape[0] + lags - minSL + 1)
    bsadf = None
    allADF = []
    
    # Loop through the time series and compute ADF statistics
    for start in startPoints:
        y_, x_ = y[start:], x[start:]
        bMean_, bStd_ = getBetas(y_, x_)
        bMean_, bStd_ = bMean_[0, 0], bStd_[0, 0] ** 0.5
        allADF.append(bMean_ / bStd_)  # ADF statistic
        
        # Update bsadf with the maximum ADF value
        if bsadf is None or allADF[-1] > bsadf:
            bsadf = allADF[-1]
    
    # Return result as a dictionary with the final bsadf
    out = {'date': logP.index[-1], 'bsadf': bsadf}
    return out

# Snippet 17.2: Preparing the datasets
def getYX(series, constant, lags):
    # Compute differences and apply lags
    series_ = series.diff().dropna()
    x = lagDF(series_, lags).dropna()
    
    seriesdf = series.to_frame()
    series_df = series_.to_frame()
    
    # Set the lagged level of the original series as the first column
    x.iloc[:, 0] = seriesdf.values[-x.shape[0]-1:-1, 0]
    y = series_df.iloc[-x.shape[0]:].values
    
    # Add constant or time trend components if needed
    if constant != 'nc':  # Add a constant if necessary
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
    if constant[:2] == 'ct':  # Add a linear time trend if necessary
        trend = np.arange(x.shape[0]).reshape(-1, 1)
        x = np.append(x, trend, axis=1)
    if constant == 'ctt':  # Add a second-degree polynomial time trend if necessary
        x = np.append(x, trend ** 2, axis=1)
    
    return y, x

# Snippet 17.3: Apply lags to a dataframe
def lagDF(df0, lags):
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

        df_.columns = [str(i) + '_' + str(lag) for i in df_.columns]
        df1 = df1.join(df_, how='outer')
    
    return df1

# Snippet 17.4: Fitting the ADF specification (regression)
def getBetas(y, x):
    # Perform the regression using the normal equations
    xy = np.dot(x.T, y)
    xx = np.dot(x.T, x)
    xxinv = np.linalg.inv(xx)
    bMean = np.dot(xxinv, xy)
    
    # Calculate the variance of the betas
    err = y - np.dot(x, bMean)
    bVar = np.dot(err.T, err) / (x.shape[0] - x.shape[1]) * xxinv
    
    return bMean, bVar

# Outer loop for SADF computation
def get_sadf(logP, minSL, constant, lags):
    gsadf_values = []
    
    # Iterate over the time series and compute bsadf for each window
    for window_end in range(minSL, len(logP)):
        window_logP = logP.iloc[:window_end + 1]  # Define the current window
        bsadf_result = get_bsadf(window_logP, minSL, constant, lags)
        gsadf_values.append(bsadf_result['bsadf'])
    
    # Return a DataFrame with sadf values and corresponding time index
    return pd.DataFrame({'date': logP.index[minSL:], 'sadf': gsadf_values})

# Loop through each CSV file
for file_name in tqdm(csv_files):
    file_path = os.path.join(stocks_data_dir, file_name)
    
    df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    col = 'log_price'
    sadf = get_sadf(df[col], 20, 'ct', 1)
    
    # Merge SADF values with the original DataFrame
    # Assuming 'sadf_result' is a DataFrame with 'Date' as the index
    df_with_sadf = df.join(sadf.set_index('date'), on='datetime')
    df_with_sadf['sadf'] = df_with_sadf['sadf'].fillna(0)
    df_with_sadf.to_csv(file_path)

# %%