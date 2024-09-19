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
