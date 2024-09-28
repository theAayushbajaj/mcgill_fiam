#%%
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import pickle
import warnings
warnings.filterwarnings('ignore')

import sys
import os
#%%
# Split the whole dataset into individual stock CSV files

# set the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv('../raw_data/hackathon_sample_v2.csv')

# add a new column 'datetime' with the date in datetime format
data['datetime'] = pd.to_datetime(data['date'], format='%Y%m%d')

# Define the minimum number of records required
min_records = 180

# Group by 'cusip' and 'permno', and order each group by 'year' and 'month'
grouped = data.groupby(['cusip', 'permno'], group_keys=False)

# Function to apply to each group: sort by 'year' and 'month', then assign the last stock_ticker
def apply_last_stock_ticker(group):
    # Check if the group size meets the minimum record requirement
    if len(group) >= min_records:
        # Sort by 'year' and 'month'
        group = group.sort_values(by=['year', 'month'])
        # Get the last stock_ticker used and assign it to all rows in the group
        last_ticker = group['stock_ticker'].iloc[-1]
        group['stock_ticker'] = last_ticker
        # Save the group as a CSV file using the format 'cusip_permno.csv'
        # file_name = f"../stocks_data/{str(last_ticker)}_{str(group['cusip'].iloc[0])}_{str(group['permno'].iloc[0])}.csv"
        file_name = f"../stocks_data/{str(last_ticker)}.csv"
        group.to_csv(file_name, index=False)
        return group

# Apply the function to each group
result = grouped.apply(apply_last_stock_ticker)




#%%
# Add the 'target' column to each stock CSV file

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

# Create t1 object, save it in \objects folder as t1.pkl

# APPL stock has all the datetime rows
path = '../stocks_data/AAPL.csv'
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

# Create a dataframe of prices for all stocks
# Save it in \objects folder as prices.pkl

# start with APPL since it has all the datetime rows
path = '../stocks_data/AAPL.csv'
df = pd.read_csv(path)
prices = pd.DataFrame()
prices.index = pd.to_datetime(df['datetime'])

# Loop through each CSV file
for file_name in tqdm(csv_files):
    file_path = os.path.join(stocks_data_dir, file_name)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    
    # Add the 'adj_price' column to the prices DataFrame
    prices[file_name] = df['adj_price']
    
# Save the prices DataFrame
with open('../objects/prices.pkl', 'wb') as f:
    pickle.dump(prices, f)





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
    df.to_csv(file_path)#%%


#%%

# Stack all the CSV files into one DataFrame

# Create an empty list to store the DataFrames
dfs = []

# Loop through each CSV file
for file_name in tqdm(csv_files):
    file_path = os.path.join(stocks_data_dir, file_name)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Append the DataFrame to the list
    dfs.append(df)
    
# Concatenate all the DataFrames in the list
stacked_data = pd.concat(dfs, ignore_index=True)
stacked_data = stacked_data.sort_values(by='datetime')

stacked_data
# save the stacked data as pickle
stacked_data.to_pickle('../objects/stacked_data.pkl')
# %%

# features
path = '../raw_data/factor_char_list.csv'
features = pd.read_csv(path)
features_list = features.values.ravel().tolist()
# Add created features
# added_features = ['log_diff', 'frac_diff']
# features_list+=added_features
X_dataset = stacked_data[features_list]
# to pickle
X_dataset.to_pickle('../objects/X_dataset.pkl')
y_dataset = stacked_data['target']
# %%


# %%