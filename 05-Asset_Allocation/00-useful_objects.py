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

# %%

# Create a dataframe of prices for all stocks
# Save it in \objects folder as prices.pkl

# start with APPL since it has all the datetime rows
path = '../stocks_data/AAPL.csv'
df = pd.read_csv(path)
prices = pd.DataFrame()
prices.index = pd.to_datetime(df['t1'])

# Loop through each CSV file
for file_name in tqdm(csv_files):
    file_path = os.path.join(stocks_data_dir, file_name)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, index_col='t1', parse_dates=True)
    
    # Add the 'adj_price' column to the prices DataFrame
    prices[file_name] = df['adj_price']
    
# Save the prices DataFrame
with open('../objects/prices.pkl', 'wb') as f:
    pickle.dump(prices, f)





# %%

# create a dataframe of signals for all stocks
# Save it in \objects folder as signals.pkl

# start with APPL since it has all the datetime rows
path = '../stocks_data/AAPL.csv'
df = pd.read_csv(path)
signals = pd.DataFrame()
signals.index = pd.to_datetime(df['t1'])

# Loop through each CSV file
for file_name in tqdm(csv_files):
    file_path = os.path.join(stocks_data_dir, file_name)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, index_col='t1', parse_dates=True)
    
    # Add the 'signal' column to the signals DataFrame
    signals[file_name] = df['prediction']*df['probability']
    
# Save the signals DataFrame
with open('../objects/signals.pkl', 'wb') as f:
    pickle.dump(signals, f)
    
#%%

# create a dataframe of market capitalizations for all stocks
# Save it in \objects folder as market_caps.pkl

# start with APPL since it has all the datetime rows
path = '../stocks_data/AAPL.csv'
df = pd.read_csv(path)
market_caps = pd.DataFrame()
market_caps.index = pd.to_datetime(df['t1'])

# Loop through each CSV file
for file_name in tqdm(csv_files):
    file_path = os.path.join(stocks_data_dir, file_name)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, index_col='t1', parse_dates=True)
    
    # Add the 'market_cap' column to the market_caps DataFrame
    market_caps[file_name] = df['market_equity']
    
# Save the market_caps DataFrame
with open('../objects/market_caps.pkl', 'wb') as f:
    pickle.dump(market_caps, f)
# %%

# create a dataframe of stockexret for all stocks
# Save it in \objects folder as stockexret.pkl

# start with APPL since it has all the datetime rows
path = '../stocks_data/AAPL.csv'
df = pd.read_csv(path)
stockexret = pd.DataFrame()
stockexret.index = pd.to_datetime(df['t1'])

# Loop through each CSV file
for file_name in tqdm(csv_files):
    file_path = os.path.join(stocks_data_dir, file_name)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, index_col='t1', parse_dates=True)
    
    # Add the 'stockexret' column to the stockexret DataFrame
    stockexret[file_name] = df['stock_exret']
    
# Save the stockexret DataFrame
with open('../objects/stockexret.pkl', 'wb') as f:
    pickle.dump(stockexret, f)
# %%
