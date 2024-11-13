"""
This script processes stock data from multiple CSV files located in the specified
directory and compiles various DataFrames for analysis. It generates four primary
DataFrames: `prices`, `signals`, `market_caps`, and `stockexret`, which are
subsequently saved as pickle files in the `objects` folder.

Key functionalities:
- Loads adjusted price data from CSV files and stores it in a DataFrame.
- Computes trading signals as the product of the 'prediction' and 'probability' columns
  from each stock's data and saves the resulting DataFrame.
- Gathers market capitalization data from each stock's data and compiles it into a
  DataFrame.
- Retrieves stock excess return data and compiles it into a DataFrame.
- Saves all compiled DataFrames as pickle files for later use.

Usage:
1. Set the current working directory to the script's location.
2. Specify the directory containing the stock CSV files.
3. Execute the script to create and save the DataFrames in the `objects` folder.
"""


import os
import pickle
import pandas as pd
from tqdm import tqdm
import json

# set the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set the directory containing your stock CSV files
STOCKS_DATA_DIR = '../stocks_data'

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(STOCKS_DATA_DIR) if f.endswith('.csv')]


# Create a dataframe of prices for all stocks
# Save it in \objects folder as prices.pkl

# Load AAPL.csv since it has all the datetime indices
try:
    PATH = '../stocks_data/CNMD.csv'
    aapl_df = pd.read_csv(PATH)
except:
    PATH = '../stocks_data/AAPL.csv'
    aapl_df = pd.read_csv(PATH)

# Initialize the datetime indices for `prices` dataframe from AAPL.csv
prices = pd.DataFrame()
prices.index = pd.to_datetime(aapl_df['t1'])

for file_name in tqdm(csv_files):
    file_path = os.path.join(STOCKS_DATA_DIR, file_name)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, index_col='t1', parse_dates=True)

    # Add the 'adj_price' column to the prices DataFrame
    prices[file_name] = df['adj_price']

# Save the prices DataFrame
with open('../objects/prices.pkl', 'wb') as f:
    pickle.dump(prices, f)


# Create a dataframe of signals for all stocks
# Save it in \objects folder as signals.pkl

# Initialize the datetime indices for `signals` dataframe from AAPL.csv
signals = pd.DataFrame()
signals.index = pd.to_datetime(aapl_df['t1'])

for file_name in tqdm(csv_files):
    file_path = os.path.join(STOCKS_DATA_DIR, file_name)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, index_col='t1', parse_dates=True)

    # Add the 'signal' column to the signals DataFrame
    signals[file_name] = df['prediction']*df['probability']

# Save the signals DataFrame
with open('../objects/signals.pkl', 'wb') as f:
    pickle.dump(signals, f)
    
# factor signals
Factor_signals = dict()
OBJECTS_DIR = "../objects"
# load factor list from object : objects/factors_list.json
with open(f"{OBJECTS_DIR}/factors_list.json", "r") as f:
    factors_list = json.load(f)
    
for factor in factors_list:
    Factor_signals[factor] = pd.DataFrame()
    Factor_signals[factor].index = pd.to_datetime(aapl_df['t1'])
    
    for file_name in tqdm(csv_files):
        file_path = os.path.join(STOCKS_DATA_DIR, file_name)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, index_col='t1', parse_dates=True)

        # Add the 'signal' column to the signals DataFrame
        Factor_signals[factor][file_name] = df[factor]
        
# Save the dictionary of factor signals
with open(f"{OBJECTS_DIR}/Factor_signals.pkl", "wb") as f:
    pickle.dump(Factor_signals, f)


# Create a dataframe of market capitalizations for all stocks
# Save it in \objects folder as market_caps.pkl

# Initialize the datetime indices for `market_caps` dataframe from AAPL.csv
market_caps = pd.DataFrame()
market_caps.index = pd.to_datetime(aapl_df['t1'])

for file_name in tqdm(csv_files):
    file_path = os.path.join(STOCKS_DATA_DIR, file_name)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, index_col='t1', parse_dates=True)

    # Add the 'market_cap' column to the market_caps DataFrame
    market_caps[file_name] = df['market_equity']

# Save the market_caps DataFrame
with open('../objects/market_caps.pkl', 'wb') as f:
    pickle.dump(market_caps, f)


# Create a dataframe of stockexret for all stocks
# Save it in \objects folder as stockexret.pkl

# Initialize the datetime indices for `stock_exret` dataframe from AAPL.csv
stockexret = pd.DataFrame()
stockexret.index = pd.to_datetime(aapl_df['t1'])

for file_name in tqdm(csv_files):
    file_path = os.path.join(STOCKS_DATA_DIR, file_name)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, index_col='t1', parse_dates=True)

    # Add the 'stockexret' column to the stockexret DataFrame
    stockexret[file_name] = df['stock_exret']

# Save the stockexret DataFrame
with open('../objects/stockexret.pkl', 'wb') as f:
    pickle.dump(stockexret, f)
