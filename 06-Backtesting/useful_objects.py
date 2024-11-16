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

OBJECTS_DIR = "../objects"

df = pd.read_csv('../objects/FULL_stacked_data_with_preds.csv', parse_dates=True)
df = df.sort_values(by="t1")

# Initialize the datetime indices for `prices` dataframe from AAPL.csv
prices = pd.DataFrame()
prices.index = pd.to_datetime(list(set(df['t1'].values)))
prices = prices.sort_index()
prices.index.name = 't1'

signals = pd.DataFrame()
signals.index = prices.index
signals.index.name = 't1'

market_caps = pd.DataFrame()
market_caps.index = prices.index
market_caps.index.name = 't1'

stockexret = pd.DataFrame()
stockexret.index = prices.index
stockexret.index.name = 't1'

# Group by 'cusip' and 'permno'
grouped = df.groupby(["cusip", "permno"], group_keys=False)

# load factor list from object : objects/factors_list.json
with open(f"{OBJECTS_DIR}/factors_list.json", "r") as f:
    factors_list = json.load(f)

factor_signals = dict()

for factor in factors_list:
    factor_signals[factor] = pd.DataFrame()
    factor_signals[factor].index = prices.index
    factor_signals[factor].index.name = 't1'

for (cusip, permno), group in tqdm(grouped, desc="Processing groups"):
    ticker = group['stock_ticker'].iloc[0]  # Get the ticker name from the group

    group['t1'] = pd.to_datetime(group['t1'])
    group.set_index('t1', inplace=True)

    # Create and store the prices DataFrame
    prices[ticker] = group[['adj_price']]

    # Create and store signals DataFrame
    signals[ticker] = group['prediction'] * group['probability']

    # Create and market_caps signals DataFrame
    market_caps[ticker] = group[['market_equity']]

    # Create and stockexret signals DataFrame
    stockexret[ticker] = group[['stock_exret']]

    for factor in factors_list:
        factor_signals[factor][ticker] = group[factor]

prices.to_csv(f'{OBJECTS_DIR}/prices.csv')
signals.to_csv(f'{OBJECTS_DIR}/signals.csv')
market_caps.to_csv(f'{OBJECTS_DIR}/market_caps.csv')
stockexret.to_csv(f'{OBJECTS_DIR}/stockexret.csv')

# Ensure the 'objects' directory exists, create it if not present
FACTOR_DIR = f'{OBJECTS_DIR}/factors'

if not os.path.exists(FACTOR_DIR):
    os.makedirs(FACTOR_DIR)

for factor in factor_signals.keys():
    factor_signals[factor].to_csv(f'{FACTOR_DIR}/{factor}.csv')
