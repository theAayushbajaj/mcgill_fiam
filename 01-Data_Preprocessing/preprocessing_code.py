"""
This script preprocesses stock data by cleaning and splitting it into individual stock CSV files.
Additionally, it processes market indicator data and saves it as a separate CSV file.
"""

import os
import warnings
import pandas as pd
from pandas.tseries.offsets import BMonthEnd, BMonthBegin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings('ignore')

# Set the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Ensure the 'stocks_data' directory exists, create it if not present
STOCKS_DATA_DIR = "../stocks_data"

if not os.path.exists(STOCKS_DATA_DIR):
    os.makedirs(STOCKS_DATA_DIR)

# Ensure the 'objects' directory exists, create it if not present
OBJECTS_DIR = "../objects"

if not os.path.exists(OBJECTS_DIR):
    os.makedirs(OBJECTS_DIR)
    
# Load benchmark data from a CSV file
market_data = pd.read_csv("../raw_data/mkt_ind.csv")
try:
    market_data['market_exret'] = market_data['sp_ret'] - market_data['RF']
except:
    market_data['market_exret'] = market_data['sp_ret'] - market_data['rf']


# Load the stock data from a CSV file
data = pd.read_csv("../raw_data/hackathon_sample_v2.csv")

# Convert 'date' column to a 't1' datetime column with the appropriate format
data["t1"] = pd.to_datetime(data["date"], format="%Y%m%d")

# Adding the alpha column to dataset
for date, group in data.groupby('t1'):
    # fill NAs in beta_60m with median
    group['beta_60m'] = group['beta_60m'].fillna(group['beta_60m'].median())
    data.loc[group.index, 'beta_60m'] = group['beta_60m']
data = pd.merge(data, market_data[['market_exret', 'month', 'year']], on=['year', 'month'], how='left')
data['alpha'] = data['stock_exret'] - data['beta_60m'] * data['market_exret']
data['target'] = data['alpha'].apply(lambda x: 1 if x >= 0 else -1)

# Rescale features
FEATURES_PATH = '../raw_data/factor_char_list.csv'
features = pd.read_csv(FEATURES_PATH)
features_list = features.values.ravel().tolist()
data_scaled = pd.DataFrame()
for date, group in data.groupby('t1'):
    # Standardize each column within the group, skipping NaNs
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(group[features_list])
    
    # Create a DataFrame with standardized values and maintain the original index
    standardized_df = pd.DataFrame(standardized_features, columns=features_list, index=group.index)
    
    # Fill NaNs with the median of each column post-standardization
    standardized_df = standardized_df.apply(lambda x: x.fillna(x.median()))
    
    # Append the standardized DataFrame to the scaled data
    data_scaled = pd.concat([data_scaled, standardized_df])
    
# Assign the scaled features back to the original DataFrame
data[features_list] = data_scaled[features_list]
    


# Define the minimum number of records required for saving
MIN_RECORDS = 120

def apply_last_stock_ticker(group):
    """
    Processes each stock ticker group by sorting records by year and month.
    Ensures the group has at least the minimum required records before saving it to a CSV file.

    Args:
        group (DataFrame): A group of stock records corresponding to a specific stock ticker.

    Returns:
        DataFrame or None: Returns the processed group if valid, otherwise None.
    """

    if len(group) >= MIN_RECORDS:
        # Sort the group by 'year' and 'month' for chronological order
        group = group.sort_values(by=["year", "month"])

        # Assign the last stock ticker to all rows in the group
        last_ticker = group["stock_ticker"].iloc[-1]
        group["stock_ticker"] = last_ticker

        # Save the group as a CSV file using the format '{ticker}.csv'
        file_path = os.path.join(STOCKS_DATA_DIR, f"{last_ticker}.csv")
        group.to_csv(file_path, index=False)

        return group

    return None

# Group the stock data by 'cusip' and 'permno',
# without adding group labels to the index
grouped = data.groupby(["cusip", "permno"], group_keys=False)

# Apply the `apply_last_stock_ticker` function to
# each group to process and save them
grouped.apply(apply_last_stock_ticker)

# Load the market indicator data from a CSV file
mkt_ind = pd.read_csv("../raw_data/mkt_ind.csv")

# Create a 't1' column as the last business day of the specified year and month
mkt_ind["t1"] = pd.to_datetime(mkt_ind[["year", "month"]].assign(day=1)) + BMonthEnd(1)

# Create a 't1_index' column as the first business day of the same period
mkt_ind["t1_index"] = mkt_ind["t1"] - BMonthBegin(1)

# Save the processed market indicator DataFrame as a CSV file
mkt_ind.to_csv("../objects/mkt_ind.csv", index=False)
