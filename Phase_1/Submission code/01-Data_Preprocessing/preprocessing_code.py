"""
This script preprocesses stock data by cleaning and splitting it into individual stock CSV files.
Additionally, it processes market indicator data and saves it as a separate CSV file.
"""

import os
import warnings
import pandas as pd
from pandas.tseries.offsets import BMonthEnd, BMonthBegin

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

# Load the stock data from a CSV file
data = pd.read_csv("../raw_data/hackathon_sample_v2.csv")

# Convert 'date' column to a 't1' datetime column with the appropriate format
data["t1"] = pd.to_datetime(data["date"], format="%Y%m%d")

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
