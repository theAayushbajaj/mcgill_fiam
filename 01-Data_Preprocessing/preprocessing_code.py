# %%
"""
This script preprocesses stock data by cleaning and splitting it into individual stock CSV files.
Additionally, it processes market indicator data and saves it as a separate CSV file.
"""

import os
import warnings
import pandas as pd
import json
from pandas.tseries.offsets import BMonthEnd, BMonthBegin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

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

FEATURES_PATH = "../raw_data/factor_char_list.csv"

FACTOR_PATH = "factors_theme.json"
with open(FACTOR_PATH, "r") as f:
    factor_themes = json.load(f)
factors_list = list(factor_themes.keys())
with open(f"{OBJECTS_DIR}/factors_list.json", "w") as f:
    json.dump(factors_list, f)


DATA_DIR = "../raw_data/hackathon_sample_v2.csv"
MARKET_DATA_DIR = "../raw_data/mkt_ind.csv"

# Load benchmark data from a CSV file
market_data = pd.read_csv(MARKET_DATA_DIR)
try:
    market_data["market_exret"] = market_data["sp_ret"] - market_data["RF"]
except KeyError:
    market_data["market_exret"] = market_data["sp_ret"] - market_data["rf"]

# Load the stock data from a CSV file
data = pd.read_csv(DATA_DIR)

# Convert 'date' column to a 't1' datetime column with the appropriate format
data["t1"] = pd.to_datetime(data["date"], format="%Y%m%d")


# Adding Theme Factors
def addFactorThemes(df, factor_themes):
    """
    Add a new column for each factor theme based on the JSON file containing theme definitions.
    Each theme aggregates factors based on signs provided in the JSON.

    Parameters:
    df (DataFrame): The input DataFrame containing factor columns.
    factor_path (str): Path to the JSON file with theme definitions.

    Returns:
    DataFrame: The DataFrame with new columns for each factor theme.
    """
    # Loop through each theme and aggregate factors based on signs
    for theme_name, factors in factor_themes.items():
        theme_sum = pd.Series(0, index=df.index)
        for factor in factors:
            # Multiply the factor values by the sign and add to theme sum
            factor_name = factor["name"]
            factor_sign = factor["sign"]
            if factor_name in df.columns:
                theme_sum += df[factor_name] * factor_sign
            else:
                print(
                    f"Warning: Factor '{factor_name}' not found in DataFrame columns."
                )

        # Add the aggregated theme column to the DataFrame
        df[theme_name] = theme_sum

    return df


# factors
# scale the features min max
# apply on these the addFactorThemes
data_factors = data.copy()
features = pd.read_csv(FEATURES_PATH)
features_list = features.values.ravel().tolist()


def minmax_fillmedian(df):
    minmax_scaler = MinMaxScaler()
    df[features_list] = minmax_scaler.fit_transform(df[features_list])
    df[features_list] = df[features_list].apply(lambda x: x.fillna(x.median()))
    return df


data_factors = data_factors.groupby("t1", group_keys=False).apply(minmax_fillmedian)
data_factors = data_factors.groupby(["cusip", "permno"], group_keys=False).apply(
    addFactorThemes, factor_themes=factor_themes
)

data[factors_list] = data_factors[factors_list]

# Fill NAs in beta_60m with the median value within each 't1' group
data["beta_60m"] = data.groupby("t1")["beta_60m"].transform(
    lambda x: x.fillna(x.median())
)

# Merge with market data
data = pd.merge(
    data,
    market_data[["market_exret", "month", "year"]],
    on=["year", "month"],
    how="left",
)

# Calculate alpha and target
data["alpha"] = data["stock_exret"] - data["beta_60m"] * data["market_exret"]
data["target"] = data["alpha"].apply(lambda x: 1 if x >= 0 else -1)
data.head()
# %%

# Rescale features


def customStandardize(df):
    """
    Standardize features_list, minmax scale factors_list
    """
    scaler = StandardScaler()
    df[features_list] = scaler.fit_transform(df[features_list])
    minmax_scaler = MinMaxScaler()
    df[factors_list] = minmax_scaler.fit_transform(df[factors_list])
    df[features_list + factors_list] = df[features_list + factors_list].apply(
        lambda x: x.fillna(x.median())
    )
    return df


data = data.groupby("t1", group_keys=False).apply(customStandardize)
data.head()
# %%

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

# %%
