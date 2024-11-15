"""
This script adds the following features to the dataset:
    1. 'target' --->>> 1 if excess return >= 0. Otherwise -1.
    2. 't1_index' --->>> First business day of the current month.
    3. 't1' --->>> Last business day of the current month.
    4. 'total_return' --->>> excess return + risk free rate.
    5. 'adj_price' --->>> Adjusted price that takes into account stock splits, dividends etc.
    6. 'log_price' --->>> Logarithm of 'adj_price'.
    7. 'log_diff' --->>> Difference between consecutive values of log price.
    8. 'weight_attr' --->>> Absolute value of excess return.
    9. 'frac_diff' --->>> Fractionally differentiated price as feature.
"""

import os
import warnings

import json

import pandas as pd
from pandas.tseries.offsets import BMonthBegin
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from helpers import frac_diff_ffd, find_min_ffd, get_sadf, getTimeDecay

warnings.filterwarnings("ignore")

# set the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set the directory containing your stock CSV files
STOCKS_DATA_DIR = "../stocks_data"

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(STOCKS_DATA_DIR) if f.endswith(".csv")]

total = len(csv_files)

# Add 't1_index' which is the first business day of the current month
# and 't1' which is the last business day of the current month.

print("Adding 't1_index', 'adj_price', 'log_price', 'frac_diff', 'sadf', 'random' and 'clfw' columns to each file...\n")
with tqdm(total=total) as pbar:
    for file_name in csv_files:
        file_path = os.path.join(STOCKS_DATA_DIR, file_name)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, index_col="t1", parse_dates=True)

        # df['t1']=df.index

        # 't1_index' which is the first business day of the current month
        df["t1_index"] = df.index - BMonthBegin(1)

        ## Adjusted price calculation
        # Calculate the total return
        df["total_return"] = df["stock_exret"] + df["rf"]
        # lag the total return
        df["total_return"] = df["total_return"].shift(1)
        df["total_return"].iloc[0] = 0

        # Start the adjusted price at the initial 'prc' value
        # initial_price = df['prc'].iloc[0]
        initial_price = 100

        # Compute the adjusted price
        df["adj_price"] = initial_price * (1 + df["total_return"]).cumprod()

        ## Add log price and log-diff columns to each stock CSV file.
        # For the first row, the log-diff is set to 0.
        # Return attribution weight as the absolute value of the stock_exret.
        df["log_price"] = df["adj_price"].apply(lambda x: 0 if x == 0 else np.log(x))

        # Calculate the log-diff column
        df["log_diff"] = df["log_price"].diff()
        df["log_diff"].iloc[0] = 0

        # Before training, needs to be scaled with
        # *= X_train.shape[0]/X_train['weight_attr'].sum()
        # df['weight_attr'] = df['stock_exret'].abs()
        df["weight_attr"] = df["alpha"].abs()
        # fill NaN values with 0
        df["weight_attr"] = df["weight_attr"].fillna(0)

        ## Add fractionally differentiated price as a feature
        d_frac = find_min_ffd(df)

        frac_diff = frac_diff_ffd(df[["log_price"]], d=d_frac, thres=0.01)
        df["frac_diff"] = frac_diff
        df["frac_diff"] = df["frac_diff"].fillna(0)

        ## Add structural breaks
        sadf = get_sadf(df["log_price"], 20, "ct", 1)

        # Merge SADF values with the original DataFrame
        # Assuming 'sadf_result' is a DataFrame with 'Date' as the index
        df_with_sadf = df.join(sadf.set_index("date"), on="t1")
        df_with_sadf["sadf"] = df_with_sadf["sadf"].fillna(0)
        df = df_with_sadf

        ## Fill missing values with the previous value
        df.fillna(method="ffill", axis=0, inplace=True)

        ## Add column with random integers
        df["random"] = np.random.randint(1, 101, size=len(df))

        ## Add time decay
        start_date = df.index.min()

        # df['days_since_start'] = (df.index - start_date).days
        time_window = (df.index - start_date).days
        # print('Time window : ', time_window.tolist())
        y = getTimeDecay(time_window.tolist(), clfLastW=0)

        y.index = df.index

        df["clfw"] = y

        df.to_csv(file_path)

        pbar.update(1)
print("\n")

dfs = []
print("Saving datasets...\n")
with tqdm(total=total) as pbar:
    for file_name in csv_files:
        file_path = os.path.join(STOCKS_DATA_DIR, file_name)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Append the DataFrame to the list
        dfs.append(df)

        pbar.update(1)
print("\n")

# Concatenate all the DataFrames in the list
# and sort by values in 't1'.
FULL_stacked_data = pd.concat(dfs, ignore_index=True)
FULL_stacked_data = FULL_stacked_data.sort_values(by="t1")

# Load relevant feature list
FEATURES_PATH = "../raw_data/factor_char_list.csv"
features = pd.read_csv(FEATURES_PATH)
features_list = features.values.ravel().tolist()

OBJECTS_DIR = "../objects"
# load factor list from object : objects/factors_list.json
with open(f"{OBJECTS_DIR}/factors_list.json", "r") as f:
    factors_list = json.load(f)
# Added features
added_features = ["log_diff", "frac_diff", "sadf"]


# Dataset creation for Causal Inference.
causal_dataset = FULL_stacked_data[features_list + ["target"]]
causal_dataset.to_csv(f"{OBJECTS_DIR}/causal_dataset.csv")

# Save to json added features, factors and features list for future use
with open(f"{OBJECTS_DIR}/added_features.json", "w") as f:
    json.dump(added_features, f)

with open(f"{OBJECTS_DIR}/features_list.json", "w") as f:
    json.dump(features_list, f)


# Scaling the new features (olf features are already scaled)
print("Scaling the new features...\n")
for date, group in FULL_stacked_data.groupby("t1"):
    # Standardize each column within the group, skipping NaNs
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(group[added_features])

    # Create a DataFrame with standardized values and maintain the original index
    standardized_df = pd.DataFrame(
        standardized_features, columns=added_features, index=group.index
    )

    # Fill NaNs with the median of each column post-standardization
    standardized_df = standardized_df.apply(lambda x: x.fillna(x.median()))

    # Assign the standardized values back to the original DataFrame
    FULL_stacked_data.loc[group.index, added_features] = standardized_df

print("\n")

X_DATASET_transformed = FULL_stacked_data[
    features_list + added_features + factors_list + ["random"]
]
# Save the standardized X_DATASET
X_DATASET_transformed.to_pickle(f"{OBJECTS_DIR}/X_DATASET.pkl")


# Y_DATASET contains all the target variables.
relevant_targets = [
    "stock_ticker",
    "stock_exret",
    "target",
    "t1",
    "t1_index",
    "weight_attr",
    "alpha",
    "market_exret",
]
Y_DATASET = FULL_stacked_data[relevant_targets]
Y_DATASET.to_pickle(f"{OBJECTS_DIR}/Y_DATASET.pkl")

# WEIGHT_SAMPLING contains all the sample weights.
WEIGHT_SAMPLING = FULL_stacked_data["weight_attr"]
WEIGHT_SAMPLING.to_pickle(f"{OBJECTS_DIR}/WEIGHT_SAMPLING.pkl")

# FULL_stacked_data contains all possible features and targets.
FULL_stacked_data.to_pickle(f"{OBJECTS_DIR}/FULL_stacked_data.pkl")
