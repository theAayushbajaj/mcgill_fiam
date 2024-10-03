"""
Stock Prediction Model using Random Forest and Bagging

This script implements a stock prediction model utilizing Random Forest and
Bagging classifiers to predict stock returns based on feature engineering and
causal discovery techniques. The model leverages historical data to train
and validate its predictions using an expanding window approach.

The script follows these primary steps:
1. Load processed datasets for features (X) and target variables (Y) from serialized files.
2. Set up MultiIndex for time-series data to facilitate indexing.
3. Perform model training and evaluation through an expanding window technique:
   - Split the dataset into training, validation, and testing sets based on defined cutoffs.
   - Adjust sample weights to balance the training process.
   - Train a Bagging classifier with a base Random Forest estimator using
     Randomized Search for hyperparameter optimization.
   - Predict the target variable probabilities and evaluate model performance using log loss.
4. Store the predictions and probabilities in the appropriate output formats.
5. Compile stock data from multiple CSV files and merge it with predictions,
   allowing for individual stock analysis.
6. Save the results to CSV files grouped by stock ticker.

Usage:
------
- Ensure that preprocessing_code.py and feature_engineering.py are
  executed prior to running this script.
- Ensure that the preprocessing and feature engineering scripts are executed prior
  to running this script to generate the required input files (`X_DATASET.pkl` and `Y_DATASET.pkl`).
- The output predictions will be stored in `predictions.csv`, and individual
  stock prediction files will be saved in the specified stocks data directory.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from scipy.stats import randint

from tqdm import tqdm

warnings.filterwarnings('ignore')

# MUST RUN 01-Data_Preprocessing/preprocessing_code.py AND
# 02-Feature_Engineering/feature_engineering_code.py BEFORE
# load objects/X_DATASET.pkl and objects/Y_DATASET.pkl
X = pd.read_pickle('../objects/X_DATASET.pkl')
Y = pd.read_pickle('../objects/Y_DATASET.pkl')

X['t1_index'] = Y['t1_index']
X.reset_index(inplace=True)
X.set_index(['t1_index', 'index'], inplace=True)

# ### X contains the feautes given and ADDED BY US
Y = Y.reset_index()
Y.set_index(['t1_index', 'index'], inplace=True)



# Load the filtered features obtained from causal discovery as the final set of features.
filtered_features = pd.read_json('../0X-Causal_discovery/filtered_features.json')
filtered_features['final'].to_list()

# List of features added
added_features = ['log_diff', 'frac_diff', 'sadf']

# Get the combined feature list (causal discovery + added features)
stock_vars = filtered_features['final'].to_list() + added_features
TGT_VAR = 'target'  # Target variable

# Define the start of the training set, counter for rolling window and output dataframe.
starting = pd.to_datetime("20000101", format="%Y%m%d")
COUNTER = 0
pred_out = pd.DataFrame()


X.index = pd.MultiIndex.from_tuples(
    [(pd.to_datetime(t1_index), other_index) for t1_index, other_index in X.index]
)
Y.index = pd.MultiIndex.from_tuples(
    [(pd.to_datetime(t1_index), other_index) for t1_index, other_index in Y.index]
)

# Expanding window to train get predictions and
# corresponding probabilities for years 2010-2023.
while (starting + pd.DateOffset(years=11 + COUNTER)) <= pd.to_datetime("20240101", format="%Y%m%d"):

    # Array to store cutoff points for train, validation and test sets
    cutoff = [
        starting,
        starting + pd.DateOffset(years=8 + COUNTER),  # Training set end date
        starting + pd.DateOffset(years=10 + COUNTER),  # Validation set end date
        starting + pd.DateOffset(years=11 + COUNTER),  # Test set end date
    ]

    print(f"Train Start: {cutoff[0]}, Train End: {cutoff[1]}, Val Start: {cutoff[1]},\
          Val End: {cutoff[2]}, Test Start: {cutoff[2]}, Test End: {cutoff[3]}")

    # Cut the sample into training, validation, and testing sets
    X_train = X[(X.index.get_level_values(0) >= cutoff[0]) &
                (X.index.get_level_values(0) < cutoff[1])]
    X_validate = X[(X.index.get_level_values(0) >= cutoff[1]) &
                   (X.index.get_level_values(0) < cutoff[2])]
    X_test = X[(X.index.get_level_values(0) >= cutoff[2]) &
               (X.index.get_level_values(0) < cutoff[3])]

    Y_train = Y[(Y.index.get_level_values(0) >= cutoff[0]) &
                (Y.index.get_level_values(0) < cutoff[1])]
    Y_validate = Y[(Y.index.get_level_values(0) >= cutoff[1]) &
                   (Y.index.get_level_values(0) < cutoff[2])]
    Y_test = Y[(Y.index.get_level_values(0) >= cutoff[2]) &
               (Y.index.get_level_values(0) < cutoff[3])]

    # Adjust sample weights in train, validation and test sets
    Y_train['weight_attr'] *= Y_train.shape[0] / Y_train['weight_attr'].sum()
    Y_validate['weight_attr'] *= Y_validate.shape[0] / Y_validate['weight_attr'].sum()
    Y_test['weight_attr'] *= Y_test.shape[0] / Y_test['weight_attr'].sum()


    # Extract only the relevant features
    X_train_vals = X_train[stock_vars].values
    X_validate_vals = X_validate[stock_vars].values
    X_test_vals = X_test[stock_vars].values

    # Stack train and validation set for hyperparameter tuning
    X_train_val = np.vstack([X_train_vals, X_validate_vals])
    Y_train_val = pd.concat([Y_train, Y_validate])

    # Create test_fold
    test_fold = np.concatenate([
        np.full(len(X_train_vals), -1),  # training set indices
        np.zeros(len(X_validate_vals))   # validation set indices
    ])

    ps = PredefinedSplit(test_fold)


    # Initialize Random Forest classifier
    base_rf = RandomForestClassifier(
        criterion="entropy",
        bootstrap=False,
        class_weight="balanced_subsample"
    )


    # Initialize Bagging Classifier
    bagging_clf = BaggingClassifier(
        estimator=base_rf,
        oob_score=True,
        n_jobs=16
    )

    # Define hyperparameter ranges
    param_distributions = {
        'estimator__n_estimators': randint(100, 500),
        'estimator__max_depth': randint(40, 50),
        'estimator__min_samples_split': randint(5, 10),
        'estimator__min_samples_leaf': randint(3, 5),
        'n_estimators': randint(50, 100),
        #'max_samples': uniform(0.1, 1.0),
        'max_features': randint(1, X_train_val.shape[1] + 1)
    }

    # Define the optimizer
    optimizer = RandomizedSearchCV(
        bagging_clf,
        param_distributions=param_distributions,
        n_iter=100,
        cv=ps,      # Use predefined split
        n_jobs=1,
        verbose=2,
        random_state=42,
        scoring='neg_log_loss'
    )


    # Fit the optimizer to obtain the best combination of hyperparameters
    optimizer.fit(
        X_train_val,
        Y_train_val[TGT_VAR].values,
        **{'sample_weight': Y_train_val['weight_attr'].values}
    )

    # Extract the best features
    best_estimator = optimizer.best_estimator_

    # Rescale the weights of train and validation as
    # this whole dataset is used to train the model.
    Y_train_val['weight_attr'] *= Y_train_val.shape[0] / Y_train_val['weight_attr'].sum()

    # Train the model with the best hyperparameter combination
    best_estimator.fit(
        X_train_val,
        Y_train_val[TGT_VAR].values,
        sample_weight=Y_train_val['weight_attr'].values
    )

    # Predict on test set
    prob = best_estimator.predict_proba(X_test_vals)
    score_ = -log_loss(Y_test[TGT_VAR].values, prob,
                       sample_weight=Y_test["weight_attr"].values, labels=best_estimator.classes_)
    print("Log Loss on Test Set:", score_)

    # Store predictions in Y_test
    Y_test['prediction'] = best_estimator.predict(X_test_vals)
    Y_test['probability'] = prob.max(axis=1)

    pred_out = pred_out.append(Y_test[['prediction', 'probability']])
    pred_out.to_csv("../objects/predictions.csv")


    COUNTER += 1

# Load the predictions file to incorporate them into `{ticker}.csv` files
preds = pd.read_csv('../objects/predictions.csv')
preds.set_index(['Unnamed: 0', 'Unnamed: 1'], inplace=True)
preds.index.names = ['t1_index', None]

# set the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set the directory containing your stock CSV files
STOCKS_DATA_DIR = '../stocks_data'

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(STOCKS_DATA_DIR) if f.endswith('.csv')]

dfs = []

for file_name in tqdm(csv_files):
    file_path = os.path.join(STOCKS_DATA_DIR, file_name)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Append the DataFrame to the list
    dfs.append(df)

stacked_data = pd.concat(dfs, ignore_index=True)
stacked_data = stacked_data.sort_values(by='t1')
stacked_data.drop(columns=['Unnamed: 0'], inplace=True)

# Set the 't1_index' to the first level
stacked_data.set_index('t1_index', inplace=True, append=True)
stacked_data = stacked_data.swaplevel(i=-1, j=0)

stacked_data['prediction'] = preds['prediction']
stacked_data['probability'] = preds['probability']

stacked_data.reset_index(level=0, inplace=True)

# Assuming 'stock_ticker' is one of the columns in FULL_stacked_data
# Split the FULL_stacked_data by 'stock_ticker'
grouped_data = stacked_data.groupby('stock_ticker')

# Loop through each group and save to CSV
for stock_ticker, group in tqdm(grouped_data):
    # Define the file path for each stock ticker
    file_path = os.path.join(STOCKS_DATA_DIR, f'{stock_ticker}.csv')

    # Save the group DataFrame to a CSV file, overwriting if it exists
    group.to_csv(file_path, index=True)
