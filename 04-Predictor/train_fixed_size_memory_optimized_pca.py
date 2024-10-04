"""
This script trains a Bagging Classifier with a Random Forest base estimator on stock market data
using a rolling window approach. PCA is applied to reduce feature dimensionality, retaining 80%
of variance, and predictions are generated for each test set.

Key Steps:
1. Load feature (`X`) and target (`Y`) datasets.
2. Define stock feature variables and set the time-series index.
3. Apply rolling windows for train (5 years), validation (2 years), and test (1 year) sets.
4. Adjust sample weights for training, validation, and test sets.
5. Scale the data and apply PCA to reduce dimensionality.
6. Train a Bagging Classifier with RandomizedSearchCV for hyperparameter tuning.
7. Make predictions on the test set and evaluate with log-loss.
8. Save predictions to CSV files and append them to stock-specific data.

Returns:
- `predictions.csv`: Combined predictions from all test sets.
- Updated stock CSV files with predictions and probabilities.
"""

import os
import warnings
import glob
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.stats import randint, uniform

from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load your data
X = pd.read_pickle('../objects/X_DATASET.pkl')
Y = pd.read_pickle('../objects/Y_DATASET.pkl')

# Prepare the data
X['t1_index'] = Y['t1_index']
X.reset_index(inplace=True)
X.set_index(['t1_index', 'index'], inplace=True)

Y = Y.reset_index()
Y.set_index(['t1_index', 'index'], inplace=True)

# Feature variables and target variable
filtered_features = pd.read_json('../0X-Causal_discovery/filtered_features.json')
added_features = ['log_diff', 'frac_diff', 'sadf']
stock_vars = filtered_features['final'].to_list() + added_features
TGT_VAR = 'target'  # Target variable

# Ensure the index is datetime
X.index = pd.MultiIndex.from_tuples(
    [(pd.to_datetime(t1_index), other_index) for t1_index, other_index in X.index]
)
Y.index = pd.MultiIndex.from_tuples(
    [(pd.to_datetime(t1_index), other_index) for t1_index, other_index in Y.index]
)

# Initialize parameters
starting = pd.to_datetime("2004-01-01")
training_window = pd.DateOffset(years=5)
validation_window = pd.DateOffset(years=2)
test_window = pd.DateOffset(years=1)
step_size = pd.DateOffset(years=1)
end_date = pd.to_datetime("2024-01-01")

COUNTER = 0
test_scores = []


while True:
    # Calculate start and end dates for each window
    train_start = starting + COUNTER * step_size
    train_end = train_start + training_window

    val_start = train_end
    val_end = val_start + validation_window

    test_start = val_end
    test_end = test_start + test_window

    # Break the loop if the test end date exceeds the dataset end date
    if test_end > end_date:
        break

    print(f"Train Start: {train_start}, Train End: {train_end}, "
          f"Val Start: {val_start}, Val End: {val_end}, "
          f"Test Start: {test_start}, Test End: {test_end}")

    # Cut the sample into training, validation, and testing sets
    X_train = X[(X.index.get_level_values(0) >= train_start) &
                (X.index.get_level_values(0) < train_end)]
    X_validate = X[(X.index.get_level_values(0) >= val_start) &
                   (X.index.get_level_values(0) < val_end)]
    X_test = X[(X.index.get_level_values(0) >= test_start) &
               (X.index.get_level_values(0) < test_end)]

    Y_train = Y[(Y.index.get_level_values(0) >= train_start) &
                (Y.index.get_level_values(0) < train_end)]
    Y_validate = Y[(Y.index.get_level_values(0) >= val_start) &
                   (Y.index.get_level_values(0) < val_end)]
    Y_test = Y[(Y.index.get_level_values(0) >= test_start) &
               (Y.index.get_level_values(0) < test_end)]

    # Adjust sample weights (if necessary)
    Y_train = Y_train.copy()
    Y_validate = Y_validate.copy()
    Y_test = Y_test.copy()

    Y_train['weight_attr'] *= Y_train.shape[0] / Y_train['weight_attr'].sum()
    Y_validate['weight_attr'] *= Y_validate.shape[0] / Y_validate['weight_attr'].sum()
    Y_test['weight_attr'] *= Y_test.shape[0] / Y_test['weight_attr'].sum()

    # Prepare training data
    X_train_vals = X_train[stock_vars].values.astype('float32')
    X_validate_vals = X_validate[stock_vars].values.astype('float32')
    X_test_vals = X_test[stock_vars].values.astype('float32')

    # **Scale the data**
    scaler = StandardScaler()
    X_train_vals_scaled = scaler.fit_transform(X_train_vals)
    X_validate_vals_scaled = scaler.transform(X_validate_vals)
    X_test_vals_scaled = scaler.transform(X_test_vals)

    # **Apply PCA to X_train_vals_scaled**
    pca = PCA(n_components=0.80, svd_solver='full')  # Retain 80% of variance
    X_train_pca = pca.fit_transform(X_train_vals_scaled)

    # Determine the number of components
    n_components = pca.n_components_
    print(f"Number of components explaining 80% variance: {n_components}")

    # Handle cases where n_components is less than 1
    if n_components < 1:
        print("PCA resulted in less than 1 component. Skipping this iteration.")
        COUNTER += 1
        continue

    # Transform validation and test data
    X_validate_pca = pca.transform(X_validate_vals_scaled)
    X_test_pca = pca.transform(X_test_vals_scaled)

    # Combine training and validation data
    X_train_val = np.vstack([X_train_pca, X_validate_pca])
    Y_train_val = pd.concat([Y_train, Y_validate])

    # Create test_fold for PredefinedSplit
    test_fold = np.concatenate([
        np.full(len(X_train_pca), -1),  # Training set indices
        np.zeros(len(X_validate_pca))   # Validation set indices
    ])

    ps = PredefinedSplit(test_fold)

    # Adjust param_distributions
    if n_components >= 2:
        max_features_dist = randint(1, n_components)
    else:
        max_features_dist = [1]  # Use a fixed value of 1

    param_distributions = {
        'estimator__n_estimators': randint(10, 100),
        'estimator__max_depth': randint(5, 20),
        'estimator__min_samples_split': randint(2, 5),
        'estimator__min_samples_leaf': randint(1, 3),
        'estimator__max_features': ['sqrt', 'log2'],
        'n_estimators': randint(5, 20),
        'max_samples': uniform(0.1, 1.0),
        'max_features': max_features_dist
    }

    # Define the base estimator and bagging classifier with n_jobs=1
    base_rf = RandomForestClassifier(
        criterion="entropy",
        bootstrap=False,
        class_weight="balanced_subsample",
        n_jobs=1
    )

    bagging_clf = BaggingClassifier(
        estimator=base_rf,
        oob_score=True,
        n_jobs=1
    )

    # Define the optimizer with n_jobs=-1
    optimizer = RandomizedSearchCV(
        bagging_clf,
        param_distributions=param_distributions,
        n_iter=20,
        cv=ps,
        n_jobs=-1,
        verbose=2,
        random_state=42,
        scoring='neg_log_loss'
    )

    # Fit the optimizer
    optimizer.fit(
        X_train_val,
        Y_train_val[TGT_VAR].values,
        sample_weight=Y_train_val['weight_attr'].values
    )

    best_estimator = optimizer.best_estimator_

    # Re-adjust weights for training
    Y_train_val['weight_attr'] *= Y_train_val.shape[0] / Y_train_val['weight_attr'].sum()

    # Fit the best estimator
    best_estimator.fit(
        X_train_val,
        Y_train_val[TGT_VAR].values,
        sample_weight=Y_train_val['weight_attr'].values
    )

    # Predict on test set
    prob = best_estimator.predict_proba(X_test_pca)
    score_ = -log_loss(Y_test[TGT_VAR].values, prob,
                       sample_weight=Y_test["weight_attr"].values, labels=best_estimator.classes_)
    print("Log Loss on Test Set:", score_)
    test_scores.append(score_)

    # Store predictions in Y_test and save to disk immediately
    Y_test['prediction'] = best_estimator.predict(X_test_pca)
    Y_test['probability'] = prob.max(axis=1)
    Y_test[['prediction', 'probability']].to_csv(f"../objects/predictions_{COUNTER}.csv",
                                                 index=True)

    # # Clean up variables to free memory
    # del X_train, X_validate, X_test, Y_train, Y_validate, Y_test
    # del X_train_vals, X_validate_vals, X_test_vals, X_train_pca, X_validate_pca,
    # X_test_pca, X_train_val, Y_train_val
    # del prob, best_estimator, optimizer, pca
    # gc.collect()

    COUNTER += 1

prediction_files = glob.glob('../objects/predictions_*.csv')
pred_out = pd.concat((pd.read_csv(f, index_col=[0, 1]) for f in prediction_files))
pred_out.to_csv("../objects/predictions.csv")


temp_pred = pd.read_csv('../objects/predictions.csv')
temp_pred.set_index(['Unnamed: 0', 'Unnamed: 1'], inplace=True)
temp_pred.index.names = ['t1_index', None]
temp_pred.head()


# set the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set the directory containing your stock CSV files
STOCKS_DATA_DIR = '../stocks_data'

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(STOCKS_DATA_DIR) if f.endswith('.csv')]

dfs = []

# Loop through each CSV file
for file_name in tqdm(csv_files):
    file_path = os.path.join(STOCKS_DATA_DIR, file_name)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Append the DataFrame to the list
    dfs.append(df)

stacked_data = pd.concat(dfs, ignore_index=True)
stacked_data = stacked_data.sort_values(by='t1')

# Set the 't1_index' to the first level
stacked_data.set_index('t1_index', inplace=True, append=True)
stacked_data = stacked_data.swaplevel(i=-1, j=0)

stacked_data['prediction'] = temp_pred['prediction']
stacked_data['probability'] = temp_pred['probability']

stacked_data.reset_index(level=0, inplace=True)

# Assuming 'stock_ticker' is one of the columns in stacked_data
# Split the stacked_data by 'stock_ticker'
grouped_data = stacked_data.groupby('stock_ticker')

# Loop through each group and save to CSV
for stock_ticker, group in tqdm(grouped_data):
    # Define the file path for each stock ticker
    file_path = os.path.join(STOCKS_DATA_DIR, f'{stock_ticker}.csv')

    # Save the group DataFrame to a CSV file, overwriting if it exists
    group.to_csv(file_path, index=True)  # Set index=True to keep 'row_index' as index
