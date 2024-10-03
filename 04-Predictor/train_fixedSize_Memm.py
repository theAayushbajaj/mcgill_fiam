# %%
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from scipy.stats import randint, uniform

import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import gc  # Import garbage collector
#%%

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
tgt_var = 'target'  # Target variable

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

counter = 0
test_scores = []
#%%

while True:
    # Calculate start and end dates for each window
    train_start = starting + counter * step_size
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
    X_train = X[(X.index.get_level_values(0) >= train_start) & (X.index.get_level_values(0) < train_end)]
    X_validate = X[(X.index.get_level_values(0) >= val_start) & (X.index.get_level_values(0) < val_end)]
    X_test = X[(X.index.get_level_values(0) >= test_start) & (X.index.get_level_values(0) < test_end)]

    Y_train = Y[(Y.index.get_level_values(0) >= train_start) & (Y.index.get_level_values(0) < train_end)]
    Y_validate = Y[(Y.index.get_level_values(0) >= val_start) & (Y.index.get_level_values(0) < val_end)]
    Y_test = Y[(Y.index.get_level_values(0) >= test_start) & (Y.index.get_level_values(0) < test_end)]

    # Adjust sample weights (if necessary)
    Y_train = Y_train.copy()
    Y_validate = Y_validate.copy()
    Y_test = Y_test.copy()

    Y_train['weight_attr'] *= Y_train.shape[0] / Y_train['weight_attr'].sum()
    Y_validate['weight_attr'] *= Y_validate.shape[0] / Y_validate['weight_attr'].sum()
    Y_test['weight_attr'] *= Y_test.shape[0] / Y_test['weight_attr'].sum()

    # Prepare training and validation data
    X_train_vals = X_train[stock_vars].values.astype('float32')
    X_validate_vals = X_validate[stock_vars].values.astype('float32')
    X_test_vals = X_test[stock_vars].values.astype('float32')

    X_train_val = np.vstack([X_train_vals, X_validate_vals])
    Y_train_val = pd.concat([Y_train, Y_validate])

    # Downcast data types to reduce memory usage
    Y_train_val['weight_attr'] = Y_train_val['weight_attr'].astype('float32')
    Y_train_val[tgt_var] = Y_train_val[tgt_var].astype('int8')

    # Create test_fold for PredefinedSplit
    test_fold = np.concatenate([
        np.full(len(X_train_vals), -1),  # Training set indices
        np.zeros(len(X_validate_vals))   # Validation set indices
    ])

    ps = PredefinedSplit(test_fold)

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

    # Adjust param_distributions to reduce model complexity
    param_distributions = {
        'estimator__n_estimators': randint(10, 100),
        'estimator__max_depth': randint(5, 20),
        'estimator__min_samples_split': randint(2, 5),
        'estimator__min_samples_leaf': randint(1, 3),
        'estimator__max_features': ['sqrt', 'log2'],
        'n_estimators': randint(5, 20),
        'max_samples': uniform(0.1, 1.0),
        'max_features': randint(1, X_train_val.shape[1])
    }

    # Define the optimizer with n_jobs=-1
    optimizer = RandomizedSearchCV(
        bagging_clf,
        param_distributions=param_distributions,
        n_iter=20,  # Reduced number of iterations
        cv=ps,
        n_jobs=-1,
        verbose=2,
        random_state=42,
        scoring='neg_log_loss'
    )

    # Fit the optimizer
    optimizer.fit(
        X_train_val,
        Y_train_val[tgt_var].values,
        sample_weight=Y_train_val['weight_attr'].values
    )

    best_estimator = optimizer.best_estimator_

    # Re-adjust weights for training
    Y_train_val['weight_attr'] *= Y_train_val.shape[0] / Y_train_val['weight_attr'].sum()

    # Fit the best estimator
    best_estimator.fit(
        X_train_val,
        Y_train_val[tgt_var].values,
        sample_weight=Y_train_val['weight_attr'].values
    )

    # Predict on test set
    prob = best_estimator.predict_proba(X_test_vals)
    score_ = -log_loss(Y_test[tgt_var].values, prob, sample_weight=Y_test["weight_attr"].values, labels=best_estimator.classes_)
    print("Log Loss on Test Set:", score_)
    test_scores.append(score_)

    # Store predictions in Y_test and save to disk immediately
    Y_test['prediction'] = best_estimator.predict(X_test_vals)
    Y_test['probability'] = prob.max(axis=1)
    Y_test[['prediction', 'probability']].to_csv(f"../objects/predictions_{counter}.csv")

    # Clean up variables to free memory
    del X_train, X_validate, X_test, Y_train, Y_validate, Y_test
    del X_train_vals, X_validate_vals, X_test_vals, X_train_val, Y_train_val
    del prob, best_estimator, optimizer
    gc.collect()

    counter += 1

#%%
import glob
#%%
prediction_files = glob.glob('../objects/predictions_*.csv')
pred_out = pd.concat((pd.read_csv(f, index_col=[0, 1]) for f in prediction_files))
# pred_out.to_csv("../objects/predictions.csv", index=True)
pred_out.to_csv("../objects/predictions.csv")

# %%
temp_pred = pd.read_csv('../objects/predictions.csv')
temp_pred.set_index(['Unnamed: 0', 'Unnamed: 1'], inplace=True)
temp_pred.index.names = ['t1_index', None]
temp_pred.head()

#%%
# set the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set the directory containing your stock CSV files
stocks_data_dir = '../stocks_data'

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(stocks_data_dir) if f.endswith('.csv')]

dfs = []

# Loop through each CSV file
for file_name in tqdm(csv_files):
    file_path = os.path.join(stocks_data_dir, file_name)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Append the DataFrame to the list
    dfs.append(df)
FULL_stacked_data = pd.concat(dfs, ignore_index=True)
FULL_stacked_data = FULL_stacked_data.sort_values(by='t1')
FULL_stacked_data.drop(columns=['Unnamed: 0'], inplace=True)
FULL_stacked_data

#%%
# Set the 't1_index' to the first level
FULL_stacked_data.set_index('t1_index', inplace=True, append=True)
FULL_stacked_data = FULL_stacked_data.swaplevel(i=-1, j=0)  # Swap the last level (t1_index) to be the first level
FULL_stacked_data.head()

FULL_stacked_data['prediction'] = temp_pred['prediction']
FULL_stacked_data['probability'] = temp_pred['probability']

#%%
# FULL_stacked_data['prediction'].head()
filter = FULL_stacked_data[FULL_stacked_data.index.get_level_values(0) >= '2009-11-31 00:00:00']
filter[['prediction', 'probability', 'stock_ticker']].head()


# %%
FULL_stacked_data.reset_index(level=0, inplace=True)

# %%
FULL_stacked_data[['probability','prediction','stock_ticker']]
# %%
# Assuming 'stock_ticker' is one of the columns in FULL_stacked_data
# Split the FULL_stacked_data by 'stock_ticker'
grouped_data = FULL_stacked_data.groupby('stock_ticker')

# Loop through each group and save to CSV
for stock_ticker, group in tqdm(grouped_data):
    # Define the file path for each stock ticker
    file_path = os.path.join(stocks_data_dir, f'{stock_ticker}.csv')
    
    # Save the group DataFrame to a CSV file, overwriting if it exists
    group.to_csv(file_path, index=True)  # Set index=True to keep 'row_index' as index

# %%
