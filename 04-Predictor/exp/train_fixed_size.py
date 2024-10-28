"""
Stock Prediction Model using Random Forest and Bagging

This script implements a stock prediction model utilizing Random Forest and
Bagging classifiers to predict stock returns based on feature engineering and
causal discovery techniques. The model leverages historical data to train
and validate its predictions using an expanding window approach.

The script follows these primary steps:
1. Load processed datasets for features (X) and target variables (Y) from serialized files.
2. Set up MultiIndex for time-series data to facilitate indexing.
3. Perform model training and evaluation through a ROLLING window technique:
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

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from scipy.stats import randint, uniform

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
filtered_features['final'].to_list()
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
starting = pd.to_datetime("2000-01-01")
training_window = pd.DateOffset(years=5)
validation_window = pd.DateOffset(years=2)
test_window = pd.DateOffset(years=1)
step_size = pd.DateOffset(years=1)
end_date = pd.to_datetime("2024-01-01")

COUNTER = 0
pred_out = pd.DataFrame()


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
    Y_train['weight_attr'] *= Y_train.shape[0] / Y_train['weight_attr'].sum()
    Y_validate['weight_attr'] *= Y_validate.shape[0] / Y_validate['weight_attr'].sum()
    Y_test['weight_attr'] *= Y_test.shape[0] / Y_test['weight_attr'].sum()

    # Prepare training and validation data
    X_train_vals = X_train[stock_vars].values
    X_validate_vals = X_validate[stock_vars].values
    X_test_vals = X_test[stock_vars].values

    X_train_val = np.vstack([X_train_vals, X_validate_vals])
    Y_train_val = pd.concat([Y_train, Y_validate])

    # Create test_fold for PredefinedSplit
    test_fold = np.concatenate([
        np.full(len(X_train_vals), -1),  # Training set indices
        np.zeros(len(X_validate_vals))   # Validation set indices
    ])

    ps = PredefinedSplit(test_fold)

    # Define the base estimator and bagging classifier
    base_rf = RandomForestClassifier(
        criterion="entropy",
        bootstrap=False,
        class_weight="balanced_subsample"
    )

    bagging_clf = BaggingClassifier(
        estimator=base_rf,
        oob_score=True,
        n_jobs=-1
    )

    param_distributions = {
        'estimator__n_estimators': randint(10, 1000),
        'estimator__max_depth': randint(5, 50),
        'estimator__min_samples_split': randint(2, 10),
        'estimator__min_samples_leaf': randint(1, 5),
        'estimator__max_features': ['sqrt', 'log2'],
        'n_estimators': randint(10, 100),
        'max_samples': uniform(0.1, 1.0),
        'max_features': randint(1, X_train_val.shape[1] + 1)
    }

    # Define the optimizer
    optimizer = RandomizedSearchCV(
        bagging_clf,
        param_distributions=param_distributions,
        n_iter=100,
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
