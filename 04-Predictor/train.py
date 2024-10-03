# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import log_loss,accuracy_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from scipy.stats import randint, uniform

import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# MUST RUN 01-Data_Preprocessing/01-preprocessing_code.py AND 02-Feature_Engineering/01-fe_code.py BEFORE
# %%
# load objects/X_DATASET.pkl and objects/Y_DATASET.pkl
X = pd.read_pickle('../objects/X_DATASET.pkl')
Y = pd.read_pickle('../objects/Y_DATASET.pkl')

# %%
X['t1_index'] = Y['t1_index']
X.reset_index(inplace=True)
X.set_index(['t1_index', 'index'], inplace=True)

# %% [markdown]
# ### X contains the feautes given and ADDED BY US

# %%
Y = Y.reset_index()
Y.set_index(['t1_index', 'index'], inplace=True)


# %% [markdown]
# ### Currently, in $Y$,
# *prediction* and *probability* are randomly generated, those are the columns
# we must fill for the investment perdiod (2008-) we our true probabilities and predictions.

# %%
filtered_features = pd.read_json('../0X-Causal_discovery/filtered_features.json')
filtered_features['final'].to_list()
added_features = ['log_diff', 'frac_diff', 'sadf']
stock_vars = filtered_features['final'].to_list() + added_features
tgt_var = 'target'  # Target variable

starting = pd.to_datetime("20000101", format="%Y%m%d")
counter = 0
pred_out = pd.DataFrame()


X.index = pd.MultiIndex.from_tuples(
    [(pd.to_datetime(t1_index), other_index) for t1_index, other_index in X.index]
)
Y.index = pd.MultiIndex.from_tuples(
    [(pd.to_datetime(t1_index), other_index) for t1_index, other_index in Y.index]
)

# Estimation with expanding window
while (starting + pd.DateOffset(years=11 + counter)) <= pd.to_datetime("20240101", format="%Y%m%d"):
    # For testing purposes
    # if counter == 1:
    #     break

    cutoff = [
        starting,
        starting + pd.DateOffset(years=8 + counter),  # Training set end date
        starting + pd.DateOffset(years=10 + counter),  # Validation set end date
        starting + pd.DateOffset(years=11 + counter),  # Test set end date
    ]

    print(f"Train Start: {cutoff[0]}, Train End: {cutoff[1]}, Val Start: {cutoff[1]}, Val End: {cutoff[2]}, Test Start: {cutoff[2]}, Test End: {cutoff[3]}")

    # Cut the sample into training, validation, and testing sets
    X_train = X[(X.index.get_level_values(0) >= cutoff[0]) & (X.index.get_level_values(0) < cutoff[1])]
    X_validate = X[(X.index.get_level_values(0) >= cutoff[1]) & (X.index.get_level_values(0) < cutoff[2])]
    X_test = X[(X.index.get_level_values(0) >= cutoff[2]) & (X.index.get_level_values(0) < cutoff[3])]

    Y_train = Y[(Y.index.get_level_values(0) >= cutoff[0]) & (Y.index.get_level_values(0) < cutoff[1])]
    Y_validate = Y[(Y.index.get_level_values(0) >= cutoff[1]) & (Y.index.get_level_values(0) < cutoff[2])]
    Y_test = Y[(Y.index.get_level_values(0) >= cutoff[2]) & (Y.index.get_level_values(0) < cutoff[3])]

    # Adjust sample weights
    Y_train['weight_attr'] *= Y_train.shape[0] / Y_train['weight_attr'].sum()
    Y_validate['weight_attr'] *= Y_validate.shape[0] / Y_validate['weight_attr'].sum()
    Y_test['weight_attr'] *= Y_test.shape[0] / Y_test['weight_attr'].sum()


    X_train_vals = X_train[stock_vars].values
    X_validate_vals = X_validate[stock_vars].values
    X_test_vals = X_test[stock_vars].values

    X_train_val = np.vstack([X_train_vals, X_validate_vals])
    Y_train_val = pd.concat([Y_train, Y_validate])

    # Create test_fold 
    test_fold = np.concatenate([
        np.full(len(X_train_vals), -1),  # training set indices
        np.zeros(len(X_validate_vals))   # validation set indices
    ])
    
    ps = PredefinedSplit(test_fold)


    base_rf = RandomForestClassifier(
        criterion="entropy",
        bootstrap=False,
        class_weight="balanced_subsample"
    )


    bagging_clf = BaggingClassifier(
        estimator=base_rf,
        oob_score=True,
        n_jobs=16
    )

 
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


    optimizer.fit(
        X_train_val,
        Y_train_val[tgt_var].values,
        **{'sample_weight': Y_train_val['weight_attr'].values}
    )


    best_estimator = optimizer.best_estimator_

    Y_train_val['weight_attr'] *= Y_train_val.shape[0] / Y_train_val['weight_attr'].sum()

    best_estimator.fit(
        X_train_val,
        Y_train_val[tgt_var].values,
        sample_weight=Y_train_val['weight_attr'].values
    )

    # Predict on test set
    prob = best_estimator.predict_proba(X_test_vals)
    score_ = -log_loss(Y_test[tgt_var].values, prob, sample_weight=Y_test["weight_attr"].values, labels=best_estimator.classes_)
    print("Log Loss on Test Set:", score_)

    # Store predictions in Y_test
    Y_test['prediction'] = best_estimator.predict(X_test_vals)
    Y_test['probability'] = prob.max(axis=1)
 
    pred_out = pred_out.append(Y_test[['prediction', 'probability']])
    pred_out.to_csv("../objects/predictions.csv")


    counter += 1
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