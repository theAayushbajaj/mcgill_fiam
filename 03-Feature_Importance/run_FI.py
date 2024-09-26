import numpy as np
import pandas as pd

import pickle

import sys
sys.path.append('../src/ch_08')

import code_ch_08 as f_ch8
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

with open('../objects/X_dataset.pkl', 'rb') as f:
    X = pickle.load(f)
    
with open('../objects/stacked_data.pkl', 'rb') as f:
    stacked_data = pickle.load(f)

path = '../raw_data/factor_char_list.csv'
features = pd.read_csv(path)
features_list = features.values.ravel().tolist()

def getCont(stacked_data, dropped_indices=None):
    cont = pd.concat([stacked_data['datetime'], stacked_data['target'], stacked_data['weight_attr']], axis=1, ignore_index=True)
    cont.rename(columns={cont.columns[0]: 't1', cont.columns[1]: 'bin', cont.columns[2]: 'w'}, inplace=True)
    
    if dropped_indices is not None:
        cont = cont.drop(dropped_indices)
    
    tmp = cont['t1'].shift(-1).dropna()
    tmp = pd.to_datetime(tmp)
    # last date
    result = tmp.iloc[-1] + pd.DateOffset(days=5) + pd.tseries.offsets.BMonthEnd(1)
    tmp = pd.concat([tmp, pd.Series([result])], ignore_index=True)
    # index as first business day of the following month
    # t1.index = pd.to_datetime(datetime) + pd.DateOffset(days=5) - pd.tseries.offsets.BMonthBegin(1)
    cont.index = pd.to_datetime(cont['t1']) - pd.DateOffset(days=40) + pd.tseries.offsets.BMonthEnd(1)
    
    cont['w'] *= cont.shape[0]/cont['w'].sum()
    
    return cont

def runFeatureImportance(data):
    # Bagging classifier on RF where max_samples is set to average uniqueness
    clf2 = RandomForestClassifier(
        n_estimators=1,  # 1 tree
        criterion="entropy",  # information gain
        bootstrap=False,  # no bootstrap
        class_weight="balanced_subsample"  # prevent minority class from being ignored
    )

    clf2 = BaggingClassifier(
        estimator=clf2,  # base estimator
        n_estimators=1_000,  # 1_000 trees
        max_samples=0.94,  # average uniqueness
        max_features=1.0  # all features for bagging
    )

    methods = ['MDI', 'MDA', 'SFI']

    n_estimators = 1000  # Number of trees in the random forest
    cv = 10  # Number of cross-validation folds
    max_samples = 1.0  # Use the entire dataset for each tree
    numThreads = 1  # Adjust based on your available computational resources
    pctEmbargo = 0  # No embargo for simplicity

    for method in methods:
        print(f"Running feature importance for {method}...")
        imp, oob, oos = f_ch8.featImportance(pd.DataFrame(data), cont, n_estimators=n_estimators, cv=cv,
                                        max_samples=max_samples, numThreads=numThreads, 
                                        pctEmbargo=pctEmbargo, method=method)
        
        # Plot the feature importance using the provided function
        f_ch8.plotFeatImportance(pathOut='./', imp=imp, oob=oob, oos=oos, method=method, tag='test', simNum=0)

# Remove NAs from X_dataset and run feature importance code
X_clean = X.copy()

print("Record count BEFORE dropping NaN records: ", len(X_clean))
X_clean.dropna(inplace=True)
print("Record count AFTER dropping NaN records: ", len(X_clean))

# Get the indices that were dropped from X_clean
dropped_indices = X.index.difference(X_clean.index)

cont = getCont(stacked_data, dropped_indices)
X_clean['datetime'] = cont.index
X_clean.set_index('datetime', inplace=True)

runFeatureImportance(X_clean)

# -------------------------X--------------------------X------------------------X----------------- #

# Replacing NaN values with 1e6

print('Number of rows with NaN records: ', X.isna().any(axis=1).sum())
X.fillna(1e6, inplace=True)
print('Number of NaN records after filling NaN values: ', X.isna().any(axis=1).sum())

cont = getCont(stacked_data)
X['datetime'] = cont.index
X.set_index('datetime', inplace=True)

runFeatureImportance(X)