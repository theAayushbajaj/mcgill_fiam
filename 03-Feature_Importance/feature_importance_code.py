"""
This script evaluates feature importance in the given dataset using following
three distinct methods:

    1. Mean Decrease Impurity (MDI): Estimates feature importance based on how much each feature
                                     decreases the impurity of decision trees within the model.

    2. Mean Decrease Accuracy (MDA): Assesses the importance of a feature by measuring the impact
                                     on model accuracy when the feature's values are shuffled.

    3. Single Feature Importance (SFI): Evaluates the importance of each feature individually by
                                        training the model using one feature at a time and
                                        observing its effect on model performance.
"""

import warnings
import pickle
import sys
import os
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(parent_dir, 'src/ch_08'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import code_ch_08 as f_ch8

warnings.filterwarnings('ignore')

# Load dataset and targets
with open(os.path.join(parent_dir, 'objects/X_DATASET.pkl'), 'rb') as f:
    X = pickle.load(f)

with open(os.path.join(parent_dir, 'objects/Y_DATASET.pkl'), 'rb') as f:
    Y = pickle.load(f)

# Load the feature list
PATH = os.path.join(parent_dir, 'raw_data/factor_char_list.csv')
features = pd.read_csv(PATH)
features_list = features.values.ravel().tolist()


# We do feature importance on given features
X = X[features_list]

def get_cont(input_df, drop_index=None):
    """
    Processes a DataFrame `input_df` to return a modified DataFrame `df` containing key
    columns for further analysis.

    Parameters:
    input_df : pd.DataFrame
        A DataFrame containing the columns 't1_index', 't1', 'target', and 'weight_attr'.
    drop_index : list, optional
        A list of indices to drop from the resulting DataFrame `df`.
        If None, no indices are dropped.

    Returns:
    pd.DataFrame
        The resulting DataFrame `df` with columns 't1_index' (set as the index),
        't1' (converted to datetime),
        'bin' (the 'target' column), and 'w' (the 'weight_attr' column).
        The weight column 'w' is normalized by the sum of all weights.
    """
    df = pd.concat([input_df['t1_index'], input_df['t1'], input_df['target'],
                      input_df['weight_attr']], axis=1, ignore_index=True)

    df.rename(columns={df.columns[0]: 't1_index', df.columns[1]: 't1',
                         df.columns[2]: 'bin', df.columns[3]: 'w'}, inplace=True)

    if drop_index is not None:
        df = df.drop(drop_index)

    df.set_index('t1_index', inplace=True)
    df['t1'] = pd.to_datetime(df['t1'])
    df.index = pd.to_datetime(df.index)

    df['w'] *= df.shape[0]/df['w'].sum()

    return df

def run_feature_importance(data, case_tag):
    """
    Performs feature importance analysis using three different methods
    (MDI, MDA, SFI) on the given dataset.
    The method utilizes a Bagging Classifier built on a RandomForestClassifier
    with entropy criterion, balanced subsampling, and no bootstrap aggregation.
    Feature importance results are computed for each method and visualized through plots.

    Parameters:
    data : pd.DataFrame
        The dataset containing features and target variables for which
        feature importance will be calculated.
    case_tag : str
        A tag used for labeling the plots and saving results, helping to
        distinguish different runs or cases.

    Returns:
    dict
        A dictionary `fi_estimates` containing feature importance estimates
        for each method ('MDI', 'MDA', 'SFI').
        For each method, the dictionary contains:
        - 'imp' : Importance scores for each feature.
        - 'oob' : Out-of-bag feature importance.
        - 'oos' : Out-of-sample feature importance.

    The function generates feature importance plots for each method and saves
    them in the specified output path.
    """

    methods = ['MDI', 'MDA', 'SFI']
    fi_estimates = {method: {} for method in methods}

    n_estimators = 1000  # Number of trees in the random forest
    cv = 10  # Number of cross-validation folds
    max_samples = 1.0  # Use the entire dataset for each tree
    num_threads = 1  # Adjust based on your available computational resources
    pct_embargo = 0  # No embargo for simplicity

    for method in methods:
        print(f"Running feature importance for {method}...")
        imp, oob, oos = f_ch8.featImportance(pd.DataFrame(data), cont,
                                             n_estimators=n_estimators, cv=cv,
                                             max_samples=max_samples, numThreads=num_threads,
                                             pctEmbargo=pct_embargo, method=method)

        fi_estimates[method]['imp'] = imp
        fi_estimates[method]['oob'] = oob
        fi_estimates[method]['oos'] = oos

        # Plot the feature importance using the provided function
        f_ch8.plotFeatImportance(pathOut='./', imp=imp, oob=oob, oos=oos,
                                 method=method, tag=case_tag, simNum=0)

    return fi_estimates

# Replacing NaN values with 1e6

print('Number of rows with NaN records: ', X.isna().any(axis=1).sum())
X.fillna(1e6, inplace=True)
print('Number of NaN records after filling NaN values: ', X.isna().any(axis=1).sum())

cont = get_cont(Y)
X['datetime'] = cont.index
X.set_index('datetime', inplace=True)

fi_estimates_fillna = run_feature_importance(X, 'fillNA')

with open('./fi_estimates_fillna.pkl', 'wb') as f:
    pickle.dump(fi_estimates_fillna, f)
