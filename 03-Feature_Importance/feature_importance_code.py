"""
This script evaluates feature importance in the given dataset using following
two distinct methods:

    1. Mean Decrease Impurity (MDI): Estimates feature importance based on how much each feature
                                     decreases the impurity of decision trees within the model.

    2. Mean Decrease Accuracy (MDA): Assesses the importance of a feature by measuring the impact
                                     on model accuracy when the feature's values are shuffled.
"""

import warnings
import pickle
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.tree import DecisionTreeClassiﬁer
from sklearn.ensemble import BaggingClassiﬁer
from sklearn.metrics import log_loss,accuracy_score
import matplotlib.pyplot as mpl

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(parent_dir, 'src/ch_08'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

class PurgedKFold(BaseCrossValidator):
    '''
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between
    '''
    def __init__(self, n_splits=3, t1=None, pct_embargo=0.):
        if not isinstance(t1, pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold, self).__init__()  # No additional parameters needed here
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, x, y=None, groups=None):
        if (x.index == self.t1.index).sum() != len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        indices = np.arange(X.shape[0])
        mbrg = int(x.shape[0] * self.pct_embargo)
        test_starts = [(i[0], i[-1] + 1) for i in
                       np.array_split(np.arange(x.shape[0]), self.n_splits)]
        self.t1 = pd.to_datetime(self.t1, errors='coerce')

        for i, j in test_starts:
            t0 = self.t1.index[i]  # start of test set
            test_indices = indices[i:j]
            max_t1_idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)

            if max_t1_idx < x.shape[0]:  # right train (with embargo)
                train_indices = np.concatenate((train_indices, indices[max_t1_idx + mbrg:]))

            yield train_indices, test_indices

    def get_n_splits(self, x=None, y=None, groups=None):
        return self.n_splits

# SNIPPET 7.4 USING THE PurgedKFold CLASS
def cv_score(clf,x,y,sample_weight,scoring='neg_log_loss',t1=None,cv=None,cv_gen=None,
    pct_embargo=None):
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('wrong scoring method.')

    # from clfSequential import PurgedKFold
    if cv_gen is None:
        cv_gen=PurgedKFold(n_splits=cv,t1=t1,pct_embargo=pct_embargo) # purged
    score=[]
    for train,test in cv_gen.split(x=x):
        ﬁt=clf.ﬁt(X=x.iloc[train,:],y=y.iloc[train],
                  sample_weight=sample_weight.iloc[train].values)
        if scoring=='neg_log_loss':
            prob=ﬁt.predict_proba(x.iloc[test,:])
            score_=-log_loss(y.iloc[test],prob,
                            sample_weight=sample_weight.iloc[test].values,
                            labels=clf.classes_)
        else:
            pred=ﬁt.predict(x.iloc[test,:])
            score_=accuracy_score(y.iloc[test],pred,sample_weight= \
                sample_weight.iloc[test].values)
        score.append(score_)
    return np.array(score)

def feat_imp_mdi(ﬁt, feat_names):
    # feat importance based on IS mean impurity reduction
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(ﬁt.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient="index")
    df0.columns = feat_names
    df0 = df0.replace(0, np.nan)  # because max_features=1
    imp = pd.concat(
        {"mean": df0.mean(), "std": df0.std() * df0.shape[0] ** -0.5}, axis=1
    )
    imp /= imp["mean"].sum()
    return imp

def feat_imp_mda(clf, x, y, cv, sample_weight, t1, pct_embargo, scoring="neg_log_loss"):
    # feat importance based on OOS score reduction
    if scoring not in ["neg_log_loss", "accuracy"]:
        raise Exception("Wrong scoring method.")

    cv_gen = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)  # purged cv
    scr0, scr1 = pd.Series(), pd.DataFrame(columns=x.columns)
    for i, (train, test) in enumerate(cv_gen.split(x=x)):
        x0, y0, w0 = x.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        x1, y1, w1 = x.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        ﬁt = clf.ﬁt(X=x0, y=y0, sample_weight=w0.values)
        if scoring == "neg_log_loss":
            prob = ﬁt.predict_proba(x1)
            scr0.loc[i] = -log_loss(
                y1, prob, sample_weight=w1.values, labels=clf.classes_
            )
        else:
            pred = ﬁt.predict(x1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)
        for j in X.columns:
            x1_ = x1.copy(deep=True)
            np.random.shufﬂe(x1_[j].values)  # permutation of a single column
            if scoring == "neg_log_loss":
                prob = ﬁt.predict_proba(x1_)
                scr1.loc[i, j] = -log_loss(
                    y1, prob, sample_weight=w1.values, labels=clf.classes_
                )
            else:
                pred = ﬁt.predict(x1_)
                scr1.loc[i, j] = accuracy_score(y1, pred, sample_weight=w1.values)
    imp = (-scr1).add(scr0, axis=0)
    if scoring == "neg_log_loss":
        imp = imp / -scr1
    else:
        imp = imp / (1.0 - scr1)
    imp = pd.concat(
        {"mean": imp.mean(), "std": imp.std() * imp.shape[0] ** -0.5}, axis=1
    )
    return imp, scr0.mean()

# SNIPPET 8.8 CALLING FEATURE IMPORTANCE FOR ANY METHOD
def feat_importance(
    trns_x,
    inp,
    n_estimators=1000,
    cv=10,
    max_samples=1.0,
    num_threads=24,
    pct_embargo=0,
    scoring="accuracy",
    method="MDA",
    min_w_leaf=0.0,
    **kargs,
):
    """
    This function runs feature importance for MDI and MDA methods.
    """
    n_jobs = -1 if num_threads > 1 else 1  # run 1 thread with ht_helper in dirac1
    # 1) prepare classifier,cv. max_features=1, to prevent masking
    clf = DecisionTreeClassiﬁer(
        criterion="entropy",
        max_features=1,
        class_weight="balanced",
        min_weight_fraction_leaf=min_w_leaf,
    )
    clf = BaggingClassiﬁer(
        estimator=clf,
        n_estimators=n_estimators,
        max_features=1.0,
        max_samples=max_samples,
        oob_score=True,
        n_jobs=n_jobs,
    )
    ﬁt = clf.ﬁt(X=trns_x, y=inp["bin"], sample_weight=inp["w"].values)
    oob = ﬁt.oob_score_
    if method == "MDI":
        imp = feat_imp_mdi(ﬁt, feat_names=trns_x.columns)
        oos = cv_score(
            clf,
            x=trns_x,
            y=inp["bin"],
            cv=cv,
            sample_weight=inp["w"],
            t1=inp["t1"],
            pct_embargo=pct_embargo,
            scoring=scoring,
        ).mean()
    elif method == "MDA":
        imp, oos = feat_imp_mda(
            clf,
            x=trns_x,
            y=inp["bin"],
            cv=cv,
            sample_weight=inp["w"],
            t1=inp["t1"],
            pct_embargo=pct_embargo,
            scoring=scoring,
        )
    return imp, oob, oos

def plot_feat_importance(path_out, imp, oob, oos, method, tag=0, sim_num=0, **kargs):
    """
    This function plots the mean importance bars.
    """
    mpl.ﬁgure(ﬁgsize=(10, imp.shape[0] / 5.0))
    imp = imp.sort_values("mean", ascending=True)
    ax = imp["mean"].plot(
        kind="barh", color="b", alpha=0.25, xerr=imp["std"], error_kw={"ecolor": "r"}
    )
    if method == "MDI":
        mpl.xlim([0, imp.sum(axis=1).max()])
        mpl.axvline(1.0 / imp.shape[0], linewidth=1, color="r", linestyle="dotted")
    ax.get_yaxis().set_visible(False)
    for i, j in zip(ax.patches, imp.index):
        ax.text(
            i.get_width() / 2,
            i.get_y() + i.get_height() / 2,
            j,
            ha="center",
            va="center",
            color="black",
        )
    mpl.title(
        "tag="
        + tag
        + " | oob="
        + str(round(oob, 4))
        + " | oos="
        + str(round(oos, 4))
    )
    mpl.saveﬁg(
        path_out + f"featImportance_{str(sim_num)}_{method}_{tag}" + ".png", dpi=100
    )
    mpl.clf()
    mpl.close()

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
    (MDI, MDA) on the given dataset.
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
        for each method ('MDI', 'MDA').
        For each method, the dictionary contains:
        - 'imp' : Importance scores for each feature.
        - 'oob' : Out-of-bag feature importance.
        - 'oos' : Out-of-sample feature importance.

    The function generates feature importance plots for each method and saves
    them in the specified output path.
    """

    methods = ['MDI', 'MDA']
    fi_estimates = {method: {} for method in methods}

    n_estimators = 1000  # Number of trees in the random forest
    cv = 10  # Number of cross-validation folds
    max_samples = 1.0  # Use the entire dataset for each tree
    num_threads = 1  # Adjust based on your available computational resources
    pct_embargo = 0  # No embargo for simplicity

    for method in methods:
        print(f"Running feature importance for {method}...")
        imp, oob, oos = feat_importance(pd.DataFrame(data), cont,
                                             n_estimators=n_estimators, cv=cv,
                                             max_samples=max_samples, num_threads=num_threads,
                                             pct_embargo=pct_embargo, method=method)

        fi_estimates[method]['imp'] = imp
        fi_estimates[method]['oob'] = oob
        fi_estimates[method]['oos'] = oos

        # Plot the feature importance using the provided function
        plot_feat_importance(path_out='./', imp=imp, oob=oob, oos=oos,
                                 method=method, tag=case_tag, sim_num=0)

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
