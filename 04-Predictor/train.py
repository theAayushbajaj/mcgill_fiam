# %%
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import log_loss,accuracy_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from scipy.stats import randint, uniform

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
top_100_features = pd.read_json('../03-Feature_Importance/top_100_features.json')
top_100_features['combined'].to_list()
added_features = ['log_diff', 'frac_diff', 'sadf']
# top 50 instead
stock_vars = top_100_features['combined'].to_list()[:50] + added_features
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
        cv=ps,      # Use predefined split
        n_jobs=-1,
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
