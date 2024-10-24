from sklearn.datasets import make_classiﬁcation
import numpy as np
import pandas as pd


import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root_dir)

from src.ch_03.code_ch_03 import *
from src.ch_04.code_ch_04 import *
from src.ch_05.code_ch_05 import *
from src.ch_07.code_ch_07 import *


# SNIPPET 8.2 MDI FEATURE IMPORTANCE
def featImpMDI(ﬁt, featNames):
    # feat importance based on IS mean impurity reduction
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(ﬁt.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient="index")
    df0.columns = featNames
    df0 = df0.replace(0, np.nan)  # because max_features=1
    imp = pd.concat(
        {"mean": df0.mean(), "std": df0.std() * df0.shape[0] ** -0.5}, axis=1
    )
    imp /= imp["mean"].sum()
    return imp


# SNIPPET 8.3 MDA FEATURE IMPORTANCE
def featImpMDA(clf, X, y, cv, sample_weight, t1, pctEmbargo, scoring="neg_log_loss"):
    # feat importance based on OOS score reduction
    if scoring not in ["neg_log_loss", "accuracy"]:
        raise Exception("wrong scoring method.")
    from sklearn.metrics import log_loss, accuracy_score

    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged cv
    scr0, scr1 = pd.Series(), pd.DataFrame(columns=X.columns)
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        ﬁt = clf.ﬁt(X=X0, y=y0, sample_weight=w0.values)
        if scoring == "neg_log_loss":
            prob = ﬁt.predict_proba(X1)
            scr0.loc[i] = -log_loss(
                y1, prob, sample_weight=w1.values, labels=clf.classes_
            )
        else:
            pred = ﬁt.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)
        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shufﬂe(X1_[j].values)  # permutation of a single column
            if scoring == "neg_log_loss":
                prob = ﬁt.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(
                    y1, prob, sample_weight=w1.values, labels=clf.classes_
                )
            else:
                pred = ﬁt.predict(X1_)
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


# SNIPPET 8.4 IMPLEMENTATION OF SFI
def auxFeatImpSFI(featNames, clf, trnsX, cont, scoring, cvGen):
    imp = pd.DataFrame(columns=["mean", "std"])
    for featName in featNames:
        df0 = cvScore(
            clf,
            X=trnsX[[featName]],
            y=cont["bin"],
            sample_weight=cont["w"],
            scoring=scoring,
            cvGen=cvGen,
        )
        imp.loc[featName, "mean"] = df0.mean()
        imp.loc[featName, "std"] = df0.std() * df0.shape[0] ** -0.5
    return imp


# SNIPPET 8.5 COMPUTATION OF ORTHOGONAL FEATURES
def get_eVec(dot, varThres):
    # compute eVec from dot prod matrix, reduce dimension
    eVal, eVec = np.linalg.eigh(dot)
    idx = eVal.argsort()[::-1]  # arguments for sorting eVal desc
    eVal, eVec = eVal[idx], eVec[:, idx]
    # 2) only positive eVals
    eVal = pd.Series(eVal, index=["PC_" + str(i + 1) for i in range(eVal.shape[0])])
    eVec = pd.DataFrame(eVec, index=dot.index, columns=eVal.index)
    eVec = eVec.loc[:, eVal.index]
    # 3) reduce dimension, form PCs
    cumVar = eVal.cumsum() / eVal.sum()
    dim = cumVar.values.searchsorted(varThres)
    eVal, eVec = eVal.iloc[: dim + 1], eVec.iloc[:, : dim + 1]
    return eVal, eVec


# -----------------------------------------------------------------
def orthoFeats(dfX, varThres=0.95):
    # Given a dataframe dfX of features, compute orthofeatures dfP
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)  # standardize
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)
    eVal, eVec = get_eVec(dot, varThres)
    dfP = np.dot(dfZ, eVec)
    return dfP


# SNIPPET 8.7 CREATING A SYNTHETIC DATASET
# def getTestData(n_features=40, n_informative=10, n_redundant=10, n_samples=10000):
#     trnsX, cont = make_classiﬁcation(
#         n_samples=n_samples,
#         n_features=n_features,
#         n_informative=n_informative,
#         n_redundant=n_redundant,
#         random_state=0,
#         shufﬂe=False,
#     )

#     df0 = pd.date_range(
#         periods=n_samples, freq=pd.tseries.offsets.BDay(), end=pd.Timestamp.today()
#     )
#     trnsX, cont = pd.DataFrame(trnsX, index=df0), pd.Series(cont, index=df0).to_frame(
#         "bin"
#     )
#     df0 = ["I_" + str(i) for i in range(n_informative)] + [
#         "R_" + str(i) for i in range(n_redundant)
#     ]
#     df0 += ["N_" + str(i) for i in range(n_features - len(df0))]
#     trnsX.columns = df0
#     cont["w"] = 1.0 / cont.shape[0]
#     # cont['t1']=pd.Series(cont.index,index=cont.index)
#     # Add 't1' as a random time, log-uniformly distributed between 1 and 10 days after the index
#     random_days = np.random.uniform(1, 30, size=n_samples)
#     cont["t1"] = cont.index + pd.to_timedelta(random_days, unit="D")
#     return trnsX, cont
def getTestData(n_features=40, n_informative=10, n_redundant=10, n_samples=10000, time_unit='D'):
    # Generate synthetic data
    trnsX, cont = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=0,
        shuffle=False,
    )

    # Define the frequency for time indexing
    if time_unit == 'H':
        # Hourly data
        df0 = pd.date_range(start='1900-01-01', periods=n_samples, freq='H')
    else:
        # Default to business days if no valid time unit is provided
        df0 = pd.date_range(start='1900-01-01', periods=n_samples, freq='B')
    
    # Create the DataFrame
    trnsX = pd.DataFrame(trnsX, index=df0)
    cont = pd.Series(cont, index=df0).to_frame('bin')

    # Generate feature names
    feature_names = ["I_" + str(i) for i in range(n_informative)] + \
                    ["R_" + str(i) for i in range(n_redundant)] + \
                    ["N_" + str(i) for i in range(n_features - n_informative - n_redundant)]
    trnsX.columns = feature_names

    # Add weights and 't1' column
    cont['w'] = 1.0 / cont.shape[0]

    # Generate random time differences (adjust for unit of time)
    if time_unit == 'H':
        random_time_diff = np.random.uniform(1, 48, size=n_samples)  # 1 to 48 hours
    else:
        random_time_diff = np.random.uniform(1, 30, size=n_samples)  # 1 to 30 days
    
    cont['t1'] = cont.index + pd.to_timedelta(random_time_diff, unit=time_unit)

    return trnsX, cont


# SNIPPET 8.8 CALLING FEATURE IMPORTANCE FOR ANY METHOD
def featImportance(
    trnsX,
    cont,
    n_estimators=1000,
    cv=10,
    max_samples=1.0,
    numThreads=24,
    pctEmbargo=0,
    scoring="accuracy",
    method="SFI",
    minWLeaf=0.0,
    **kargs,
):
    # feature importance from a random forest
    from sklearn.tree import DecisionTreeClassiﬁer
    from sklearn.ensemble import BaggingClassiﬁer

    # from mpEngine import mpPandasObj
    n_jobs = -1 if numThreads > 1 else 1  # run 1 thread with ht_helper in dirac1
    # 1) prepare classifier,cv. max_features=1, to prevent masking
    clf = DecisionTreeClassiﬁer(
        criterion="entropy",
        max_features=1,
        class_weight="balanced",
        min_weight_fraction_leaf=minWLeaf,
    )
    clf = BaggingClassiﬁer(
        estimator=clf,
        n_estimators=n_estimators,
        max_features=1.0,
        max_samples=max_samples,
        oob_score=True,
        n_jobs=n_jobs,
    )
    ﬁt = clf.ﬁt(X=trnsX, y=cont["bin"], sample_weight=cont["w"].values)
    oob = ﬁt.oob_score_
    if method == "MDI":
        imp = featImpMDI(ﬁt, featNames=trnsX.columns)
        oos = cvScore(
            clf,
            X=trnsX,
            y=cont["bin"],
            cv=cv,
            sample_weight=cont["w"],
            t1=cont["t1"],
            pctEmbargo=pctEmbargo,
            scoring=scoring,
        ).mean()
    elif method == "MDA":
        imp, oos = featImpMDA(
            clf,
            X=trnsX,
            y=cont["bin"],
            cv=cv,
            sample_weight=cont["w"],
            t1=cont["t1"],
            pctEmbargo=pctEmbargo,
            scoring=scoring,
        )
    elif method == "SFI":
        cvGen = PurgedKFold(n_splits=cv, t1=cont["t1"], pctEmbargo=pctEmbargo)
        oos = cvScore(
            clf,
            X=trnsX,
            y=cont["bin"],
            sample_weight=cont["w"],
            scoring=scoring,
            cvGen=cvGen,
        ).mean()
        clf.n_jobs = 1  # paralellize auxFeatImpSFI rather than clf
        imp = mpPandasObj(
            auxFeatImpSFI,
            ("featNames", trnsX.columns),
            numThreads,
            clf=clf,
            trnsX=trnsX,
            cont=cont,
            scoring=scoring,
            cvGen=cvGen,
        )
    return imp, oob, oos


# SNIPPET 8.9 CALLING ALL COMPONENTS
def testFunc(
    n_features=40,
    n_informative=10,
    n_redundant=10,
    n_estimators=1000,
    n_samples=10000,
    cv=10,
):
    # test the performance of the feat importance functions on artificial data
    # Nr noise features = n_features—n_informative—n_redundant
    trnsX, cont = getTestData(n_features, n_informative, n_redundant, n_samples)
    dict0 = {
        "minWLeaf": [0.0],
        "scoring": ["accuracy"],
        "method": ["MDI", "MDA", "SFI"],
        "max_samples": [1.0],
    }
    jobs, out = (dict(izip(dict0, i)) for i in product(*dict0.values())), []
    kargs = {
        "pathOut": "./testFunc/",
        "n_estimators": n_estimators,
        "tag": "testFunc",
        "cv": cv,
    }
    for job in jobs:
        job["simNum"] = (
            job["method"]
            + "_"
            + job["scoring"]
            + "_"
            + "%.2f" % job["minWLeaf"]
            + "_"
            + str(job["max_samples"])
        )
        print(job["simNum"])
        kargs.update(job)
        imp, oob, oos = featImportance(trnsX=trnsX, cont=cont, **kargs)
        plotFeatImportance(imp=imp, oob=oob, oos=oos, **kargs)
        df0 = imp[["mean"]] / imp["mean"].abs().sum()
        df0["type"] = [i[0] for i in df0.index]
        df0 = df0.groupby("type")["mean"].sum().to_dict()
        df0.update({"oob": oob, "oos": oos})
        df0.update(job)
        out.append(df0)
    out = pd.DataFrame(out).sort_values(
        ["method", "scoring", "minWLeaf", "max_samples"]
    )
    out = out[
        "method", "scoring", "minWLeaf", "max_samples", "I", "R", "N", "oob", "oos"
    ]
    out.to_csv(kargs["pathOut"] + "stats.csv")
    return


# SNIPPET 8.10 FEATURE IMPORTANCE PLOTTING FUNCTION
def plotFeatImportance(pathOut, imp, oob, oos, method, tag=0, simNum=0, **kargs):
    # plot mean imp bars with std
    import matplotlib.pyplot as mpl  # Correcting the import for matplotlib

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
        pathOut + f"featImportance_{str(simNum)}_{method}_{tag}" + ".png", dpi=100
    )
    mpl.clf()
    mpl.close()
    return
