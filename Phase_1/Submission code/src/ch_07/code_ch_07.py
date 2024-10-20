import pandas as pd

# SNIPPET 7.1 PURGING OBSERVATION IN THE TRAINING SET

def getTrainTimes(t1,testTimes):
    """
    Given testTimes, find the times of the training observations.
    —t1.index: Time when the observation started.
    —t1.value: Time when the observation ended.
    —testTimes: Times of testing observations.
    """
    trn=t1.copy(deep=True)
    for i,j in testTimes.items():
        df0=trn[(i<=trn.index)&(trn.index<=j)].index # train starts within test
        df1=trn[(i<=trn)&(trn<=j)].index # train ends within test
        df2=trn[(trn.index<=i)&(j<=trn)].index # train envelops test
        trn=trn.drop(df0.union(df1).union(df2))
    return trn

# SNIPPET 7.2 EMBARGO ON TRAINING OBSERVATIONS

def getEmbargoTimes(times,pctEmbargo):
    # Get embargo time for each bar
    step=int(times.shape[0]*pctEmbargo)
    if step==0:
        mbrg=pd.Series(times,index=times)
    else:
        mbrg=pd.Series(times[step:],index=times[:-step])
        # mbrg=mbrg.append(pd.Series(times[-1],index=times[-step:]))
        mbrg = pd.concat([mbrg, pd.Series(times[-1], index=times[-step:])])
    return mbrg
#———————————————————————————————————————
# testTimes=pd.Series(mbrg[dt1],index=[dt0]) # include embargo before purge
# trainTimes=getTrainTimes(t1,testTimes)
# testTimes=t1.loc[dt0:dt1].index

# SNIPPET 7.3 CROSS-VALIDATION CLASS WHEN OBSERVATIONS OVERLAP
# class PurgedKFold(_BaseKFold):
from sklearn.model_selection import BaseCrossValidator
import numpy as np
import pandas as pd

class PurgedKFold(BaseCrossValidator):
    '''
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between
    '''
    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.):
        if not isinstance(t1, pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold, self).__init__()  # No additional parameters needed here
        self.n_splits = n_splits
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        self.t1 = pd.to_datetime(self.t1, errors='coerce')

        for i, j in test_starts:
            t0 = self.t1.index[i]  # start of test set
            test_indices = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            
            if maxT1Idx < X.shape[0]:  # right train (with embargo)
                train_indices = np.concatenate((train_indices, indices[maxT1Idx + mbrg:]))
            
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
            
# SNIPPET 7.4 USING THE PurgedKFold CLASS
def cvScore(clf,X,y,sample_weight,scoring='neg_log_loss',t1=None,cv=None,cvGen=None,
    pctEmbargo=None):
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss,accuracy_score
    # from clfSequential import PurgedKFold
    if cvGen is None:
        cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
    score=[]
    for train,test in cvGen.split(X=X):
        ﬁt=clf.ﬁt(X=X.iloc[train,:],y=y.iloc[train],
                  sample_weight=sample_weight.iloc[train].values)
        if scoring=='neg_log_loss':
            prob=ﬁt.predict_proba(X.iloc[test,:])
            score_=-log_loss(y.iloc[test],prob, 
                            sample_weight=sample_weight.iloc[test].values,
                            labels=clf.classes_)
        else:
            pred=ﬁt.predict(X.iloc[test,:])
            score_=accuracy_score(y.iloc[test],pred,sample_weight= \
                sample_weight.iloc[test].values)
        score.append(score_)
    return np.array(score)