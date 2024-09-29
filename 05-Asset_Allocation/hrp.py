#%%
import sys
import os

#%%
# SNIPPET 16.4 FULL IMPLEMENTATION OF THE HRP ALGORITHM
import matplotlib.pyplot as mpl
import scipy.cluster.hierarchy as sch,random,numpy as np,pandas as pd

from sklearn.covariance import LedoitWolf
#———————————————————————————————————————

def getIVP(cov, **kargs):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp
#———————————————————————————————————————
def getClusterVar(cov,cItems):
    # Compute variance per cluster
    cov_=cov.loc[cItems,cItems] # matrix slice
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar

def getClusterMean(mu, cItems):
    # Compute expected return per cluster
    mu_ = mu.loc[cItems]
    w_ = np.ones(len(mu_)) / len(mu_)  # Equal weights within the cluster
    cMean = np.dot(w_, mu_)
    return cMean

#———————————————————————————————————————
def getQuasiDiag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0]).sort_index()  # concatenate and sort
        sortIx.index = range(sortIx.shape[0])  # re-index
    return sortIx.tolist()
#———————————————————————————————————————
# def getRecBipart(cov, sortIx):
#     # Compute HRP alloc
#     w = pd.Series(1, index=sortIx)
#     cItems = [sortIx]  # initialize all items in one cluster
#     while len(cItems) > 0:
#         cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
#         for i in range(0, len(cItems), 2):  # parse in pairs
#             cItems0 = cItems[i]  # cluster 1
#             cItems1 = cItems[i + 1]  # cluster 2
#             cVar0 = getClusterVar(cov, cItems0)
#             cVar1 = getClusterVar(cov, cItems1)
#             alpha = 1 - cVar0 / (cVar0 + cVar1)
#             alpha = 2 * alpha - 1

#             w[cItems0] *= alpha  # weight 1
#             w[cItems1] *= 1 - alpha if alpha >= 0 else -1 - alpha  # weight 2
#     # w/=w.abs().sum()
#     return w

def getRecBipart(cov, sortIx, mu):
    # Compute HRP alloc
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            # Calculate cluster variances and expected returns
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            cMean0 = getClusterMean(mu, cItems0)
            cMean1 = getClusterMean(mu, cItems1)
            # Calculate Sharpe ratios for clusters
            sr0 = cMean0 /np.sqrt(cVar0) if cVar0 > 0 else 0
            sr1 = cMean1 / np.sqrt(cVar1) if cVar1 > 0 else 0
            
            denom = abs(sr0) + abs(sr1)
            # Allocate weights proportional to Sharpe ratios
            alpha = sr1 / denom if denom != 0 else 0.5
            w[cItems0] *= sr0/ denom
            w[cItems1] *= sr1 / denom
    return w

#———————————————————————————————————————
def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist = ((1 - corr) / 2.)**.5  # distance matrix
    return dist
#———————————————————————————————————————
def plotCorrMatrix(path,corr,labels=None):
    # Heatmap of the correlation matrix
    if labels is None:
        labels=[]
    mpl.pcolor(corr)
    mpl.colorbar()
    mpl.yticks(np.arange(.5,corr.shape[0]+.5),labels)
    mpl.xticks(np.arange(.5,corr.shape[0]+.5),labels)
    mpl.saveﬁg(path)
    mpl.clf();mpl.close() # reset pylab
    return
#———————————————————————————————————————
# def generateData(nObs,size0,size1,sigma1):
#     # Time series of correlated variables
#     #1) generating some uncorrelated data
#     np.random.seed(seed=12345);random.seed(12345)
#     x=np.random.normal(0,1,size=(nObs,size0)) # each row is a variable
#     #2) creating correlation between the variables
#     cols=[random.randint(0,size0–1) for i in xrange(size1)]
#     y=x[:,cols]+np.random.normal(0,sigma1,size=(nObs,len(cols)))
#     x=np.append(x,y,axis=1)
#     x=pd.DataFrame(x,columns=range(1,x.shape[1]+1))
#     return x,cols
#———————————————————————————————————————
def main():
    # LOAD /Users/paulkelendji/Desktop/GitHub_paul/mcgill_fiam/objects/prices.pkl
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    path = '../objects/prices.pkl'

    prices = pd.read_pickle(path)
    prices

    # Step 1) Gather the returns
    Start_Date = '2015-01-01'
    End_Date = '2019-12-01'

    training = prices.loc[Start_Date:End_Date].dropna(axis=1)
    x = training.pct_change().dropna()
    cov,corr=x.cov(),x.corr()
    
    lw = LedoitWolf()
    shrunk_cov_matrix = lw.fit(x).covariance_
    cov = shrunk_cov_matrix
    # array to DataFrame
    cov = pd.DataFrame(cov, index=x.columns, columns=x.columns)
    
    #2) compute and plot correl matrix
    plotCorrMatrix('HRP3_corr0.png',corr,labels=corr.columns)
    
    #3) cluster
    dist=correlDist(corr)
    link=sch.linkage(dist,'single')
    sortIx=getQuasiDiag(link)
    sortIx=corr.index[sortIx].tolist() # recover labels
    df0=corr.loc[sortIx,sortIx] # reorder
    plotCorrMatrix('HRP3_corr1.png',df0,labels=df0.columns)
    
    #4) Capital allocation
    hrp=getRecBipart(cov,sortIx)
    print('Showing weights asociated with each ticker: ')
    print(hrp)
    print('Market exposure : ', hrp.sum())
    print('Sum of absolute values of weights: ', hrp.abs().sum())
#———————————————————————————————————————
#%%
if __name__=='__main__':
    main()
# %%
