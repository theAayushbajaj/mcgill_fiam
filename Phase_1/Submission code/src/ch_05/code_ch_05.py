"""Code to solve exercises in Chapter 5 of Advances in Financial Machine 
Learning by Marcos Lopez de Prado.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# SNIPPET 5.1 WEIGHTING FUNCTION
def getWeights(d, size):
    # thres > 0 drops insignificant weights
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

# Function to plot weights
def plotWeights(dRange, nPlots, size):
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = getWeights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[d])
        w = w.join(w_, how='outer')
    ax = w.plot()
    ax.legend(loc='lower right')  # Use 'lower right' to place the legend in the right bottom corner
    plt.show()
    return

# SNIPPET 5.2 STANDARD FRACDIFF (EXPANDING WINDOW)
def fracDiff(series, d, thres=.01):
    '''
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    #1) Compute weights for the longest series
    w = getWeights(d, series.shape[0])
    
    #2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    
    #3) Apply weights to values
    df = {}
    for name in series.columns:
        seriesF = series[[name]].fillna(method='ffill').dropna()
        seriesF = pd.to_numeric(seriesF[name], errors='coerce')  # Convert to numeric, coercing errors to NaN
        df_ = pd.Series(dtype='float64')
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            if not np.isfinite(seriesF.loc[loc]):  # Check if the value is finite
                continue  # exclude NAs
            
            # Adjust the np.dot function to handle 1D array
            weight_subset = w[-(iloc + 1):].reshape(-1)  # Reshape to 1D array
            value_subset = seriesF.loc[:loc].values  # Get the values to match the weights
            
            df_[loc] = np.dot(weight_subset, value_subset)
            
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


# SNIPPET 5.3 THE NEW FIXED-WIDTH WINDOW FRACDIFF METHOD
def fracDiff_FFD(series, d, thres=1e-5):
    '''
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''
    #1) Compute weights for the longest series
    w = getWeights_FFD(d, thres)
    width = len(w) - 1
    #2) Apply weights to values
    df = {}
    for name in series.columns:
        # Corrected the fill method to 'ffill'
        seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1-width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue  # exclude NAs
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


# def getWeights_FFD(d, thres):
#     # Appends while the last weight is above the threshold
#     w = [1.]
#     while abs(w[-1]) > thres:
#         w_ = -w[-1] / len(w) * (d - len(w) + 1)
#         w.append(w_)
#     w = np.array(w[::-1]).reshape(-1, 1)
#     return w


def getWeights_FFD(d, thres):
    # Appends while the last weight is above the threshold
    w = [1.]
    while True:
        w_ = (-w[-1] / len(w)) * (d - len(w) + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
    return np.array(w[::-1]).reshape(-1, 1)


# SNIPPET 5.4 FINDING THE MINIMUM D VALUE THAT PASSES THE ADF TEST

# def plotMinFFD():
#     from statsmodels.tsa.stattools import adfuller
#     path,instName='./','ES1_Index_Method12'
#     out=pd.DataFrame(columns=['adfStat','pVal','lags','nObs','95% conf','corr'])
#     df0=pd.read_csv(path+instName+'.csv',index_col=0,parse_dates=True)
#     for d in np.linspace(0,1,11):
#         df1=np.log(df0[['Close']]).resample('1D').last() # downcast to daily obs
#         df2=fracDiff_FFD(df1,d,thres=.01)
#         corr=np.corrcoef(df1.loc[df2.index,'Close'],df2['Close'])[0,1]
#         df2=adfuller(df2['Close'],maxlag=1,regression='c',autolag=None)
#         out.loc[d]=list(df2[:4])+[df2[4]['5%']]+[corr] # with critical value
#     out.to_csv(path+instName+'_testMinFFD.csv')
#     out[['adfStat','corr']].plot(secondary_y='adfStat')
#     mpl.axhline(out['95% conf'].mean(),linewidth=1,color='r',linestyle='dotted')
#     mpl.saveﬁg(path+instName+'_testMinFFD.png')
#     return

def plotMinFFD(df, threshold=1e-5, applyLog = False, **kwargs):
    from statsmodels.tsa.stattools import adfuller
    import matplotlib.pyplot as plt
    # if df is a Series, convert it to a DataFrame with 'close' as the column name
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df, columns=['close'])
        
    # if index is not datetime, create a daily datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        idx = pd.date_range(start='2000-01-01', periods=len(df), freq='D')
        df.index = idx
    
    
    d_found = False
    
    out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    
    if applyLog:
        df1 = np.log(df[['close']]).resample('1D').last()  # Downcast to daily observations
    else:
        df1 = df[['close']].resample('1D').last()
    
    
    # Assuming 'Close' is the column name in df
    for d in np.linspace(0, 1, 21):
        df2 = fracDiff_FFD(df1, d, thres=threshold)
        corr = np.corrcoef(df1.loc[df2.index, 'close'], df2['close'])[0, 1]
        # adf_result = adfuller(df2['close'], maxlag=1, regression='c', autolag=None)
        adf_result = adfuller(df2['close'], **kwargs)
        out.loc[d] = list(adf_result[:4]) + [adf_result[4]['5%']] + [corr]  # With critical value
        # print(f'Fractional differentiation order: {d}, ADF Statistic: {adf_result[0]}, Correlation: {corr}')
        if not d_found and adf_result[1] < 0.05:
            d_found = True
            d_req = d
            print(f'Fractional differentiation order found: {d_req}')

    # Plotting the results
    out[['adfStat', 'corr']].plot(secondary_y='adfStat')
    plt.axhline(out['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
    plt.title('ADF Statistic and Correlation vs. Differentiation Order')
    plt.show()
    
    return out, d_req


def plotMinEXP(df, threshold=1e-5):
    from statsmodels.tsa.stattools import adfuller
    import matplotlib.pyplot as plt
    d_found = False
    
    out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    
    # Assuming 'Close' is the column name in df
    for d in np.linspace(0, 1, 21):
        df1 = np.log(df[['close']]).resample('1D').last()  # Downcast to daily observations
        df2 = fracDiff(df1, d, thres=threshold)
        corr = np.corrcoef(df1.loc[df2.index, 'close'], df2['close'])[0, 1]
        adf_result = adfuller(df2['close'], maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(adf_result[:4]) + [adf_result[4]['5%']] + [corr]  # With critical value
        # print(f'Fractional differentiation order: {d}, ADF Statistic: {adf_result[0]}, Correlation: {corr}')
        if not d_found and adf_result[1] < 0.05:
            d_found = True
            d_req = d
            print(f'Fractional differentiation order found: {d_req}')

    # Plotting the results
    out[['adfStat', 'corr']].plot(secondary_y='adfStat')
    plt.axhline(out['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
    plt.title('ADF Statistic and Correlation vs. Differentiation Order')
    plt.show()
    
    return out, d_req