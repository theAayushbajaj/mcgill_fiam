#%%
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import pickle



#%%

# Add the 'target' column to each stock CSV file

# set the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set the directory containing your stock CSV files
stocks_data_dir = '../stocks_data'

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(stocks_data_dir) if f.endswith('.csv')]

#%%

# Stack all the CSV files into one DataFrame

# Create an empty list to store the DataFrames
dfs = []

# Loop through each CSV file
for file_name in tqdm(csv_files):
    file_path = os.path.join(stocks_data_dir, file_name)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Append the DataFrame to the list
    dfs.append(df)
    
# Concatenate all the DataFrames in the list
stacked_data = pd.concat(dfs, ignore_index=True)
stacked_data = stacked_data.sort_values(by='datetime')

stacked_data
# %%

# features
path = '../raw_data/factor_char_list.csv'
features = pd.read_csv(path)
features_list = features.values.ravel().tolist()
# Add created features
added_features = ['log_diff', 'frac_diff']
features_list+=added_features
X_dataset = stacked_data[features_list]
# to pickle
X_dataset.to_pickle('../objects/X_dataset.pkl')
y_dataset = stacked_data['target']
# %%
stacked_data
# save the stacked data as pickle
stacked_data.to_pickle('../objects/stacked_data.pkl')

# %%

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


# %%
X_pca = orthoFeats(X_dataset)
X_pca = pd.DataFrame(X_pca, index=X_dataset.index)
# name each column "pca_i" where i is the index of the column
X_pca.columns = ["pca_%d" % i for i in range(X_pca.shape[1])]
X_pca

# %%
df = X_dataset
nan_rows = df[df.isna().any(axis=1)]
nan_rows
# %%
