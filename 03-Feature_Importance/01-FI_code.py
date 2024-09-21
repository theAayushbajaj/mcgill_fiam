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
import numpy as np
import pandas as pd

# SNIPPET 8.5 COMPUTATION OF ORTHOGONAL FEATURES (Modified for Variance and Loadings)
def get_eVec(dot, varThres):
    # Compute eigenvalues (eVal) and eigenvectors (eVec) from dot product matrix
    eVal, eVec = np.linalg.eigh(dot)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = eVal.argsort()[::-1]  # Sort eigenvalues in descending order
    eVal, eVec = eVal[idx], eVec[:, idx]
    
    # Keep only positive eigenvalues
    eVal = pd.Series(eVal, index=["PC_" + str(i + 1) for i in range(eVal.shape[0])])
    eVec = pd.DataFrame(eVec, index=dot.index, columns=eVal.index)
    
    # Compute cumulative variance explained
    cumVar = eVal.cumsum() / eVal.sum()
    
    # Select the number of principal components that explain at least varThres variance
    dim = cumVar.values.searchsorted(varThres)
    
    # Keep only the selected principal components
    eVal, eVec = eVal.iloc[: dim + 1], eVec.iloc[:, : dim + 1]
    
    # Return eigenvalues (variance explained) and eigenvectors (loadings)
    return eVal, eVec, cumVar.iloc[: dim + 1]


# Function to standardize features and compute orthogonal features (PCA)
def orthoFeats(dfX, varThres=0.95):
    # Standardize the feature matrix
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)
    
    # Compute the dot product (covariance matrix)
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)
    
    # Get eigenvalues (variance explained) and eigenvectors (loadings)
    eVal, eVec, cumVar = get_eVec(dot, varThres)
    
    # Transform the original features into the new principal components
    dfP = np.dot(dfZ, eVec)
    
    return dfP, eVal, eVec, cumVar


# Apply the function to your dataset
X_pca, eigenvalues, loadings, cumulative_variance = orthoFeats(X_dataset, varThres=0.95)

# Convert PCA-transformed data into a DataFrame with appropriate column names
X_pca = pd.DataFrame(X_pca, index=X_dataset.index)
X_pca.columns = ["pca_%d" % i for i in range(X_pca.shape[1])]

# Print the variance explained by each principal component (eigenvalues)
print("Variance Explained (Eigenvalues):")
print(eigenvalues)

# Print the cumulative variance explained
print("\nCumulative Variance Explained:")
print(cumulative_variance)

# Print the loadings (eigenvectors)
print("\nLoadings (Eigenvectors):")
print(loadings)

# Now X_pca contains the PCA-transformed features, eigenvalues contain variance explained,
# and loadings give the contribution of each original feature to each principal component.


# %%
import numpy as np
import pandas as pd

variance_explained = eigenvalues / eigenvalues.sum()


# Function to get top important features based on variance explained and loadings
def get_top_features(variance_explained, loadings, top_n=20):
    """
    Rank features by their importance using the variance explained by each principal component 
    and the absolute value of the feature's loadings.
    
    Arguments:
    - variance_explained: Series, variance explained by each principal component.
    - loadings: DataFrame, loadings (eigenvectors) where columns are principal components and rows are features.
    - top_n: Number of top features to return.
    
    Returns:
    - ranked_features: DataFrame with features ranked by importance.
    """
    # Ensure the absolute values of the loadings are used
    abs_loadings = loadings.abs()
    
    # Multiply each feature's loading by the variance explained of the respective principal component
    feature_importance = abs_loadings.mul(variance_explained, axis=1)
    
    # Sum the weighted contributions across all principal components for each feature
    feature_importance['total_importance'] = feature_importance.sum(axis=1)
    
    # Sort features by their total importance in descending order
    ranked_features = feature_importance[['total_importance']].sort_values(by='total_importance', ascending=False)
    
    # Return the top N important features
    return ranked_features.head(top_n)

# Apply the function to your data
top_20_features = get_top_features(variance_explained, loadings, top_n=20)

# Print the top 20 important features
print("Top 20 Important Features:")
print(top_20_features)


# %%
