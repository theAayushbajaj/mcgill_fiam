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

# ADDING TARGET COLUMN TO EACH CSV FILE

# Loop through each CSV file
for file_name in csv_files:
    file_path = os.path.join(stocks_data_dir, file_name)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # ['prediction', 'probability']
    # add column, prediction, random number from -1, 1
    df['prediction'] = np.random.uniform(-1, 1, df.shape[0])
    df['probability'] = np.abs(df['prediction'])
    df['prediction'] = np.sign(df['prediction'])
    
    df.to_csv(file_path, index=False)
    
# %%

# path = '../stocks_data/AAPL_03783310_14593.csv'
# df_appl = pd.read_csv(path)
# df_appl[['stock_exret', 'probability']]

# %%
