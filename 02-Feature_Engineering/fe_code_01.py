#%%
import pandas as pd
import os



#%%
# Add the 'target' column to each stock CSV file

# set the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set the directory containing your stock CSV files
stocks_data_dir = '../stocks_data'

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(stocks_data_dir) if f.endswith('.csv')]

# Loop through each CSV file
for file_name in csv_files:
    file_path = os.path.join(stocks_data_dir, file_name)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Check if 'stock_exret' column exists
    if 'stock_exret' in df.columns:
        # Create the 'target' column as the sign of 'stock_exret'
        df['target'] = df['stock_exret'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        # Save the updated DataFrame back to the CSV file (or to a new file)
        df.to_csv(file_path, index=False)
    else:
        print(f"'stock_exret' column not found in {file_name}")

# %%

# Add Weight Sampling
# NOO, WILL BE IN THE PIPELINE OF THE MODEL, SINCE IT IS TESTED 
