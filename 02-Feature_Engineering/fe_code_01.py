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

#%%

import pandas as pd
import os
import yfinance as yf
import time
import logging

# Set up logging
logging.basicConfig(filename='add_closing_price.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# Directory containing your stock CSV files
stocks_data_dir = '../stocks_data'

# List all CSV files in the directory
csv_files = [f for f in os.listdir(stocks_data_dir) if f.endswith('.csv')]

for file_name in csv_files:
    file_path = os.path.join(stocks_data_dir, file_name)
    
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Ensure 'date' column is in datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Get the stock ticker
        stock_ticker = df['stock_ticker'].iloc[-1]
        
        # Check for valid stock ticker
        if pd.isna(stock_ticker) or stock_ticker.strip() == '':
            logging.warning(f"Invalid or missing stock ticker in {file_name}")
            continue  # Skip this file
        
        # Get the date range from your data
        start_date = df['date'].min()
        end_date = df['date'].max()
        
        # Fetch historical data using yfinance
        ticker = yf.Ticker(stock_ticker)
        hist = ticker.history(start=start_date - pd.Timedelta(days=1), end=end_date + pd.Timedelta(days=1))
        
        if hist.empty:
            logging.warning(f"No historical data found for {stock_ticker} in {file_name}")
            continue  # Skip this file
        
        hist = hist.reset_index()
        hist.rename(columns={'Date': 'date'}, inplace=True)
        
        # Merge the closing prices
        df = df.merge(hist[['date', 'Close']], on='date', how='left')
        
        # Handle missing 'Close' values
        if df['Close'].isnull().any():
            logging.info(f"Missing 'Close' values in {file_name}. Applying forward-fill.")
            df['Close'] = df['Close'].fillna(method='ffill')
        
        # Save the updated DataFrame
        df.to_csv(file_path, index=False)
        
        logging.info(f"Successfully updated {file_name} with closing prices.")
        
    except Exception as e:
        logging.error(f"Error processing {file_name}: {e}")
    
    # Sleep to avoid rate limits
    time.sleep(1)

# %%
