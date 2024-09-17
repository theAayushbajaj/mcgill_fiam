import pandas as pd


data = pd.read_csv('<path-to-hackathon_sample_v2.csv>')

# Define the minimum number of records required
min_records = 180

# Group by 'cusip' and 'permno', and order each group by 'year' and 'month'
grouped = data.groupby(['cusip', 'permno'], group_keys=False)

# Function to apply to each group: sort by 'year' and 'month', then assign the last stock_ticker
def apply_last_stock_ticker(group):
    # Check if the group size meets the minimum record requirement
    if len(group) >= min_records:
        # Sort by 'year' and 'month'
        group = group.sort_values(by=['year', 'month'])
        # Get the last stock_ticker used and assign it to all rows in the group
        last_ticker = group['stock_ticker'].iloc[-1]
        group['stock_ticker'] = last_ticker
        # Save the group as a CSV file using the format 'cusip_permno.csv'
        file_name = f"<path-to-store-tick-data>/{str(group['cusip'].iloc[0])}_{str(group['permno'].iloc[0])}.csv"
        group.to_csv(file_name, index=False)
        return group

# Apply the function to each group
result = grouped.apply(apply_last_stock_ticker)