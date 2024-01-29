import pandas as pd

# Load the data
raw_data = pd.read_csv('data/PET_SWIT.csv')

# Define the column names
raw_data.columns = ['x1', 'x2', 'y']

# Generate dates starting from Jan 2nd, 1980
start_date = '1980-01-02'
raw_data['time'] = pd.date_range(start=start_date, periods=len(raw_data), freq='D')

# Change column names
raw_data = raw_data[['time', 'x1', 'x2', 'y']]  # Reorder columns

# Save the data
raw_data.to_csv('data/data_2.csv', sep=',', index=False, encoding='utf-8')