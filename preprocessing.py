import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

variables = 'masked_SWIT'
# variables = 'unmasked_SWIT_PET'

if variables == 'masked_SWIT':

    # Load raw X data (snow melted + precipitation) - just element X1 of the loaded matlab object
    raw_data = scipy.io.loadmat('data/SWIT_28x18.mat')['X1']

    # Load mask - element ismap of the loaded matlab object
    mask = scipy.io.loadmat('data/mask_28x18.mat')['ismap']

    # Apply (dot_product) mask on the raw data to generate masked data (X)
    X = np.average(raw_data * mask, axis=(2, 3))

    # Save the masked data
    np.save('data/SWIT_masked.npy', X, allow_pickle=False, fix_imports=False)

    # Load y data (stream flow) - element 'Q1' of the loaded matlab object
    y = scipy.io.loadmat('data/y.mat')['Q1']

    # # Plot the data
    # plt.plot(X)
    # plt.plot(y)
    # plt.show()

    # Save the data to 
    # Generate dates starting from Jan 1st, 1980
    start_date = '1980-01-01'
    dates = pd.date_range(start=start_date, periods=len(X), freq='D')

    # Add an empty element at the beginning of y
    y = np.insert(y, 0, None)

    # Create a DataFrame
    data = pd.DataFrame({'time': dates, 'X': X.flatten(), 'y': y.flatten()})

    # Normalize the data
    scaler = MinMaxScaler()
    data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:])

    # Save the database
    data.to_csv('data/data.csv', sep=',', index=False, encoding='utf-8')

else:

    # Load raw X data (snow melted + precipitation) - just element 'data' of the loaded matlab object
    raw_data = scipy.io.loadmat('data/SWIT_PET_y_masked.mat')['data']
    
    # Build SWIT, PET and Q data from data_for_ATT_time
    # Get the first window of SWIT and PET and Q
    data = raw_data[0][:,(0,1,-1)]

    # Loop through the rest of the windows and append the last row
    for i in range(1, len(raw_data)):
        data = np.append(data, [raw_data[i][-1,(0,1,-1)]], axis=0) # Reshape the second array to have the same number of dimensions

    data = pd.DataFrame(data, columns=['x1', 'x2', 'y'])
    
    # Generate dates starting from Jan 2nd, 1980
    start_date = '1980-01-01'
    data['time'] = pd.date_range(start=start_date, periods=len(data), freq='D')

    # Change column names
    data = data[['time', 'x1', 'x2', 'y']]  # Reorder columns

    # Normalize the data
    scaler = MinMaxScaler()
    data.iloc[:, 1:] = scaler.fit_transform(data.iloc[:, 1:])

    # Save the data
    data.to_csv('data/data.csv', sep=',', index=False, encoding='utf-8')