import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# variables = 'masked_SWIT'
variables = 'unmasked_SWIT_PET'

if variables == 'masked_PET':

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
    df = pd.DataFrame({'time': dates, 'X': X.flatten(), 'y': y.flatten()})

    # Save the database
    df.to_csv('data/data.csv', sep=',', index=False, encoding='utf-8')

else:

    # Load raw X data (snow melted + precipitation) - just element 'data' of the loaded matlab object
    raw_data = scipy.io.loadmat('data/SWIT_PET_y_masked.mat')['data']

    # Build SWIT, PET and Q data from data_for_ATT_time
    # Get the first window of SWIT and PET and Q
    data = raw_data[0][:,(0,1,-1)]

    # Loop through the rest of the windows and append the last row
    for i in range(1, len(raw_data)):
        data = np.append(data, [raw_data[i][-1,(0,1,-1)]], axis=0) # Reshape the second array to have the same number of dimensions

    # Save the data to a csv file
    np.savetxt('data/SWIT_PET_y_masked.csv', data, delimiter=',', fmt='%f')

    # Load the data
    raw_data = pd.read_csv('data/SWIT_PET_y_masked.csv')

    # Define the column names
    raw_data.columns = ['x1', 'x2', 'y']

    # Generate dates starting from Jan 2nd, 1980
    start_date = '1980-01-01'
    raw_data['time'] = pd.date_range(start=start_date, periods=len(raw_data), freq='D')

    # Change column names
    raw_data = raw_data[['time', 'x1', 'x2', 'y']]  # Reorder columns

    # Save the data
    raw_data.to_csv('data/data_2.csv', sep=',', index=False, encoding='utf-8')