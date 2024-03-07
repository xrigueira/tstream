import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load raw X data (snow melted + precipitation) - just element 'data' of the loaded matlab object
raw_data = scipy.io.loadmat('data/SWIT_PET_y_masked_v3.mat')['data']

# Build SWIT, PET and Q data from data_for_ATT_time
# Get the first window of SWIT and PET and Q
data = raw_data[0][:,(0,1,-1)]

# Loop through the rest of the windows and append the last row
for i in range(1, len(raw_data)):
    data = np.append(data, [raw_data[i][-1,(0,1,-1)]], axis=0) # Reshape the second array to have the same number of dimensions

# Save the data to a csv file
np.savetxt('data/SWIT_PET_y_masked.csv', data, delimiter=',', fmt='%f')
