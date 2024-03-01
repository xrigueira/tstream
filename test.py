import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
# Load raw X data (snow melted + precipitation) - just element X1 of the loaded matlab object
raw_data = scipy.io.loadmat('data/data_for_ATT_time.mat')['data']
print(raw_data.shape)

raw_data = scipy.io.loadmat('data/data_for_ATT_time1y_v3.mat')['data']
print(raw_data.shape)

raw_data = scipy.io.loadmat('data/X_28x18.mat')['X1']
print(raw_data.shape)
