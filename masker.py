import scipy
import numpy as np
import matplotlib.pyplot as plt

# Load raw X data (snow melted + precipitation) - just element X1 of the loaded matlab object
raw_data = scipy.io.loadmat('X_28x18.mat')['X1']

# Load mask - element ismap of the loaded matlab object
mask = scipy.io.loadmat('mask_28x18.mat')['ismap']

# Apply (dot_product) mask on the raw data to generate masked data (X)
X = np.average(raw_data * mask, axis=(2, 3))

# Save the masked data
np.save('X_masked.npy', X, allow_pickle=False, fix_imports=False)

# Load y data (stream flow) - element 'Q1' of the loaded matlab object
y = scipy.io.loadmat('y.mat')['Q1']

# Plot the data
# plt.plot(X)
# plt.plot(y)
# plt.show()

# import torch

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name())
# print(torch.cuda.current_device())

# Continue with the example from Autotransformer
# https://github.com/thuml/Autoformer/blob/main/predict.ipynb
# https://github.com/thuml/Autoformer/tree/main