import scipy
import numpy as np
import matplotlib.pyplot as plt

# Load raw X data (snow melted + precipitation) - just element X1 of the loaded matlab object
raw_data = scipy.io.loadmat('SWIT_28x18_python_new.mat')['X1']

# Load mask - element ismap of the loaded matlab object
mask = scipy.io.loadmat('ismap_28x18_updated')['ismap']

# Apply (dot_product) mask on the raw data to generate masked data (X)
X = np.average(raw_data * mask, axis=(2, 3))

# Load y data (stream flow) - element 'Q1' of the loaded matlab object
y = scipy.io.loadmat('Q_LR.mat')['Q1']

# Plot the data
plt.plot(X)
plt.plot(y)
plt.show()

# Continue with the example from Autotransformer
# https://github.com/thuml/Autoformer/blob/main/predict.ipynb
# https://github.com/thuml/Autoformer/tree/main