import numpy as np

file_path = "best_parameters.npy"
data = np.load(file_path, allow_pickle=True)
print(data)
