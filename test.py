import numpy as np

all_mha_weights = np.load('all_mha_weights.npy', allow_pickle=False, fix_imports=False)
all_sa_weights = np.load('all_sa_weights.npy', allow_pickle=False, fix_imports=False)

print(all_mha_weights[100])

