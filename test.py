import numpy as np

all_mha_weights = np.load('all_mha_weights.npy', allow_pickle=False, fix_imports=False)
for i in all_mha_weights:
    print(i)

