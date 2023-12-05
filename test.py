import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

timestamp_col_name = 'time'
test_size = 0.2

tgt_y_truth = np.load('tgt_y_truth.npy', allow_pickle=False, fix_imports=False)
tgt_y_hat = np.load('tgt_y_hat.npy', allow_pickle=False, fix_imports=False)

tgt_y_truth_train, tgt_y_truth_test = tgt_y_truth[:-(round(len(tgt_y_truth)*test_size))], tgt_y_truth[(round(len(tgt_y_truth)*(1-test_size))):]
tgt_y_hat_train, tgt_y_hat_test = tgt_y_hat[:-(round(len(tgt_y_truth)*test_size))], tgt_y_hat[(round(len(tgt_y_truth)*(1-test_size))):]

def nash_sutcliffe_efficiency(observed, modeled):
    
    mean_observed = np.mean(observed)
    numerator = np.sum((observed - modeled)**2)
    denominator = np.sum((observed - mean_observed)**2)
    
    nse = 1 - (numerator / denominator)
    return nse

# Calculate NSE for training data
nse_train = nash_sutcliffe_efficiency(tgt_y_truth_train, tgt_y_hat_train)
print("NSE for training data:", nse_train)

# Calculate NSE for test data
nse_test = nash_sutcliffe_efficiency(tgt_y_truth_test, tgt_y_hat_test)
print("NSE for test data:", nse_test)
