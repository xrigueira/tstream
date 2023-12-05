import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.notebook.plotting import init_notebook_plotting, render

import utils
import dataset as ds
import transformer as tst
import main as mn

"""Hyperparameter tuning with Bayesian optimization."""
# https://ax.dev/tutorials/tune_cnn_service.html

init_notebook_plotting()

# Define seed
torch.manual_seed(0)

# Hyperparams
test_size = 0.2
batch_size = 128
src_variables = ['X']
tgt_variables = ['y']
input_variables = src_variables + tgt_variables
timestamp_col_name = "time"

# Only use data from this date and onwards
cutoff_date = datetime.datetime(1980, 1, 1)

# d_model = 32
# n_heads = 2
# n_encoder_layers = 1
# n_decoder_layers = 1
encoder_sequence_len = 1461 # length of input given to encoder used to create the pre-summarized windows (4 years of data) 1461
crushed_encoder_sequence_len = 53 # Encoder sequence length afther summarizing the data when defining the dataset 53
decoder_sequence_len = 1 # length of input given to decoder
output_sequence_length = 1 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = encoder_sequence_len + output_sequence_length # used to slice data into sub-sequences
step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
# in_features_encoder_linear_layer = 32
# in_features_decoder_linear_layer = 32
max_sequence_len = encoder_sequence_len
batch_first = True

# Define seed
torch.manual_seed(0)

# Get device
device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using {device} device')

# Read data
data = utils.read_data(timestamp_col_name=timestamp_col_name)

# Extract train and test data
training_data = data[:-(round(len(data)*test_size))]
testing_data = data[(round(len(data)*(1-test_size))):]

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Fit scaler on the training set
scaler.fit(training_data.iloc[:, 1:])

training_data.iloc[:, 1:] = scaler.transform(training_data.iloc[:, 1:])
testing_data.iloc[:, 1:] = scaler.transform(testing_data.iloc[:, 1:])

# Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chuncks
training_indices = utils.get_indices(data=training_data, window_size=window_size, step_size=step_size)
testing_indices = utils.get_indices(data=testing_data, window_size=window_size, step_size=step_size)

# Make instance of the custom dataset class
training_data = ds.TransformerDataset(data=torch.tensor(training_data[input_variables].values).float(),
                                    indices=training_indices, encoder_sequence_len=encoder_sequence_len, 
                                    decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_length)
testing_data = ds.TransformerDataset(data=torch.tensor(testing_data[input_variables].values).float(),
                                    indices=testing_indices, encoder_sequence_len=encoder_sequence_len, 
                                    decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_length)

# Define data for inference
inference_data = training_data + testing_data

# Set up dataloaders
training_data = DataLoader(training_data, batch_size, shuffle=True)
testing_data = DataLoader(testing_data, batch_size, shuffle=True)
inference_data = DataLoader(inference_data, batch_size=1)

# Update the encoder sequence length to its crushed version
encoder_sequence_len = crushed_encoder_sequence_len

# Make src mask for the decoder with size
# [batch_size*n_heads, output_sequence_length, encoder_sequence_len]
src_mask = utils.masker(dim1=output_sequence_length, dim2=encoder_sequence_len).to(device)

# Make tgt mask for decoder with size
# [batch_size*n_heads, output_sequence_length, output_sequence_length]
tgt_mask = utils.masker(dim1=output_sequence_length, dim2=output_sequence_length).to(device)

# Initialize client
ax_client = AxClient()

# Set up experiment
# An experiment consists of a search space (parameters and parameter constraints) 
# and optimization configuration (objective name, minimization setting, and outcome constraints).
# Create an experiment with required arguments: name, parameters, and objective_name.
ax_client.create_experiment(
    name="tune_cnn_on_mnist",  # The name of the experiment.
    parameters=[
        {
            "name": "lr",  # The name of the parameter.
            "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
            "bounds": [1e-5, 0.001],  # The bounds for range parameters. 
            # "values" The possible values for choice parameters .
            # "value" The fixed value for fixed parameters.
            "value_type": "float",  # Optional, the value type ("int", "float", "bool" or "str"). Defaults to inference from type of "bounds".
            "log_scale": True,  # Optional, whether to use a log scale for range parameters. Defaults to False.
            # "is_ordered" Optional, a flag for choice parameters.
        },
        {
            "name": "d_model",
            "type": "choice",
            "values": [32, 64, 128, 256, 512],
            "value_type": "int",
            "log_scale": False,
        },
        {
            "name": "n_heads",
            "type": "choice",
            "values": [2, 4, 8],
            "value_type": "int",
            "log_scale": False,
        },
        {
            "name": "n_encoder_layers",
            "type": "choice",
            "values": [1, 2, 4],
            "value_type": "int",
            "log_scale": False,
        },
        {
            "name": "n_decoder_layers",
            "type": "choice",
            "values": [1, 2, 4],
            "value_type": "int",
            "log_scale": False,
        },
        {
            "name": "in_features_encoder_linear_layer",
            "type": "choice",
            "values": [32, 64, 128],
            "value_type": "int",
            "log_scale": False,
        },
        {
            "name": "in_features_decoder_linear_layer",
            "type": "choice",
            "values": [32, 64, 128],
            "value_type": "int",
            "log_scale": False,
        },
        
    ],
    
    objectives={"nse": ObjectiveProperties(minimize=False)},  # The objective name and minimization setting.
    # parameter_constraints: Optional, a list of strings of form "p1 >= p2" or "p1 + p2 <= some_bound".
    # outcome_constraints: Optional, a list of strings of form "constrained_metric <= some_bound".
)

# Define function to train, test and evaluate the model
def train_test_infer(parameterization):
    
    # Extract hyperparameters
    lr = parameterization["lr"]
    d_model = parameterization["d_model"]
    n_heads = parameterization["n_heads"]
    n_encoder_layers = parameterization["n_encoder_layers"]
    n_decoder_layers = parameterization["n_decoder_layers"]
    in_features_encoder_linear_layer = parameterization["in_features_encoder_linear_layer"]
    in_features_decoder_linear_layer = parameterization["in_features_decoder_linear_layer"]
    
    # Instantiate the transformer model and send it to device
    model = tst.TimeSeriesTransformer(input_size=len(src_variables), decoder_sequence_len=decoder_sequence_len, 
            batch_first=batch_first, d_model=d_model, n_encoder_layers=n_encoder_layers, 
            n_decoder_layers=n_decoder_layers, n_heads=n_heads, dropout_encoder=0.2, 
            dropout_decoder=0.2, dropout_pos_encoder=0.1, dim_feedforward_encoder=in_features_encoder_linear_layer, 
            dim_feedforward_decoder=in_features_decoder_linear_layer, num_predicted_features=len(tgt_variables)).to(device)
    
    # Define optimizer and loss function
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Update model in the training process and test it
    epochs = 180
    start_time = time.time()
    df_training = pd.DataFrame(columns=('epoch', 'loss_train'))
    df_testing = pd.DataFrame(columns=('epoch', 'loss_test'))
    for t in range(epochs):
        mn.train(training_data, model, src_mask, tgt_mask, loss_function, optimizer, device, df_training, epoch=t)
        mn.test(testing_data, model, src_mask, tgt_mask, loss_function, device, df_testing, epoch=t)
    print("Done! ---Execution time: %s seconds ---" % (time.time() - start_time))
    
    # Get NSE metric on inference
    tgt_y_truth, tgt_y_truth_train, tgt_y_truth_test, tgt_y_hat, tgt_y_hat_train, tgt_y_hat_test = mn.inference(inference_data, model, src_mask, tgt_mask, device, test_size)

    nse = mn.nash_sutcliffe_efficiency(tgt_y_truth_test, tgt_y_hat_test)
    
    return nse

# Run optimization loop
# Attach the trial
ax_client.attach_trial(
    parameters={"lr": 0.00001, "d_model": 32, "n_heads": 2, "n_encoder_layers": 1, "n_decoder_layers": 1,
                "in_features_encoder_linear_layer": 32, "in_features_decoder_linear_layer": 32}
)

# Get the parameters and run the trial 
baseline_parameters = ax_client.get_trial_parameters(trial_index=0)
ax_client.complete_trial(trial_index=0, raw_data=train_test_infer(baseline_parameters))

for i in range(100):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=train_test_infer(parameters))

# Maximize the number of trials in parallel
ax_client.get_max_parallelism()

# Retrieve best parameters
best_parameters, values = ax_client.get_best_parameters()
np.save('best_parameters.npy', best_parameters)
print(best_parameters)

mean, covariance = values