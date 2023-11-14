import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
import dataset as ds
import timeseries_transformer as tst

# Hyperparams
test_size = 0.1
batch_size = 128
tgt_col_name = "FCR_N_PriceEUR"
timestamp_col_name = "timestamp"

# Only use data from this date and onwards
cutoff_date = datetime.datetime(2017, 1, 1) 

## Params
d_model = 512
n_heads = 8
n_decoder_layers = 1
n_encoder_layers = 1
encoder_sequence_len = 192 # length of input given to encoder
decoder_sequence_len = 48 # length of input given to decoder
output_sequence_length = 48 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = encoder_sequence_len + output_sequence_length # used to slice data into sub-sequences
step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_sequence_len = encoder_sequence_len
batch_first = True

# Get device
device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using {device} device')

# Define input variables 
exogenous_vars = [] # should contain strings. Each string must correspond to a column name
input_variables = [tgt_col_name] + exogenous_vars
tgt_idx = 0 # index position of target in batched tgt_y

input_size = len(input_variables)

# Read data
data = utils.read_data(timestamp_col_name=timestamp_col_name)

# Extract train and test data
training_data = data[:-(round(len(data)*test_size))]
testing_data = data[(round(len(data)*test_size)):]

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

# Make dataloaders
training_data = DataLoader(training_data, batch_size)
testing_data = DataLoader(testing_data, batch_size)

# Instantiate the transformer model and send it to device
model = tst.TimeSeriesTransformer(input_size=len(input_variables), decoder_sequence_len=decoder_sequence_len, 
                                batch_first=batch_first, num_predicted_features=1).to(device)

# Make src mask for the decoder with size
# [batch_size*n_heads, output_sequence_length, encoder_sequence_len]
src_mask = utils.masker(dim1=output_sequence_length, dim2=encoder_sequence_len)

# Make tgt mask for decoder with size
# [batch_size*n_heads, output_sequence_length, output_sequence_length]
tgt_mask = utils.masker(dim1=output_sequence_length, dim2=output_sequence_length)

# Define optimizer and loss function
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the training step
def train(dataloader, model, loss_function, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for i, batch in enumerate(dataloader):
        src, tgt, tgt_y = batch
        src, tgt, tgt_y = src.to(device), tgt.to(device), tgt.to(device)
        
        # Compute prediction error
        pred = model(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        loss = loss_function(pred, tgt_y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('Current batch', i)
        if i % 5 == 0:
            loss, current = loss.item(), (i + 1) * len(src)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Define testing step
def test(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, batch in dataloader:
            src, trg, trg_y = batch
            src, trg, trg_y = src.to(device), trg.to(device), trg_y.to(device)
            
            pred = model(src=src, tgt=trg, src_mask=src_mask, tgt_mask=tgt_mask)
            pred = pred.squeeze(-1)
            test_loss += loss_function(pred, trg_y).item()
    
    test_loss /+ num_batches
    print(f"Avg loss: {test_loss:>8f}")

# Update model in the training process and test it
epochs = 5
for t in range (epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(training_data, model, loss_function, optimizer, device)
    test(testing_data, model, loss_function)
print('Done!')

# # Save the model
# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")