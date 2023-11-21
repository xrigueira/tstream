import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
import dataset as ds
import transformer as tst

# Hyperparams
test_size = 0.1
batch_size = 128
src_variables = ['X']
tgt_variables = ['y']
input_variables = src_variables + tgt_variables
timestamp_col_name = "time"

# Only use data from this date and onwards
cutoff_date = datetime.datetime(1980, 1, 1) 

# Params
d_model = 128
n_heads = 2
n_decoder_layers = 1
n_encoder_layers = 1
encoder_sequence_len = 1461 # length of input given to encoder used to create the pre-summarized windows (4 years of data)
crushed_encoder_sequence_len = 53 # Encoder sequence length afther summarizing the data when defining the dataset
decoder_sequence_len = 1 # length of input given to decoder
output_sequence_length = 1 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = encoder_sequence_len + output_sequence_length # used to slice data into sub-sequences
step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 32
in_features_decoder_linear_layer = 32
max_sequence_len = encoder_sequence_len
batch_first = True

# Get device
device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using {device} device')

# Read data
data = utils.read_data(timestamp_col_name=timestamp_col_name)

# Extract train and test data
training_data = data[:-(round(len(data)*test_size))]
testing_data = data[(round(len(data)*(1-test_size))):]

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

# Make dataloaders
training_data = DataLoader(training_data, batch_size)
testing_data = DataLoader(testing_data, batch_size)

# Update the encoder sequence length to its crushed version
encoder_sequence_len = crushed_encoder_sequence_len

# Instantiate the transformer model and send it to device
model = tst.TimeSeriesTransformer(input_size=len(src_variables), decoder_sequence_len=decoder_sequence_len, 
                                batch_first=batch_first, num_predicted_features=len(tgt_variables)).to(device)

# Make src mask for the decoder with size
# [batch_size*n_heads, output_sequence_length, encoder_sequence_len]
src_mask = utils.masker(dim1=output_sequence_length, dim2=encoder_sequence_len).to(device)

# Make tgt mask for decoder with size
# [batch_size*n_heads, output_sequence_length, output_sequence_length]
tgt_mask = utils.masker(dim1=output_sequence_length, dim2=output_sequence_length).to(device)

# Define optimizer and loss function
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the training step
def train(dataloader, model, loss_function, optimizer, device, df_training, epoch):
    size = len(dataloader.dataset)
    model.train()
    training_loss = [] # For plotting purposes
    for i, batch in enumerate(dataloader):
        src, tgt, tgt_y = batch
        src, tgt, tgt_y = src.to(device), tgt.to(device), tgt.to(device)
        
        # Compute prediction error
        pred = model(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask).to(device)
        loss = loss_function(pred, tgt_y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Save results for plotting
        training_loss.append(loss.item())
        epoch_train_loss = np.mean(training_loss)
        df_training.loc[epoch] = [epoch, epoch_train_loss]
        
        print('Current batch', i)
        if i % 1== 0:
            loss, current = loss.item(), (i + 1) * len(src)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Define testing step
def test(dataloader, model, loss_function, device, df_testing, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    testing_loss = [] # For plotting purposes
    with torch.no_grad():
        for batch in dataloader:
            src, tgt, tgt_y = batch
            src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)
            
            pred = model(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask).to(device)
            test_loss += loss_function(pred, tgt_y.unsqueeze(2))
            
            # Save results for plotting
            testing_loss.append(test_loss.item())
            epoch_test_loss = np.mean(testing_loss)
            df_testing.loc[epoch] = [epoch, epoch_test_loss]
    
    test_loss /+ num_batches
    print(f"Avg loss: {test_loss:>8f}")

# Update model in the training process and test it
epochs = 1
df_training = pd.DataFrame(columns=('epoch', 'loss_train'))
df_testing = pd.DataFrame(columns=('epoch', 'loss_test'))
for t in range (epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(training_data, model, loss_function, optimizer, device, df_training, epoch=t)
    test(testing_data, model, loss_function, device, df_testing, epoch=t)
print('Done!')

# # Save the model
# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")

# Inference
# Get ground truth
tgt_y_truth = torch.zeros(len(inference_data))
for i, (src, tgt, tgt_y) in enumerate(inference_data):
    tgt_y_truth[i] = tgt_y

tgt_y_hat = torch.zeros((len(inference_data)), device=device)

# Perform inference
with torch.no_grad():
    for i, sample in enumerate(inference_data):
        src, tgt, tgt_y = sample
        src, tgt, tgt_y = src.unsqueeze(0), tgt.unsqueeze(0), tgt_y.unsqueeze(0)
        
        src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)

        pred = model(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask).to(device)
        tgt_y_hat[i] = pred

# Pass target_y_hat to cpu for plotting purposes
tgt_y_hat = tgt_y_hat.cpu()

# Plot loss
plt.figure(1);plt.clf()
plt.plot(df_training['epoch'], df_training['loss_train'], '-o', label='loss train')
plt.plot(df_training['epoch'], df_testing['loss_test'], '-o', label='loss test')
plt.xlabel(r'epoch')
plt.ylabel(r'loss')
plt.legend()
plt.show()

# Plot inference
plt.figure(2);plt.clf()
plt.plot(tgt_y_truth, label='observed')
plt.plot(tgt_y_hat, label='predicted')
plt.xlabel(r'time (days)')
plt.ylabel(r'y')
plt.legend()
plt.show()