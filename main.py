import time
import datetime
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from prettytable import PrettyTable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
import dataset as ds
import transformer as tst

# Define the training step
def train(dataloader, model, src_mask, tgt_mask, loss_function, optimizer, device, df_training, epoch):
    size = len(dataloader.dataset)
    model.train()
    training_loss = [] # For plotting purposes
    for i, batch in enumerate(dataloader):
        src, tgt, tgt_y = batch
        src, tgt, tgt_y = src.to(device), tgt.to(device), tgt.to(device)

        # Zero out gradients for every batch
        optimizer.zero_grad()
        
        # Compute prediction error
        pred = model(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask).to(device)
        loss = loss_function(pred, tgt_y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Save results for plotting
        training_loss.append(loss.item())
        epoch_train_loss = np.mean(training_loss)
        df_training.loc[epoch] = [epoch, epoch_train_loss]
        
        # if i % 20 == 0:
        #     print('Current batch', i)
        #     loss, current = loss.item(), (i + 1) * len(src)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Define testing step
def test(dataloader, model, src_mask, tgt_mask, loss_function, device, df_testing, epoch):
    num_batches = len(dataloader)
    model.eval()
    testing_loss = [] # For plotting purposes
    with torch.no_grad():
        for batch in dataloader:
            src, tgt, tgt_y = batch
            src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)
            
            pred = model(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask).to(device)
            loss = loss_function(pred, tgt_y.unsqueeze(2))
            
            # Save results for plotting
            testing_loss.append(loss.item())
            epoch_test_loss = np.mean(testing_loss)
            df_testing.loc[epoch] = [epoch, epoch_test_loss]
    
    loss /= num_batches
    # print(f"Avg test loss: {loss:>8f}")

# Define inference step
def inference(inference_data, model, src_mask, tgt_mask, device, test_size):
    # Get ground truth
    tgt_y_truth = torch.zeros(len(inference_data))
    for i, (src, tgt, tgt_y) in enumerate(inference_data):
        tgt_y_truth[i] = tgt_y

    # Define tensor to store the predictions
    tgt_y_hat = torch.zeros((len(inference_data)), device=device)

    # Perform inference
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(inference_data):
            src, tgt, tgt_y = sample
            src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)

            pred = model(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask).to(device)
            # print(pred, tgt_y)
            tgt_y_hat[i] = pred

    # Pass target_y_hat to cpu for plotting purposes
    tgt_y_hat = tgt_y_hat.cpu()

    tgt_y_truth_train, tgt_y_truth_test = tgt_y_truth[:-(round(len(tgt_y_truth)*test_size))].numpy(), tgt_y_truth[(round(len(tgt_y_truth)*(1-test_size))):].numpy()
    tgt_y_hat_train, tgt_y_hat_test = tgt_y_hat[:-(round(len(tgt_y_truth)*test_size))].numpy(), tgt_y_hat[(round(len(tgt_y_truth)*(1-test_size))):].numpy()
    # np.save('tgt_y_truth.npy', tgt_y_truth, allow_pickle=False, fix_imports=False)
    # np.save('tgt_y_hat.npy', tgt_y_hat, allow_pickle=False, fix_imports=False)
    
    return tgt_y_truth, tgt_y_truth_train, tgt_y_truth_test, tgt_y_hat, tgt_y_hat_train, tgt_y_hat_test

# Define function to get and format the number of parameters
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    
    print(table)
    print(f"Total trainable parameters: {total_params}")
    
    return total_params

# Define Nash-Sutcliffe efficiency
def nash_sutcliffe_efficiency(observed, modeled):
    mean_observed = np.mean(observed)
    numerator = np.sum((observed - modeled)**2)
    denominator = np.sum((observed - mean_observed)**2)
    
    nse = 1 - (numerator / denominator)
    
    return nse

if __name__ == '__main__':
    
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

    d_model = 32
    n_heads = 2
    n_decoder_layers = 1
    n_encoder_layers = 1
    encoder_sequence_len = 1461 # length of input given to encoder used to create the pre-summarized windows (4 years of data) 1461
    crushed_encoder_sequence_len = 53 # Encoder sequence length afther summarizing the data when defining the dataset 53
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

    # Instantiate the transformer model and send it to device
    model = tst.TimeSeriesTransformer(input_size=len(src_variables), decoder_sequence_len=decoder_sequence_len, 
                                    batch_first=batch_first, d_model=d_model, n_encoder_layers=n_encoder_layers, 
                                    n_decoder_layers=n_decoder_layers, n_heads=n_heads, dropout_encoder=0.2, 
                                    dropout_decoder=0.2, dropout_pos_encoder=0.1, dim_feedforward_encoder=in_features_encoder_linear_layer, 
                                    dim_feedforward_decoder=in_features_decoder_linear_layer, num_predicted_features=len(tgt_variables)).to(device)
    
    # Print model and number of parameters
    print('Defined model:\n', model)
    count_parameters(model)
    
    # Make src mask for the decoder with size
    # [batch_size*n_heads, output_sequence_length, encoder_sequence_len]
    src_mask = utils.masker(dim1=output_sequence_length, dim2=encoder_sequence_len).to(device)
    
    # Make tgt mask for decoder with size
    # [batch_size*n_heads, output_sequence_length, output_sequence_length]
    tgt_mask = utils.masker(dim1=output_sequence_length, dim2=output_sequence_length).to(device)
    
    # Define optimizer and loss function
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Update model in the training process and test it
    epochs = 250
    start_time = time.time()
    df_training = pd.DataFrame(columns=('epoch', 'loss_train'))
    df_testing = pd.DataFrame(columns=('epoch', 'loss_test'))
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_data, model, src_mask, tgt_mask, loss_function, optimizer, device, df_training, epoch=t)
        test(testing_data, model, src_mask, tgt_mask, loss_function, device, df_testing, epoch=t)
    print("Done! ---Execution time: %s seconds ---" % (time.time() - start_time))

    # # Save the model
    # torch.save(model, "models/model.pth")
    # print("Saved PyTorch entire model to models/model.pth")

    # # Load the model
    # model = torch.load("models/model.pth").to(device)
    # print('Loaded PyTorch model from models/model.pth')

    # Inference
    tgt_y_truth, tgt_y_truth_train, tgt_y_truth_test, tgt_y_hat, tgt_y_hat_train, tgt_y_hat_test = inference(inference_data, model, src_mask, tgt_mask, device, test_size)
    
    # Plot loss
    plt.figure(1);plt.clf()
    plt.plot(df_training['epoch'], df_training['loss_train'], '-o', label='loss train')
    plt.plot(df_training['epoch'], df_testing['loss_test'], '-o', label='loss test')
    plt.yscale('log')
    plt.xlabel(r'epoch')
    plt.ylabel(r'loss')
    plt.legend()
    plt.show()

    # Plot inference
    plt.figure(2);plt.clf()
    plt.plot(tgt_y_truth, label='observed')
    plt.plot(range(len(tgt_y_hat_train)), tgt_y_hat_train, label='predicted train')
    plt.plot(range(len(tgt_y_hat_train), len(tgt_y_hat)), tgt_y_hat_test, color='lightskyblue', label='predicted test')
    plt.xlabel(r'time (days)')
    plt.ylabel(r'y')
    plt.legend()
    plt.show()

    # Metrics
    from sklearn.metrics import mean_squared_error

    nse_train = nash_sutcliffe_efficiency(tgt_y_truth_train, tgt_y_hat_train)
    mse_train = mean_squared_error(tgt_y_truth_train, tgt_y_hat_train)
    print('-- training result ')
    print('NSE = ', nse_train)
    print('MSE = ', mse_train)

    nse_test = nash_sutcliffe_efficiency(tgt_y_truth_test, tgt_y_hat_test)
    mse_test = mean_squared_error(tgt_y_truth_test, tgt_y_hat_test)
    print('\n-- test result ')
    print('NSE = ', nse_test)
    print('MSE = ', mse_test)