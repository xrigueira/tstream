import time
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

# Define the training step
def train(dataloader, model, src_mask, tgt_mask, loss_function, optimizer, device, df_training, epoch):
    
    size = len(dataloader.dataset)
    model.train()
    training_loss = [] # For plotting purposes
    for i, batch in enumerate(dataloader):
        src, tgt, tgt_y = batch
        src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)

        # Zero out gradients for every batch
        optimizer.zero_grad()
        
        # Compute prediction error
        pred, sa_weights, mha_weights = model(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        pred = pred.to(device)
        loss = loss_function(pred, tgt_y.unsqueeze(2))
        
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
def val(dataloader, model, src_mask, tgt_mask, loss_function, device, df_validation, epoch):
    
    num_batches = len(dataloader)
    model.eval()
    validation_loss = [] # For plotting purposes
    with torch.no_grad():
        for batch in dataloader:
            src, tgt, tgt_y = batch
            src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)
            
            pred, sa_weights, mha_weights = model(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask)
            pred = pred.to(device)
            loss = loss_function(pred, tgt_y.unsqueeze(2))
            
            # Save results for plotting
            validation_loss.append(loss.item())
            epoch_val_loss = np.mean(validation_loss)
            df_validation.loc[epoch] = [epoch, epoch_val_loss]
    
    loss /= num_batches
    # print(f"Avg test loss: {loss:>8f}")

# Define inference step
def test(dataloader, model, src_mask, tgt_mask, device):
    
    # Get ground truth
    tgt_y_truth = torch.zeros(len(dataloader))
    for i, (src, tgt, tgt_y) in enumerate(dataloader):
        tgt_y_truth[i] = tgt_y

    # Define tensor to store the predictions
    tgt_y_hat = torch.zeros((len(dataloader)), device=device)

    # Define list to store the multi-head self attention weights
    all_sa_weights_inference = []
    all_mha_weights_inference = []
    
    # Perform inference
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            src, tgt, tgt_y = sample
            src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)

            pred, sa_weights, mha_weights = model(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask)
            all_sa_weights_inference.append(sa_weights)
            all_mha_weights_inference.append(mha_weights)
            pred = pred.to(device)

            tgt_y_hat[i] = pred

    # Save inference attention for the last step
    # np.save('all_sa_weights.npy', [sa_weight.cpu() for sa_weight in all_sa_weights_inference], allow_pickle=False, fix_imports=False)
    # np.save('all_mha_weights.npy', [mha_weight.cpu() for mha_weight in all_mha_weights_inference], allow_pickle=False, fix_imports=False)
    
    # Pass target_y_hat to cpu for plotting purposes
    tgt_y_hat = tgt_y_hat.cpu()

    tgt_y_truth, tgt_y_hat = tgt_y_truth.numpy(), tgt_y_hat.numpy()

    # Save ground truth and predictions
    # np.save('tgt_y_truth.npy', tgt_y_truth, allow_pickle=False, fix_imports=False)
    # np.save('tgt_y_hat.npy', tgt_y_hat, allow_pickle=False, fix_imports=False)
    
    return tgt_y_truth, tgt_y_hat

if __name__ == '__main__':
    
    # Define seed
    torch.manual_seed(0)
    
    # Hyperparams
    batch_size = 128
    validation_size = 0.125
    src_variables = ['X']
    tgt_variables = ['y']
    input_variables = src_variables + tgt_variables
    timestamp_col_name = "time"

    # Only use data from this date and onwards
    cutoff_date = datetime.datetime(1980, 1, 1) 

    d_model = 32
    n_heads = 2
    n_decoder_layers = 0 # Remember that with the current implementation it always has a decoder layer that returns the weights
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
    training_val_lower_bound = datetime.datetime(1980, 10, 1)
    training_val_upper_bound = datetime.datetime(2010, 9, 30)

    # Extract train/validation and test data
    training_val_data = data[(training_val_lower_bound <= data.index) & (data.index <= training_val_upper_bound)]
    testing_data = data[data.index > training_val_upper_bound]
    
    # Normalize the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    # Fit scaler on the training set
    scaler.fit(training_val_data.iloc[:, 1:])

    # Transform the training and test sets
    training_val_data.iloc[:, 1:] = scaler.transform(training_val_data.iloc[:, 1:])
    testing_data.iloc[:, 1:] = scaler.transform(testing_data.iloc[:, 1:])

    # Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chuncks
    training_val_indices = utils.get_indices(data=training_val_data, window_size=window_size, step_size=step_size)

    # Divide train data into train and validation data with a 8:1 ratio using the indices.
    # This is done this way because we need 4 years of data to create the summarized nonuniform timesteps,
    # what limits the size of the validation set. However, with this implementation, we use data from the
    # traning part to build the summarized nonuniform timesteps for the validation set. For example, if
    # we use the current validation size, the set would have less than four years of data and would not
    # be able to create the summarized nonuniform timesteps.
    training_indices = training_val_indices[:-(round(len(training_val_indices)*validation_size))]
    validation_indices = training_val_indices[(round(len(training_val_indices)*(1-validation_size))):]
    
    testing_indices = utils.get_indices(data=testing_data, window_size=window_size, step_size=step_size)
    
    # Make instance of the custom dataset class
    training_data = ds.TransformerDataset(data=torch.tensor(training_val_data[input_variables].values).float(),
                                        indices=training_indices, encoder_sequence_len=encoder_sequence_len, 
                                        decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_length)
    validation_data = ds.TransformerDataset(data=torch.tensor(training_val_data[input_variables].values).float(),
                                        indices=validation_indices, encoder_sequence_len=encoder_sequence_len, 
                                        decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_length)
    testing_data = ds.TransformerDataset(data=torch.tensor(testing_data[input_variables].values).float(),
                                        indices=testing_indices, encoder_sequence_len=encoder_sequence_len, 
                                        decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_length)
    
    # Set up dataloaders
    training_val_data = training_data + validation_data # For testing puporses
    training_data = DataLoader(training_data, batch_size, shuffle=True)
    validation_data = DataLoader(validation_data, batch_size, shuffle=True)
    testing_data = DataLoader(testing_data, batch_size=1)
    training_val_data = DataLoader(training_val_data, batch_size=1) # For testing puporses
    
    # Update the encoder sequence length to its crushed version
    encoder_sequence_len = crushed_encoder_sequence_len

    # Instantiate the transformer model and send it to device
    model = tst.TimeSeriesTransformer(input_size=len(src_variables), decoder_sequence_len=decoder_sequence_len, 
                                    batch_first=batch_first, d_model=d_model, n_encoder_layers=n_encoder_layers, 
                                    n_decoder_layers=n_decoder_layers, n_heads=n_heads, dropout_encoder=0.2, 
                                    dropout_decoder=0, dropout_pos_encoder=0.1, dim_feedforward_encoder=in_features_encoder_linear_layer, 
                                    dim_feedforward_decoder=in_features_decoder_linear_layer, num_predicted_features=len(tgt_variables)).to(device)
    # Send model to device
    model.to(device)
    
    # Print model and number of parameters
    print('Defined model:\n', model)
    utils.count_parameters(model)
    
    # Make src mask for the decoder with size
    # [batch_size*n_heads, output_sequence_length, encoder_sequence_len]
    src_mask = utils.unmasker(dim1=output_sequence_length, dim2=encoder_sequence_len).to(device)
    
    # Make tgt mask for decoder with size
    # [batch_size*n_heads, output_sequence_length, output_sequence_length]
    tgt_mask = utils.masker(dim1=output_sequence_length, dim2=output_sequence_length).to(device)
    
    # Define optimizer and loss function
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Update model in the training process and test it
    epochs = 400 # 250
    start_time = time.time()
    df_training = pd.DataFrame(columns=('epoch', 'loss_train'))
    df_validation = pd.DataFrame(columns=('epoch', 'loss_test'))
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_data, model, src_mask, tgt_mask, loss_function, optimizer, device, df_training, epoch=t)
        val(validation_data, model, src_mask, tgt_mask, loss_function, device, df_validation, epoch=t)
    print("Done! ---Execution time: %s seconds ---" % (time.time() - start_time))

    # # # Save the model
    # # torch.save(model, "models/model.pth")
    # # print("Saved PyTorch entire model to models/model.pth")

    # # # Load the model
    # # model = torch.load("models/model.pth").to(device)
    # # print('Loaded PyTorch model from models/model.pth')

    # Inference
    tgt_y_truth_train_val, tgt_y_hat_train_val = test(training_val_data, model, src_mask, tgt_mask, device)
    tgt_y_truth_test, tgt_y_hat_test = test(testing_data, model, src_mask, tgt_mask, device)
    
    # Plot loss
    plt.figure(1);plt.clf()
    plt.plot(df_training['epoch'], df_training['loss_train'], '-o', label='loss train')
    plt.plot(df_training['epoch'], df_validation['loss_test'], '-o', label='loss test')
    plt.yscale('log')
    plt.xlabel(r'epoch')
    plt.ylabel(r'loss')
    plt.legend()
    plt.show()

    # Plot testing results
    plt.figure(2);plt.clf()
    plt.plot(tgt_y_truth_train_val, label='observed')
    plt.plot(tgt_y_hat_train_val, label='predicted')
    plt.title('Training and validation results')
    plt.xlabel(r'time (days)')
    plt.ylabel(r'y')
    plt.legend()
    plt.show()

    plt.figure(2);plt.clf()
    plt.plot(tgt_y_truth_test, label='observed')
    plt.plot(tgt_y_hat_test, label='predicted')
    plt.title('Testing results')
    plt.xlabel(r'time (days)')
    plt.ylabel(r'y')
    plt.legend()
    plt.show()

    # Metrics
    from sklearn.metrics import mean_squared_error

    nse_train_val = utils.nash_sutcliffe_efficiency(tgt_y_truth_train_val, tgt_y_hat_train_val)
    rmse_train_val = np.sqrt(mean_squared_error(tgt_y_truth_train_val, tgt_y_hat_train_val))
    pbias_train_val = utils.pbias(tgt_y_truth_train_val, tgt_y_hat_train_val)
    kge_train_val = utils.kge(tgt_y_truth_train_val, tgt_y_hat_train_val)
    print('\n-- Train/val results')
    print('NSE = ', nse_train_val)
    print('RMSE = ', rmse_train_val)
    print('PBIAS = ', pbias_train_val)
    print('KGE = ', kge_train_val)
    
    nse_test = utils.nash_sutcliffe_efficiency(tgt_y_truth_test, tgt_y_hat_test)
    rmse_test = np.sqrt(mean_squared_error(tgt_y_truth_test, tgt_y_hat_test))
    pbias_test = utils.pbias(tgt_y_truth_test, tgt_y_hat_test)
    kge_test = utils.kge(tgt_y_truth_test, tgt_y_hat_test)
    print('\n-- Testing results')
    print('NSE = ', nse_test)
    print('RMSE = ', rmse_test)
    print('PBIAS = ', pbias_test)
    print('KGE = ', kge_test)
