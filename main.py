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
def train(dataloader, model, src_mask, memory_mask, tgt_mask, loss_function, optimizer, device, df_training, epoch):
    
    size = len(dataloader.dataset)
    model.train()
    training_loss = [] # For plotting purposes
    for i, batch in enumerate(dataloader):
        src, tgt, tgt_y, src_p, tgt_p = batch
        src, tgt, tgt_y, src_p, tgt_p = src.to(device), tgt.to(device), tgt_y.to(device), src_p.to(device), tgt_p.to(device)

        # Zero out gradients for every batch
        optimizer.zero_grad()
        
        # Compute prediction error
        pred, sa_weights_encoder, sa_weights, mha_weights = model(src=src, tgt=tgt, src_mask=src_mask, memory_mask=memory_mask, tgt_mask=tgt_mask)
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
def val(dataloader, model, src_mask, memory_mask, tgt_mask, loss_function, device, df_validation, epoch):
    
    num_batches = len(dataloader)
    model.eval()
    validation_loss = [] # For plotting purposes
    with torch.no_grad():
        for batch in dataloader:
            src, tgt, tgt_y, src_p, tgt_p = batch
            src, tgt, tgt_y, src_p, tgt_p = src.to(device), tgt.to(device), tgt_y.to(device), src_p.to(device), tgt_p.to(device)
            
            pred, sa_weights_encoder, sa_weights, mha_weights = model(src=src, tgt=tgt, src_mask=src_mask, memory_mask=memory_mask, tgt_mask=tgt_mask)
            pred = pred.to(device)
            loss = loss_function(pred, tgt_y.unsqueeze(2))
            
            # Save results for plotting
            validation_loss.append(loss.item())
            epoch_val_loss = np.mean(validation_loss)
            df_validation.loc[epoch] = [epoch, epoch_val_loss]
    
    loss /= num_batches
    # print(f"Avg test loss: {loss:>8f}")

# Define inference step
def test(dataloader, model, src_mask, memory_mask, tgt_mask, device):
    
    # Get ground truth
    tgt_y_truth = torch.zeros(len(dataloader))
    for i, (src, tgt, tgt_y, src_p, tgt_p) in enumerate(dataloader):
        tgt_y_truth[i] = tgt_y

    # Define tensor to store the predictions
    tgt_y_hat = torch.zeros((len(dataloader)), device=device)

    # Define list to store the attention weights
    all_sa_weights_encoder_inference = []
    all_sa_weights_inference = []
    all_mha_weights_inference = []
    
    # Perform inference
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            src, tgt, tgt_y, src_p, tgt_p = sample
            src, tgt, tgt_y, src_p, tgt_p = src.to(device), tgt.to(device), tgt_y.to(device), src_p.to(device), tgt_p.to(device)

            pred, sa_weights_encoder, sa_weights, mha_weights = model(src=src, tgt=tgt, src_mask=src_mask, memory_mask=memory_mask, tgt_mask=tgt_mask)
            all_sa_weights_encoder_inference.append(sa_weights_encoder)
            all_sa_weights_inference.append(sa_weights)
            all_mha_weights_inference.append(mha_weights)
            pred = pred.to(device)

            # # Save src, tgt and tgt_y, and pred for plotting purposes
            # np.save(f'results/src_p_{i}.npy', src_p.cpu(), allow_pickle=False, fix_imports=False)
            # np.save(f'results/tgt_p_{i}.npy', tgt_p.cpu(), allow_pickle=False, fix_imports=False)
            # np.save(f'results/tgt_y_hat_{i}.npy', pred.cpu(), allow_pickle=False, fix_imports=False)
            
            tgt_y_hat[i] = pred
    
    # # Save inference attention for the last step
    # np.save('results/all_sa_encoder_weights.npy', np.stack([sa_weight_encoder.cpu().numpy() for sa_weight_encoder in all_sa_weights_encoder_inference]), allow_pickle=False, fix_imports=False)
    # np.save('results/all_sa_weights.npy', np.stack([sa_weight.cpu().numpy() for sa_weight in all_sa_weights_inference]), allow_pickle=False, fix_imports=False)
    # np.save('results/all_mha_weights.npy', np.stack([mha_weight.cpu().numpy() for mha_weight in all_mha_weights_inference]), allow_pickle=False, fix_imports=False)
    
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

    # Define run number
    run = 26
    
    # Hyperparams
    batch_size = 128
    validation_size = 0.125
    # src_variables = ['x1']
    src_variables = ['x1', 'x2']
    tgt_variables = ['y']
    input_variables = src_variables + tgt_variables
    timestamp_col_name = "time"

    # Only use data from this date and onwards
    cutoff_date = datetime.datetime(1980, 1, 1) 

    d_model = 32
    n_heads = 2
    n_encoder_layers = 1
    n_decoder_layers = 1 # Remember that with the current implementation it always has a decoder layer that returns the weights
    encoder_sequence_len = 1461 # length of input given to encoder used to create the pre-summarized windows (4 years of data) 1461
    crushed_encoder_sequence_len = 53 # Encoder sequence length afther summarizing the data when defining the dataset 53
    decoder_sequence_len = 1 # length of input given to decoder
    output_sequence_len = 1 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
    window_size = encoder_sequence_len + output_sequence_len # used to slice data into sub-sequences
    step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
    in_features_encoder_linear_layer = 32
    in_features_decoder_linear_layer = 32
    max_sequence_len = encoder_sequence_len
    batch_first = True

    # Run parameters
    lr = 0.00015
    epochs = 250

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
                                        decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_len)
    validation_data = ds.TransformerDataset(data=torch.tensor(training_val_data[input_variables].values).float(),
                                        indices=validation_indices, encoder_sequence_len=encoder_sequence_len, 
                                        decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_len)
    testing_data = ds.TransformerDataset(data=torch.tensor(testing_data[input_variables].values).float(),
                                        indices=testing_indices, encoder_sequence_len=encoder_sequence_len, 
                                        decoder_sequence_len=decoder_sequence_len, tgt_sequence_len=output_sequence_len)
    
    # Set up dataloaders
    training_val_data = training_data + validation_data # For testing puporses
    training_data = DataLoader(training_data, batch_size, shuffle=False)
    validation_data = DataLoader(validation_data, batch_size, shuffle=False)
    testing_data = DataLoader(testing_data, batch_size=1)
    training_val_data = DataLoader(training_val_data, batch_size=1) # For testing puporses
    
    # Update the encoder sequence length to its crushed version
    encoder_sequence_len = crushed_encoder_sequence_len

    # Instantiate the transformer model and send it to device
    model = tst.TimeSeriesTransformer(input_size=len(src_variables), decoder_sequence_len=decoder_sequence_len, 
                                    batch_first=batch_first, d_model=d_model, n_encoder_layers=n_encoder_layers, 
                                    n_decoder_layers=n_decoder_layers, n_heads=n_heads, dropout_encoder=0.2, 
                                    dropout_decoder=0.2, dropout_pos_encoder=0.1, dim_feedforward_encoder=in_features_encoder_linear_layer, 
                                    dim_feedforward_decoder=in_features_decoder_linear_layer, num_src_features=len(src_variables), 
                                    num_predicted_features=len(tgt_variables)).to(device)

    # Send model to device
    model.to(device)
    
    # Print model and number of parameters
    print('Defined model:\n', model)
    utils.count_parameters(model)
    
    # Make src mask for the decoder with size
    # [batch_size*n_heads, output_sequence_length, encoder_sequence_len]
    src_mask = utils.masker(dim1=encoder_sequence_len, dim2=encoder_sequence_len).to(device)
    # src_mask = utils.generate_square_subsequent_mask(size=encoder_sequence_len).to(device)
    
    # Make the memory mask for the decoder
    # memory_mask = utils.unmasker(dim1=output_sequence_len, dim2=encoder_sequence_len).to(device)
    memory_mask = None

    # Make tgt mask for decoder with size
    # [batch_size*n_heads, output_sequence_length, output_sequence_length]
    tgt_mask = utils.masker(dim1=output_sequence_len, dim2=output_sequence_len).to(device)
    # tgt_mask = utils.generate_square_subsequent_mask(size=decoder_sequence_len).to(device)
    
    # Define optimizer and loss function
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Update model in the training process and test it
    start_time = time.time()
    df_training = pd.DataFrame(columns=('epoch', 'loss_train'))
    df_validation = pd.DataFrame(columns=('epoch', 'loss_val'))
    for t in range(epochs): # epochs is defined in the hyperparameters section above
        print(f"Epoch {t+1}\n-------------------------------")
        train(training_data, model, src_mask, memory_mask, tgt_mask, loss_function, optimizer, device, df_training, epoch=t)
        val(validation_data, model, src_mask, memory_mask, tgt_mask, loss_function, device, df_validation, epoch=t)
    print("Done! ---Execution time: %s seconds ---" % (time.time() - start_time))

    # # Save the model
    # torch.save(model, "models/model.pth")
    # print("Saved PyTorch entire model to models/model.pth")

    # # Load the model
    # model = torch.load("models/model.pth").to(device)
    # print('Loaded PyTorch model from models/model.pth')

    # Inference
    tgt_y_truth_train_val, tgt_y_hat_train_val = test(training_val_data, model, src_mask, memory_mask, tgt_mask, device)
    tgt_y_truth_test, tgt_y_hat_test = test(testing_data, model, src_mask, memory_mask, tgt_mask, device)
    
    # Save results
    utils.logger(run=run, batches=batch_size, d_model=d_model, n_heads=n_heads,
                encoder_layers=n_encoder_layers, decoder_layers=n_decoder_layers,
                dim_ll_encoder=in_features_encoder_linear_layer, dim_ll_decoder=in_features_decoder_linear_layer,
                lr=lr, epochs=epochs)

    # Plot loss
    plt.figure(1);plt.clf()
    plt.plot(df_training['epoch'], df_training['loss_train'], '-o', label='loss train')
    plt.plot(df_training['epoch'], df_validation['loss_val'], '-o', label='loss val')
    plt.yscale('log')
    plt.xlabel(r'epoch')
    plt.ylabel(r'loss')
    plt.legend()
    # plt.show()

    plt.savefig(f'results/run_{run}/loss.png', dpi=300)

    # Plot testing results
    utils.plots(tgt_y_truth_train_val, tgt_y_hat_train_val, 'train_val', run=run)
    utils.plots(tgt_y_truth_test, tgt_y_hat_test, 'test', run=run)

    # Metrics
    utils.metrics(tgt_y_truth_train_val, tgt_y_hat_train_val, 'train_val')
    utils.metrics(tgt_y_truth_test, tgt_y_hat_test, 'test')
