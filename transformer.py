import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class PositionalEncoder(nn.Module):
    
    """
    The authors of the original transformer paper describe very succinctly what 
    the positional encoding layer does and why it is needed:
    
    "Since our model contains no recurrence and no convolution, in order for the 
    model to make use of the order of the sequence, we must inject some 
    information about the relative or absolute position of the tokens in the 
    sequence." (Vaswani et al, 2017)
    
    Adapted from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    
    def __init__(self, dropout: float, max_seq_len: int, d_model: int, batch_first: bool=True) -> None:
        super().__init__()
        self.d_model = d_model # The dimension of the output of the sub-layers in the model
        self.dropout = nn.Dropout(p=dropout) 
        self.batch_first = batch_first
        
        position = torch.arange(max_seq_len).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # PE(pos,2i)   = sin(pos/(10000^(2i/d_model)))
        # PE(pos,2i+1) = cos(pos/(10000^(2i/d_model)))
        
        if self.batch_first:
            pe = torch.zeros(1, max_seq_len, d_model)
            
            pe[0, :, 0::2] = torch.sin(position * div_term)
            
            pe[0, :, 1::2] = torch.cos(position * div_term)
        
        else:
            pe = torch.zeros(max_seq_len, 1, d_model)
        
            pe[:, 0, 0::2] = torch.sin(position * div_term)
        
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
        x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
                        [enc_seq_len, batch_size, dim_val]
        """
        if self.batch_first:
            x = x + self.pe[:,:x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]

        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    
    """
    This class implements a transformer model that can be used for times series
    forecasting. This time series transformer model is based on the paper by
    Wu et al (2020) [1]. The paper will be referred to as "the paper".

    In cases where the paper does not specify what value was used for a specific
    configuration/hyperparameter, this class uses the values from Vaswani et al
    (2017) [2] or from PyTorch source code.

    Unlike the paper, this class assumes that input layers, positional encoding 
    layers and linear mapping layers are separate from the encoder and decoder, 
    i.e. the encoder and decoder only do what is depicted as their sub-layers 
    in the paper. For practical purposes, this assumption does not make a 
    difference - it merely means that the linear and positional encoding layers
    are implemented inside the present class and not inside the 
    Encoder() and Decoder() classes.

    [1] Wu, N., Green, B., Ben, X., O'banion, S. (2020). 
    'Deep Transformer Models for Time Series Forecasting: 
    The Influenza Prevalence Case'. 
    arXiv:2001.08317 [cs, stat] [Preprint]. 
    Available at: http://arxiv.org/abs/2001.08317 (Accessed: 1 December 2023).

    [2] Vaswani, A. et al. (2017) 
    'Attention Is All You Need'.
    arXiv:1706.03762 [cs] [Preprint]. 
    Available at: http://arxiv.org/abs/1706.03762 (Accessed: 1 December 2023).
    """
    
    def __init__(self, input_size: int, decoder_sequence_len: int, batch_first: bool,
                d_model: int, n_encoder_layers: int, n_decoder_layers: int, n_heads: int,
                dropout_encoder: float, dropout_decoder: float, dropout_pos_encoder: float,
                dim_feedforward_encoder: int, dim_feedforward_decoder: int, num_predicted_features: int):
        super().__init__()
        
        """
        Arguments:
        input_size (int): number of input variables. 1 if univariate.
        decoder_sequence_len (int): the length of the input sequence fed to the decoder
        d_model (int): all sub-layers in the model produce outputs of dimension d_model
        n_encoder_layers (int): number of stacked encoder layers in the encoder
        n_decoder_layers (int): number of stacked encoder layers in the decoder
        n_heads (int): the number of attention heads (aka parallel attention layers)
        dropout_encoder (float): the dropout rate of the encoder
        dropout_decoder (float): the dropout rate of the decoder
        dropout_pos_encoder (float): the dropout rate of the positional encoder
        dim_feedforward_encoder (int): number of neurons in the linear layer of the encoder
        dim_feedforward_decoder (int): number of neurons in the linear layer of the decoder
        num_predicted_features (int) : the number of features you want to predict. Most of the time, 
        this will be 1 because we're only forecasting FCR-N prices in DK2, but in we wanted to also 
        predict FCR-D with the same model, num_predicted_features should be 2.
        """
        
        self.decoder_sequence_len = decoder_sequence_len
        
        # print("input_size is: {}".format(input_size))
        # print("d_model is: {}".format(d_model))
        
        # Create the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(in_features=input_size, out_features=d_model)
        
        self.decoder_input_layer = nn.Linear(in_features=num_predicted_features, out_features=d_model)
        
        self.linear_mapping = nn.Linear(in_features=d_model, out_features=num_predicted_features)
        
        # Create the positional encoder dropout, max_seq_len, d_model, batch_first
        self.positional_encoding_layer = PositionalEncoder(dropout=dropout_pos_encoder, max_seq_len=5000, d_model=d_model, batch_first=batch_first)
        
        # The encoder layer used in the paper is identical to the one used by Vaswani et al (2017) 
        # on which the PyTorch transformer model is based
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward_encoder,
                                                dropout=dropout_encoder, batch_first=batch_first)

        # Stack the encoder layers in nn.TransformerEncoder
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_encoder_layers, norm=None)
        
        # Define the decoder layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward_decoder,
                                                dropout=dropout_decoder, batch_first=batch_first)
        
        # Stack the decoder layers in nn.TransformerDecoder
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_decoder_layers, norm=None)
        
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor=None, tgt_mask: Tensor=None) -> Tensor:
        
        """
        Arguments:
        src: the output sequence of the encoder. Shape: (S, E) for unbatched input, (N, S, E) if 
            batch_first=True, where S is the source of the sequence length, N is the batch size, and E is the number of 
            features (1 if univariate).
        tgt: the sequence to the decoder. Shape (T, E) for unbatched input, or (N, T, E) if batch_first=True, 
            where T is the target sequence length, N is the batch size, and E is the number of features (1 if univariate).
        src_mask: the mask for the src sequence to prevent the model from using data points from the target sequence.
        tgt_mask: the mask fro the tgt sequence to prevent the model from using data points from the target sequence.
        
        Returns:
        Tensor of shape: [target_sequence_length, batch_size, num_predicted_features]
        """
        
        # print("From model.forward(): Size of src as given to forward(): {}".format(src.size()))
        # print("From model.forward(): tgt size = {}".format(tgt.size()))
        
        # Pass through the input layer right before the encoder
        # src shape: [batch_size, src length, d_model] regardless of number of input features
        src = self.encoder_input_layer(src)
        # print("From model.forward(): Size of src after input layer: {}".format(src.size()))
        
        
        # Pass through the positional encoding layer            
        # src shape: [batch_size, src length, d_model] regardless of number of input features
        # print(src.shape) # Maybe I just have to use a squeeze on source to add a dimension of 1 for the batch size. Probably have to do it with tgt and maybe tgt_y too
        src = self.positional_encoding_layer(src)
        # print("From model.forward(): Size of src after pos_enc layer: {}".format(src.size()))
        
        # Pass through all the stacked encoder layers in the encoder
        # Masking is only needed in the encoder if input sequences are padded which they are 
        # not in this time series use case, because all my  input sequences are naturally of 
        # the same length. 
        # src shape: [batch_size, encoder_sequence_len, d_model]
        src = self.encoder(src=src)
        # print("From model.forward(): Size of src after encoder: {}".format(src.size()))

        # Pass decoder input through decoder input layer
        # tgt shape: [target sequence length, batch_size, d_model] regardless of number of input features
        # print(f"From model.forward(): Size of tgt before the decoder = {tgt.size()}")
        decoder_output = self.decoder_input_layer(tgt)
        # print("From model.forward(): Size of decoder_output after linear decoder layer: {}".format(decoder_output.size()))
        
        # if src_mask is not None:
        #     print("From model.forward(): Size of src_mask: {}".format(src_mask.size()))
        # if tgt_mask is not None:
        #     print("From model.forward(): Size of tgt_mask: {}".format(tgt_mask.size()))
        
        # Pass through the decoder
        # Output shape: [batch_size, target seq len, d_model]
        decoder_output = self.decoder(tgt=decoder_output, memory=src, tgt_mask=tgt_mask, memory_mask=src_mask)
        # print("From model.forward(): decoder_output shape after decoder: {}".format(decoder_output.shape))
        
        # Pass through the linear mapping
        # shape [batch_size, target seq len]
        decoder_output = self.linear_mapping(decoder_output)
        # print("From model.forward(): decoder_output size after linear_mapping = {}".format(decoder_output.size()))
        
        return decoder_output