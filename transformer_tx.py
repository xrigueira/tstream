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
    
class customTransformerEncoderLayer(nn.Module):
    """
    This class implements a custom encoder layer
    """

    def __init__(self, d_model: int, n_heads: int, dim_feedforward_encoder: int,
                dropout_encoder: float, device, batch_first: bool = True, 
                activation = F.relu, layer_norm_eps: float = 1e-5):
        super().__init__()

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward_encoder, device=device)
        self.dropout = nn.Dropout(dropout_encoder)
        self.linear2 = nn.Linear(dim_feedforward_encoder, d_model, device=device)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device)
        self.dropout1 = nn.Dropout(dropout_encoder)
        self.dropout2 = nn.Dropout(dropout_encoder)

        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout_encoder,
                                                device=device, batch_first=batch_first)

        # Define activation function options
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu
    
    def forward(self, src: Tensor, src_mask: Tensor=None, src_key_padding_mask = None,
                is_causal = False) -> Tensor:
        
        if self.training:
            need_weights = False
        else:
            need_weights = True
        
        x = src
        x = self.norm1(x + self._sa_block(x, src_mask, need_weights, is_causal=is_causal))
        x = self.norm2(x + self._ff_block(x))

        return x

    # Self-attention block
    def _sa_block(self, x, attn_mask, need_weights, is_causal = False):
        x = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            need_weights=need_weights, is_causal=is_causal)[0]
        return self.dropout1(x)
    
    # Feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    

class customTransformerDecoderLayer(nn.Module):
    """
    This class implements a custom decoder layer
    """

    def __init__(self, d_model: int, n_heads: int, dim_feedforward_decoder: int,
                dropout_decoder: float, device, batch_first: bool=True, 
                activation = F.relu, layer_norm_eps: float = 1e-5, 
                norm_first: bool=False):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout_decoder, batch_first=batch_first,
                                            device=device)
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout_decoder, batch_first=batch_first,
                                            device=device)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward_decoder, device=device)
        self.dropout = nn.Dropout(dropout_decoder)
        self.linear2 = nn.Linear(dim_feedforward_decoder, d_model, device=device)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device)
        self.dropout1 = nn.Dropout(dropout_decoder)
        self.dropout2 = nn.Dropout(dropout_decoder)
        self.dropout3 = nn.Dropout(dropout_decoder)

        # Define activation function
        self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal = False, memory_is_causal = True) -> Tensor:
        
        x = tgt
        x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_is_causal))
        tmp, att_weight = self._mha_block(x, memory, memory_mask, memory_is_causal)
        x = self.norm2(x + tmp)
        x = self.norm3(x + self._ff_block(x))

        return x, att_weight

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal = False):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           is_causal=is_causal)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x, mem, attn_mask, key_padding_mask, is_causal: bool = False):
        x, att_weights = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                is_causal=is_causal)
        return self.dropout2(x), att_weights

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class TimeSeriesTransformer(nn.Module):
    
    """
    This class implements a transformer model that can be used for times series
    forecasting.
    """
    # d_embed from Tianfang transformer is d_model from my case
    def __init__(self, input_size: int, decoder_sequence_len: int, batch_first: bool,
                d_model: int, n_encoder_layers: int, n_decoder_layers: int, n_heads: int,
                dropout_encoder: float, dropout_decoder: float, dropout_pos_encoder: float,
                dim_feedforward_encoder: int, dim_feedforward_decoder: int, num_predicted_features: int,
                device):
        super(TimeSeriesTransformer, self).__init__()
        
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
        
        self.model_tpye = 'Transformer'
        self.device = device

        self.decoder_sequence_len = decoder_sequence_len
        
        # print("input_size is: {}".format(input_size))
        # print("d_model is: {}".format(d_model))
        
        # Create the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(in_features=input_size, out_features=d_model)
        
        self.decoder_input_layer = nn.Linear(in_features=num_predicted_features, out_features=d_model)
        
        self.linear_mapping = nn.Linear(in_features=d_model, out_features=num_predicted_features)
        
        # Create the positional encoder dropout, max_seq_len, d_model, batch_first
        self.positional_encoding_layer = PositionalEncoder(dropout=dropout_pos_encoder, max_seq_len=5000, d_model=d_model, batch_first=batch_first)
        
        # Define the encoder layer
        encoder_layer = customTransformerEncoderLayer(d_model=d_model, n_heads=n_heads, dim_feedforward_encoder=dim_feedforward_encoder,
                                                        dropout_encoder=dropout_encoder, device=device, batch_first=batch_first)

        # Stack the encoder layers in nn.TransformerEncoder
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_encoder_layers, norm=None)
        
        # Define the decoder layer
        decoder_layer = customTransformerDecoderLayer(d_model=d_model, n_heads=n_heads, dim_feedforward_decoder=dim_feedforward_decoder,
                                                        dropout_decoder=dropout_decoder, device=device, batch_first=batch_first)
        
        # # Define the last decoder layer that returns the attention weights
        # decoder_layer_attention_weights = nn.AttentionWeightsTransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward_decoder,
        #                                         dropout=dropout_decoder, batch_first=batch_first)
        
        # Stack the decoder layers in nn.TransformerDecoder
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_decoder_layers, norm=None)

        # Initialize the weights
        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor=None, memory_mask: Tensor=None, tgt_mask: Tensor=None) -> Tensor:
        
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
        
        # Pass through the positional encoding layer            
        # src shape: [batch_size, tgt length, d_model] regardless of number of input features
        decoder_output = self.positional_encoding_layer(decoder_output)

        # Pass through the decoder
        # Output shape: [batch_size, target seq len, d_model]
        decoder_output, mha_weights = self.decoder(tgt=decoder_output, memory=src, tgt_mask=tgt_mask, memory_mask=memory_mask)
        # print("From model.forward(): decoder_output shape after decoder: {}".format(decoder_output.shape))
        
        # Pass through the linear mapping
        # shape [batch_size, target seq len]
        decoder_output = self.linear_mapping(decoder_output)
        # print("From model.forward(): decoder_output size after linear_mapping = {}".format(decoder_output.size()))
        
        return decoder_output, mha_weights