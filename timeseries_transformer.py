import torch.nn as nn
import torch.nn.functional as F
import positional_encoder as pe

class TimeSeriesTransformer(nn.Module):
    
    """
    This class implements a transformer model that can be used for times series
    forecasting. This time series transformer model is based on the paper by
    Wu et al (2020) [1]. The paper will be referred to as "the paper".

    A detailed description of the code can be found in my article here:

    https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e

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
    Available at: http://arxiv.org/abs/2001.08317 (Accessed: 9 March 2022).

    [2] Vaswani, A. et al. (2017) 
    'Attention Is All You Need'.
    arXiv:1706.03762 [cs] [Preprint]. 
    Available at: http://arxiv.org/abs/1706.03762 (Accessed: 9 March 2022).
    """
    
    def __init__(self, input_size: int, dec_seq_len: int, batch_first: bool, out_seq_len: int=58,
                dim_val: int=512, n_encoder_layers: int=4, n_decoder_layers: int=4, n_heads: int=8,
                dropout_encoder: float=0.2, dropout_decoder: float=0.2, dropout_pos_enc: float=0.1,
                dim_feedforward_encoder: int=2048, dim_feedforward_decoder: int=2048, num_predicted_features: int=1):
        super().__init__()
        """
        Arguments:
        input_size (int): number of input variables. 1 if univariate.
        dec_seq_len (int): the length of the input sequence fed to the decoder
        dim_val (int): aka d_model. All sub-layers in the model produce outputs of dimension dim_val
        n_encoder_layers (int): number of stacked encoder layers in the encoder
        n_decoder_layers (int): number of stacked encoder layers in the decoder
        n_heads (int): the number of attention heads (aka parallel attention layers)
        dropout_encoder (float): the dropout rate of the encoder
        dropout_decoder (float): the dropout rate of the decoder
        dropout_pos_enc (float): the dropout rate of the positional encoder
        dim_feedforward_encoder (int): number of neurons in the linear layer of the encoder
        dim_feedforward_decoder (int): number of neurons in the linear layer of the decoder
        num_predicted_features (int) : the number of features you want to predict. Most of the time, 
        this will be 1 because we're only forecasting FCR-N prices in DK2, but in we wanted to also 
        predict FCR-D with the same model, num_predicted_features should be 2.
        """
        
        