import os
import pandas as pd
from typing import Tuple

import torch
from torch.utils.data import Dataset

class TransformerDataset(Dataset):
    
    """
    Dataset class used for transformer models.
    ----------
    Arguments:
    data: tensor, the entire train, validation or test data sequence before any slicing. 
    If univariate, data.size() will be [number of samples, number of variables] where the 
    number of variables will be equal to 1 + the number of exogenous variables. Number of 
    exogenous variables would be 0 if univariate.

    indices: a list of tuples. Each tuple has two elements:
        1) the start index of a sub-sequence
        2) the end index of a sub-sequence. 
        The sub-sequence is split into src, tgt and tgt_y later.  

    encoder_sequence_len: int, the desired length of the input sequence given to the the first layer of
        the transformer model.

    tgt_sequence_len: int, the desired length of the target sequence (the output of the model)

    tgt_idx: The index position of the target variable in data. Data is a 2D tensor
    """
    
    def __init__(self, data: torch.tensor, indices: list, encoder_sequence_len: int, decoder_sequence_len: int, tgt_sequence_len: int) -> None:
        super().__init__()

        self.indices = indices
        self.data = data
        self.encoder_sequence_len = encoder_sequence_len
        self.decoder_sequence_len = decoder_sequence_len
        self.tgt_sequence_len = tgt_sequence_len

        print("From get_src_tgt: data size = {}".format(data.size()))


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) tgt (the decoder input)
        3) tgt_y (the target)
        """
        
        # Get the first element of the i'th tuple in the list self.indices
        start_idx = self.indices[index][0]

        # Get the second (and last) element of the i'th tuple in the list self.indices
        end_idx = self.indices[index][1]

        sequence = self.data[start_idx:end_idx]

        src, tgt, tgt_y = self._get_src_tgt(sequence=sequence, encoder_sequence_len=self.encoder_sequence_len,
            decoder_sequence_len=self.decoder_sequence_len, tgt_sequence_len=self.tgt_sequence_len)

        return src, tgt, tgt_y
    
    def _get_src_tgt(self, sequence: torch.Tensor, encoder_sequence_len: int, decoder_sequence_len: int, 
                    tgt_sequence_len: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

        """
        Generate the src (encoder input), tgt (decoder input) and tgt_y (the target)
        sequences from a sequence. 
        ----------
        Arguments:
        sequence: tensor, a 1D tensor of length n where  n = encoder input length + target 
            sequence length 
        encoder_sequence_len: int, the desired length of the input to the transformer encoder
        tgt_sequence_len: int, the desired length of the target sequence (the one against 
            which the model output is compared)

        Return: 
        src: tensor, 1D, used as input to the transformer model
        tgt: tensor, 1D, used as input to the transformer model
        tgt_y: tensor, 1D, the target sequence against which the model output is compared
            when computing loss. 
        """
        
        assert len(sequence) == encoder_sequence_len + tgt_sequence_len, "Sequence length does not equal (input length + target length)"
        
        # encoder input
        src = sequence[:encoder_sequence_len] 
        
        # decoder input. As per the paper, it must have the same dimension as the 
        # target sequence, and it must contain the last value of src, and all
        # values of tgt_y except the last (i.e. it must be shifted right by 1)
        tgt = sequence[encoder_sequence_len-1:len(sequence)-1]
        
        assert len(tgt) == tgt_sequence_len, "Length of tgt does not match target sequence length"

        # The target sequence against which the model output will be compared to compute loss
        tgt_y = sequence[-tgt_sequence_len:]

        assert len(tgt_y) == tgt_sequence_len, "Length of tgt_y does not match target sequence length"

        return src, tgt, tgt_y.squeeze(-1) # change size from [batch_size, tgt_seq_len, num_features] to [batch_size, tgt_seq_len] 