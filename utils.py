import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import Tensor

from pathlib import Path
from typing import Optional, Any, Union, Callable, Tuple

def masker(dim1: int, dim2: int) -> Tensor:
    
    """
    Generates an upper-triangular matrix of -inf, with
    zeros on the diagonal. Modified from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    ----------
    Arguments:
    dim1: int, for both src and tgt masking, this must be
        a sequence length
    dim2: int, for src masking this must be encoder sequence
        length (i.e. the length of the input sequence to the model),
        and for tgt masking , this must be target sequence length
    
    Return:
    A tensor of shape [dim1, dim2]
    """
    
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

def get_indices(data: pd.DataFrame, window_size: int, step_size: int) -> list:
    
    """
    Produce all the start and end index position that is needed to obtain the sub-sequences.
    
    Returns a list of tuples. Each tuple is (start_idx, end_idx) of a subsequence. These tuples
    should be used to slice the dataset into sub-sequences. These sub-sequences should then be
    passed into a function that sliced them into input and target sequences.
    ----------
    Arguments:
    data (pd.DataFrame): loaded database to generate the subsequences from.
    window_size (int): the desired length of each sub-sequence. Should be (input_sequence_length + 
        tgt_sequence_length). E.g. if you want the model to consider the past 100 time steps in 
        order to predict the future 50 time_steps, window_size = 100 + 50 = 150.
    step_size (int): size of each step as the data sequence is traversed by the moving window.
    
    Return:
    indices: a lits of tuples.
    """
    
    # Define the stop position
    stop_position = len(data) - 1 # because of 0 indexing in Python
    
    # Start the first sub-sequence at index 0
    subseq_first_idx = 0
    subseq_last_idx = window_size
    
    indices = []
    while subseq_last_idx <= stop_position:
        
        indices.append((subseq_first_idx, subseq_last_idx))
        
        subseq_first_idx += step_size
        subseq_last_idx += step_size
        
    return indices

def read_data(data_dir: Union[str, Path] = 'data/utah', timestamp_col_name: str='time') -> pd.DataFrame:
    
    """Read data from csv file and return a pd.DataFrame object.
    ----------
    Arguments:
    data_dir: str or Path object specifying the path to the directory containing the data.
    tgt_col_name: str, the name of the column containing the target variable
    timestamp_col_name: str, the name of the column or named index containing the timestamps
    
    Returns:
    data (pd.DataFrame): data read an loaded as a Pandas DataFrame
    """
    
    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)
    
    # Read csv file
    csv_files = list(data_dir.glob("*.csv"))
    
    if len(csv_files) > 1:
        raise ValueError("data_dir contains more than 1 csv file. Must only contain 1")
    elif len(csv_files) == 0:
        raise ValueError("data_dir must contain at least 1 csv file.")

    data_path = csv_files[0]

    print("Reading file in {}".format(data_path))
    
    data = pd.read_csv(data_path, parse_dates=[timestamp_col_name], index_col=[timestamp_col_name],  low_memory=False)
    
    # Make sure all "n/e" values have been removed from df. 
    if ne_check(data):
        raise ValueError("data frame contains 'n/e' values. These must be handled")

    # Downcast columns to smallest possible version
    data = to_numeric_and_downcast_data(data)

    # Make sure data is in ascending order by timestamp
    data.sort_values(by=[timestamp_col_name], inplace=True)

    return data

def ne_check(df:pd.DataFrame):
    
    """
    Some raw data files contain cells with "n/e". This function checks whether
    any column in a df contains a cell with "n/e". Returns False if no columns
    contain "n/e", True otherwise
    """
    
    for col in df.columns:

        true_bool = (df[col] == "n/e")

        if any(true_bool):
            return True

    return False

def to_numeric_and_downcast_data(df: pd.DataFrame):
    
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    
    fcols = df.select_dtypes('float').columns
    
    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    return df