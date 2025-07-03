"""
Utility functions for data processing and other tasks.
"""
import torch

def pad_sequences(sequences, padding_value=0):
    """
    Pads a list of sequences to the length of the longest sequence.
    Args:
        sequences (list of Tensors): List of sequences to pad.
        padding_value (int): The value to use for padding.

    Returns:
        Tensor: A tensor of padded sequences.
    """
    max_len = max(len(s) for s in sequences)
    padded_sequences = []
    for s in sequences:
        pad_len = max_len - len(s)
        padded_s = torch.cat([s, torch.full((pad_len,), padding_value, dtype=s.dtype)])
        padded_sequences.append(padded_s)
    return torch.stack(padded_sequences) 