import numpy as np
import torch

def load_tokens(filename):
    """
    Load tokenized data from a file and convert it to a PyTorch tensor.

    Args:
        filename (str): Path to the tokenized dataset file.

    Returns:
        Tensor: A PyTorch tensor containing the tokenized data.
    """
    npt = np.load(filename)
    npt = npt.astype(np.int32)  # Ensure 32-bit integers
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt
