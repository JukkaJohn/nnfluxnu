import torch
import torch.nn.functional
import os
import numpy as np
from typing import Tuple


def get_fk_table(filename: str, parent_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads a FastKernel (FK) table and corresponding x_alpha nodes from text files,
    converts them to PyTorch tensors, and reshapes x_alpha to a column vector.

    Parameters
    ----------
    filename : str
        Name of the FK table file to load (relative to parent_dir).
    parent_dir : str
        Path to the directory containing the FK table file and relative location
        of the x_alpha file.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - x_alpha: Tensor of shape (N, 1) containing x_alpha grid nodes.
        - fk_table: Tensor containing the FK table data.
    """
    file_path = os.path.join(parent_dir, filename)
    fk_table = np.loadtxt(file_path, delimiter=None)

    file_path = os.path.join(parent_dir, "../../../Data/gridnodes/x_alpha.dat")
    x_alpha = np.loadtxt(file_path, delimiter=None)

    x_alpha = torch.tensor(x_alpha, dtype=torch.float32).view(-1, 1)
    fk_table = torch.tensor(fk_table, dtype=torch.float32)

    return x_alpha, fk_table
