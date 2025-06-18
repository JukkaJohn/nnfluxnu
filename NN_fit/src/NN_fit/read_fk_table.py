import pandas as pd
import torch
import torch.nn.functional
import os
import numpy as np


def get_fk_table(filename, parent_dir):
    """This function reads the fk table for the neutrino flux and pads them for computational efficiency later on

    Returns:
        tuple: x_alphas(grid points) and the fk table in tensor to fit torch
    """

    file_path = os.path.join(parent_dir, filename)
    fk_table = np.loadtxt(file_path, delimiter=None)
    # df = pd.read_csv(file_path, sep="\s+", header=None)

    # fk_table = df.to_numpy()

    file_path = os.path.join(parent_dir, "../../../Data/gridnodes/x_alpha.dat")
    x_alpha = np.loadtxt(file_path, delimiter=None)

    x_alpha = torch.tensor(x_alpha, dtype=torch.float32).view(-1, 1)
    fk_table = torch.tensor(fk_table, dtype=torch.float32)

    return x_alpha, fk_table


def get_fk_table_n_p(filename_n, filename_p, parent_dir):
    """This function reads the fk table for the neutrino flux and pads them for computational efficiency later on

    Returns:
        tuple: x_alphas(grid points) and the fk table in tensor to fit torch
    """

    # file_path = f"data/{filenames[i]}"
    # df = pd.read_csv(file_path, sep="\s+", header=None)
    # fk_table = df.to_numpy()

    file_path_n = os.path.join(parent_dir, filename_n)
    df_n = pd.read_csv(file_path_n, sep="\s+", header=None)
    fk_table_n = df_n.to_numpy()

    file_path_p = os.path.join(parent_dir, filename_p)
    df_p = pd.read_csv(file_path_p, sep="\s+", header=None)
    fk_table_p = df_p.to_numpy()

    fk_table = 74 / 183 * fk_table_p + (183 - 74) / 183 * fk_table_n

    x_alpha = fk_table[0, :]
    x_alpha = x_alpha.reshape(len(x_alpha), 1)

    # strip first row to get fk table
    fk_table = fk_table[1:, :]

    x_alpha = torch.tensor(x_alpha, dtype=torch.float32).view(-1, 1)
    fk_table = torch.tensor(fk_table, dtype=torch.float32)

    return x_alpha, fk_table
