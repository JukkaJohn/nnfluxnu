import numpy as np
import pandas as pd

import torch


# def read_LHEF_data(starting_index, num_obs):
def read_LHEF_data():
    """Read histograms of data

    Args:
        num_obs (int): number of files to read data from

    Returns:
        tuple: returns tuple of lists and arrays containing the data
    """

    filenames = ["Enu.dat"]
    (binwidths, events, xvals_per_obs, max_events, min_events, xlabels) = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # for i in range(starting_index, num_obs):
    # low_bin, high_bin, val, err = np.loadtxt(f"data/{filenames[i]}", unpack=True)
    low_bin, high_bin, val, err = np.loadtxt(f"data/{filenames[0]}", unpack=True)
    binwidth = high_bin[0] - low_bin[0]
    val *= binwidth

    events = val
    xvals_per_obs = high_bin
    min_events = val - err
    max_events = val + err
    binwidths = binwidth
    # xlabels = filenames[i].replace(".dat", "")
    xlabels = filenames[0].replace(".dat", "")

    events_per_obs = events
    binwidths = torch.tensor(binwidths, dtype=torch.float32)
    return (
        events,
        max_events,
        min_events,
        xvals_per_obs,
        binwidths,
        xlabels,
        events_per_obs,
    )


# def get_fk_table(start_index, num_obs):
def get_fk_table():
    """This function reads the fk table for the neutrino flux and pads them for computational efficiency later on

    Returns:
        tuple: x_alphas(grid points) and the fk table in tensor to fit torch
    """

    filenames = ["FK_Enu.dat"]

    # for i in range(start_index, num_obs):
    # file_path = f"data/{filenames[i]}"
    file_path = f"data/{filenames[0]}"
    df = pd.read_csv(file_path, sep="\s+", header=None)
    fk_table = df.to_numpy()

    x_alpha = fk_table[0, :]
    # x_alpha = x_alpha.reshape(len(x_alpha), 1)

    # strip first row to get fk table
    fk_table = fk_table[1:, :]

    x_alpha = torch.tensor(x_alpha, dtype=torch.float32).view(-1, 1)
    fk_table = torch.tensor(fk_table, dtype=torch.float32)

    return x_alpha, fk_table
