import numpy as np
import torch


def read_LHEF_data(starting_index, num_obs):
    """Read histograms of data

    Args:
        num_obs (int): number of files to read data from

    Returns:
        tuple: returns tuple of lists and arrays containing the data
    """

    filenames = ["Enu", "El.dat", "Eh.dat", "theta.dat"]
    (binwidths, events, xvals_per_obs, max_events, min_events, xlabels) = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for i in range(starting_index, num_obs):
        val_n = np.loadtxt(f"data/data_{filenames[i]}n.dat", unpack=True)

        val_p = np.loadtxt(f"data/data_{filenames[i]}.dat", unpack=True)

        binwidth = 300 - 20 / 25
        val_n *= binwidth
        val_p *= binwidth

        val = 74 / 183 * val_p + (183 - 74) / 183 * val_n
        err = val / 10
        high_bin = np.linspace(25, 6000, 21)
        print(f"x_vals = {high_bin}")
        events.append(val)
        xvals_per_obs.append(high_bin[:-1])
        min_events.append(val - err)
        max_events.append(val + err)
        binwidths.append(binwidth)

        xlabels.append(filenames[i].replace(".dat", ""))

    xlabels = np.array(xlabels)
    events_per_obs = events
    events = np.concatenate(events)
    min_events = np.concatenate(min_events)
    max_events = np.concatenate(max_events)
    binwidths = np.array(binwidths)
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
