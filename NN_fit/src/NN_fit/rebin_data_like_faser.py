import numpy as np
from read_fk_table import get_fk_table


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
    lumin_factor = 65.5 / 150
    low, high, val, err = np.loadtxt(f"data/{filenames[0]}", unpack=True)
    diff = high[0] - low[0]
    val *= diff
    events = val
    xvals_per_obs = high[:-1]
    min_events.append(val - err)
    max_events.append(val + err)
    binwidths.append(diff)

    xlabels.append(filenames[0].replace(".dat", ""))

    events_per_obs = events

    return (
        events,
        max_events,
        min_events,
        xvals_per_obs,
        binwidths,
        xlabels,
        events_per_obs,
    )


def rebin_fk():
    # Read data
    data, data_min, data_max, xvals_per_obs, binwidths, xlabels, events_per_obs = (
        read_LHEF_data()
    )

    x_alphas, fk_tables = get_fk_table()
    faser_bins = [300, 600, 1000]
    xvals_per_obs = np.array(xvals_per_obs)
    # data = np.array(data)
    print(data[:20])
    print(xvals_per_obs[:20])
    data_faser = []
    min_range = 0
    for bin_val in faser_bins:
        indices = np.where(xvals_per_obs < bin_val)[0]
        print(indices)
        # data = np.append(data[max(indices)], np.sum(data[min(indices) : max(indices)]))
        max_range = max(indices) + min_range + 1
        data_faser.append(np.sum(data[min_range:max_range]))
        # print(len(xvals_per_obs[0]))
        print(min_range, max_range)
        xvals_per_obs = xvals_per_obs[max(indices) + 1 :]
        # xvals_per_obs = xvals_per_obs[max(indices) :]
        # print(xvals_per_obs)
        min_range = max_range
    data_faser.append(np.sum(data[max_range:]))
    print(data_faser)

    # print(f"data = {data}")
    # print(f" xvals = {xvals_per_obs}")
    # if rebin == 0:
    #     data = np.append(data[:-5], np.sum(data[-5:], axis=0))
    #     data_min = np.append(data_min[:-5], np.sum(data_min[-5:], axis=0))
    #     data_max = np.append(data_max[:-5], np.sum(data_max[-5:], axis=0))
    #     events_per_obs = np.append(events_per_obs[:-5], np.sum(events_per_obs[-5:]))
    #     xvals_per_obs = xvals_per_obs[:-4]

    #     summed_column = torch.sum(fk_tables[-5:, :], axis=0)

    #     fk_tables = torch.cat([fk_tables[:-5, :], summed_column.unsqueeze(0)], dim=0)
    #     return (
    #         data,
    #         data_min,
    #         data_max,
    #         xvals_per_obs,
    #         binwidths,
    #         xlabels,
    #         events_per_obs,
    #         fk_tables,
    #         x_alphas,
    #     )
    # num_sum_bins = 4
    # data = np.sum(data.reshape(-1, num_sum_bins), axis=1)
    # data_min = np.sum(data_min.reshape(-1, num_sum_bins), axis=1)
    # data_max = np.sum(data_max.reshape(-1, num_sum_bins), axis=1)
    # events_per_obs = np.sum(events_per_obs.reshape(-1, num_sum_bins), axis=1)
    # rebin_xvals_per_obs = []
    # for i in range(len(data)):
    #     rebin_xvals_per_obs.append(xvals_per_obs[(i + 1) * num_sum_bins - 1])

    # fk_tables = fk_tables.reshape(5, 4, 50).sum(dim=1)  # Shape: (4, 50)

    # return (
    #     data,
    #     data_min,
    #     data_max,
    #     rebin_xvals_per_obs,
    #     binwidths,
    #     xlabels,
    #     events_per_obs,
    #     fk_tables,
    #     x_alphas,
    # )


rebin_fk()
