import numpy as np
import torch
from read_fk_table import get_fk_table
from ML_fit_enu.src.ML_fit_neutrinos.obsolete.read_LHEF import read_LHEF_data


def rebin_fk(rebin):
    # Read data
    data, data_min, data_max, xvals_per_obs, binwidths, xlabels, events_per_obs = (
        read_LHEF_data()
    )

    x_alphas, fk_tables = get_fk_table()
    if rebin == 0:
        data = np.append(data[:-5], np.sum(data[-5:], axis=0))
        data_min = np.append(data_min[:-5], np.sum(data_min[-5:], axis=0))
        data_max = np.append(data_max[:-5], np.sum(data_max[-5:], axis=0))
        events_per_obs = np.append(events_per_obs[:-5], np.sum(events_per_obs[-5:]))
        xvals_per_obs = xvals_per_obs[:-4]

        summed_column = torch.sum(fk_tables[-5:, :], axis=0)

        fk_tables = torch.cat([fk_tables[:-5, :], summed_column.unsqueeze(0)], dim=0)
        return (
            data,
            data_min,
            data_max,
            xvals_per_obs,
            binwidths,
            xlabels,
            events_per_obs,
            fk_tables,
            x_alphas,
        )
    if rebin == 1:
        num_sum_bins = 4
        data = np.sum(data.reshape(-1, num_sum_bins), axis=1)
        data_min = np.sum(data_min.reshape(-1, num_sum_bins), axis=1)
        data_max = np.sum(data_max.reshape(-1, num_sum_bins), axis=1)
        events_per_obs = np.sum(events_per_obs.reshape(-1, num_sum_bins), axis=1)
        rebin_xvals_per_obs = []
        for i in range(len(data)):
            rebin_xvals_per_obs.append(xvals_per_obs[(i + 1) * num_sum_bins - 1])

        fk_tables = fk_tables.reshape(5, 4, 50).sum(dim=1)  # Shape: (4, 50)

        return (
            data,
            data_min,
            data_max,
            rebin_xvals_per_obs,
            binwidths,
            xlabels,
            events_per_obs,
            fk_tables,
            x_alphas,
        )

    if rebin == 2:
        data = np.append(data[:-5], np.sum(data[-5:], axis=0))
        data_min = np.append(data_min[:-5], np.sum(data_min[-5:], axis=0))
        data_max = np.append(data_max[:-5], np.sum(data_max[-5:], axis=0))
        events_per_obs = np.append(events_per_obs[:-5], np.sum(events_per_obs[-5:]))
        xvals_per_obs = xvals_per_obs[:-4]

        summed_column = torch.sum(fk_tables[-5:, :], axis=0)

        fk_tables = torch.cat([fk_tables[:-5, :], summed_column.unsqueeze(0)], dim=0)

        return (
            data,
            data_min,
            data_max,
            xvals_per_obs,
            binwidths,
            xlabels,
            events_per_obs,
            fk_tables,
            x_alphas,
        )
