import numpy as np
import torch


def read_hist(filename):
    """Read histograms of data

    Args:
        filename (str): name of data file

    Returns:
        tuple: returns tuple of lists, which contain the data
    """
    low, high, val, err = np.loadtxt(f"data/{filename}", unpack=True)
    return low, high, val, err


class ReadPOWHEGData:
    """This Class reads and analyzes data generated with POWHEG code"""

    def __init__(self, luminosity_factor, num_obs):
        self.luminosity_factor = luminosity_factor
        self.num_obs = num_obs
        """Define variables 

        Args:
            luminosity_factor (int): factor to multiply the number of events to decrease relative statistical error"""

    def raw_data_to_N_events(self):
        """This function reads the data and computes the number of events from the data by multiplying tha data by the energy binwidth and by a luminosity factor to increase the number of events.

        Args:
            luminosity_factor int: factor of 1 is 150 fb^-1

        Returns:
            tuple: tuple of lists containing the number of events and its uncertainty, and the neutrino energy
        """

        observables = ["Enu", "El", "Eh", "theta"]
        (
            binwidths,
            events,
            xvals_per_obs,
            max_events,
            min_events,
        ) = (
            [],
            [],
            [],
            [],
            [],
        )

        for i in range(self.num_obs):
            LO_filename_n = f"pwgPOWHEG+PYTHIA8-output-11-{observables[i]}-LOn.dat"
            LO_filename_min_n = f"pwgPOWHEG+PYTHIA8-output-min-{observables[i]}-LOn.dat"
            LO_filename_max_n = f"pwgPOWHEG+PYTHIA8-output-max-{observables[i]}-LOn.dat"
            LO_filename_p = f"pwgPOWHEG+PYTHIA8-output-11-{observables[i]}-LOp.dat"
            LO_filename_min_p = f"pwgPOWHEG+PYTHIA8-output-min-{observables[i]}-LOp.dat"
            LO_filename_max_p = f"pwgPOWHEG+PYTHIA8-output-max-{observables[i]}-LOp.dat"

            LO_filename_n_anti = (
                f"pwgPOWHEG+PYTHIA8-output-11-{observables[i]}-LOn-anti.dat"
            )
            LO_filename_min_n_anti = (
                f"pwgPOWHEG+PYTHIA8-output-min-{observables[i]}-LOn-anti.dat"
            )
            LO_filename_max_n_anti = (
                f"pwgPOWHEG+PYTHIA8-output-max-{observables[i]}-LOn-anti.dat"
            )
            LO_filename_p_anti = (
                f"pwgPOWHEG+PYTHIA8-output-11-{observables[i]}-LOp-anti.dat"
            )
            LO_filename_min_p_anti = (
                f"pwgPOWHEG+PYTHIA8-output-min-{observables[i]}-LOp-anti.dat"
            )
            LO_filename_max_p_anti = (
                f"pwgPOWHEG+PYTHIA8-output-max-{observables[i]}-LOp-anti.dat"
            )

            LO_low, LO_high, LO_nvalb, LO_err = read_hist(LO_filename_n_anti)
            binwidth = LO_high[0] - LO_low[0]

            LO_low_min, LO_high_min, LO_nval_minb, LO_err = read_hist(
                LO_filename_min_n_anti
            )
            LO_low_max, LO_high_max, LO_nval_maxb, LO_err = read_hist(
                LO_filename_max_n_anti
            )
            LO_low, LO_high, LO_pvalb, LO_err = read_hist(LO_filename_p_anti)
            LO_low_min, LO_high_min, LO_pval_minb, LO_err = read_hist(
                LO_filename_min_p_anti
            )
            LO_low_max, LO_high_max, LO_pval_maxb, LO_err = read_hist(
                LO_filename_max_p_anti
            )

            LO_low, LO_high, LO_nval, LO_err = read_hist(LO_filename_n)
            LO_low_min, LO_high_min, LO_nval_min, LO_err = read_hist(LO_filename_min_n)
            LO_low_max, LO_high_max, LO_nval_max, LO_err = read_hist(LO_filename_max_n)
            LO_low, LO_high, LO_pval, LO_err = read_hist(LO_filename_p)
            LO_low_min, LO_high_min, LO_pval_min, LO_err = read_hist(LO_filename_min_p)
            LO_low_max, LO_high_max, LO_pval_max, LO_err = read_hist(LO_filename_max_p)

            LO_val = 74 / 183 * (LO_pval + LO_pvalb) + (183 - 74) / 183 * (
                LO_nval + LO_nvalb
            )
            LO_val_min = 74 / 183 * (LO_pval_min + LO_pval_minb) + (183 - 74) / 183 * (
                LO_nval_min + LO_nval_minb
            )
            LO_val_max = 74 / 183 * (LO_pval_max + LO_pval_maxb) + (183 - 74) / 183 * (
                LO_nval_max + LO_nval_minb
            )

            LO_val *= self.luminosity_factor
            LO_val_min *= self.luminosity_factor
            LO_val_max *= self.luminosity_factor

            events.append(LO_val)
            xvals_per_obs.append(LO_high)
            min_events.append(LO_val_min)
            max_events.append(LO_val_max)
            binwidths.append(binwidth)

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
            observables,
            events_per_obs,
        )
