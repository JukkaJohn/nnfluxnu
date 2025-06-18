from read_data import ReadPOWHEGData
import numpy as np


class RebinData:
    """This Class can rebin the imported data by making the binwidths larger. One can choose if the binwidths need to be the same size or not."""

    def __init__(self, luminosity_factor, init_rebinning, min_num_events):
        """Define variables

        Args:
            luminosity_factor (int): factor to multiply the number of events to decrease relative statistical error
            init_rebinning (int): new binwidth
        """
        self.luminosity_factor = luminosity_factor
        self.init_rebinning = init_rebinning
        self.min_num_events = min_num_events

    def rebin_eq_binwidth(self):
        """This rebins the imported data by making larger binwidths

        Returns:
            tuple: lists of rebinned data and the original binwidth
        """
        data = ReadPOWHEGData(self.luminosity_factor)
        NLO_low, NLO_val, NLO_val_min, NLO_val_max, binwidth = (
            data.raw_data_to_N_events()
        )
        init_rebinning = int(len(NLO_low) / self.init_rebinning)

        NLO_val = np.sum(NLO_val.reshape(-1, init_rebinning), axis=1)
        NLO_val_min = np.sum(NLO_val_min.reshape(-1, init_rebinning), axis=1)
        NLO_val_max = np.sum(NLO_val_max.reshape(-1, init_rebinning), axis=1)
        NLO_low = NLO_low[init_rebinning::init_rebinning]
        # Also incluce last datapoint
        NLO_low = np.append(NLO_low, 6000)
        print(f"num events = {sum(NLO_val)}")

        return NLO_low, NLO_val, NLO_val_min, NLO_val_max, binwidth

    def rebin_uneq_binwidth(self):
        """Rebins the data by using different binwidths i.e. larger binwidth when # events is low to keep srelative tatistical error low.

        Returns:
            tuple: binned number of events and its uncertainties, the binned neutrino energy and the original binwidth
        """

        NLO_low, NLO_val, NLO_val_min, NLO_val_max, binwidth = self.rebin_eq_binwidth()
        num_events_NLO, num_events_max_NLO, num_events_min_NLO = 0, 0, 0
        NLO_low_binned, NLO_val_binned, NLO_val_max_binned, NLO_val_min_binned = (
            [],
            [],
            [],
            [],
        )

        for i in range(0, len(NLO_val)):
            num_events_NLO += NLO_val[i]
            num_events_min_NLO += NLO_val_min[i]
            num_events_max_NLO += NLO_val_max[i]

            if num_events_NLO >= self.min_num_events:
                NLO_val_binned.append(num_events_NLO)
                NLO_low_binned.append(NLO_low[i])
                NLO_val_max_binned.append(num_events_max_NLO)
                NLO_val_min_binned.append(num_events_min_NLO)

                num_events_NLO = 0
                num_events_max_NLO = 0
                num_events_min_NLO = 0

        # Add the remainder events if there are any
        if num_events_NLO > 0:
            NLO_low_binned.append(NLO_low[-1])

            NLO_val_binned.append(num_events_NLO)
            NLO_val_min_binned.append(num_events_min_NLO)
            NLO_val_max_binned.append(num_events_max_NLO)

        NLO_low_binned = np.array(NLO_low_binned)
        NLO_val_binned = np.array(NLO_val_binned)
        NLO_val_max_binned = np.array(NLO_val_max_binned)
        NLO_val_min_binned = np.array(NLO_val_min_binned)

        return (
            NLO_low_binned,
            NLO_val_binned,
            NLO_val_min_binned,
            NLO_val_max_binned,
            binwidth,
        )
