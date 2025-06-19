# Author: Jukka John
# This file produces MC reps for the fits using the uncertainties
import numpy as np
import torch
from typing import List, Tuple


def generate_MC_replicas(
    REPLICAS: int,
    data: np.ndarray,
    sig_sys: np.ndarray,
    sig_stat: np.ndarray,
    seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Generate level 2 data MC replicas for the NN fit by adding a level 1 and then a level 2 gaussian noise to the data

    Returns:
        tuple: level 0,1 and 2 data
    """

    level0, level1, level2 = [], [], []
    rng_level1 = np.random.default_rng(seed=seed)
    r_sys_1 = rng_level1.normal(0, 1, len(data)) * sig_sys
    r_stat_1 = rng_level1.normal(0, 1, len(data)) * sig_stat

    for _ in range(REPLICAS):
        data_level1 = data + r_sys_1 + r_stat_1
        r_sys_2 = np.random.normal(0, 1, len(data)) * sig_sys
        r_stat_2 = np.random.normal(0, 1, len(data)) * sig_stat

        data_level2 = data_level1 + r_sys_2 + r_stat_2

        level0.append(torch.tensor(data, dtype=torch.float32))
        level1.append(torch.tensor(data_level1, dtype=torch.float32))
        level2.append(torch.tensor(data_level2, dtype=torch.float32))

    return level0, level1, level2
