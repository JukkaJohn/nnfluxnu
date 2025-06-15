import numpy as np
import torch


def generate_MC_replicas(REPLICAS, data, sig_sys, sig_stat, seed):
    """Generate level 2 data MC replicas for the NN fit by adding a level 1 and then a level 2 gaussian noise to the data

    Returns:
        list: MC replica data
    """

    # r_sys = np.random.normal(0, 1, len(data)) * sig_sys / 10
    # r_stat = np.random.normal(0, 1, len(data)) * np.sqrt(data)
    # r_sys = np.random.normal(0,sig_sys)
    # r_stat = np.random.normal(0,np.sqrt(data))

    level0, level1, level2 = [], [], []
    # data_level1 = data + r_sys + r_stat
    # print(sig_sys)
    # print(sig_stat)
    # perhaps specifying seed for postift measures?
    rng_level1 = np.random.default_rng(seed=seed)
    r_sys_1 = rng_level1.normal(0, 1, len(data)) * sig_sys
    r_stat_1 = rng_level1.normal(0, 1, len(data)) * sig_stat

    for _ in range(REPLICAS):
        # print("1")
        # print(r_sys_1, r_stat_1)
        data_level1 = data + r_sys_1 + r_stat_1
        r_sys_2 = np.random.normal(0, 1, len(data)) * sig_sys
        r_stat_2 = np.random.normal(0, 1, len(data)) * sig_stat
        # print("2")
        # print(r_sys_2, r_stat_2)
        # r_sys = np.random.normal(0,sig_sys)
        # r_stat = np.random.normal(0,np.sqrt(data))

        data_level2 = data_level1 + r_sys_2 + r_stat_2

        level0.append(torch.tensor(data, dtype=torch.float32))
        level1.append(torch.tensor(data_level1, dtype=torch.float32))
        level2.append(torch.tensor(data_level2, dtype=torch.float32))

    return level0, level1, level2
