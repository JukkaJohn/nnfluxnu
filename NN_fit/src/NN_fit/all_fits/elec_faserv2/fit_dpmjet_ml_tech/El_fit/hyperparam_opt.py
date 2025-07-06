import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
import torch.nn as nn
from itertools import product
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_dpmjet import data_needed_for_fit
from control_file_dpmjet import hyperparams
from hyperpram import perform_fit


REPLICAS = 1
(
    preproc,
    lr,
    fit_level,
    max_counter,
    num_nodes,
    num_layers,
    act_functions,
    postfit_criteria,
    postfit_measures,
    wd,
    range_alpha,
    range_beta,
    range_gamma,
    extended_loss,
    validation,
    max_num_epochs,
) = hyperparams()
from read_faserv_pdf import read_pdf

# from read_LHEF import read_LHEF_data
from data_errors import compute_errors
from MC_data_reps import generate_MC_replicas
from postfit_criteria import Postfit
from form_loss_fct import complete_loss_fct, raw_loss_fct
from postfit_measures import Measures
from logspace_grid import generate_grid
from read_fk_table import get_fk_table
# from rebin_fk_data import rebin_fk


# from read_fk_table import get_fk_table

import pandas as pd

# import torch

filename_data_mu = (
    "../../../FKtables/data/data/data_El_FASERv_Run3_DPMJET+DPMJET_7TeV_numu_W.dat"
)
filename_data_mub = (
    "../../../FKtables/data/data/data_El_FASERv_Run3_DPMJET+DPMJET_7TeV_nubmu_W.dat"
)
filename_uncert_mu = "../../../FKtables/data/uncertainties/uncertainties_El_FASERv_Run3_DPMJET+DPMJET_7TeV_numu_W.dat"
filename_uncert_mub = "../../../FKtables/data/uncertainties/uncertainties_El_FASERv_Run3_DPMJET+DPMJET_7TeV_nubmu_W.dat"
filename_fk_mub_n = "../../../FKtables/data/fastkernel/FK_El_final_nubmu_n.dat"
filename_fk_mub_p = "../../../FKtables/data/fastkernel/FK_El_final_nubmu_p.dat"
filename_fk_mu_n = "../../../FKtables/data/fastkernel/FK_El_final_numu_n.dat"
filename_fk_mu_p = "../../../FKtables/data/fastkernel/FK_El_final_numu_p.dat"
filename_binsize = "../../../FKtables/data/binning/FK_El_binsize.dat"


level_1_instance = 42
(
    data,
    sig_sys,
    sig_stat,
    pdf,
    binwidths_mu,
    binwidths_mub,
    cov_matrix,
    pred,
    x_vals,
    x_alphas,
    level1,
    fk_tables_mu,
    fk_tables_mub,
    low_bin_mu,
    low_bin_mub,
    high_bin_mu,
    high_bin_mub,
) = data_needed_for_fit(
    fit_level,
    seed=level_1_instance,
    filename_data_mu=filename_data_mu,
    filename_data_mub=filename_data_mub,
    filename_uncert_mu=filename_uncert_mu,
    filename_uncert_mub=filename_uncert_mub,
    filename_fk_mub_n=filename_fk_mub_n,
    filename_fk_mub_p=filename_fk_mub_p,
    filename_fk_mu_n=filename_fk_mu_n,
    filename_fk_mu_p=filename_fk_mu_p,
    filename_binsize=filename_binsize,
)

seed = 10
hyper_loss = []


lrs = np.arange(0.001, 0.05, 0.002)
max_counters = np.arange(1, 10000, 100)
num_nodess = np.arange(3, 20, 1)
num_layerss = np.arange(3, 10, 1)
act_functionss = [
    torch.nn.Softplus(),
    torch.nn.ReLU(),
    torch.nn.Sigmoid(),
    torch.nn.Tanh(),
]
wds = np.logspace(-6, -1, 10)

max_num_epochss = np.arange(2000, 50000, 1000)


# for lr, max_counter, num_nodes, num_layers, act_functions, wd, max_num_epochs in (
#     lrs,
#     max_counters,
#     num_nodess,
#     num_layerss,
#     act_functionss,
#     wds,
#     max_num_epochss,
# ):
# lr, max_counter, nodes, layers, activation, weight_decay, epochs = hyperparam
# for lr in lrs:
#     for max_counter in max_counters:
#         for num_nodes in num_nodess:
#             for num_layers in num_layerss:
#                 for act_function in act_functionss:
#                     for wd in wds:
#                         for max_num_epoch in max_num_epochss:


num_samples = 100

for i in range(num_samples):
    act_functions = []
    rng = np.random.default_rng(i)
    lr = rng.choice(lrs)
    max_counter = rng.choice(max_counters)
    num_nodes = rng.choice(num_nodess)
    num_layers = rng.choice(num_layerss)
    act_function = rng.choice(act_functionss)
    wd = rng.choice(wds)
    max_num_epoch = rng.choice(max_num_epochss)
    for _ in range(num_layers + 1):
        act_functions.append(act_function)
    print("hyperparam")
    print("new run")
    # print(hyperparam)
    print(
        lr,
        max_counter,
        num_nodes,
        num_layers,
        act_function,
        wd,
        max_num_epoch,
    )
    loss = perform_fit(
        pred,
        REPLICAS,
        range_alpha,
        range_beta,
        range_gamma,
        lr,
        wd,
        max_counter,
        x_alphas,
        fk_tables_mu,
        fk_tables_mub,
        binwidths_mu,
        binwidths_mub,
        cov_matrix,
        extended_loss,
        act_functions,
        num_nodes,
        num_layers,
        x_vals,
        preproc,
        validation,
        seed,
        max_num_epoch,
    )

    hyper_loss.append(loss)
    print(f"loss = {loss}")

print("argmin")
print(hyper_loss)
print(np.argmin(hyper_loss))
