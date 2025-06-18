import numpy as np
import torch
import torch.nn.functional
import sys
import os

# parent_dir = os.path.abspath(
#     os.path.join(
#         os.getcwd(),
#         "/data/theorie/jjohn/git/faser_nufluxes_ml/ML_fit_enu/src/ML_fit_neutrinos/",
#     )
# )
# sys.path.append(parent_dir)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from read_faserv_pdf import read_pdf
from MC_data_reps import generate_MC_replicas
from logspace_grid import generate_grid
from data_errors import compute_errors

import pandas as pd
from read_fk_table import get_fk_table


def get_data(pdf, x_alphas, fk_tables_mu, fk_tables_mub, binwidths_mu, binwidths_mub):
    faser_pdf, x_faser = read_pdf(pdf, x_alphas.flatten(), 12)
    faser_pdf = torch.tensor(faser_pdf, dtype=torch.float32).view(-1, 1)
    data_mu = torch.matmul(fk_tables_mu, faser_pdf) * binwidths_mu
    data_mu = data_mu.detach().numpy().flatten()
    data_max_mu = data_mu + data_mu / 20
    data_min_mu = data_mu - data_mu / 20

    faser_pdf, x_faser = read_pdf(pdf, x_alphas.flatten(), -12)
    faser_pdf = torch.tensor(faser_pdf, dtype=torch.float32).view(-1, 1)

    data_mub = torch.matmul(fk_tables_mub, faser_pdf) * binwidths_mub
    data_mub = data_mub.detach().numpy().flatten()
    data_max_mub = data_mub + data_mub / 20
    data_min_mub = data_mub - data_mub / 20
    print('data_mu')
    print(data_mu)	
    return data_mu, data_min_mu, data_max_mu, data_mub, data_min_mub, data_max_mub


def combine_mu_mub(mu, mub):
    mu[-1] = mu[-1] + mub[-1]
    mub = mub[:-1]
    mub = mub[::-1]
    combined = np.hstack((mu, mub))
    return combined


def data_needed_for_fit(fit_level, seed):
    filename = "data_muon_sim_faser/FK_Enu_7TeV_nu_W.dat"
    x_alphas, fk_tables_mu = get_fk_table(
        filename=filename, parent_dir=parent_dir, x_alpha_grid=True
    )
    filename = "data_muon_sim_faser/FK_Enu_7TeV_nub_W.dat"
    x_alphas, fk_tables_mub = get_fk_table(
        filename=filename, parent_dir=parent_dir, x_alpha_grid=True
    )

    pdf = "FASERv_EPOS+POWHEG_7TeV"
    binwidths_mu = [200, 300, 400, 900]
    binwidths_mu = torch.tensor(binwidths_mu, dtype=torch.float32).view(-1, 1)
    binwidths_mub = [200, 700, 900]
    binwidths_mub = torch.tensor(binwidths_mub, dtype=torch.float32).view(-1, 1)

    data_mu, data_min_mu, data_max_mu, data_mub, data_min_mub, data_max_mub = get_data(
        pdf, x_alphas, fk_tables_mu, fk_tables_mub, binwidths_mu, binwidths_mub
    )

    sig_sys_mu, sig_tot_mu, cov_matrix_mu = compute_errors(
        data_mu, data_min_mu, data_max_mu
    )
    sig_sys_mub, sig_tot_mub, cov_matrix_mub = compute_errors(
        data_mub, data_min_mub, data_max_mub
    )

    sig_stat_mu = np.sqrt(data_mu)
    sig_stat_mub = np.sqrt(data_mub)
    sig_stat = combine_mu_mub(sig_stat_mu, sig_stat_mub)

    sig_sys = combine_mu_mub(sig_sys_mu, sig_sys_mub)

    data = combine_mu_mub(data_mu, data_mub)

    sig_tot = sig_stat**2 + sig_sys**2 + .1
    cov_matrix = np.diag(sig_tot)
    cov_matrix = np.linalg.inv(cov_matrix)
    cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32, requires_grad=False)

    level0, level1, level2 = generate_MC_replicas(1, data, sig_sys, sig_stat, seed)

    x_vals = np.logspace(-5, 0, 1000)

    if fit_level == 0:
        pred = level0
    if fit_level == 1:
        pred = level1
    if fit_level == 2:
        pred = level2

    return (
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
    )


# maybe create a class or something (of overkoepelende functie die alles aanroept en returnt)
