import numpy as np
import torch
import torch.nn.functional
from read_faserv_pdf import read_pdf
from MC_data_reps import generate_MC_replicas
from logspace_grid import generate_grid
from read_fk_table import get_fk_table
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def get_data(pdf, x_alphas, fk_tables_mu, fk_tables_mub, binwidths_mu, binwidths_mub):
    faser_pdf, x_faser = read_pdf(pdf, x_alphas.flatten(), 14)
    faser_pdf = torch.tensor(faser_pdf, dtype=torch.float32).view(-1, 1)
    data_mu = torch.matmul(fk_tables_mu, faser_pdf) * binwidths_mu
    data_mu = data_mu.detach().numpy().flatten()
    data_max_mu = data_mu + data_mu / 20
    data_min_mu = data_mu - data_mu / 20

    faser_pdf, x_faser = read_pdf(pdf, x_alphas.flatten(), -14)
    faser_pdf = torch.tensor(faser_pdf, dtype=torch.float32).view(-1, 1)

    data_mub = torch.matmul(fk_tables_mub, faser_pdf) * binwidths_mub
    data_mub = data_mub.detach().numpy().flatten()
    data_max_mub = data_mub + data_mub / 20
    data_min_mub = data_mub - data_mub / 20

    return data_mu, data_min_mu, data_max_mu, data_mub, data_min_mub, data_max_mub


def combine_mu_mub(mu, mub):
    mu[-1] = mu[-1] + mub[-1]
    mub = mub[:-1]
    mub = mub[::-1]
    combined = np.hstack((mu, mub))
    return combined


def data_needed_for_fit(fit_level, seed):
    filename = "data_muon_sim_faser/FK_Enu_7TeV_nu_W.dat"
    x_alphas, _ = get_fk_table(
        filename=filename, parent_dir=parent_dir, x_alpha_grid=True
    )
    # events = [44.1, 92.7, 68.5, 66.8, 44.3, 21.9]
    events = [223.16, 368.27, 258.92, 205.8, 108.74, 77.845]
    # std_errev = [3.1, 2.5, 1.7, 1.6, 2.3, 2.9]
    std_errev = [0, 0, 0, 0, 0, 0]
    # 39
    sig_sys = std_errev
    sig_sys = np.array(sig_sys)
    # sig_stat =np.sqrt(events)
    sig_stat = [72.011, 78.987, 64.535, 41.695, 24.934, 29.098]
    sig_stat = np.array(sig_stat)

    data = np.array(events)

    pdf = "FASERv_EPOS+POWHEG_7TeV"
    binwidths_mu = [200, 300, 400, 900]
    binwidths_mu = torch.tensor(binwidths_mu, dtype=torch.float32).view(-1, 1)

    binwidths_mub = [200, 700, 900]
    binwidths_mub = torch.tensor(binwidths_mub, dtype=torch.float32).view(-1, 1)

    # cov_matrix = np.diag(sig_sys**2 + sig_stat**2)
    # cov_matrix = np.linalg.inv(cov_matrix)

    cov_matrix = np.array(
        [
            [5186, -1623, 340, -69, 2, 5],
            [-1623, 6239, -1952, 281, -19, -4],
            [340, -1952, 4165, -734, 56, -27],
            [-69, 281, -734, 1738, -130, 15],
            [2, -19, 56, -130, 622, -147],
            [5, -4, -27, 15, -147, 847],
        ]
    )

    cov_matrix = np.linalg.inv(cov_matrix)

    cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32, requires_grad=False)
    level0, level1, level2 = generate_MC_replicas(1, data, sig_sys, sig_stat, seed)
    # lowx = -8
    # n = 250
    # x_vals = generate_grid(lowx, n)

    if fit_level == 0:
        pred = level0
    if fit_level == 1:
        pred = level1
    if fit_level == 2:
        pred = level2

    # lowx = -10
    # n = 500
    # x_vals = generate_grid(lowx, n)

    x_vals = np.logspace(-5, 0, 1000)

    # SIMULATED DATA

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

    simulated_data = combine_mu_mub(data_mu, data_mub)

    err_sim = np.sqrt(simulated_data)

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
        simulated_data,
        err_sim,
    )
