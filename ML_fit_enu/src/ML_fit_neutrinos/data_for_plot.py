import numpy as np
import torch
import torch.nn.functional
import sys
import os

import matplotlib.pyplot as plt
import lhapdf

lhapdf.setVerbosity(0)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
x_alpha_grid = False

from read_faserv_pdf import read_pdf
from MC_data_reps import generate_MC_replicas
from logspace_grid import generate_grid
from data_errors import compute_errors

import pandas as pd
from read_fk_table import get_fk_table


def get_data(
    filename_fk_mub_n,
    filename_fk_mub_p,
    filename_fk_mu_n,
    filename_fk_mu_p,
    filename_binsize,
    pdf,
):
    x_alphas, fk_tables_mu_n = get_fk_table(
        filename=filename_fk_mu_n, parent_dir=parent_dir, x_alpha_grid=x_alpha_grid
    )
    # filename = "data_dpmjet/FK_Eh_final_numu_p.dat"
    x_alphas, fk_tables_mu_p = get_fk_table(
        filename=filename_fk_mu_p, parent_dir=parent_dir, x_alpha_grid=x_alpha_grid
    )

    fk_tables_mu = fk_tables_mu_n * 0.5956284 + fk_tables_mu_p * 0.4043716

    # filename = "data_dpmjet/FK_Eh_final_nubmu_n.dat"
    x_alphas, fk_tables_mub_n = get_fk_table(
        filename=filename_fk_mub_n, parent_dir=parent_dir, x_alpha_grid=x_alpha_grid
    )
    # filename = "data_dpmjet/FK_Eh_final_nubmu_p.dat"
    x_alphas, fk_tables_mub_p = get_fk_table(
        filename=filename_fk_mub_p, parent_dir=parent_dir, x_alpha_grid=x_alpha_grid
    )
    fk_tables_mub = fk_tables_mub_n * 0.5956284 + fk_tables_mub_p * 0.4043716

    # filename = "data_dpmjet/FK_Eh_binsize.dat"
    file_path = os.path.join(parent_dir, filename_binsize)
    low_bin, high_bin, binwidths_mu = np.loadtxt(f"{file_path}", unpack=True)
    binwidths_mub = binwidths_mu

    faser_pdf, x = read_pdf(pdf, x_alphas.detach().numpy().flatten(), 14)
    faser_pdf = torch.tensor(faser_pdf, dtype=torch.float32)
    data_mu = (
        (torch.matmul(fk_tables_mu, faser_pdf * 1.16186e-09) * binwidths_mu)
        .detach()
        .numpy()
        .flatten()
    )

    faser_pdf, x = read_pdf(pdf, x_alphas.detach().numpy().flatten(), -14)
    faser_pdf = torch.tensor(faser_pdf, dtype=torch.float32)
    data_mub = (
        (torch.matmul(fk_tables_mub, faser_pdf * 1.16186e-09) * binwidths_mub)
        .detach()
        .numpy()
        .flatten()
    )
    error_mu = np.sqrt(data_mu)
    error_mub = np.sqrt(data_mub)

    return (
        data_mu,
        data_mub,
        error_mu,
        error_mub,
        fk_tables_mu,
        fk_tables_mub,
        low_bin,
        high_bin,
        binwidths_mu,
        binwidths_mub,
        x_alphas,
    )


def aggregate_entries_with_indices(
    fk_tables,
    data,
    binwidths,
    low_bin,
    high_bin,
):
    threshold = 20

    (
        rebin_data,
        rebin_fk_table_mu,
        rebin_binwidhts,
        rebin_low_bin,
        rebin_high_bin,
    ) = (
        [],
        [],
        [],
        [],
        [],
    )
    current_sum = 0
    start_idx = 0
    raw_data = data / binwidths

    for i, value in enumerate(data):
        current_sum += value

        if current_sum >= threshold:
            rebin_data.append(sum(data[start_idx : i + 1]))
            sum_binwidth = 0.0

            sum_binwidth = np.sum(data[start_idx : i + 1]) / np.sum(
                raw_data[start_idx : i + 1]
            )

            rebin_binwidhts.append(sum_binwidth)

            rebin_low_bin.append(low_bin[i])
            rebin_high_bin.append(high_bin[i])
            summed_column_mu = torch.sum(fk_tables[start_idx : i + 1, :], axis=0)
            summed_column_mu = summed_column_mu.unsqueeze(0)
            rebin_fk_table_mu.append(summed_column_mu)

            previous_idx = start_idx
            start_idx = i + 1
            current_sum = 0

    if current_sum >= 0:
        rebin_data[-1] += sum(data[start_idx:])

        sum_binwidth = 0.0

        sum_binwidth = np.sum(data[previous_idx:]) / np.sum(raw_data[previous_idx:])

        rebin_binwidhts[-1] = sum_binwidth

        rebin_fk_table_mu[-1] += torch.sum(fk_tables[start_idx:, :], axis=0).unsqueeze(
            0
        )

        rebin_low_bin[-1] = low_bin[previous_idx]
        rebin_high_bin.append(high_bin[-1])

    rebin_fk_table_mu = torch.cat(rebin_fk_table_mu, dim=0)

    data = np.array(data)
    rebin_binwidhts = np.array(rebin_binwidhts)
    rebin_low_bin = np.array(rebin_low_bin)

    return (
        rebin_data,
        rebin_fk_table_mu,
        rebin_binwidhts,
        rebin_low_bin,
        rebin_high_bin,
    )


def data_needed_for_fit(
    filename_fk_mub_n,
    filename_fk_mub_p,
    filename_fk_mu_n,
    filename_fk_mu_p,
    filename_binsize,
    pdf,
):
    (
        data_mu,
        data_mub,
        error_mu,
        error_mub,
        fk_tables_mu,
        fk_tables_mub,
        low_bin,
        high_bin,
        binwidths_mu,
        binwidths_mub,
        x_alphas,
    ) = get_data(
        filename_fk_mub_n,
        filename_fk_mub_p,
        filename_fk_mu_n,
        filename_fk_mu_p,
        filename_binsize,
        pdf,
    )

    print("data before")
    print(len(data_mu))

    (
        data_mu,
        fk_tables_mu,
        binwidths_mu,
        low_bin_mu,
        high_bin_mu,
    ) = aggregate_entries_with_indices(
        fk_tables_mu,
        data_mu,
        binwidths_mu,
        low_bin,
        high_bin,
    )
    print("data after")
    print((data_mu))

    (
        data_mub,
        fk_tables_mub,
        binwidths_mub,
        low_bin_mub,
        high_bin_mub,
    ) = aggregate_entries_with_indices(
        fk_tables_mub,
        data_mub,
        binwidths_mub,
        low_bin,
        high_bin,
    )
    data = np.hstack((data_mu, data_mub))
    print("data,sig_tot")
    print(data)

    binwidths_mu = torch.tensor(binwidths_mu, dtype=torch.float32).view(-1, 1)
    binwidths_mub = torch.tensor(binwidths_mub, dtype=torch.float32).view(-1, 1)
    # binwidths_mub = binwidths_mu

    sig_stat = np.sqrt(data)
    sig_sys = 0
    sig_tot = sig_stat**2
    print(sig_tot)
    seed = 42
    level0, level1, level2 = generate_MC_replicas(1, data, sig_sys, sig_stat, seed)

    return (
        data_mu,
        data_mub,
        low_bin_mu,
        low_bin_mub,
        high_bin_mu,
        high_bin_mub,
        level1,
    )


# maybe create a class or something (of overkoepelende functie die alles aanroept en returnt)
