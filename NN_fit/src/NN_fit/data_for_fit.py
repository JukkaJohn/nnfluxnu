import numpy as np
import torch
import torch.nn.functional
import sys
import os

import matplotlib.pyplot as plt
import lhapdf

lhapdf.setVerbosity(0)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from read_faserv_pdf import read_pdf
from MC_data_reps import generate_MC_replicas

import pandas as pd
from read_fk_table import get_fk_table


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
            rebin_data.append(sum(data[start_idx : i + 1]))  # Sum based on indices
            sum_binwidth = 0.0
            # print("new one")
            # for k in range(start_idx, i + 1):
            # print(k)
            sum_binwidth = np.sum(data[start_idx : i + 1]) / np.sum(
                raw_data[start_idx : i + 1]
            )

            # sum_binwidth + binwidths[k] * (
            #     data[k] / np.sum(data[start_idx : i + 1], dtype=np.float64)
            # )
            rebin_binwidhts.append(sum_binwidth)

            # print(i)
            rebin_low_bin.append(low_bin[i])
            rebin_high_bin.append(high_bin[i])
            summed_column_mu = torch.sum(fk_tables[start_idx : i + 1, :], axis=0)
            summed_column_mu = summed_column_mu.unsqueeze(0)
            rebin_fk_table_mu.append(summed_column_mu)

            # print("check sums")
            # print(fk_tables_mu[start_idx : i + 1, :])
            # print(summed_column_mu)
            # print(current_sum)
            previous_idx = start_idx
            start_idx = i + 1
            current_sum = 0
    # If there are remaining events that haven't been added, keep them as the last entry
    if current_sum >= 0:
        # print("lenmgth end")
        # print(len(data[start_idx:]))
        lemgth = len(data[start_idx:])
        print(lemgth)
        rebin_data[-1] += sum(data[start_idx:])
        # rebin_data.append(sum(data[start_idx:]))

        sum_binwidth = 0.0
        # print("new one")

        # need previous start_idx
        sum_binwidth = np.sum(data[previous_idx:]) / np.sum(raw_data[previous_idx:])
        # for k in range(start_idx, len(data)):
        #     print(k)
        #     sum_binwidth = sum_binwidth + binwidths[k] * (
        #         data[k] / np.sum(data[start_idx:], dtype=np.float64)
        #     )
        # rebin_binwidhts.append(sum_binwidth)

        rebin_binwidhts[-1] = sum_binwidth

        rebin_fk_table_mu[-1] += torch.sum(fk_tables[start_idx:, :], axis=0).unsqueeze(
            0
        )
        # rebin_fk_table_mu.append(
        #     torch.sum(fk_tables[start_idx:, :], axis=0).unsqueeze(0)
        # )

        # rebin_low_bin[-1] = low_bin[-1]
        rebin_low_bin[-1] = low_bin[previous_idx]
        rebin_high_bin.append(high_bin[-1])

    rebin_fk_table_mu = torch.cat(rebin_fk_table_mu, dim=0)

    data = np.array(data)
    rebin_binwidhts = np.array(rebin_binwidhts)
    rebin_low_bin = np.array(rebin_low_bin)
    # rebin_data *= rebin_binwidhts

    # print("bins en data")
    # print(len(rebin_low_bin))
    # print(len(rebin_data))

    return (
        rebin_data,
        rebin_fk_table_mu,
        rebin_binwidhts,
        rebin_low_bin,
        rebin_high_bin,
    )


def data_needed_for_fit(
    fit_level,
    seed,
    filename_data_mu,
    filename_data_mub,
    filename_uncert_mu,
    filename_uncert_mub,
    filename_fk_mub_n,
    filename_fk_mub_p,
    filename_fk_mu_n,
    filename_fk_mu_p,
    filename_binsize,
    pdf,
    data,
    error,
    fk_tables,
    low_bin,
    high_bin,
    binwidths,
    x_alphas,
    num_output_layers,
):
    faser_pdf, x = read_pdf(pdf, x_alphas.detach().numpy().flatten(), 12)
    faser_pdf = torch.tensor(faser_pdf, dtype=torch.float32)

    conv_mu = np.matmul(fk_tables, faser_pdf).flatten() * binwidths
    # plt.plot(low_bin, conv, label="before")
    print("data before")
    print(conv_mu)

    # plt.title("before")
    # plt.show()
    # print(f"data mub = {data_mub}")

    (
        data,
        fk_tables,
        binwidths,
        low_bin,
        high_bin,
    ) = aggregate_entries_with_indices(
        fk_tables,
        data,
        binwidths,
        low_bin,
        high_bin,
    )

    # print(f"after = {after}")
    # binwidths_mu[-1] = binwidths_mu[-1] / 11
    # # binwidths_mu[-1] = binwidths_mu[-1] / 11
    # binwidths_mu[1] = binwidths_mu[1] / 2
    # binwidths_mu[0] = binwidths_mu[0] / 6

    conv = np.matmul(fk_tables, faser_pdf).flatten() * binwidths
    print("data after")

    # print(conv)
    # print(data_mu)
    # print(low_bin_mu)

    # plt.plot(low_bin_mu, conv, label="after")
    # plt.legend()
    # plt.show()
    # print(f"data mu after = {data_mu / binwidths_mu}")

    # 10 tot 11 keer te groot de before vergeleken met after, dus binwidths 10/11 keer te groot

    # binwidths_mub[-1] = binwidths_mub[-1] / 18
    # # binwidths_mu[-1] = binwidths_mu[-1] / 11
    # binwidths_mub[1] = binwidths_mub[1] / 2
    # binwidths_mub[0] = binwidths_mub[0] / 6

    # print(after / before)
    # conv = np.matmul(fk_tables_mu, faser).flatten() * binwidths_mu
    # print(f"conv after = {conv}")
    # plt.plot(low_bin, conv, label="after")
    # plt.legend()
    # # plt.title("after")
    # plt.show()

    binwidths = torch.tensor(binwidths, dtype=torch.float32).view(-1, 1)

    sig_stat = np.sqrt(data)
    sig_sys = 0
    sig_tot = sig_stat**2
    print(sig_tot)

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
        binwidths,
        cov_matrix,
        pred,
        x_vals,
        x_alphas,
        level1,
        fk_tables,
        low_bin,
        high_bin,
    )


# maybe create a class or something (of overkoepelende functie die alles aanroept en returnt)
