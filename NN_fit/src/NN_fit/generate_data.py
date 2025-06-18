import numpy as np
import torch
import torch.nn.functional
import sys
import os

import matplotlib.pyplot as plt
import lhapdf
import yaml

lhapdf.setVerbosity(0)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from read_faserv_pdf import read_pdf
from read_faserv_pdf import read_pdf
from MC_data_reps import generate_MC_replicas
from logspace_grid import generate_grid
from data_errors import compute_errors

import pandas as pd
from read_fk_table import get_fk_table


if len(sys.argv) < 2:
    print("Usage: python script.py <input_file>")
    sys.exit(1)

config_path = sys.argv[1]


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


config = load_config(config_path)


def compute_pseudo_data(
    filename_fk_mub_n,
    filename_fk_mub_p,
    filename_fk_mu_n,
    filename_fk_mu_p,
    filename_binsize,
    pid,
    pdf_name,
    pdf_set,
):
    x_alphas, fk_tables_mu_n = get_fk_table(
        filename=filename_fk_mu_n, parent_dir=parent_dir
    )

    x_alphas, fk_tables_mu_p = get_fk_table(
        filename=filename_fk_mu_p, parent_dir=parent_dir
    )

    fk_tables_mu = fk_tables_mu_n * 0.5956284 + fk_tables_mu_p * 0.4043716

    x_alphas, fk_tables_mub_n = get_fk_table(
        filename=filename_fk_mub_n, parent_dir=parent_dir
    )

    x_alphas, fk_tables_mub_p = get_fk_table(
        filename=filename_fk_mub_p, parent_dir=parent_dir
    )
    fk_tables_mub = fk_tables_mub_n * 0.5956284 + fk_tables_mub_p * 0.4043716

    file_path = os.path.join(parent_dir, filename_binsize)
    low_bin, high_bin, binwidths_mu = np.loadtxt(f"{file_path}", unpack=True)
    binwidths_mub = binwidths_mu

    faser_pdf, x = read_pdf(pdf_name, x_alphas.detach().numpy().flatten(), pid, pdf_set)
    faser_pdf = torch.tensor(faser_pdf, dtype=torch.float32)
    data_mu = (
        (torch.matmul(fk_tables_mu, faser_pdf) * binwidths_mu)
        .detach()
        .numpy()
        .flatten()
    )

    faser_pdf, x = read_pdf(
        pdf_name, x_alphas.detach().numpy().flatten(), -pid, pdf_set
    )
    faser_pdf = torch.tensor(faser_pdf, dtype=torch.float32)
    data_mub = (
        (torch.matmul(fk_tables_mub, faser_pdf) * binwidths_mub)
        .detach()
        .numpy()
        .flatten()
    )
    data_mu = np.array(data_mu)
    data_mub = np.array(data_mub)
    data_mu = np.where(data_mu < 0, 0, data_mu)
    data_mu = np.where(data_mu < 0, 0, data_mu)
    data_mu = np.where(data_mu == 0, 0.1, data_mu)
    data_mub = np.where(data_mub == 0, 0.1, data_mub)

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
    )


def aggregate_entries_with_indices(
    fk_tables,
    data,
    binwidths,
    low_bin,
    high_bin,
    threshold,
):
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

    if current_sum >= 0:
        lemgth = len(data[start_idx:])
        print(lemgth)
        rebin_data[-1] += sum(data[start_idx:])

        sum_binwidth = 0.0

        sum_binwidth = np.sum(data[previous_idx:]) / np.sum(raw_data[previous_idx:])

        rebin_binwidhts[-1] = sum_binwidth

        rebin_fk_table_mu[-1] += torch.sum(fk_tables[start_idx:, :], axis=0).unsqueeze(
            0
        )
        rebin_low_bin[-1] = low_bin[previous_idx]
        rebin_high_bin[-1] = high_bin[-1]

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


def write_data(
    filename_fk_mub_n,
    filename_fk_mub_p,
    filename_fk_mu_n,
    filename_fk_mu_p,
    filename_binsize,
    pid,
    pdf_name,
    pdf_set,
    filename_to_store_events,
    filename_to_store_stat_error,
    filename_to_store_sys_error,
    filename_to_store_cov_matrix,
    min_num_events,
    observable,
    combine_nu_nub_data,
    division_factor_sys_error,
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
    ) = compute_pseudo_data(
        filename_fk_mub_n,
        filename_fk_mub_p,
        filename_fk_mu_n,
        filename_fk_mu_p,
        filename_binsize,
        pid,
        pdf_name,
        pdf_set,
    )

    if combine_nu_nub_data:
        data = data_mu + data_mub
        binwidths = binwidths_mu
        fk_tables = fk_tables_mu + fk_tables_mub
        (
            data,
            fk_tables,
            binwidths,
            low_bin,
            high_bin,
        ) = aggregate_entries_with_indices(
            fk_tables, data, binwidths, low_bin, high_bin, min_num_events
        )
        stack_binning = np.column_stack((low_bin, high_bin, binwidths))

        error_stat = np.sqrt(data)
        error_sys = np.array(data) / division_factor_sys_error

        np.savetxt(
            f"../../../Data/uncertainties/{filename_to_store_stat_error}_comb_min_{min_num_events}_events",
            error_stat,
        )

        np.savetxt(
            f"../../../Data/data/{filename_to_store_events}_comb_min_{min_num_events}_events",
            data,
        )
        cov_matrix = np.diag(error_sys**2 + error_stat**2)
        cov_matrix = np.linalg.inv(cov_matrix)
        np.savetxt(
            f"../../../Data/uncertainties/{filename_to_store_cov_matrix}_comb_min_{min_num_events}_events",
            cov_matrix,
        )
        np.savetxt(
            f"../../../Data/uncertainties/{filename_to_store_sys_error}_comb_min_{min_num_events}_events",
            error_sys,
        )
        np.savetxt(
            f"../../../Data/binning/FK_{observable}_binsize_nu_min_{min_num_events}_events",
            stack_binning,
        )
        np.savetxt(
            f"../../../Data/binning/FK_{observable}_binsize_nub_min_{min_num_events}_events",
            stack_binning,
        )
        np.savetxt(
            f"../../../Data/fastkernel/FK_{observable}_comb_min_{min_num_events}_events",
            fk_tables,
        )

    else:
        (
            data_mu,
            fk_tables_mu,
            binwidths_mu,
            low_bin_mu,
            high_bin_mu,
        ) = aggregate_entries_with_indices(
            fk_tables_mu, data_mu, binwidths_mu, low_bin, high_bin, min_num_events
        )

        (
            data_mub,
            fk_tables_mub,
            binwidths_mub,
            low_bin_mub,
            high_bin_mub,
        ) = aggregate_entries_with_indices(
            fk_tables_mub, data_mub, binwidths_mub, low_bin, high_bin, min_num_events
        )
        error_stat_nu = np.sqrt(data_mu)
        error_stat_nub = np.sqrt(data_mub)
        error_sys_nu = np.array(data_mu) / division_factor_sys_error
        error_sys_nub = np.array(data_mub) / division_factor_sys_error
        stacked_data = np.hstack((data_mu, data_mub))
        error_tot_nu = error_stat_nu**2 + error_sys_nu**2
        error_tot_nub = error_stat_nub**2 + error_sys_nub**2
        stacked_error = np.hstack((error_tot_nu, error_tot_nub))
        error_stat_tot = np.hstack((error_stat_nu, error_stat_nub))
        error_sys_tot = np.hstack((error_sys_nu, error_sys_nub))
        np.savetxt(
            f"../../../Data/data/{filename_to_store_events}_nu_min_{min_num_events}_events",
            data_mu,
        )
        np.savetxt(
            f"../../../Data/data/{filename_to_store_events}_nub_min_{min_num_events}_events",
            data_mub,
        )
        np.savetxt(
            f"../../../Data/data/{filename_to_store_events}_comb_min_{min_num_events}_events",
            stacked_data,
        )
        np.savetxt(
            f"../../../Data/uncertainties/{filename_to_store_stat_error}_comb_min_{min_num_events}_events",
            error_stat_tot,
        )
        np.savetxt(
            f"../../../Data/uncertainties/{filename_to_store_sys_error}_comb_min_{min_num_events}_events",
            error_sys_tot,
        )

        cov_matrix = np.diag(stacked_error)
        cov_matrix = np.linalg.inv(cov_matrix)
        np.savetxt(
            f"../../../Data/uncertainties/{filename_to_store_cov_matrix}_comb_min_{min_num_events}_events",
            cov_matrix,
        )

        np.savetxt(
            f"../../../Data/fastkernel/FK_{observable}_nu_mu_min_{min_num_events}_events",
            fk_tables_mu,
        )
        np.savetxt(
            f"../../../Data/fastkernel/FK_{observable}_nu_mub_min_{min_num_events}_events",
            fk_tables_mub,
        )

        stack_binning_mu = np.column_stack((low_bin_mu, high_bin_mu, binwidths_mu))
        np.savetxt(
            f"../../../Data/binning/FK_{observable}_binsize_mub_min_{min_num_events}_events",
            stack_binning_mu,
        )

        stack_binning_mub = np.column_stack((low_bin_mub, high_bin_mub, binwidths_mub))
        np.savetxt(
            f"../../../Data/binning/FK_{observable}_binsize_mub_min_{min_num_events}_events",
            stack_binning_mub,
        )

    print("The data has been written to the Data directory")


pdf_name = config["data"]["pdf"]
min_num_events = config["data"]["min_num_events"]
observable = config["data"]["observable"]
combine_nu_nub_data = config["data"]["combine_nu_nub_data"]
pid = config["data"]["particle_id"]
pdf_set = config["data"]["pdf_set"]
filename_fk_table = config["data"]["filename_fk_table"]
filename_binwidth = config["data"]["filename_binwidth"]
filename_to_store_events = config["data"]["filename_to_store_events"]
filename_to_store_sys_error = config["data"]["filename_to_store_sys_error"]
filename_to_store_stat_error = config["data"]["filename_to_store_stat_error"]
filename_to_store_cov_matrix = config["data"]["filename_to_store_cov_matrix"]
division_factor_sys_error = config["data"]["division_factor_sys_error"]


filename_fk_mub_n = f"../../../Data/fastkernel/{filename_fk_table}_nubmu_n.dat"
filename_fk_mub_p = f"../../../Data/fastkernel/{filename_fk_table}_nubmu_p.dat"
filename_fk_mu_n = f"../../../Data/fastkernel/{filename_fk_table}_numu_n.dat"
filename_fk_mu_p = f"../../../Data/fastkernel/{filename_fk_table}_numu_p.dat"
filename_binsize = f"../../../Data/binning/{filename_binwidth}.dat"

write_data(
    filename_fk_mub_n,
    filename_fk_mub_p,
    filename_fk_mu_n,
    filename_fk_mu_p,
    filename_binsize,
    pid,
    pdf_name,
    pdf_set,
    filename_to_store_events,
    filename_to_store_stat_error,
    filename_to_store_sys_error,
    filename_to_store_cov_matrix,
    min_num_events,
    observable,
    combine_nu_nub_data,
    division_factor_sys_error,
)
