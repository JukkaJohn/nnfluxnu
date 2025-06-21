# Author: Jukka John
# This file generates pseudo data using an input flux and an fk-table
import numpy as np
import torch
import torch.nn.functional
import sys
import os
import lhapdf
import yaml
from typing import Tuple, List

lhapdf.setVerbosity(0)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from read_faserv_pdf import read_pdf
from read_faserv_pdf import read_pdf
from MC_data_reps import generate_MC_replicas

from read_fk_table import get_fk_table


if len(sys.argv) < 2:
    print("Usage: python script.py <input_file>")
    sys.exit(1)

config_path = sys.argv[1]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


config = load_config(config_path)


def compute_pseudo_data(
    filename_fk_mub_n: str,
    filename_fk_mub_p: str,
    filename_fk_mu_n: str,
    filename_fk_mu_p: str,
    filename_binsize: str,
    pid: int,
    pdf_name: str,
    pdf_set: int,
) -> Tuple[
    np.ndarray,  # data_mu
    np.ndarray,  # data_mub
    np.ndarray,  # error_mu
    np.ndarray,  # error_mub
    torch.Tensor,  # fk_tables_mu
    torch.Tensor,  # fk_tables_mub
    np.ndarray,  # low_bin
    np.ndarray,  # high_bin
    np.ndarray,  # binwidths_mu
    np.ndarray,  # binwidths_mub
]:
    """
    Computes pseudo-data for neutrino and anti-neutrino scattering using FK tables and PDFs.

    This function processes FK tables (FastKernel weight tables) and convolves them with
    parton distribution functions (PDFs) to produce synthetic ("pseudo") data for both
    neutrinos (mu) and anti-neutrinos (mub). It also calculates associated statistical errors.

    Parameters
    ----------
    filename_fk_mub_n : str
        Filename for the anti-neutrino FK table (neutron target).
    filename_fk_mub_p : str
        Filename for the anti-neutrino FK table (proton target).
    filename_fk_mu_n : str
        Filename for the neutrino FK table (neutron target).
    filename_fk_mu_p : str
        Filename for the neutrino FK table (proton target).
    filename_binsize : str
        Filename containing binning information (low, high bin edges, and widths).
    pid : int
        PDG ID of the relevant parton (e.g., 12, 14, 16 for neutrino flavors).
    pdf_name : str
        Name of the PDF set (e.g., "CT18", "NNPDF4.0").
    pdf_set : int
        Index of the replica or member within the PDF set.

    Returns
    -------
    data_mu : np.ndarray
        Pseudo-data for neutrino cross sections, binned.
    data_mub : np.ndarray
        Pseudo-data for anti-neutrino cross sections, binned.
    error_mu : np.ndarray
        Statistical uncertainty (sqrt(N)) for neutrino data.
    error_mub : np.ndarray
        Statistical uncertainty (sqrt(N)) for anti-neutrino data.
    fk_tables_mu : torch.Tensor
        Final combined FK table for neutrinos.
    fk_tables_mub : torch.Tensor
        Final combined FK table for anti-neutrinos.
    low_bin : np.ndarray
        Lower bin edges of the energy bins.
    high_bin : np.ndarray
        Upper bin edges of the energy bins.
    binwidths_mu : np.ndarray
        Widths of bins used for neutrino integration.
    binwidths_mub : np.ndarray
        Widths of bins used for anti-neutrino integration (same as `binwidths_mu`).

    Notes
    -----
    - This function uses hard-coded weights: 59.56% neutron and 40.44% proton contributions.
    - Any negative or zero values in the pseudo-data are replaced with small positive values
      (0.1) to avoid numerical issues.
    - Requires FK tables and bin sizes to be precomputed and available as text files.
    """
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
    fk_tables: torch.Tensor,
    data: np.ndarray,
    binwidths: np.ndarray,
    low_bin: np.ndarray,
    high_bin: np.ndarray,
    threshold: float,
) -> Tuple[
    List[float],  # rebin_data
    torch.Tensor,  # rebin_fk_table_mu
    np.ndarray,  # rebin_binwidhts
    np.ndarray,  # rebin_low_bin
    np.ndarray,  # rebin_high_bin
]:
    """
    Aggregates FK table rows and data bins until a minimum threshold of events is reached.

    This function rebins cross section data, corresponding FK table rows, and bin widths
    such that each new bin has at least a specified number of events (threshold). This
    is often necessary for achieving meaningful statistical analysis in low-statistics bins.

    Parameters
    ----------
    fk_tables : torch.Tensor
        The FastKernel table with shape (n_bins, n_x).
    data : np.ndarray
        Event data per bin (e.g., cross section  bin width).
    binwidths : np.ndarray
        Width of each original bin.
    low_bin : np.ndarray
        Lower edges of the original bins.
    high_bin : np.ndarray
        Upper edges of the original bins.
    threshold : float
        Minimum number of events required to form a new rebinned bin.

    Returns
    -------
    rebin_data : list of float
        Aggregated event counts after rebinning.
    rebin_fk_table_mu : torch.Tensor
        Rebinned FK table rows (shape: new_n_bins  n_x).
    rebin_binwidhts : np.ndarray
        Bin widths corresponding to rebinned bins.
    rebin_low_bin : np.ndarray
        Lower edges of the rebinned bins.
    rebin_high_bin : np.ndarray
        Upper edges of the rebinned bins.

    Notes
    -----
    - Remaining data after the last full threshold bin is added to the final bin.
    - Bin widths are recomputed using weighted averages to ensure consistency.
    """
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
    filename_fk_mub_n: str,
    filename_fk_mub_p: str,
    filename_fk_mu_n: str,
    filename_fk_mu_p: str,
    filename_binsize: str,
    pid: int,
    pdf_name: str,
    pdf_set: int,
    filename_to_store_events: str,
    filename_to_store_stat_error: str,
    filename_to_store_sys_error: str,
    filename_to_store_cov_matrix: str,
    min_num_events: int,
    observable: str,
    combine_nu_nub_data: bool,
    division_factor_sys_error: float,
) -> None:
    """
    Computes pseudo-data from FK tables and PDFs, optionally rebins them, and writes the results to disk.

    This function is the main pipeline to produce and store pseudo-experimental data, its
    statistical and systematic uncertainties, covariance matrix, binning information, and
    FastKernel tables, all ready for use in PDF fits or phenomenology studies.

    Parameters
    ----------
    filename_fk_mub_n : str
        FK table for anti-neutrino interactions on neutrons.
    filename_fk_mub_p : str
        FK table for anti-neutrino interactions on protons.
    filename_fk_mu_n : str
        FK table for neutrino interactions on neutrons.
    filename_fk_mu_p : str
        FK table for neutrino interactions on protons.
    filename_binsize : str
        Filename for bin edges and widths (low, high, width).
    pid : int
        PDG ID of the target parton species.
    pdf_name : str
        Name of the LHAPDF set to use.
    pdf_set : int
        Index of the PDF replica or member.
    filename_to_store_events : str
        Base filename for storing rebinned event data.
    filename_to_store_stat_error : str
        Base filename for storing statistical uncertainties.
    filename_to_store_sys_error : str
        Base filename for storing systematic uncertainties.
    filename_to_store_cov_matrix : str
        Base filename for storing the inverse of the covariance matrix.
    min_num_events : int
        Minimum number of events per bin in the rebinned dataset.
    observable : str
        Observable label (e.g., "energy", "pt") used in output filenames.
    combine_nu_nub_data : bool
        If True, neutrino and anti-neutrino data are summed into one dataset.
    division_factor_sys_error : float
        Factor to divide event counts for estimating systematic uncertainties.

    Returns
    -------
    None
        Writes output files directly to disk.

    Output Files
    ------------
    ../../../Data/data/:
        - Re-binned event counts (combined, mu, mub)

    ../../../Data/uncertainties/:
        - Statistical and systematic uncertainties
        - Covariance matrix (inverted, diagonal only)

    ../../../Data/binning/:
        - Re-binned bin edges and widths (mu, mub, or combined)

    ../../../Data/fastkernel/:
        - Re-binned FK tables

    Notes
    -----
    - Assumes FK and bin files are already precomputed and exist in the expected format.
    - The covariance matrix is stored in inverted form (1/σ² on the diagonal).
    - Output filenames are automatically labeled with PID and threshold settings.
    - Handles both the case where ν and ν̄ data are stored separately or combined.
    """
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
            f"../../../Data/uncertainties/{filename_to_store_stat_error}_comb_min_{min_num_events}_events_{pid}",
            error_stat,
        )

        np.savetxt(
            f"../../../Data/data/{filename_to_store_events}_comb_min_{min_num_events}_events_{pid}",
            data,
        )
        cov_matrix = np.diag(error_sys**2 + error_stat**2)
        cov_matrix = np.linalg.inv(cov_matrix)
        np.savetxt(
            f"../../../Data/uncertainties/{filename_to_store_cov_matrix}_comb_min_{min_num_events}_events_{pid}",
            cov_matrix,
        )
        np.savetxt(
            f"../../../Data/uncertainties/{filename_to_store_sys_error}_comb_min_{min_num_events}_events_{pid}",
            error_sys,
        )
        np.savetxt(
            f"../../../Data/binning/FK_{observable}_binsize_nu_min_{min_num_events}_events_{pid}",
            stack_binning,
        )
        np.savetxt(
            f"../../../Data/binning/FK_{observable}_binsize_nub_min_{min_num_events}_events_{pid}",
            stack_binning,
        )
        np.savetxt(
            f"../../../Data/fastkernel/FK_{observable}_comb_min_{min_num_events}_events_{pid}",
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
            f"../../../Data/data/{filename_to_store_events}_nu_min_{min_num_events}_events_{pid}",
            data_mu,
        )
        np.savetxt(
            f"../../../Data/data/{filename_to_store_events}_nub_min_{min_num_events}_events_{-pid}",
            data_mub,
        )
        np.savetxt(
            f"../../../Data/data/{filename_to_store_events}_comb_min_{min_num_events}_events_{pid}",
            stacked_data,
        )
        np.savetxt(
            f"../../../Data/uncertainties/{filename_to_store_stat_error}_comb_min_{min_num_events}_events_{pid}",
            error_stat_tot,
        )
        np.savetxt(
            f"../../../Data/uncertainties/{filename_to_store_sys_error}_comb_min_{min_num_events}_events_{pid}",
            error_sys_tot,
        )

        cov_matrix = np.diag(stacked_error)
        cov_matrix = np.linalg.inv(cov_matrix)
        np.savetxt(
            f"../../../Data/uncertainties/{filename_to_store_cov_matrix}_comb_min_{min_num_events}_events_{pid}",
            cov_matrix,
        )

        np.savetxt(
            f"../../../Data/fastkernel/FK_{observable}_mu_min_{min_num_events}_events_{pid}",
            fk_tables_mu,
        )
        np.savetxt(
            f"../../../Data/fastkernel/FK_{observable}_mub_min_{min_num_events}_events_{-pid}",
            fk_tables_mub,
        )

        stack_binning_mu = np.column_stack((low_bin_mu, high_bin_mu, binwidths_mu))
        np.savetxt(
            f"../../../Data/binning/FK_{observable}_binsize_mu_min_{min_num_events}_events_{pid}",
            stack_binning_mu,
        )

        stack_binning_mub = np.column_stack((low_bin_mub, high_bin_mub, binwidths_mub))
        np.savetxt(
            f"../../../Data/binning/FK_{observable}_binsize_mub_min_{min_num_events}_events_{-pid}",
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
