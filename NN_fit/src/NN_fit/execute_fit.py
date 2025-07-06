# Author: Jukka John
# This files execute a fit by reading the data, calling the perform_fit function, performing postfit measures and criteria and plots the result of the fit

import yaml
import sys
import os
import numpy as np
import torch
from MC_data_reps import generate_MC_replicas
from help_read_files import safe_loadtxt
from execute_postfit import postfit_execution


if len(sys.argv) < 2:
    print("Usage: python script.py <input_file>")
    sys.exit(1)

config_path = f"runcards/{sys.argv[1]}"


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

config = load_config(config_path)

hidden_layers = config["model"]["hidden_layers"]
activation_function = config["model"]["activation_function"]
preproc = config["model"]["preproc"]
extended_loss = config["model"]["extended_loss"]
num_output_layers = config["model"]["num_output_layers"]
num_input_layers = config["model"]["num_input_layers"]

fit_level = config["closure_test"]["fit_level"]
num_reps = config["closure_test"]["num_reps"]
num_l1_inst = config["closure_test"]["diff_l1_inst"]
diff_l1_inst = config["closure_test"]["diff_l1_inst"]

patience = config["training"]["patience"]
max_epochs = config["training"]["max_epochs"]
lr = config["training"]["lr"]
wd = config["training"]["wd"]
range_alpha = config["training"]["range_alpha"]
range_beta = config["training"]["range_beta"]
range_gamma = config["training"]["range_gamma"]
optimizer = config["training"]["optimizer"]
validation_split = config["training"]["validation_split"]
max_chi_sq = config["training"]["max_chi_sq"]
lag_mult_pos = config["training"]["lag_mult_pos"]
lag_mult_int = config["training"]["lag_mult_int"]
x_int = config["training"]["x_int"]

observable = config["dataset"]["observable"]
filename_data = config["dataset"]["filename_data"]
grid_node = config["dataset"]["grid_node"]
filename_stat_error = config["dataset"]["filename_stat_error"]
filename_sys_error = config["dataset"]["filename_sys_error"]
filename_cov_matrix = config["dataset"]["filename_cov_matrix"]
filename_binning = config["dataset"]["filename_binning"]
fit_faser_data = config["dataset"]["fit_faser_data"]

postfit_measures = config["postfit"]["postfit_measures"]
postfit_criteria = config["postfit"]["postfit_criteria"]
lhapdf_path = config["postfit"]["lhapdf_path"]

filename_postfit = "postfit_measures.txt"
pdf = config["dataset"]["pdf"]
pdf_set = config["dataset"]["pdf_set"]
particle_id_nu = config["postfit"]["particle_id_nu"]
particle_id_nub = config["postfit"]["particle_id_nub"]
dir_for_data = config["postfit"]["dir_for_data"]
neutrino_pdf_fit_name_lhapdf = config["postfit"]["neutrino_pdf_fit_name_lhapdf"]

produce_plot = config["postfit"]["produce_plot"]

# Electron neutrino fit
if num_output_layers == 1:
    from perform_fit_comb import perform_fit

    low_bin, high_bin, binwidths_mu = safe_loadtxt(
        f"../../../Data/binning/{filename_binning}_{particle_id_nu}", unpack=True
    )
    fk_tables = safe_loadtxt(
        f"../../../Data/fastkernel/FK_{observable}_comb_min_20_events_{particle_id_nu}",
    )
    binwidths_mu = torch.tensor(binwidths_mu, dtype=torch.float32).view(-1, 1)
    fk_tables = torch.tensor(fk_tables, dtype=torch.float32)

    data = safe_loadtxt(
        f"../../../Data/data/{filename_data}_{particle_id_nu}", delimiter=None
    )
    stat_error = safe_loadtxt(
        f"../../../Data/uncertainties/{filename_stat_error}_{particle_id_nu}",
        delimiter=None,
    )
    sys_error = safe_loadtxt(
        f"../../../Data/uncertainties/{filename_sys_error}_{particle_id_nu}",
        delimiter=None,
    )
    cov_matrix = safe_loadtxt(
        f"../../../Data/uncertainties/{filename_cov_matrix}_{particle_id_nu}",
        delimiter=None,
    )
    cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32, requires_grad=False)

# Muon neutrino fit
elif num_output_layers == 2:
    from perform_fit_nu_nub import perform_fit

    low_bin_mu, high_bin_mu, binwidths_mu = safe_loadtxt(
        f"../../../Data/binning/FK_{observable}_binsize_mu_min_20_events_{particle_id_nu}",
        unpack=True,
    )
    low_bin_mub, high_bin_mub, binwidths_mub = safe_loadtxt(
        f"../../../Data/binning/FK_{observable}_binsize_mub_min_20_events_{particle_id_nub}",
        unpack=True,
    )
    fk_tables_nu = safe_loadtxt(
        f"../../../Data/fastkernel/FK_{observable}_mu_min_20_events_{particle_id_nu}",
    )
    fk_tables_nub = safe_loadtxt(
        f"../../../Data/fastkernel/FK_{observable}_mub_min_20_events_{particle_id_nub}",
    )
    data = safe_loadtxt(
        f"../../../Data/data/{filename_data}_{particle_id_nu}", delimiter=None
    )
    stat_error = safe_loadtxt(
        f"../../../Data/uncertainties/{filename_stat_error}_{particle_id_nu}",
        delimiter=None,
    )
    sys_error = safe_loadtxt(
        f"../../../Data/uncertainties/{filename_sys_error}_{particle_id_nu}",
        delimiter=None,
    )
    cov_matrix = safe_loadtxt(
        f"../../../Data/uncertainties/{filename_cov_matrix}_{particle_id_nu}",
        delimiter=None,
    )
    cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32, requires_grad=False)

    binwidths_mu = torch.tensor(binwidths_mu, dtype=torch.float32).view(-1, 1)
    binwidths_mub = torch.tensor(binwidths_mub, dtype=torch.float32).view(-1, 1)
    fk_tables_nu = torch.tensor(fk_tables_nu, dtype=torch.float32)
    fk_tables_nub = torch.tensor(fk_tables_nub, dtype=torch.float32)
    cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32, requires_grad=False)
else:
    print("please choose a number of layers between 1 and 2")
    exit()

x_alphas = np.loadtxt("../../../Data/gridnodes/x_alpha.dat", unpack=True)
x_alphas = torch.tensor(x_alphas, dtype=torch.float32).view(-1, 1)
x_vals = np.logspace(-5, 0, 1000)


(
    mean_pdf_all_fits_mu,
    mean_pdf_all_fits_mub,
    total_std_mu,
    total_std_mub,
    total_preds_Enu,
    total_std_preds_Enu,
    total_preds_Enu_mub,
    total_std_preds_Enu_mub,
) = 0, 0, 0, 0, 0, 0, 0, 0
for i in range(diff_l1_inst):
    seed = i + 1

    level0, level1, level2 = generate_MC_replicas(
        num_reps, data, sys_error, stat_error, seed, fit_level
    )

    if fit_level == 0:
        pred = level0
    elif fit_level == 1:
        pred = level1
    elif fit_level == 2:
        pred = level2
    else:
        print("please select 0,1 or 2 for fit level")

    if num_output_layers == 1:
        (
            chi_squares,
            N_event_pred,
            neutrino_pdfs,
            model,
            chi_square_for_postfit,
            train_indices,
            val_indices,
            training_lengths,
            val_losses,
        ) = perform_fit(
            pred,
            num_reps,
            range_alpha,
            range_beta,
            range_gamma,
            lr,
            wd,
            patience,
            x_alphas,
            fk_tables,
            binwidths_mu,
            cov_matrix,
            extended_loss,
            activation_function,
            num_input_layers,
            num_output_layers,
            hidden_layers,
            x_vals,
            preproc,
            validation_split,
            max_epochs,
            max_chi_sq,
            lag_mult_pos,
            lag_mult_int,
            x_int,
        )
        N_event_pred = np.array(N_event_pred)
        neutrino_pdfs = np.array(neutrino_pdfs)
    if num_output_layers == 2:
        (
            chi_squares,
            N_event_pred_nu,
            N_event_pred_nub,
            neutrino_pdfs_mu,
            neutrino_pdfs_mub,
            model,
            chi_square_for_postfit,
            train_indices,
            val_indices,
            training_lengths,
            val_losses,
        ) = perform_fit(
            pred,
            num_reps,
            range_alpha,
            range_beta,
            range_gamma,
            lr,
            wd,
            patience,
            x_alphas,
            fk_tables_nu,
            fk_tables_nub,
            binwidths_mu,
            binwidths_mub,
            cov_matrix,
            extended_loss,
            activation_function,
            num_input_layers,
            num_output_layers,
            hidden_layers,
            x_vals,
            preproc,
            validation_split,
            max_epochs,
            max_chi_sq,
            fit_faser_data,
            lag_mult_pos,
            lag_mult_int,
            x_int,
        )
        N_event_pred = np.hstack((N_event_pred_nu, N_event_pred_nub))
        N_event_pred = np.array(N_event_pred)
        neutrino_pdfs_mu = np.array(neutrino_pdfs_mu)
        neutrino_pdfs_mub = np.array(neutrino_pdfs_mub)

    os.makedirs(dir_for_data, exist_ok=True)

    chi_squares = np.array(chi_squares)

    chi_square_for_postfit = np.array(chi_square_for_postfit)
    pred = np.array(pred)
    training_lengths = np.array([training_lengths])

    if num_output_layers == 1:
        neutrino_pdfs_mu = None
        neutrino_pdfs_mub = None
        N_event_pred_nu = None
        N_event_pred_nub = None
        low_bin_mu = None
        low_bin_mub = None
        high_bin_mu = None
        high_bin_mub = None
    if num_output_layers == 2:
        neutrino_pdfs = None
        low_bin = None
        high_bin = None

    postfit_execution(
        postfit_criteria,
        validation_split,
        data,
        cov_matrix,
        num_output_layers,
        chi_square_for_postfit,
        neutrino_pdfs_mu,
        neutrino_pdfs_mub,
        neutrino_pdfs,
        postfit_measures,
        train_indices,
        val_indices,
        level1,
        N_event_pred,
        pred,
        dir_for_data,
        filename_postfit,
        i,
        fit_level,
        x_alphas,
        pdf,
        pdf_set,
        particle_id_nu,
        particle_id_nub,
        lr,
        wd,
        max_epochs,
        patience,
        chi_squares,
        neutrino_pdf_fit_name_lhapdf,
        x_vals,
        produce_plot,
        training_lengths,
        stat_error,
        sys_error,
        low_bin,
        high_bin,
        N_event_pred_nu,
        N_event_pred_nub,
        low_bin_mu,
        high_bin_mu,
        low_bin_mub,
        high_bin_mub,
        val_losses,
        lhapdf_path,
    )

    if diff_l1_inst > 1:
        if num_output_layers == 2:
            mean_pdf_all_fits_mu += np.mean(neutrino_pdfs_mu, axis=0) / diff_l1_inst
            mean_pdf_all_fits_mub += np.mean(neutrino_pdfs_mub, axis=0) / diff_l1_inst

            total_std_mu += np.std(neutrino_pdfs_mu, axis=0) ** 2
            total_std_mub += np.std(neutrino_pdfs_mub, axis=0) ** 2

            total_preds_Enu += np.mean(N_event_pred_nu, axis=0) / diff_l1_inst
            total_std_preds_Enu += np.std(N_event_pred_nu, axis=0) ** 2

            total_preds_Enu_mub += np.mean(N_event_pred_nub, axis=0) / diff_l1_inst
            total_std_preds_Enu_mub += np.std(N_event_pred_nub, axis=0) ** 2
        if num_output_layers == 1:
            mean_pdf_all_fits_mu += np.mean(neutrino_pdfs, axis=0) / diff_l1_inst

            total_std_mu += np.std(neutrino_pdfs, axis=0) ** 2

            total_preds_Enu += np.mean(N_event_pred, axis=0) / diff_l1_inst
            total_std_preds_Enu += np.std(N_event_pred, axis=0) ** 2


if diff_l1_inst > 1:
    total_std_mu = np.sqrt(total_std_mu)
    total_std_mub = np.sqrt(total_std_mub)
    total_std_preds_Enu = np.sqrt(total_std_preds_Enu)
    total_std_preds_Enu_mub = np.sqrt(total_std_preds_Enu_mub)
    if num_output_layers == 2:
        from plot_for_diff_level_1_shifts_nu_nub import plot

        sig_tot = np.sqrt(stat_error**2 + sys_error**2)

        plot(
            x_vals,
            mean_pdf_all_fits_mu,
            mean_pdf_all_fits_mub,
            total_std_mu,
            total_std_mub,
            data,
            total_preds_Enu,
            total_preds_Enu_mub,
            total_std_preds_Enu,
            total_std_preds_Enu_mub,
            sig_tot,
            particle_id_nu,
            low_bin_mu,
            high_bin_mu,
            low_bin_mub,
            high_bin_mub,
            pdf,
            pdf_set,
            dir_for_data,
            lhapdf_path,
        )
    if num_output_layers == 1:
        total_std_mu = np.sqrt(total_std_mu)
        total_std_preds_Enu = np.sqrt(total_std_preds_Enu)
        from plot_diff_level1_comb import plot

        sig_tot = np.sqrt(stat_error**2 + sys_error**2)
        (x_vals,)

        plot(
            x_vals,
            mean_pdf_all_fits_mu,
            total_std_mu,
            data,
            total_preds_Enu,
            total_std_preds_Enu,
            sig_tot,
            particle_id_nu,
            low_bin,
            high_bin,
            pdf,
            pdf_set,
            dir_for_data,
        )
