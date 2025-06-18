import yaml
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from read_fk_table import get_fk_table
from MC_data_reps import generate_MC_replicas
from postfit_criteria import Postfit
from postfit_measures import Measures
from write_pdf_to_lhapdf import write_lhapdf_grid, customize_info_file

if len(sys.argv) < 2:
    print("Usage: python script.py <input_file>")
    sys.exit(1)

config_path = sys.argv[1]


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

observable = config["dataset"]["observable"]
filename_data = config["dataset"]["filename_data"]
grid_node = config["dataset"]["grid_node"]
filename_stat_error = config["dataset"]["filename_stat_error"]
filename_sys_error = config["dataset"]["filename_sys_error"]
filename_cov_matrix = config["dataset"]["filename_cov_matrix"]
filename_binning = config["dataset"]["filename_binning"]

postfit_measures = config["postfit"]["postfit_measures"]
postfit_criteria = config["postfit"]["postfit_criteria"]

filename_postfit = "postfit_measures.txt"
pdf = config["dataset"]["pdf"]
pdf_set = config["dataset"]["pdf_set"]

dir_for_data = config["postfit"]["dir_for_data"]

# Simplify perhaps because only fk table and binwidths is combined

data = np.loadtxt(f"../../../Data/data/{filename_data}", delimiter=",")
stat_error = np.loadtxt(
    f"../../../Data/uncertainties/{filename_stat_error}", delimiter=","
)
sys_error = np.loadtxt(
    f"../../../Data/uncertainties/{filename_sys_error}", delimiter=","
)
cov_matrix = np.loadtxt(
    f"../../../Data/uncertainties/{filename_cov_matrix}", delimiter=","
)
cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32, requires_grad=False)
if num_output_layers == 1:
    from perform_fit_comb import perform_fit

    low_bin, high_bin, binwidths_mu = np.loadtxt(
        f"../../../Data/binning/{filename_binning}", unpack=True
    )
    fk_tables = np.loadtxt(
        f"../../../Data/fastkernel/FK_{observable}_comb_min_20_events",
    )
    binwidths_mu = torch.tensor(binwidths_mu, dtype=torch.float32).view(-1, 1)
    fk_tables = torch.tensor(fk_tables, dtype=torch.float32)


elif num_output_layers == 2:
    from perform_fit_nu_nub import perform_fit

    low_bin_mu, high_bin_mu, binwidths_mu = np.loadtxt(
        f"../../../Data/uncertainties/FK_{observable}_binsize_nub", unpack=True
    )
    low_bin, high_bin, binwidths_mub = np.loadtxt(
        f"../../../Data/uncertainties/FK_{observable}_binsize_nub", unpack=True
    )
    fk_tables_nu = get_fk_table(
        filename=f"FK_{observable}_nu.dat", parent_dir=parent_dir
    )
    fk_tables_nub = get_fk_table(
        filename=f"FK_{observable}_nub.dat", parent_dir=parent_dir
    )
    binwidths_mu = torch.tensor(binwidths_mu, dtype=torch.float32).view(-1, 1)
    binwidths_mub = torch.tensor(binwidths_mub, dtype=torch.float32).view(-1, 1)
    cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32, requires_grad=False)
else:
    print("please choose a number of layers between 1 and 2")
    exit()

x_alphas = np.loadtxt("../../../Data/gridnodes/x_alpha.dat", unpack=True)
x_alphas = torch.tensor(x_alphas, dtype=torch.float32).view(-1, 1)
x_vals = np.logspace(-5, 0, 1000)

for i in range(diff_l1_inst):
    seed = i + 1

    level0, level1, level2 = generate_MC_replicas(
        num_reps, data, stat_error, sys_error, seed
    )

    if fit_level == 0:
        pred = level0
    if fit_level == 1:
        pred = level1
    if fit_level == 2:
        pred = level2

    if fit_level == 1:
        (
            chi_squares,
            N_event_pred,
            neutrino_pdfs,
            model,
            chi_square_for_postfit,
            train_indices,
            val_indices,
            training_lengths,
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
        )
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
        )
        N_event_pred = np.hstack((N_event_pred_nu, N_event_pred_nub))

    os.makedirs(dir_for_data, exist_ok=True)

    print(chi_squares)
    chi_squares = np.array(chi_squares)
    N_event_pred = np.array(N_event_pred)
    neutrino_pdfs = np.array(neutrino_pdfs)
    chi_square_for_postfit = np.array(chi_square_for_postfit)
    pred = np.array(pred)
    training_lengths = np.array([training_lengths])

    if postfit_criteria:
        train_indices = train_indices.reshape(1, -1)
        val_indices = val_indices.reshape(1, -1)

        level1 = level1[0]
        num_reps = np.shape(N_event_pred)[1]

        if validation_split != 0.0:
            train_indices = train_indices[0]
            val_indices = val_indices[0]
            train_indices = train_indices.astype(int)
            val_indices = val_indices.astype(int)

            N_event_pred_train = N_event_pred[:, train_indices]
            pred_train = pred[:, train_indices]

            N_event_pred_val = N_event_pred[:, val_indices]
            data_val = data[val_indices]
            pred_val = pred[:, val_indices]

            level1_val = level1[val_indices]

            val_indices = torch.tensor(val_indices)

            cov_matrix_val = cov_matrix[val_indices][:, val_indices]

        if num_output_layers == 1:

            def compute_postfit_criteria(neutrino_pdfs, N_event_pred, pred):
                # if postfit_criteria:
                closure_fit = Postfit()
                neutrino_pdfs, N_event_pred, pred = closure_fit.apply_postfit_criteria(
                    chi_square_for_postfit, N_event_pred, neutrino_pdfs, pred
                )
                return (neutrino_pdfs, N_event_pred, pred)

            if postfit_criteria and validation_split != 0.0:
                neutrino_pdfs, N_event_pred_train, pred_train = (
                    compute_postfit_criteria(
                        neutrino_pdfs, N_event_pred_train, pred_train
                    )
                )
            if postfit_criteria and validation_split == 0:
                neutrino_pdfs, N_event_pred, pred = compute_postfit_criteria(
                    neutrino_pdfs, N_event_pred, pred
                )
        if num_output_layers == 2:

            def compute_postfit_criteria(
                neutrino_pdfs_mu, neutrino_pdfs_mub, N_event_pred, pred
            ):
                # if postfit_criteria:
                closure_fit = Postfit()
                neutrino_pdfs_mu, _, _ = closure_fit.apply_postfit_criteria(
                    chi_square_for_postfit, N_event_pred, neutrino_pdfs_mu, pred
                )
                neutrino_pdfs_mub, N_event_pred, pred = (
                    closure_fit.apply_postfit_criteria(
                        chi_square_for_postfit, N_event_pred, neutrino_pdfs_mub, pred
                    )
                )

            if postfit_criteria and validation_split != 0.0:
                compute_postfit_criteria(
                    neutrino_pdfs_mu, neutrino_pdfs_mub, N_event_pred_train, pred_train
                )
            if postfit_criteria and validation_split == 0:
                compute_postfit_criteria(
                    neutrino_pdfs_mu, neutrino_pdfs_mub, N_event_pred, pred
                )

    if postfit_measures:
        with open(f"{dir_for_data}/{filename_postfit}", "a") as file:
            file.write("postfit report faser sim fit:\n")
            file.write("100 replicas:\n")

        def compute_postfit_measures(cov_matrix, N_event_pred, data, level1, pred):
            compute_postfit = Measures(cov_matrix, pdf, N_event_pred)
            if fit_level != 0:
                delta_chi = compute_postfit.compute_delta_chi(
                    data,
                    N_event_pred,
                    level1,
                    x_alphas.detach().numpy().squeeze(),
                )
                print(f"mean delta chi = {delta_chi}")
                with open(f"{dir_for_data}/{filename_postfit}", "a") as file:
                    file.write(f"delta chi^2 = {delta_chi}:\n")

                accuracy = compute_postfit.compute_accuracy(
                    x_alphas.detach().numpy().flatten(), neutrino_pdfs, pdf, 1, pdf_set
                )
                print(f"accuracy = {accuracy}")
                with open(f"{dir_for_data}/{filename_postfit}", "a") as file:
                    file.write(f"accuracy = {accuracy}:\n")

            # if fit_level != 3:
            phi = compute_postfit.compute_phi(data, chi_square_for_postfit)
            print(f"phi = {phi}")
            with open(f"{dir_for_data}/{filename_postfit}", "a") as file:
                file.write(f"phi = {phi}:\n")

            if fit_level == 2:
                bias_to_var = compute_postfit.compute_bias_to_variance(
                    data, pred, N_event_pred, len(N_event_pred)
                )
                print(f"bias to var = {bias_to_var}")
                with open(f"{dir_for_data}/{filename_postfit}", "a") as file:
                    file.write(f"bias_to_var = {bias_to_var}:\n")

        if postfit_measures and validation_split != 0.0:
            compute_postfit_measures(
                cov_matrix_val, N_event_pred_val, data_val, level1_val, pred_val
            )
        if postfit_measures and validation_split == 0.0:
            compute_postfit_measures(cov_matrix, N_event_pred, data, level1, pred)

        with open(f"{dir_for_data}/{filename_postfit}", "a") as file:
            file.write(f"mean chi^2 = {np.mean(chi_square_for_postfit)}:\n")
            # file.write(f"average training length = {np.mean(training_lengths)}:\n")
            file.write("settings used:\n")
            file.write(f"learning rate = {lr}:\n")
            file.write(f"weigth decay = {wd}:\n")
            file.write(f"max training lenght = {max_epochs}:\n")
            file.write(f"patience = {patience}:\n")

    with open(f"{dir_for_data}/chi_square.txt", "a") as f:
        np.savetxt(f, chi_squares, delimiter=",")

    with open(f"{dir_for_data}/chi_squares_for_postfit.txt", "a") as f:
        np.savetxt(f, chi_square_for_postfit, delimiter=",")

    with open(f"{dir_for_data}/events.txt", "a") as f:
        np.savetxt(f, N_event_pred, delimiter=",")

    with open(f"{dir_for_data}/pdf.txt", "a") as f:
        np.savetxt(f, neutrino_pdfs, delimiter=",")

    # write to lhapdf grid
    template_path = "/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/template_.info"
    path = "/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/testgrid/testgrid.info"
    set_index = int(np.random.rand() * 1e7)
    customize_info_file(template_path, path, set_index, 12)
    mean_pdf = np.mean(neutrino_pdfs, axis=0)
    std_pdf = np.std(neutrino_pdfs, axis=0)
    path = "/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/testgrid/testgrid_0000.dat"
    write_lhapdf_grid(x_vals, mean_pdf, path, 12)
    path = "/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/testgrid/testgrid_0001.dat"
    write_lhapdf_grid(x_vals, std_pdf, path, 12)

    if chi_square_for_postfit.size != 0:
        with open(f"{dir_for_data}/pred.txt", "a") as f:
            np.savetxt(f, pred, delimiter=",")

        with open(f"{dir_for_data}/train_indices.txt", "a") as f:
            np.savetxt(f, train_indices, delimiter=",")
        with open(f"{dir_for_data}/val_indices.txt", "a") as f:
            np.savetxt(f, val_indices, delimiter=",")

        with open(f"{dir_for_data}/training_lengths.txt", "a") as f:
            np.savetxt(f, training_lengths, delimiter=",")
