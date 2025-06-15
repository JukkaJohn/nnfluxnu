import numpy as np
import os
import sys
import torch

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Data for plot

from data_qgsjet import data_needed_for_fit
from control_file_qgsjet import hyperparams
from postfit_measures import Measures
from postfit_criteria import Postfit
from plot_qgsjet import plot

# Get number of reps from make runscripts
filename_postfit = "fit_report_dpm_Eh.txt"
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

obs = "El"
pdf_name = "FASERv_Run3_QGSJET+POWHEG_7TeV"
filename_data_mu = (
    "../../../FKtables/data/data/data_Eh_FASERv_Run3_DPMJET+DPMJET_7TeV_numu_W.dat"
)
filename_data_mub = (
    "../../../FKtables/data/data/data_Eh_FASERv_Run3_DPMJET+DPMJET_7TeV_nubmu_W.dat"
)
filename_uncert_mu = "../../../FKtables/data/uncertainties/uncertainties_Eh_FASERv_Run3_DPMJET+DPMJET_7TeV_numu_W.dat"
filename_uncert_mub = "../../../FKtables/data/uncertainties/uncertainties_Eh_FASERv_Run3_DPMJET+DPMJET_7TeV_nubmu_W.dat"
filename_fk_mub_n = f"../../../FKtables/data/fastkernel/FK_{obs}_final_nubmu_n.dat"
filename_fk_mub_p = f"../../../FKtables/data/fastkernel/FK_{obs}_final_nubmu_p.dat"
filename_fk_mu_n = f"../../../FKtables/data/fastkernel/FK_{obs}_final_numu_n.dat"
filename_fk_mu_p = f"../../../FKtables/data/fastkernel/FK_{obs}_final_numu_p.dat"
filename_binsize = f"../../../FKtables/data/binning/FK_{obs}_binsize.dat"

(
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
) = data_needed_for_fit(
    0,
    42,
    filename_data_mu=filename_data_mu,
    filename_data_mub=filename_data_mub,
    filename_uncert_mu=filename_uncert_mu,
    filename_uncert_mub=filename_uncert_mub,
    filename_fk_mub_n=filename_fk_mub_n,
    filename_fk_mub_p=filename_fk_mub_p,
    filename_fk_mu_n=filename_fk_mu_n,
    filename_fk_mu_p=filename_fk_mu_p,
    filename_binsize=filename_binsize,
    pdf=pdf_name,
)

level1 = level1[0]
chi_squares = np.loadtxt("chi_square.txt", delimiter=",")
chi_squares_postfit = np.loadtxt("chi_squares_for_postfit.txt", delimiter=",")
N_event_pred = np.loadtxt("events.txt", delimiter=",")

neutrino_pdfs = np.loadtxt("pdf.txt", delimiter=",")
train_indices = np.loadtxt("train_indices.txt", delimiter=",")
val_indices = np.loadtxt("val_indices.txt", delimiter=",")
# training_lengths = np.loadtxt("training_lengths.txt", delimiter=",")

pred = np.loadtxt("pred.txt", delimiter=",")
num_reps = np.shape(N_event_pred)[0]
if validation != 0.0:
    train_indices = train_indices[0]
    val_indices = val_indices[0]
    train_indices = train_indices.astype(int)
    val_indices = val_indices.astype(int)

    print(train_indices)
    print(val_indices)
    N_event_pred_train = N_event_pred[:, train_indices]
    pred_train = pred[:, train_indices]

    N_event_pred_val = N_event_pred[:, val_indices]
    data_val = data[val_indices]
    pred_val = pred[:, val_indices]

    level1_val = level1[val_indices]

    val_indices = torch.tensor(val_indices)

    print(cov_matrix.shape)
    cov_matrix_val = cov_matrix[val_indices][:, val_indices]


def compute_postfit_criteria(neutrino_pdfs_mu, N_event_pred, pred):
    # if postfit_criteria:
    closure_fit = Postfit()
    neutrino_pdfs_mu, N_event_pred, pred = closure_fit.apply_postfit_criteria(
        chi_squares_postfit, N_event_pred, neutrino_pdfs_mu, pred
    )


if postfit_criteria and validation != 0.0:
    compute_postfit_criteria(neutrino_pdfs, N_event_pred_train, pred_train)
if postfit_criteria and validation == 0:
    compute_postfit_criteria(neutrino_pdfs, N_event_pred, pred)

with open(filename_postfit, "a") as file:
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
        with open(filename_postfit, "a") as file:
            file.write(f"delta chi^2 = {delta_chi}:\n")

        accuracy = compute_postfit.compute_accuracy(
            x_alphas.detach().numpy().flatten(), neutrino_pdfs, pdf, 1
        )
        print(f"accuracy = {accuracy}")
        with open(filename_postfit, "a") as file:
            file.write(f"accuracy = {accuracy}:\n")

    if fit_level != 3:
        phi = compute_postfit.compute_phi(data, chi_squares_postfit)
        print(f"phi = {phi}")
        with open(filename_postfit, "a") as file:
            file.write(f"phi = {phi}:\n")

    if fit_level == 2:
        bias_to_var = compute_postfit.compute_bias_to_variance(
            data, pred, N_event_pred, num_reps
        )
        print(f"bias to var = {bias_to_var}")
        with open(filename_postfit, "a") as file:
            file.write(f"bias_to_var = {bias_to_var}:\n")


if postfit_measures and validation != 0.0:
    compute_postfit_measures(
        cov_matrix_val, N_event_pred_val, data_val, level1_val, pred_val
    )
if postfit_measures and validation == 0.0:
    compute_postfit_measures(cov_matrix, N_event_pred, data, level1, pred)

with open(filename_postfit, "a") as file:
    file.write(f"mean chi^2 = {np.mean(chi_squares_postfit)}:\n")
    # file.write(f"average training length = {np.mean(training_lengths)}:\n")
    file.write("settings used:\n")
    file.write(f"learning rate = {lr}:\n")
    file.write(f"weigth decay = {wd}:\n")
    file.write(f"max training lenght = {max_num_epochs}:\n")
    file.write(f"patience = {max_counter}:\n")

sig_tot = np.sqrt(sig_sys**2 + sig_stat**2)

plot(
    x_vals, neutrino_pdfs, data, N_event_pred, sig_tot, 12, low_bin, high_bin, pdf_name
)
