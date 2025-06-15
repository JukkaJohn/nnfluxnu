import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.nn.functional
import sys
import os

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from read_faserv_pdf import read_pdf
from MC_data_reps import generate_MC_replicas
from data_for_faser_fit import data_needed_for_fit
from form_loss_fct import complete_loss_fct, raw_loss_fct
from logspace_grid import generate_grid

# from plot_results_faser_data import plot
from control_file_fits import hyperparams
from read_fk_table import get_fk_table
from faser_fit import perform_fit


# Training_split =0.8
seed = 1
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


filename = "data_muon_sim_faser/FK_Enu_7TeV_nu_W.dat"
x_alphas, fk_tables_mu = get_fk_table(
    filename=filename, parent_dir=parent_dir, x_alpha_grid=True
)
filename = "data_muon_sim_faser/FK_Enu_7TeV_nub_W.dat"
x_alphas, fk_tables_mub = get_fk_table(
    filename=filename, parent_dir=parent_dir, x_alpha_grid=True
)

level_1_instance = 63
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
    simulated_data,
    err_sim,
) = data_needed_for_fit(fit_level, seed=level_1_instance)

seed = 1

(
    chi_squares,
    N_event_pred,
    neutrino_pdfs_mu,
    neutrino_pdfs_mub,
    model,
    chi_square_for_postfit,
    train_indices,
    val_indices,
    training_lengths,
) = perform_fit(
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
    max_num_epochs,
)


chi_squares = np.array(chi_squares)
N_event_pred = np.array(N_event_pred)
neutrino_pdfs_mu = np.array(neutrino_pdfs_mu)
neutrino_pdfs_mub = np.array(neutrino_pdfs_mub)
chi_square_for_postfit = np.array(chi_square_for_postfit)
training_lengths = np.array([training_lengths])
pred = np.array(pred)

# train_indices = train_indices.flatten()
# val_indices = val_indices.flatten()
train_indices = train_indices.reshape(1, -1)
val_indices = val_indices.reshape(1, -1)


print(train_indices.shape)
print(val_indices.shape)
with open("../chi_square.txt", "a") as f:
    np.savetxt(f, chi_squares, delimiter=",")

with open("../chi_squares_for_postfit.txt", "a") as f:
    np.savetxt(f, chi_square_for_postfit, delimiter=",")

with open("../events.txt", "a") as f:
    np.savetxt(f, N_event_pred, delimiter=",")

with open("../mu_pdf.txt", "a") as f:
    np.savetxt(f, neutrino_pdfs_mu, delimiter=",")

with open("../mub_pdf.txt", "a") as f:
    np.savetxt(f, neutrino_pdfs_mub, delimiter=",")

if chi_square_for_postfit.size != 0:
    with open("../pred.txt", "a") as f:
        np.savetxt(f, pred, delimiter=",")

    with open("../train_indices.txt", "a") as f:
        np.savetxt(f, train_indices, delimiter=",")
    with open("../val_indices.txt", "a") as f:
        np.savetxt(f, val_indices, delimiter=",")
    with open("../training_lengths.txt", "a") as f:
        np.savetxt(f, training_lengths, delimiter=",")
