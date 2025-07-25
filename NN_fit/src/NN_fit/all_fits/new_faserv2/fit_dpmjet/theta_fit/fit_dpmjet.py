import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from data_dpmjet import data_needed_for_fit

from control_file_dpmjet import hyperparams
from fit_file import perform_fit

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
from read_faserv_pdf import read_pdf

# from read_LHEF import read_LHEF_data
from data_errors import compute_errors
from MC_data_reps import generate_MC_replicas
from postfit_criteria import Postfit
from form_loss_fct import complete_loss_fct, raw_loss_fct
from postfit_measures import Measures
from logspace_grid import generate_grid
from read_fk_table import get_fk_table
# from rebin_fk_data import rebin_fk


# from read_fk_table import get_fk_table

import pandas as pd

# import torch
#
obs = "theta"
pdf_name = "FASERv2_DPMJET+DPMJET_7TeV"
filename_data_mu = (
    "../../../FKtables/data/data/data_Eh_FASERv_Run3_DPMJET+DPMJET_7TeV_numu_W.dat"
)
filename_data_mub = (
    "../../../FKtables/data/data/data_Eh_FASERv_Run3_DPMJET+DPMJET_7TeV_nubmu_W.dat"
)
filename_uncert_mu = "../../../FKtables/data/uncertainties/uncertainties_Eh_FASERv_Run3_DPMJET+DPMJET_7TeV_numu_W.dat"
filename_uncert_mub = "../../../FKtables/data/uncertainties/uncertainties_Eh_FASERv_Run3_DPMJET+DPMJET_7TeV_nubmu_W.dat"
filename_fk_mub_n = f"../../../FKtables/data/fastkernel/FK_{obs}_fine_nubmu_n.dat"
filename_fk_mub_p = f"../../../FKtables/data/fastkernel/FK_{obs}_fine_nubmu_p.dat"
filename_fk_mu_n = f"../../../FKtables/data/fastkernel/FK_{obs}_fine_numu_n.dat"
filename_fk_mu_p = f"../../../FKtables/data/fastkernel/FK_{obs}_fine_numu_p.dat"
filename_binsize = f"../../../FKtables/data/binning/FK_{obs}_fine_binsize.dat"

level_1_instance = 42

for _ in range(50):
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
        fit_level,
        seed=level_1_instance,
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

    seed = 10
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
        REPLICAS,
        range_alpha,
        range_beta,
        range_gamma,
        lr,
        wd,
        max_counter,
        x_alphas,
        fk_tables,
        binwidths,
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
    neutrino_pdfs = np.array(neutrino_pdfs)
    chi_square_for_postfit = np.array(chi_square_for_postfit)
    pred = np.array(pred)
    training_lengths = np.array([training_lengths])

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

    with open("../pdf.txt", "a") as f:
        np.savetxt(f, neutrino_pdfs, delimiter=",")

    if chi_square_for_postfit.size != 0:
        with open("../pred.txt", "a") as f:
            np.savetxt(f, pred, delimiter=",")

        with open("../train_indices.txt", "a") as f:
            np.savetxt(f, train_indices, delimiter=",")
        with open("../val_indices.txt", "a") as f:
            np.savetxt(f, val_indices, delimiter=",")

        with open("../training_lengths.txt", "a") as f:
            np.savetxt(f, training_lengths, delimiter=",")
