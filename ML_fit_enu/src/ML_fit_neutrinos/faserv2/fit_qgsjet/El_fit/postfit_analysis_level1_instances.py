import numpy as np
import os
import sys
import torch
import matplotlib.pyplot as plt

# Add the parent directory to sys.path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Data for plot

from data_qgsjet import data_needed_for_fit
from control_file_qgsjet import hyperparams
from postfit_measures import Measures
from postfit_criteria import Postfit
from plot_diff_level1 import plot
# Get number of reps from make runscripts

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
gen = "QGSJET+POWHEG"
geom = "Run3"

filename_data_mu = (
    f"../../../FKtables/data/data/data_{obs}_FASERv_{geom}_{gen}_7TeV_numu_W.dat"
)
filename_data_mub = (
    f"../../../FKtables/data/data/data_{obs}_FASERv_{geom}_{gen}_7TeV_nubmu_W.dat"
)
filename_uncert_mu = f"../../../FKtables/data/uncertainties/uncertainties_{obs}_FASERv_{geom}_{gen}_7TeV_numu_W.dat"
filename_uncert_mub = f"../../../FKtables/data/uncertainties/uncertainties_{obs}_FASERv_{geom}_{gen}_7TeV_nubmu_W.dat"
filename_fk_mub_n = f"../../../FKtables/data/fastkernel/FK_{obs}_fine_nubmu_n.dat"
filename_fk_mub_p = f"../../../FKtables/data/fastkernel/FK_{obs}_fine_nubmu_p.dat"
filename_fk_mu_n = f"../../../FKtables/data/fastkernel/FK_{obs}_fine_numu_n.dat"
filename_fk_mu_p = f"../../../FKtables/data/fastkernel/FK_{obs}_fine_numu_p.dat"
filename_binsize = f"../../../FKtables/data/binning/FK_{obs}_binsize.dat"


def compute_postfit_criteria(neutrino_pdfs_mu, neutrino_pdfs_mub, N_event_pred, pred):
    # if postfit_criteria:
    closure_fit = Postfit()
    neutrino_pdfs_mu, _, _ = closure_fit.apply_postfit_criteria(
        chi_squares_postfit, N_event_pred, neutrino_pdfs_mu, pred
    )
    neutrino_pdfs_mub, N_event_pred, pred = closure_fit.apply_postfit_criteria(
        chi_squares_postfit, N_event_pred, neutrino_pdfs_mub, pred
    )


def compute_postfit_measures(cov_matrix, N_event_pred, data, level1, pred):
    compute_postfit = Measures(cov_matrix, pdf, N_event_pred)
    delta_chi = compute_postfit.compute_delta_chi(
        data,
        N_event_pred,
        level1,
        x_alphas.detach().numpy().squeeze(),
    )
    print(f"mean delta chi = {delta_chi}")

    accuracy = compute_postfit.compute_accuracy(
        x_alphas.detach().numpy().flatten(), neutrino_pdfs_mu, pdf, 1
    )
    print(f"accuracy = {accuracy}")

    bias_to_var = compute_postfit.compute_bias_to_variance(
        data, pred, N_event_pred, num_reps
    )
    print(f"bias to var = {bias_to_var}")

    phi = compute_postfit.compute_phi(data, chi_squares_postfit)

    return delta_chi, accuracy, bias_to_var, phi


mean_pdf_all_fits_mu = 0
mean_pdf_all_fits_mub = 0
(
    total_std_mu,
    total_std_mub,
    total_preds_Enu,
    total_std_preds_Enu,
    total_preds_Enu_mub,
    total_std_preds_Enu_mub,
) = 0, 0, 0, 0, 0, 0
num_level1_instances = 2

delta_chis, accuracies, bias_to_vars, rel_uncertainties, chi_level1s = (
    [],
    [],
    [],
    [],
    [],
)
for i in range(1, num_level1_instances + 1):
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
        fk_tables_mu,
        fk_tables_mub,
        low_bin_mu,
        low_bin_mub,
        high_bin_mu,
        high_bin_mub,
    ) = data_needed_for_fit(
        0,
        i,
        filename_data_mu=filename_data_mu,
        filename_data_mub=filename_data_mub,
        filename_uncert_mu=filename_uncert_mu,
        filename_uncert_mub=filename_uncert_mub,
        filename_fk_mub_n=filename_fk_mub_n,
        filename_fk_mub_p=filename_fk_mub_p,
        filename_fk_mu_n=filename_fk_mu_n,
        filename_fk_mu_p=filename_fk_mu_p,
        filename_binsize=filename_binsize,
    )
    level1 = level1[0]
    chi_squares = np.loadtxt(
        f"diff_level_1_runs/runscripts_{i}/chi_square.txt", delimiter=","
    )
    chi_squares_postfit = np.loadtxt(
        f"diff_level_1_runs/runscripts_{i}/chi_squares_for_postfit.txt", delimiter=","
    )
    # N_event_pred = np.loadtxt(
    #     f"diff_level_1_runs/runscripts_{i}/events.txt", delimiter=","
    # )
    N_event_pred_mu = np.loadtxt(
        f"diff_level_1_runs/runscripts_{i}/events_mu.txt", delimiter=","
    )
    N_event_pred_mub = np.loadtxt(
        f"diff_level_1_runs/runscripts_{i}/events_mub.txt", delimiter=","
    )

    #N_event_pred_mu = N_event_pred_mu[:, :]
    #N_event_pred_mub = N_event_pred_mub[:, :]
    N_event_pred = np.hstack((N_event_pred_mu, N_event_pred_mub))

    neutrino_pdfs_mu = np.loadtxt(
        f"diff_level_1_runs/runscripts_{i}/mu_pdf.txt", delimiter=","
    )
    neutrino_pdfs_mub = np.loadtxt(
        f"diff_level_1_runs/runscripts_{i}/mub_pdf.txt", delimiter=","
    )

    pred = np.loadtxt(f"diff_level_1_runs/runscripts_{i}/pred.txt", delimiter=",")
    train_indices = np.loadtxt(
        f"diff_level_1_runs/runscripts_{i}/train_indices.txt", delimiter=","
    )

    val_indices = np.loadtxt(
        f"diff_level_1_runs/runscripts_{i}/val_indices.txt", delimiter=","
    )

    training_lengths = np.loadtxt(
        f"diff_level_1_runs/runscripts_{i}/training_lengths.txt", delimiter=","
    )

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
        # return indices and all is fine????

        N_event_pred_val = N_event_pred[:, val_indices]
        data_val = data[val_indices]
        pred_val = pred[:, val_indices]

        level1_val = level1[val_indices]

        val_indices = torch.tensor([val_indices])
        cov_matrix_val = cov_matrix[val_indices][:, val_indices]

    if postfit_criteria and validation != 0.0:
        compute_postfit_criteria(
            neutrino_pdfs_mu, neutrino_pdfs_mub, N_event_pred_train, pred_train
        )
    if postfit_criteria and validation == 0:
        compute_postfit_criteria(
            neutrino_pdfs_mu, neutrino_pdfs_mub, N_event_pred, pred
        )

    # compute_postfit = Measures(cov_matrix, pdf, N_event_pred)

    if postfit_measures and validation != 0.0:
        delta_chi, accuracy, bias_to_var, phi = compute_postfit_measures(
            cov_matrix_val, N_event_pred_val, data_val, level1_val, pred_val
        )
    if postfit_measures and validation == 0.0:
        delta_chi, accuracy, bias_to_var, phi = compute_postfit_measures(
            cov_matrix, N_event_pred, data, level1, pred
        )

    rel_uncertainty = (np.std(neutrino_pdfs_mu, axis=0)) / np.mean(
        neutrino_pdfs_mu, axis=0
    )

    mean_N_event_fits = np.mean(N_event_pred, axis=0)
    mean_N_event_fits = torch.tensor(mean_N_event_fits, dtype=torch.float32)
    level1 = torch.tensor(level1, dtype=torch.float32)

    diff = mean_N_event_fits - level1
    diffcov = torch.matmul(cov_matrix, diff)
    chi_level1 = torch.dot(diff.view(-1), diffcov.view(-1))
    chi_level1s.append(chi_level1)

    with open("fit_report_diff_level_1.txt", "a") as file:
        # file.write(f"delta chi^2 = {delta_chi}:\n")
        file.write(f"accuracy = {accuracy}:\n")
        file.write(f"bias to variance ratio = {bias_to_var}:\n")
        file.write(f"phi = {phi}:\n")
        file.write(f"average training length = {np.mean(training_lengths)}:\n")
        file.write(f"mean chi^2 = {np.mean(chi_squares_postfit)}:\n")
        file.write(f"chi^2 level1 = {chi_level1}:\n")

    rel_uncertainties.append(rel_uncertainty.flatten())
    delta_chis.append(delta_chi)

    accuracies.append(accuracy)
    bias_to_vars.append(bias_to_var)

    mean_pdf_all_fits_mu += np.mean(neutrino_pdfs_mu, axis=0) / num_level1_instances
    mean_pdf_all_fits_mub += np.mean(neutrino_pdfs_mub, axis=0) / num_level1_instances

    total_std_mu += np.std(neutrino_pdfs_mu, axis=0) ** 2
    total_std_mub += np.std(neutrino_pdfs_mub, axis=0) ** 2

    total_preds_Enu += np.mean(N_event_pred_mu, axis=0) / num_level1_instances
    total_std_preds_Enu += np.std(N_event_pred_mu, axis=0) ** 2

    total_preds_Enu_mub += np.mean(N_event_pred_mub, axis=0) / num_level1_instances
    total_std_preds_Enu_mub += np.std(N_event_pred_mub, axis=0) ** 2

    # var_fits = np.var(y_fits, axis=0)
    # mean_var_errors = np.mean(y_errors**2, axis=0)
    # total_std = np.sqrt(var_fits + mean_var_errors)

    # total_mu_pdfs_err.append(neutrino_pdfs_mu)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],  # Computer Modern
    }
)
plt.grid(color="grey", linestyle="-", linewidth=0.25)
plt.plot(range(len(chi_level1s)), chi_level1s)
plt.title(
    r"$ \mathrm{pseudo \ \ data, \ \ 10 \ \ level \ \ 1 \ \ instances}$", loc="right"
)
plt.ylabel(r"$\chi^2[ \left\langle T_{rep} \right\rangle ,D_1] $")
plt.xlabel("fit number")
plt.savefig("chi_squared.pdf")
# plt.show()

total_std_mu = np.sqrt(total_std_mu) / num_level1_instances
total_std_mub = np.sqrt(total_std_mub) / num_level1_instances
total_std_preds_Enu = np.sqrt(total_std_preds_Enu) / num_level1_instances

print(f"mean rel uncertainty = {np.mean(rel_uncertainties, axis=0)}")
print(f"std rel uncertainty = {np.std(rel_uncertainties, axis=0)}")
# print(f"mean delta chi = {np.mean(delta_chis)}")
print(f"mean accuracy = {np.mean(accuracies)}")
print(f"mean bias to var ratio = {np.mean(bias_to_vars)}")
# add mean chi square and mean training lengths + all measures for individual fits I guess
mean_rel_unc = np.mean(rel_uncertainties, axis=0)
std_rel_unc = np.std(rel_uncertainties, axis=0)
# plt.plot(x_vals, mean_rel_unc,color = 'blue')
print(mean_rel_unc.shape)
plt.figure()

plt.grid(color="grey", linestyle="-", linewidth=0.25)
plt.title(
    r"$ \mathrm{pseudo \ \ data, \ \ 10 \ \ level \ \ 1 \ \ instances}$", loc="right"
)

plt.fill_between(
    x_vals.flatten(),
    (mean_rel_unc + std_rel_unc),
    (mean_rel_unc - std_rel_unc),
    color="red",
    label="rel unc",
)
# plt.legend()
plt.xlabel(r"$x_\nu$")
plt.xscale("log")
plt.ylabel(r"$ \frac{\sigma_{T}}{<T>_{rep}} $")
# plt.ylabel("relative uncertainty")
plt.savefig("rel_unc.pdf")
print(mean_rel_unc)
print(std_rel_unc)
# plt.show()

# N_event_pred = np.loadtxt("diff_level_1_runs/level0_fit/events.txt", delimiter=",")
N_event_pred_mu = np.loadtxt(
    "diff_level_1_runs/level0_fit/events_mu.txt", delimiter=","
)
N_event_pred_mub = np.loadtxt(
    "diff_level_1_runs/level0_fit/events_mub.txt", delimiter=","
)

#N_event_pred_mu = N_event_pred_mu[:, :]
#N_event_pred_mub = N_event_pred_mub[:, :]
N_event_pred = np.hstack((N_event_pred_mu, N_event_pred_mub))

level0_fit = np.mean(N_event_pred_mu, axis=0)
err_level0_fit = np.std(N_event_pred_mu, axis=0)

level0_fit_mub = np.mean(N_event_pred_mub, axis=0)
err_level0_fit_mub = np.std(N_event_pred_mub, axis=0)

neutrino_pdfs_mu = np.loadtxt("diff_level_1_runs/level0_fit/mu_pdf.txt", delimiter=",")
neutrino_pdfs_mub = np.loadtxt(
    "diff_level_1_runs/level0_fit/mub_pdf.txt", delimiter=","
)
mu_level0_fit = np.mean(neutrino_pdfs_mu, axis=0)
mub_level0_fit = np.mean(neutrino_pdfs_mub, axis=0)
err_mu_level0_fit = np.std(neutrino_pdfs_mu, axis=0)
err_mub_level0_fit = np.std(neutrino_pdfs_mub, axis=0)
err_sim = np.sqrt(sig_sys**2 + sig_stat**2)
plot(
    x_vals,
    mean_pdf_all_fits_mu,
    mean_pdf_all_fits_mub,
    total_std_mu,
    total_std_mub,
    data,
    total_preds_Enu,
    total_std_preds_Enu,
    total_preds_Enu_mub,
    total_std_preds_Enu_mub,
    14,
    data,
    err_sim,
    level0_fit,
    err_level0_fit,
    level0_fit_mub,
    err_level0_fit_mub,
    mu_level0_fit,
    err_mu_level0_fit,
    mub_level0_fit,
    err_mub_level0_fit,
    high_bin_mu,
    low_bin_mu,
    low_bin_mub,
    high_bin_mub,
)
