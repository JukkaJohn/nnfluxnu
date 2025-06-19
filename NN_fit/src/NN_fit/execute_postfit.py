# Author: Jukka John
# This file does all the postfit stuff after the fit has been completed: postfit criteria, measures, writes to lhapdf grid and plots the result of the fit
import numpy as np
import torch
from postfit_criteria import Postfit
from postfit_measures import Measures
from write_all_pdfs_to_lhapdf import write_lhapdf_grid, customize_info_file
import os
from typing import Optional, Union


def postfit_execution(
    postfit_criteria: bool,
    validation_split: float,
    data: np.ndarray,
    cov_matrix: np.ndarray,
    num_output_layers: int,
    chi_square_for_postfit: np.ndarray,
    neutrino_pdfs_mu: Optional[np.ndarray],
    neutrino_pdfs_mub: Optional[np.ndarray],
    neutrino_pdfs: Optional[np.ndarray],
    postfit_measures: bool,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    level1: np.ndarray,
    N_event_pred: np.ndarray,
    pred: np.ndarray,
    dir_for_data: str,
    filename_postfit: str,
    diff_lev_1: Union[str, int],
    fit_level: int,
    x_alphas: torch.Tensor,
    pdf: object,
    pdf_set: str,
    particle_id_nu: int,
    particle_id_nub: int,
    lr: float,
    wd: float,
    max_epochs: int,
    patience: int,
    chi_squares: np.ndarray,
    neutrino_pdf_fit_name_lhapdf: str,
    x_vals: np.ndarray,
    produce_plot: bool,
    training_lengths: np.ndarray,
    stat_error: np.ndarray,
    sys_error: np.ndarray,
    low_bin: int,
    high_bin: int,
    N_event_pred_nu: Optional[np.ndarray],
    N_event_pred_nub: Optional[np.ndarray],
    low_bin_mu: Optional[int],
    high_bin_mu: Optional[int],
    low_bin_mub: Optional[int],
    high_bin_mub: Optional[int],
    val_losses: list,
):
    """
    Execute post-fit processing after a PDF (Parton Distribution Function) neural fit.

    This function performs various operations following a neural network-based PDF fit:
    - Applies post-fit criteria to PDFs and predictions.
    - Calculates post-fit measures (e.g., delta chi-squared, phi, bias-to-variance).
    - Logs results and configuration to file.
    - Writes PDF replicas and uncertainties to LHAPDF-compatible grid files.
    - Optionally generates plots of the results.

    Parameters:
        postfit_criteria (bool): Whether to apply post-fit criteria to the predictions and PDFs.
        validation_split (float): Fraction of data used for validation. If 0, no validation is used.
        data (np.ndarray): Measured experimental data.
        cov_matrix (np.ndarray): Covariance matrix of the data.
        num_output_layers (int): Number of neural network outputs (1 for single PDF, 2 for neutrino/antineutrino PDFs).
        chi_square_for_postfit (np.ndarray): Chi-square values for the post-fit predictions.
        neutrino_pdfs_mu (np.ndarray): Neutrino PDFs for muon neutrino (only if `num_output_layers==2`).
        neutrino_pdfs_mub (np.ndarray): Neutrino PDFs for anti-muon neutrino (only if `num_output_layers==2`).
        neutrino_pdfs (np.ndarray): Neutrino PDFs for the single-output model.
        postfit_measures (bool): Whether to compute post-fit performance metrics.
        train_indices (np.ndarray): Training sample indices.
        val_indices (np.ndarray): Validation sample indices.
        level1 (np.ndarray): Level-1 shifts or corrections.
        N_event_pred (np.ndarray): Predicted number of events from the model.
        pred (np.ndarray): Model predictions.
        dir_for_data (str): Directory to save results and intermediate files.
        filename_postfit (str): Name of the post-fit report file.
        diff_lev_1 (str/int): Identifier for the level-1 difference, used in LHAPDF naming.
        fit_level (int): Level of fit used; affects which postfit measures are computed.
        x_alphas (torch.Tensor): Neural network output alphas (PDF coefficients).
        pdf (object): Reference PDF used during fitting.
        pdf_set (str): Name of the PDF set used as baseline.
        particle_id_nu (int): PDG ID for the neutrino used.
        particle_id_nub (int): PDG ID for the anti-neutrino used.
        lr (float): Learning rate used in the training.
        wd (float): Weight decay used during training.
        max_epochs (int): Maximum number of training epochs.
        patience (int): Early stopping patience.
        chi_squares (np.ndarray): Chi-square values for the fit.
        neutrino_pdf_fit_name_lhapdf (str): Name to use for the LHAPDF set.
        x_vals (np.ndarray): X-values (momentum fraction) for PDF grids.
        produce_plot (bool): Whether to produce summary plots of fit results.
        training_lengths (np.ndarray): Number of epochs run for each replica.
        stat_error (np.ndarray): Statistical error on the data.
        sys_error (np.ndarray): Systematic error on the data.
        low_bin (int): Lower bound for binning (single PDF).
        high_bin (int): Upper bound for binning (single PDF).
        N_event_pred_nu (np.ndarray): Event predictions for muon neutrino (two-output case).
        N_event_pred_nub (np.ndarray): Event predictions for anti-muon neutrino (two-output case).
        low_bin_mu (int): Lower bin index for muon neutrino.
        high_bin_mu (int): Upper bin index for muon neutrino.
        low_bin_mub (int): Lower bin index for anti-muon neutrino.
        high_bin_mub (int): Upper bin index for anti-muon neutrino.

    Outputs:
        - Writes various `.txt` files with statistical and fit information.
        - Creates LHAPDF-compatible grid files with the fitted PDFs.
        - Optionally generates plots summarizing the fit quality and predictions.

    Notes:
        - Assumes presence of `Postfit`, `Measures`, and LHAPDF writing utilities.
        - Assumes external plotting utilities (`plot_comb_pdf_cl`, `plot_nu_nub_cl`) are available.
        - Handles both single-output (combined neutrino) and two-output (neutrino/antineutrino) models.
    """
    if postfit_criteria:
        train_indices = train_indices.reshape(1, -1)
        val_indices = val_indices.reshape(1, -1)

        level1 = level1[0]

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
        with open(f"{dir_for_data}/{filename_postfit}", "w") as file:
            file.write(f"level 1 shift {diff_lev_1}:\n")
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

                with open(f"{dir_for_data}/{filename_postfit}", "w") as file:
                    file.write(f"delta chi^2 = {delta_chi}:\n")

                if num_output_layers == 1:
                    accuracy = compute_postfit.compute_accuracy(
                        x_alphas.detach().numpy().flatten(),
                        neutrino_pdfs,
                        pdf,
                        1,
                        pdf_set,
                        particle_id_nu,
                    )

                    with open(f"{dir_for_data}/{filename_postfit}", "w") as file:
                        file.write(f"accuracy = {accuracy}:\n")
                if num_output_layers == 2:
                    accuracy_nu = compute_postfit.compute_accuracy(
                        x_alphas.detach().numpy().flatten(),
                        neutrino_pdfs_mu,
                        pdf,
                        1,
                        pdf_set,
                        particle_id_nu,
                    )

                    accuracy_nub = compute_postfit.compute_accuracy(
                        x_alphas.detach().numpy().flatten(),
                        neutrino_pdfs_mub,
                        pdf,
                        1,
                        pdf_set,
                        particle_id_nub,
                    )

                    with open(f"{dir_for_data}/{filename_postfit}", "w") as file:
                        file.write(f"accuracy nu = {accuracy_nu}:\n")
                        file.write(f"accuracy nub = {accuracy_nub}:\n")

            phi = compute_postfit.compute_phi(data, chi_square_for_postfit)

            with open(f"{dir_for_data}/{filename_postfit}", "w") as file:
                file.write(f"phi = {phi}:\n")

            if fit_level == 2:
                bias_to_var = compute_postfit.compute_bias_to_variance(
                    data, pred, N_event_pred, len(N_event_pred)
                )

                with open(f"{dir_for_data}/{filename_postfit}", "w") as file:
                    file.write(f"bias_to_var = {bias_to_var}:\n")

        if postfit_measures and validation_split != 0.0:
            compute_postfit_measures(
                cov_matrix_val, N_event_pred_val, data_val, level1_val, pred_val
            )
        if postfit_measures and validation_split == 0.0:
            compute_postfit_measures(cov_matrix, N_event_pred, data, level1, pred)

        with open(f"{dir_for_data}/{filename_postfit}", "w") as file:
            file.write(f"mean chi^2 = {np.mean(chi_square_for_postfit)}:\n")
            file.write(f"average training length = {np.mean(training_lengths)}:\n")
            file.write("settings used:\n")
            file.write(f"learning rate = {lr}:\n")
            file.write(f"weigth decay = {wd}:\n")
            file.write(f"max training lenght = {max_epochs}:\n")
            file.write(f"patience = {patience}:\n")

    with open(f"{dir_for_data}/chi_square.txt", "w") as f:
        np.savetxt(f, chi_squares, delimiter=",")

    with open(f"{dir_for_data}/chi_squares_for_postfit.txt", "w") as f:
        np.savetxt(f, chi_square_for_postfit, delimiter=",")

    with open(f"{dir_for_data}/events.txt", "w") as f:
        np.savetxt(f, N_event_pred, delimiter=",")

    os.makedirs(
        f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{neutrino_pdf_fit_name_lhapdf}_{diff_lev_1}",
        exist_ok=True,
    )
    template_path = "/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/template_.info"
    path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{neutrino_pdf_fit_name_lhapdf}_{diff_lev_1}/{neutrino_pdf_fit_name_lhapdf}.info"
    set_index = int(np.random.rand() * 1e7)
    pdf_dict_central = {}
    pdf_dict_error = {}

    if num_output_layers == 1:
        customize_info_file(template_path, path, set_index, f"{particle_id_nu}", 2)
        mean_pdf = np.mean(neutrino_pdfs, axis=0)
        std_pdf = np.std(neutrino_pdfs, axis=0)
        path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{neutrino_pdf_fit_name_lhapdf}_{diff_lev_1}/{neutrino_pdf_fit_name_lhapdf}_0000.dat"
        pdf_dict_error[12] = mean_pdf
        pdf_dict_central[12] = std_pdf
        write_lhapdf_grid(x_vals, pdf_dict_central, path)
        path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{neutrino_pdf_fit_name_lhapdf}_{diff_lev_1}/{neutrino_pdf_fit_name_lhapdf}_0001.dat"
        write_lhapdf_grid(x_vals, pdf_dict_error, path)
    if num_output_layers == 2:
        customize_info_file(
            template_path, path, set_index, f"{particle_id_nu}, {particle_id_nub}", 2
        )
        mean_pdf_nu = np.mean(neutrino_pdfs_mu, axis=0)
        mean_pdf_nub = np.mean(neutrino_pdfs_mub, axis=0)
        std_pdf_nu = np.std(neutrino_pdfs_mu, axis=0)
        std_pdf_nub = np.std(neutrino_pdfs_mub, axis=0)
        path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{neutrino_pdf_fit_name_lhapdf}_{diff_lev_1}/{neutrino_pdf_fit_name_lhapdf}_0000.dat"

        pdf_dict_error[14] = mean_pdf_nu
        pdf_dict_error[-14] = mean_pdf_nub
        pdf_dict_central[14] = std_pdf_nu
        pdf_dict_central[-14] = std_pdf_nub
        write_lhapdf_grid(x_vals, pdf_dict_central, path)

        path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{neutrino_pdf_fit_name_lhapdf}_{diff_lev_1}/{neutrino_pdf_fit_name_lhapdf}_0001.dat"
        write_lhapdf_grid(x_vals, pdf_dict_error, path)

    if len(val_losses) > 0:
        val_losses = np.array(val_losses)
        with open(f"{dir_for_data}/val_losses.txt", "w") as f:
            np.savetxt(f, val_losses, delimiter=",")

    if chi_square_for_postfit.size != 0:
        with open(f"{dir_for_data}/pred.txt", "w") as f:
            np.savetxt(f, pred, delimiter=",")

        with open(f"{dir_for_data}/train_indices.txt", "w") as f:
            np.savetxt(f, train_indices, delimiter=",")
        with open(f"{dir_for_data}/val_indices.txt", "w") as f:
            np.savetxt(f, val_indices, delimiter=",")

        with open(f"{dir_for_data}/training_lengths.txt", "w") as f:
            np.savetxt(f, training_lengths, delimiter=",")

    if produce_plot:
        if num_output_layers == 1:
            from plot_comb_pdf_cl import plot

            sig_tot = np.sqrt(stat_error**2 + sys_error**2)
            plot(
                x_vals,
                neutrino_pdfs,
                data,
                N_event_pred,
                sig_tot,
                particle_id_nu,
                low_bin,
                high_bin,
                pdf,
                pdf_set,
                dir_for_data,
            )
        if num_output_layers == 2:
            from plot_nu_nub_cl import plot

            sig_tot = np.sqrt(stat_error**2 + sys_error**2)
            plot(
                x_vals,
                neutrino_pdfs_mu,
                neutrino_pdfs_mub,
                data,
                N_event_pred_nu,
                N_event_pred_nub,
                sig_tot,
                particle_id_nu,
                low_bin_mu,
                high_bin_mu,
                low_bin_mub,
                high_bin_mub,
                pdf,
                pdf_set,
                dir_for_data,
            )
