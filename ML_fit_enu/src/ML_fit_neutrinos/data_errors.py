import numpy as np
import torch


def compute_errors(pred, pred_min, pred_max):
    """Compute total sigma error on data and covariance matrix

    Args:
        pred (list): central data
        pred_min (list): minimum data
        pred_max (list): max data

    Returns:
        tuple: covariance matrix and total error
    """
    # compute systematic error on data

    # for _ in range(num_obs):
    delta_plus = pred_max - pred
    delta_min = pred_min - pred

    # print(f"delta plus = {delta_plus}")

    semi_diff = (delta_plus + delta_min) / 2
    average = (delta_plus - delta_min) / 2
    sig_sys = np.sqrt(average * average + 2 * semi_diff * semi_diff)
    # print(f"sig sys = {sig_sys}")
    # compute covariance matrix of data
    pred = np.where(pred == 0, 1, pred)

    sig_tot = sig_sys**2 + pred

    # cov_matrix = np.zeros
    cov_matrix = np.diag(sig_tot)
    cov_matrix = np.linalg.inv(cov_matrix)
    cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32, requires_grad=False)

    return sig_sys, sig_tot, cov_matrix
