# Author: Jukka John
# This file compute sthe losses used in the NN
import torch


def complete_loss_fct_nu_nub(
    pred: torch.Tensor,
    data: torch.Tensor,
    cov_matrix: torch.Tensor,
    f: torch.Tensor,
    int_point_nu: torch.Tensor,
    int_point_nub: torch.Tensor,
    x_int: torch.Tensor,
    lag_mult_pos: float,
    lag_mult_int: float,
) -> torch.Tensor:
    """
    Extended chi-squared loss for separate neutrino and antineutrino predictions, with constraints.

    Parameters
    ----------
    pred : torch.Tensor
        Model predictions for observed data (shape: N).
    data : torch.Tensor
        Observed pseudo-data (shape: N).
    cov_matrix : torch.Tensor
        Covariance matrix for the data (shape: N x N).
    f : torch.Tensor
        Raw (non-preprocessed) neural network outputs (shape: N x 2), with:
            - f[:, 0]: neutrino component (ν)
            - f[:, 1]: antineutrino component (ν̄)
    int_point_nu : torch.Tensor
        Integral constraint values for ν (shape: N).
    int_point_nub : torch.Tensor
        Integral constraint values for ν̄ (shape: N).
    x_int : torch.Tensor
        x-values for integral evaluation (shape: N).
    lag_mult_pos : float
        Lagrange multiplier for enforcing positivity.
    lag_mult_int : float
        Lagrange multiplier for enforcing normalization via integral.

    Returns
    -------
    torch.Tensor
        Total loss including chi-squared term, positivity penalty, and integral constraint.
    """
    diff = pred - data
    diffcov = torch.matmul(cov_matrix, diff)

    fnu = f[:, 0]
    fnub = f[:, 1]
    fnu = torch.where(fnu > 0, fnu, torch.tensor(0.0))
    fnub = torch.where(fnub > 0, fnub, torch.tensor(0.0))

    loss = (
        (1 / pred.size(0)) * torch.dot(diff.view(-1), diffcov.view(-1))
        + abs(torch.sum(fnu)) * lag_mult_pos
        + abs(torch.sum(fnub)) * lag_mult_pos
        + abs(torch.sum(int_point_nu * x_int)) * lag_mult_int
        + abs(torch.sum(int_point_nub * x_int)) * lag_mult_int
    )

    return loss


def complete_loss_fct_comb(
    pred: torch.Tensor,
    data: torch.Tensor,
    cov_matrix: torch.Tensor,
    f: torch.Tensor,
    int_point_nu: torch.Tensor,
    x_int: torch.Tensor,
    lag_mult_pos: float,
    lag_mult_int: float,
) -> torch.Tensor:
    """
    Extended chi-squared loss for combined neutrino + antineutrino prediction, with constraints.

    Parameters
    ----------
    pred : torch.Tensor
        Model predictions (shape: N).
    data : torch.Tensor
        Observed pseudo-data (shape: N).
    cov_matrix : torch.Tensor
        Covariance matrix for the data (shape: N x N).
    f : torch.Tensor
        Raw NN output without preprocessing (shape: N).
    int_point_nu : torch.Tensor
        Integral constraint vector (shape: N).
    x_int : torch.Tensor
        x-values for integral constraint (shape: N).
    lag_mult_pos : float
        Lagrange multiplier for positivity constraint.
    lag_mult_int : float
        Lagrange multiplier for integral normalization.

    Returns
    -------
    torch.Tensor
        Total loss.
    """
    diff = pred - data
    diffcov = torch.matmul(cov_matrix, diff)

    f = torch.where(f > 0, 0, f)

    loss = (
        (1 / pred.size(0)) * torch.dot(diff.view(-1), diffcov.view(-1))
        + abs(torch.sum(f)) * lag_mult_pos
        + abs(torch.sum(int_point_nu * x_int)) * lag_mult_int
    )

    return loss


def raw_loss_fct(
    pred: torch.Tensor, data: torch.Tensor, cov_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Standard chi-squared loss without any constraints.

    Parameters
    ----------
    pred : torch.Tensor
        Model predictions (shape: N).
    data : torch.Tensor
        Observed pseudo-data (shape: N).
    cov_matrix : torch.Tensor
        Covariance matrix (shape: N x N).

    Returns
    -------
    torch.Tensor
        Chi-squared loss.
    """
    diff = pred - data
    diffcov = torch.matmul(cov_matrix, diff)

    loss = (1 / pred.size(0)) * torch.dot(diff.view(-1), diffcov.view(-1))

    return loss
