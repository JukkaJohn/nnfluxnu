# Author: Jukka John
# This files executes a hyperparam optimization for electron neutrinos
import torch
import numpy as np
from structure_NN import (
    PreprocessedMLP,
    CustomLoss,
)
from typing import List, Tuple
from sklearn.model_selection import KFold


def perform_fit(
    pred: List[np.ndarray],
    range_alpha: float,
    range_beta: float,
    range_gamma: float,
    lr: float,
    wd: float,
    patience: int,
    x_alphas: torch.Tensor,
    fk_tables: torch.Tensor,
    binwidths: torch.Tensor,
    cov_matrix: np.ndarray,
    extended_loss: bool,
    activation_function: str,
    num_input_layers: int,
    num_output_layers: int,
    hidden_layers: List[int],
    x_vals: np.ndarray,
    preproc: str,
    max_epochs: int,
    lag_mult_pos: float,
    lag_mult_int: float,
    x_int: np.ndarray,
    num_folds: int,
) -> Tuple[
    List[float],  # chi_squares
    List[np.ndarray],  # N_event_pred
    List[np.ndarray],  # neutrino_pdfs
    PreprocessedMLP,  # model (last accepted fit)
    List[float],  # chi_square_for_postfit
    np.ndarray,  # train_indices
    np.ndarray,  # val_indices
    int,  # training_length
]:
    """
    Trains a neural network model to fit pseudo-data for electron neutrino event predictions
    using K-fold cross-validation and physics-informed constraints.

    This function fits a parameterized neural network (PreprocessedMLP) using a custom loss function
    that incorporates statistical and physical constraints such as positivity and normalization.
    K-fold cross-validation is used to evaluate the model's generalization performance across
    different data splits. Random initialization of the preprocessing parameters (alpha, beta, gamma)
    enables exploration of a hyperparameter space.

    Parameters
    ----------
    pred : List[np.ndarray]
        List containing prediction arrays (pseudo-data) for electron neutrino event counts.
    range_alpha : float
        Maximum value for randomly sampling the alpha preprocessing parameter.
    range_beta : float
        Maximum value for randomly sampling the beta preprocessing parameter.
    range_gamma : float
        Maximum value for randomly sampling the gamma preprocessing parameter.
    lr : float
        Learning rate for the Adam optimizer.
    wd : float
        Weight decay (L2 regularization) used during optimization.
    patience : int
        Early stopping patience threshold (number of epochs without improvement before stopping).
    x_alphas : torch.Tensor
        Input tensor used to evaluate the model's predicted PDFs.
    fk_tables : torch.Tensor
        Forward-folding kernel that maps PDF space to observable event space.
    binwidths : torch.Tensor
        Bin widths used to scale the convolved predictions.
    cov_matrix : np.ndarray
        Covariance matrix of the pseudo-data, used for uncertainty-aware loss computation.
    extended_loss : bool
        Whether to include extended physics constraints (e.g., positivity, integrals) in the loss.
    activation_function : str
        Name of the activation function to be used in the MLP (e.g., 'relu', 'tanh').
    num_input_layers : int
        Number of input neurons to the network (typically 1 for univariate PDFs).
    num_output_layers : int
        Number of output neurons (typically 1 for electron neutrinos).
    hidden_layers : List[int]
        List of hidden layer sizes (e.g., [50, 50] for a 2-layer MLP with 50 neurons each).
    x_vals : np.ndarray
        Input values over which the final PDF predictions will be evaluated.
    preproc : str
        Type of preprocessing function used on the PDFs (e.g., 'log', 'powerlaw').
    max_epochs : int
        Maximum number of training epochs per fold.
    lag_mult_pos : float
        Lagrange multiplier for the positivity constraint in the loss.
    lag_mult_int : float
        Lagrange multiplier for the integral (normalization) constraint in the loss.
    x_int : np.ndarray
        Input values used for evaluating the integral constraints on the PDF.

    Returns
    -------
    chi_squares : List[float]
        History of chi-squared values during training (saved periodically).
    N_event_pred : List[np.ndarray]
        Placeholder for predicted event counts (not currently populated in this version).
    neutrino_pdfs : List[np.ndarray]
        Placeholder for final PDF outputs (not currently populated in this version).
    model : PreprocessedMLP
        Trained neural network model from the final fold.
    chi_square_for_postfit : List[float]
        Final loss value (chi-squared) for each fold.
    train_indices : np.ndarray
        Indices of the training samples used in the final fold.
    val_indices : np.ndarray
        Indices of the validation samples used in the final fold.
    training_length : int
        Number of epochs completed during the final fold training.
    num_folds: int
        number of k-folds

    Notes
    -----
    - The function uses 3-fold cross-validation to evaluate generalization.
    - Preprocessing parameters (alpha, beta, gamma) are randomized for each fold.
    - This implementation supports only one prediction channel and assumes
      symmetric treatment of integrals and positivity constraints.
    - `N_event_pred` and `neutrino_pdfs` are currently not returned meaningfully.
    """
    (
        chi_squares,
        k_fold_losses,
    ) = (
        [],
        [],
    )
    x_vals = torch.tensor(x_vals, dtype=torch.float32).view(-1, 1)

    x_int = torch.tensor(x_int, dtype=torch.float32).view(-1, 1)

    indices = np.arange(pred[0].shape[0])
    k = num_folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    folds = []
    for train_index, test_index in kf.split(indices):
        folds.append((train_index, test_index))

    for j in range(k):
        train_indices = folds[j][0]
        val_size = max(1, int(0.1 * len(train_indices)))
        val_indices = np.random.choice(train_indices, size=val_size, replace=False)
        k_fold_indices = folds[j][1]

        alpha, beta, gamma = (
            np.random.rand() * range_alpha,
            np.random.rand() * range_beta,
            np.random.rand() * range_gamma,
        )

        model = PreprocessedMLP(
            alpha,
            beta,
            gamma,
            activation_function,
            hidden_layers,
            num_input_layers,
            num_output_layers,
            preproc,
        )

        criterion = CustomLoss(
            extended_loss,
            num_output_layers,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        losses = []

        model.train()
        best_loss = 1e13  # initial loss
        counter = 0
        training_length = 0

        while counter < patience:
            if max_epochs < training_length:
                break

            training_length += 1

            optimizer.zero_grad()
            y_pred = model(x_alphas)

            y_preds = torch.matmul(fk_tables, y_pred[:, 0]) * binwidths.flatten()
            y_preds = y_preds.squeeze()
            y_int_mu = model(x_int)
            y_int_mub = y_int_mu

            y_train = y_preds[train_indices]
            pred_train = pred[0][train_indices]
            cov_matrix_train = cov_matrix[train_indices][:, train_indices]

            loss = criterion(
                y_train,
                pred_train,
                cov_matrix_train,
                y_int_mu,
                y_int_mub,
                y_pred,
                x_int,
                lag_mult_pos,
                lag_mult_int,
            )
            loss.backward()

            y_val = y_preds[val_indices]
            pred_val = pred[0][val_indices]
            cov_matrix_val = cov_matrix[val_indices][:, val_indices]

            loss_val = criterion(
                y_val,
                pred_val,
                cov_matrix_val,
                y_int_mu,
                y_int_mub,
                y_pred,
                x_int,
                lag_mult_pos,
                lag_mult_int,
            )

            if training_length % 500 == 0:
                chi_squares.append(loss.detach().numpy())
                print(loss.detach().numpy())

            losses.append(loss.detach().numpy())
            optimizer.step()

            if loss_val < best_loss:
                best_loss = loss_val
                counter = 0
            else:
                counter += 1

        y_k_fold = y_preds[k_fold_indices]
        pred_k_fold = pred[0][k_fold_indices]
        cov_matrix_k_fold = cov_matrix[k_fold_indices][:, k_fold_indices]

        loss_k_fold = criterion(
            y_k_fold,
            pred_k_fold,
            cov_matrix_k_fold,
            y_int_mu,
            y_int_mub,
            y_pred,
            x_int,
            lag_mult_pos,
            lag_mult_int,
        )

        k_fold_loss = loss_k_fold.item()
        k_fold_losses.append(k_fold_loss)
        print("k_fold_losses")
        print(k_fold_losses)

    return np.mean(k_fold_losses)
