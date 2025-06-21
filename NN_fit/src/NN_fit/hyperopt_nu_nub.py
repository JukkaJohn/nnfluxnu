# Author: Jukka John
# This files executes a hyperparam optimization for muon neutrinos
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
    fk_tables_mu: torch.Tensor,
    fk_tables_mub: torch.Tensor,
    binwidths_mu: torch.Tensor,
    binwidths_mub: torch.Tensor,
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
    Performs k-fold cross-validation training and evaluation of a neural network
    for muon neutrino flux prediction using Bayesian-inspired randomized hyperparameters.

    This function trains a `PreprocessedMLP` model to predict neutrino and antineutrino
    event distributions, optimizing a custom loss function that incorporates physical
    constraints and covariance information. It performs training with early stopping
    based on validation loss, and evaluates generalization via a held-out fold.

    Parameters:
    ----------
    pred : List[np.ndarray]
        List containing ground truth predicted neutrino events for training and evaluation.
    range_alpha : float
        Upper bound for random initialization of the alpha hyperparameter.
    range_beta : float
        Upper bound for random initialization of the beta hyperparameter.
    range_gamma : float
        Upper bound for random initialization of the gamma hyperparameter.
    lr : float
        Learning rate for the optimizer.
    wd : float
        Weight decay (L2 regularization) for the optimizer.
    patience : int
        Number of epochs to wait without validation improvement before early stopping.
    x_alphas : torch.Tensor
        Input tensor used for prediction by the model.
    fk_tables_mu : torch.Tensor
        Forward-folding kernel table for muon neutrinos.
    fk_tables_mub : torch.Tensor
        Forward-folding kernel table for anti-muon neutrinos.
    binwidths_mu : torch.Tensor
        Bin widths used for muon neutrino predictions.
    binwidths_mub : torch.Tensor
        Bin widths used for anti-muon neutrino predictions.
    cov_matrix : np.ndarray
        Covariance matrix for uncertainty propagation in loss computation.
    extended_loss : bool
        Whether to include extended regularization terms in the custom loss.
    activation_function : str
        Activation function to use in the model (e.g., "relu", "tanh").
    num_input_layers : int
        Number of input features/layers for the model.
    num_output_layers : int
        Number of outputs (e.g., neutrino types).
    hidden_layers : List[int]
        Sizes of hidden layers in the MLP.
    x_vals : np.ndarray
        Input data values used for training and evaluation.
    preproc : str
        Type of input preprocessing to apply (e.g., "standard", "log").
    max_epochs : int
        Maximum number of training epochs per fold.
    lag_mult_pos : float
        Lagrange multiplier weight for positivity constraint in the loss.
    lag_mult_int : float
        Lagrange multiplier weight for integral constraint in the loss.
    x_int : np.ndarray
        Points at which the integrals for regularization are evaluated.
    num_folds: int
        Number of k-folds

    Returns:
    -------
    Tuple[
        List[float],           # chi_squares over training
        List[np.ndarray],      # Predicted neutrino event counts
        List[np.ndarray],      # Predicted neutrino PDFs
        PreprocessedMLP,       # Trained model instance
        List[float],           # Chi-square values for post-fit analysis
        np.ndarray,            # Training indices from final fold
        np.ndarray,            # Validation indices from final fold
        int                    # Total number of training iterations in last fold
    ]
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

            y_pred_mu = (
                torch.matmul(fk_tables_mu, y_pred[:, 0]) * binwidths_mu.flatten()
            )
            y_pred_mub = (
                torch.matmul(fk_tables_mub, y_pred[:, 1]) * binwidths_mub.flatten()
            )

            y_pred_mu = y_pred_mu.squeeze()
            y_pred_mub = y_pred_mub.squeeze()

            y_preds = torch.hstack((y_pred_mu, y_pred_mub))

            y_train = y_preds[train_indices]
            pred_train = pred[0][train_indices]
            cov_matrix_train = cov_matrix[train_indices][:, train_indices]
            y_int_mu = model(x_int)[:, 0]
            y_int_mub = model(x_int)[:, 1]

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
            # print(loss)

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
