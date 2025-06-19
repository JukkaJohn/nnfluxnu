# Author: Jukka John
# This file performs the NN fit
import torch
import numpy as np
from structure_NN import (
    PreprocessedMLP,
    CustomLoss,
)
from typing import List, Tuple


def perform_fit(
    pred: List[np.ndarray],
    num_reps: int,
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
    validation_split: float,
    max_epochs: int,
    max_chi_sq: float,
    lag_mult_pos: float,
    lag_mult_int: float,
    x_int: np.ndarray,
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
    Performs repeated training of neural networks to fit pseudo-data predictions using a
    physics-constrained loss function and neural PDF parameterization.

    Each replica (`num_reps`) of the prediction is fitted using a randomly initialized
    neural network (based on PreprocessedMLP), and the corresponding predictions and losses
    are recorded. Optionally uses a validation split.

    Parameters
    ----------
    pred : list of np.ndarray
        List of length `num_reps`, each containing the pseudo-data event predictions.
    num_reps : int
        Number of replicas (i.e., independent fits with random initialization).
    range_alpha : float
        Upper bound for uniform sampling of `alpha` preprocessing parameter.
    range_beta : float
        Upper bound for uniform sampling of `beta` preprocessing parameter.
    range_gamma : float
        Upper bound for uniform sampling of `gamma` preprocessing parameter.
    lr : float
        Learning rate for the Adam optimizer.
    wd : float
        Weight decay (L2 regularization) for the optimizer.
    patience : int
        Early stopping patience (currently unused but declared).
    x_alphas : torch.Tensor
        Input x-values used to evaluate PDF predictions for data loss.
    fk_tables : torch.Tensor
        FastKernel tables to convert PDFs into observable space.
    binwidths : torch.Tensor
        Widths of each bin used in rebinning the predictions.
    cov_matrix : np.ndarray
        Covariance matrix used for weighted loss calculation.
    extended_loss : bool
        If True, uses extended loss with constraints (e.g., normalization, positivity).
    activation_function : str
        Activation function used in the neural network (e.g., 'relu', 'tanh').
    num_input_layers : int
        Number of input layers before hidden layers.
    num_output_layers : int
        Number of output layers after hidden layers.
    hidden_layers : list of int
        Number of neurons in each hidden layer.
    x_vals : np.ndarray
        x-values used to store final fitted PDFs.
    preproc : str
        Type of preprocessing function (e.g., 'powerlaw', 'exp') applied to PDFs.
    validation_split : float
        Fraction of data used for validation (between 0 and 1).
    max_epochs : int
        Maximum number of training epochs.
    max_chi_sq : float
        Maximum allowed chi-squared for a fit to be accepted.
    lag_mult_pos : float
        Lagrange multiplier for positivity constraint.
    lag_mult_int : float
        Lagrange multiplier for integral constraint.
    x_int : np.ndarray
        x-values used for computing integral constraints.

    Returns
    -------
    chi_squares : list of float
        Training loss (chi-squared) values saved periodically during training.
    N_event_pred : list of np.ndarray
        Predicted event yields after applying the FastKernel convolution.
    neutrino_pdfs : list of np.ndarray
        Final predicted PDFs (postprocessed) from successful fits.
    model : PreprocessedMLP
        Final trained model (from last accepted fit).
    chi_square_for_postfit : list of float
        Final chi-squared values for each accepted fit (for post-fit evaluation).
    train_indices : np.ndarray
        Indices used for training set in the last run (if validation was used).
    val_indices : np.ndarray
        Indices used for validation set in the last run (if validation was used).
    training_length : int
        Number of training steps run in the final (last) model.

    Notes
    -----
    - Only models with `loss < max_chi_sq` are retained in the final output.
    - The PDFs are preprocessed using a parameterized function with random α, β, γ values.
    - Assumes the model class `PreprocessedMLP` and loss class `CustomLoss` are defined externally.
    - Model and predictions use PyTorch; inputs must be tensors where appropriate.
    - Currently no explicit early stopping logic is implemented (but patience is reserved).
    """
    (
        neutrino_pdfs,
        N_event_pred,
        chi_squares,
        preproc_pdfs,
        nn_pdfs,
        chi_square_for_postfit,
        val_losses,
    ) = [], [], [], [], [], [], []
    x_vals = torch.tensor(x_vals, dtype=torch.float32).view(-1, 1)
    x_int = torch.tensor(x_int, dtype=torch.float32).view(-1, 1)

    for i in range(num_reps):
        training_length = 0
        counter = 0
        best_loss = 1e13  # initial loss
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

        dataset_size = pred[i].shape[0]
        indices = np.arange(dataset_size)

        np.random.shuffle(indices)
        val_size = int(dataset_size * validation_split)

        train_indices, val_indices = indices[val_size:], indices[:val_size]

        if validation_split != 0:
            print("we are doing training validation split")
            pred_train = pred[i][train_indices]
            cov_matrix_train = cov_matrix[train_indices][:, train_indices]
            cov_matrix_val = cov_matrix[val_indices][:, val_indices]
            pred_val = pred[i][val_indices]
        else:
            pred[i] = pred[i].squeeze()
        print(pred[i])

        losses = []

        model.train()

        while counter < max_epochs:
            if max_epochs < training_length:
                break

            training_length += 1
            optimizer.zero_grad()
            y_pred = model(x_alphas)

            y_preds = torch.matmul(fk_tables, y_pred.flatten()) * binwidths.flatten()

            y_preds = y_preds.squeeze()

            y_int_mu = model(x_int)
            y_int_mub = y_int_mu

            if validation_split != 0.0:
                y_train = y_preds[train_indices]
                y_val = y_preds[val_indices]

                loss_val = criterion(
                    y_val,
                    pred_val,
                    cov_matrix_val,
                    y_pred,
                    y_int_mu,
                    y_int_mub,
                    x_int,
                    lag_mult_pos,
                    lag_mult_int,
                )

                loss = criterion(
                    y_train,
                    pred_train,
                    cov_matrix_train,
                    y_pred,
                    y_int_mu,
                    y_int_mub,
                    x_int,
                    lag_mult_pos,
                    lag_mult_int,
                )
            else:
                loss = criterion(
                    y_preds,
                    pred[i],
                    cov_matrix,
                    y_pred,
                    y_int_mu,
                    y_int_mub,
                    x_int,
                    lag_mult_pos,
                    lag_mult_int,
                )

            loss.backward()

            if training_length % 500 == 0:
                chi_squares.append(loss.detach().numpy())
                if validation_split != 0.0:
                    val_losses.append(loss_val)

            losses.append(loss.detach().numpy())
            optimizer.step()

            if loss < best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1

        if loss < max_chi_sq:
            f_nu = model(x_vals).detach().numpy().flatten()

            chi_square_for_postfit.append(loss.detach().numpy())

            preproc_pdfs.append(model.preproces(x_vals).detach().numpy().flatten())
            nn_pdf = model.neuralnet(x_vals)[:, 0].detach().numpy().flatten()
            nn_pdfs.append(nn_pdf)

            N_event_pred.append(y_preds.detach().numpy())

            neutrino_pdfs.append(f_nu)

    return (
        chi_squares,
        N_event_pred,
        neutrino_pdfs,
        model,
        chi_square_for_postfit,
        train_indices,
        val_indices,
        training_length,
        val_losses,
    )
