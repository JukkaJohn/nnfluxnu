# Author: Jukka John
# This file is accessed to build the NN model
import torch
import torch.nn as nn
from form_loss_fct import raw_loss_fct, complete_loss_fct_nu_nub, complete_loss_fct_comb


class SimplePerceptron(torch.nn.Module):
    """
    A feedforward multilayer perceptron (MLP) with configurable activation functions and layer sizes.

    Parameters
    ----------
    act_functions : list of str
        List of activation function names (e.g., ['relu', 'relu', 'softplus']) for each layer.
    num_input_layers : int
        Number of input features.
    hidden_layers : list of int
        List of integers specifying the number of units in each hidden layer.
    num_output_layers : int
        Number of output features.

    Attributes
    ----------
    layers : nn.Sequential
        Composed list of linear and activation layers forming the MLP.

    Notes
    -----
    - Supported activation functions: 'relu', 'softplus'.
    - The last activation is applied after the final output layer.
    """

    def __init__(
        self,
        act_functions: list[str],
        num_input_layers: int,
        hidden_layers: list[int],
        num_output_layers: int,
    ):
        super().__init__()

        activation_map = {
            "softplus": nn.Softplus,
            "relu": nn.ReLU,
        }

        activation_names = act_functions

        act_functions = [activation_map[name]() for name in activation_names]

        layers = []

        layers.append(nn.Linear(num_input_layers, hidden_layers[0]))
        layers.append(act_functions[0])

        for i in range(0, len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(act_functions[i])

        layers.append(nn.Linear(hidden_layers[-1], num_output_layers))
        layers.append(act_functions[-1])
        print(layers)

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_input_layers)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_output_layers)
        """
        return self.layers(x)


class CustomPreprocessing(nn.Module):
    """
    Applies a parameterized functional preprocessing to the input based on power-law forms.

    The form is:
        f(x) = γ * (1 - x)^β * x^(1 - α)

    Parameters
    ----------
    alpha : float
        Initial value for α parameter.
    beta : float
        Initial value for β parameter.
    gamma : float
        Initial value for γ parameter.
    preproc : bool
        If True, alpha, beta, and gamma are learnable parameters. Otherwise, they are fixed.

    Notes
    -----
    - Input values are clamped between [1e-6, 1 - 1e-6] for numerical stability.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        preproc: bool,
    ):
        super(CustomPreprocessing, self).__init__()

        if preproc:
            self.alpha = nn.Parameter(
                torch.tensor(alpha, dtype=torch.float32, requires_grad=True)
            )
            self.beta = nn.Parameter(
                torch.tensor(beta, dtype=torch.float32, requires_grad=True)
            )
            self.gamma = nn.Parameter(
                torch.tensor(gamma, dtype=torch.float32, requires_grad=True)
            )
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
            self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32))
            self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the preprocessing function.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, 1).

        Returns
        -------
        torch.Tensor
            Preprocessed output of same shape.
        """
        x = torch.clamp(x, 1e-6, 1 - 1e-6)
        beta = torch.nn.functional.relu(self.beta)

        return self.gamma * (1 - x) ** beta * x ** (1 - self.alpha)


class PreprocessedMLP(nn.Module):
    """
    A neural network combining a preprocessing layer with an MLP.

    Parameters
    ----------
    alpha, beta, gamma : float
        Parameters for the preprocessing function.
    activation_function : list of str
        Activation functions for the MLP.
    hidden_layers : list of int
        Sizes of hidden layers.
    num_input_layers : int
        Number of input features.
    num_output_layers : int
        Number of output features.
    preproc : bool
        Whether to use preprocessing.

    Attributes
    ----------
    preprocessing : CustomPreprocessing
        The preprocessing module applied before the MLP.
    mlp : SimplePerceptron
        The neural network model applied to the preprocessed inputs.

    Methods
    -------
    forward(x)
        Applies preprocessing (if enabled) followed by the MLP.
    neuralnet(x)
        Returns raw MLP output (no preprocessing).
    preproces(x)
        Returns the preprocessing factor only.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        activation_function: list[str],
        hidden_layers: list[int],
        num_input_layers: int,
        num_output_layers: int,
        preproc: bool,
    ):
        super(PreprocessedMLP, self).__init__()

        self.preprocessing = CustomPreprocessing(alpha, beta, gamma, preproc)
        self.mlp = SimplePerceptron(
            activation_function, num_input_layers, hidden_layers, num_output_layers
        )
        self.preproc = preproc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass through preprocessing and MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of the combined preprocessing and MLP.
        """
        if self.preproc:
            f_preproc = self.preprocessing(x)
            f_NN = self.mlp(x)
            f_nu = f_preproc * f_NN
            return f_nu
        else:
            f_NN = self.mlp(x)
            return f_NN

    def neuralnet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through only the MLP (no preprocessing).

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        f_NN = self.mlp(x)
        return f_NN

    def preproces(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the preprocessing term γ * (1 - x)^β * x^(1 - α)

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        f_preproc = self.preprocessing(x)
        return f_preproc


class CustomLoss(nn.Module):
    """
    Custom loss function wrapper supporting multiple modes: raw chi-squared, or extended with constraints.

    Parameters
    ----------
    extended_loss : bool
        If True, uses extended loss with positivity and normalization constraints.
    num_output_layers : int
        Determines loss mode: 1 for combined ν+ν̄, 2 for separate ν and ν̄ losses.

    Methods
    -------
    forward(pred, data, cov_matrix, small_x_point1, small_x_point2, model, x_int, lag_mult_pos, lag_mult_int)
        Computes the loss.
    """

    def __init__(self, extended_loss: bool, num_output_layers: int):
        super(CustomLoss, self).__init__()
        self.extended_loss = extended_loss
        self.num_output_layers = num_output_layers

    def forward(
        self,
        pred: torch.Tensor,
        data: torch.Tensor,
        cov_matrix: torch.Tensor,
        small_x_point1: torch.Tensor,
        small_x_point2: torch.Tensor,
        model: torch.Tensor,
        x_int: torch.Tensor,
        lag_mult_pos: float,
        lag_mult_int: float,
    ) -> torch.Tensor:
        """
        Computes the loss function.

        Parameters
        ----------
        pred : torch.Tensor
            Model prediction.
        data : torch.Tensor
            Target data (pseudo-data).
        cov_matrix : np.ndarray or torch.Tensor
            Covariance matrix for data.
        small_x_point1 : torch.Tensor
            Neural prediction before preprocessing (ν or combined).
        small_x_point2 : torch.Tensor
            Neural prediction before preprocessing (ν̄, if using two outputs).
        model : torch.nn.Module
            Reference to the model (used for constraints).
        x_int : torch.Tensor
            x-points for integral constraint.
        lag_mult_pos : float
            Lagrange multiplier for positivity.
        lag_mult_int : float
            Lagrange multiplier for integral constraint.

        Returns
        -------
        torch.Tensor
            Final loss scalar.
        """
        if self.extended_loss:
            if self.num_output_layers == 1:
                loss = complete_loss_fct_comb(
                    pred,
                    data,
                    cov_matrix,
                    small_x_point1,
                    model,
                    x_int,
                    lag_mult_pos,
                    lag_mult_int,
                )
            if self.num_output_layers == 2:
                loss = complete_loss_fct_nu_nub(
                    pred,
                    data,
                    cov_matrix,
                    small_x_point1,
                    small_x_point2,
                    model,
                    x_int,
                    lag_mult_pos,
                    lag_mult_int,
                )

        else:
            loss = raw_loss_fct(pred, data, cov_matrix)
        return loss
