import torch
import torch.nn as nn
from form_loss_fct import raw_loss_fct, complete_loss_fct_nu_nub, complete_loss_fct_comb


class SimplePerceptron(torch.nn.Module):
    def __init__(
        self, act_functions, num_input_layers, hidden_layers, num_output_layers
    ):
        super().__init__()

        activation_map = {
            "softplus": nn.Softplus,  # no parentheses
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

        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], num_output_layers))
        layers.append(act_functions[-1])  # Optional activation
        print(layers)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class CustomPreprocessing(nn.Module):
    def __init__(self, alpha, beta, gamma, preproc):
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

    def forward(self, x):
        x = torch.clamp(x, 1e-6, 1 - 1e-6)
        beta = torch.nn.functional.relu(self.beta)
        # beta = torch.nn.functional.leaky_relu(self.beta)
        # alpha = 1 - torch.nn.functional.softplus(self.alpha)
        return self.gamma * (1 - x) ** beta * x ** (1 - self.alpha)


class PreprocessedMLP(nn.Module):
    def __init__(
        self,
        alpha,
        beta,
        gamma,
        activation_function,
        hidden_layers,
        num_input_layers,
        num_output_layers,
        preproc,
    ):
        super(PreprocessedMLP, self).__init__()

        self.preprocessing = CustomPreprocessing(alpha, beta, gamma, preproc)
        self.mlp = SimplePerceptron(
            activation_function, num_input_layers, hidden_layers, num_output_layers
        )
        self.preproc = preproc

    def forward(self, x):
        if self.preproc:
            f_preproc = self.preprocessing(x)
            f_NN = self.mlp(x)
            f_nu = f_preproc * f_NN
            return f_nu
        else:
            f_NN = self.mlp(x)
            return f_NN

    def neuralnet(self, x):
        f_NN = self.mlp(x)
        return f_NN

    def preproces(self, x):
        f_preproc = self.preprocessing(x)
        return f_preproc


class CustomLoss(nn.Module):
    def __init__(
        self,
        extended_loss,
        num_output_layers,
    ):
        super(CustomLoss, self).__init__()
        self.extended_loss = extended_loss
        self.num_output_layers = num_output_layers

    def forward(
        self,
        pred,
        data,
        cov_matrix,
        small_x_point1,
        small_x_point2,
        model,
        x_int,
        lag_mult_pos,
        lag_mult_int,
    ):
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
