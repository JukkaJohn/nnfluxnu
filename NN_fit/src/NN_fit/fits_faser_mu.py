import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.nn.functional
import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(
    os.path.join(
        os.getcwd(),
        "/Users/jukkajohn/Masterscriptie/faser_nufluxes_ml/ML_fit_enu/src/ML_fit_neutrinos/",
    )
)
sys.path.append(parent_dir)

from read_faserv_pdf import read_pdf
from MC_data_reps import generate_MC_replicas
from data_for_faser_fit import data_needed_for_fit
from form_loss_fct import complete_loss_fct, raw_loss_fct
from logspace_grid import generate_grid

# from plot_results_faser_data import plot
from control_file_fits import hyperparams
from read_fk_table import get_fk_table

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
) = hyperparams()


filename = "data_muon_sim_faser/FK_Enu_7TeV_nu_W.dat"
x_alphas, fk_tables_mu = get_fk_table(filename=filename, parent_dir=parent_dir)
filename = "data_muon_sim_faser/FK_Enu_7TeV_nub_W.dat"
x_alphas, fk_tables_mub = get_fk_table(filename=filename, parent_dir=parent_dir)

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
    _,
) = data_needed_for_fit(fit_level)


class SimplePerceptron(torch.nn.Module):
    def __init__(self):
        super().__init__()

        layers = []
        layers.append(nn.Linear(1, 2))
        layers.append(act_functions[0])
        layers.append(nn.Linear(2, num_nodes))
        layers.append(act_functions[1])
        for i in range(num_layers - 2):
            layers.append(nn.Linear(num_nodes, num_nodes))
            layers.append(act_functions[i + 2])
        layers.append(nn.Linear(num_nodes, 2))
        self.layers = nn.Sequential(*layers)

        # def _initialize_weights(self):
        # for layer in self.layers:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.uniform_(layer.weight, a=-10, b=10)  # adjust a, b as needed
        #         if layer.bias is not None:
        #             nn.init.uniform_(layer.bias, a=-10, b=10)

    def forward(self, x):
        return self.layers(x)


if not preproc:

    class CustomPreprocessing(nn.Module):
        def __init__(self, alpha, beta, gamma):
            super(CustomPreprocessing, self).__init__()
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
            self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32))
            self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

        def forward(self, x):
            x = torch.clamp(x, 1e-6, 1 - 1e-6)
            return self.gamma * (1 - x) ** self.beta * x ** (1 - self.alpha)

    class PreprocessedMLP(nn.Module):
        def __init__(self, alpha, beta, gamma):
            super(PreprocessedMLP, self).__init__()
            self.preprocessing = CustomPreprocessing(alpha, beta, gamma)
            self.mlp = SimplePerceptron()

        def forward(self, x):
            f_NN = self.mlp(x)
            return f_NN

        def neuralnet(self, x):
            f_NN = self.mlp(x)
            return f_NN

        def preproc(self, x):
            f_preproc = self.preprocessing(x)
            return f_preproc

else:

    class CustomPreprocessing(nn.Module):
        def __init__(self, alpha, beta, gamma):
            super(CustomPreprocessing, self).__init__()

            self.alpha = nn.Parameter(
                torch.tensor(alpha, dtype=torch.float32, requires_grad=True)
            )
            self.beta = nn.Parameter(
                torch.tensor(beta, dtype=torch.float32, requires_grad=True)
            )
            self.gamma = nn.Parameter(
                torch.tensor(gamma, dtype=torch.float32, requires_grad=True)
            )

        def forward(self, x):
            x = torch.clamp(x, 1e-6, 1 - 1e-6)

            return self.gamma * (1 - x) ** self.beta * x ** (1 - self.alpha)

    class PreprocessedMLP(nn.Module):
        def __init__(self, alpha, beta, gamma):
            super(PreprocessedMLP, self).__init__()
            self.preprocessing = CustomPreprocessing(alpha, beta, gamma)
            self.mlp = SimplePerceptron()

        def forward(self, x):
            f_preproc = self.preprocessing(x)
            f_NN = self.mlp(x)
            f_nu = f_preproc * f_NN
            return f_nu

        def neuralnet(self, x):
            f_NN = self.mlp(x)
            return f_NN

        def preproc(self, x):
            f_preproc = self.preprocessing(x)
            return f_preproc


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, pred, data, cov_matrix, small_x_point1, small_x_point2, model):
        if extended_loss:
            complete_loss_fct(
                pred, data, cov_matrix, small_x_point1, small_x_point2, model
            )
        else:
            loss = raw_loss_fct(pred, data, cov_matrix)
        return loss


x_vals = torch.tensor(x_vals, dtype=torch.float32).view(-1, 1)
(
    neutrino_pdfs_mu,
    neutrino_pdfs_mub,
    N_event_pred,
    arc_lenghts,
    chi_squares,
    int_penaltys,
    pos_penaltys,
    preproc_pdfs,
    nn_pdfs,
    chi_square_for_postfit,
) = [], [], [], [], [], [], [], [], [], []


def perform_fit(pred, REPLICAS):
    for i in range(REPLICAS):
        alpha, beta, gamma = (
            np.random.rand() * range_alpha,
            np.random.rand() * range_beta,
            np.random.rand() * range_gamma,
        )
        model = PreprocessedMLP(alpha, beta, gamma)

        criterion = CustomLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        losses = []
        pred[i] = pred[i].squeeze()
        model.train()
        best_loss = 1e13  # initial loss
        counter = 0
        # num_epochs = 12000
        # for ep in range(num_epochs):
        while counter < max_counter:
            optimizer.zero_grad()
            y_pred = model(x_alphas)
            # y_pred =  model(x_alphas) * x_alphas
            # y_pred_mu = torch.matmul(fk_tables_mu,y_pred[:,0]- model(torch.tensor([1.], dtype=torch.float32).view(-1,1))[:,0]  )  * binwidths_mu.flatten()
            # y_pred_mub = torch.matmul(fk_tables_mub,y_pred[:,1]- model(torch.tensor([1.], dtype=torch.float32).view(-1,1))[:,1]  )  * binwidths_mub.flatten()
            y_pred_mu = (
                torch.matmul(fk_tables_mu, y_pred[:, 0]) * binwidths_mu.flatten()
            )
            y_pred_mub = (
                torch.matmul(fk_tables_mub, y_pred[:, 1]) * binwidths_mub.flatten()
            )

            y_pred_mu = y_pred_mu.squeeze()
            y_pred_mub = y_pred_mub.squeeze()

            y_pred_mu[-1] = y_pred_mu[-1] + y_pred_mub[-1]

            y_pred_mub = y_pred_mub[:-1]

            y_pred_mub = torch.flip(y_pred_mub, dims=[0])

            y_preds = torch.hstack((y_pred_mu, y_pred_mub))

            x_int = torch.tensor([1e-4, 1.0], dtype=torch.float32).view(-1, 1)
            y_int_mu = model(x_int)[:, 0]
            y_int_mub = model(x_int)[:, 1]

            loss = criterion(y_preds, pred[i], cov_matrix, y_int_mu, y_int_mub, x_int)

            loss.backward()
            # print(loss)
            if counter % 100 == 0:
                print(loss)
                chi_squares.append(loss.detach().numpy())

            # for name, param in model.named_parameters():
            #     if param.grad is not None and torch.isnan(param.grad).any():
            #         print(f"NaN detected in gradients of {name}!")
            #         print(y_preds)

            losses.append(loss.detach().numpy())
            optimizer.step()

            if loss < best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1

        if loss < 5.0:
            print(f"rep {i + 1} done out of {REPLICAS}")
            print(f"reduced chi^2 level 2 = {loss}")

            print(f"Constrained alpha: {(model.preprocessing.alpha.item())}")
            print(f"Constrained beta: {(model.preprocessing.beta.item())}")
            print(f"Constrained gamma: {model.preprocessing.gamma.item()}")

            f_nu_mub = model(x_vals)[:, 1].detach().numpy().flatten()
            f_nu_mu = model(x_vals)[:, 0].detach().numpy().flatten()

            chi_square_for_postfit.append(loss.detach().numpy())

            preproc_pdfs.append(model.preproc(x_vals).detach().numpy().flatten())
            nn_pdf = model.neuralnet(x_vals)[:, 0].detach().numpy().flatten()
            nn_pdfs.append(nn_pdf)

            N_event_pred.append(y_preds.detach().numpy())
            neutrino_pdfs_mu.append(f_nu_mu)
            neutrino_pdfs_mub.append(f_nu_mub)
    return (
        chi_squares,
        N_event_pred,
        neutrino_pdfs_mu,
        neutrino_pdfs_mub,
        model,
        chi_square_for_postfit,
    )


(
    chi_squares,
    N_event_pred,
    neutrino_pdfs_mu,
    neutrino_pdfs_mub,
    model,
    chi_square_for_postfit,
) = perform_fit(pred, REPLICAS)


chi_squares = np.array(chi_squares)
N_event_pred = np.array(N_event_pred)
neutrino_pdfs_mu = np.array(neutrino_pdfs_mu)
neutrino_pdfs_mub = np.array(neutrino_pdfs_mub)
chi_square_for_postfit = np.array(chi_square_for_postfit)

with open("../chi_square.txt", "a") as f:
    np.savetxt(f, chi_squares, delimiter=",")

with open("../chi_squares_for_postfit.txt", "a") as f:
    np.savetxt(f, chi_square_for_postfit, delimiter=",")

with open("../events.txt", "a") as f:
    np.savetxt(f, N_event_pred, delimiter=",")

# Append to the "mu_pdf.txt" file
with open("../mu_pdf.txt", "a") as f:
    np.savetxt(f, neutrino_pdfs_mu, delimiter=",")

# Append to the "mub_pdf.txt" file
with open("../mub_pdf.txt", "a") as f:
    np.savetxt(f, neutrino_pdfs_mub, delimiter=",")

with open("../pred.txt", "a") as f:
    np.savetxt(f, pred[0].detach().numpy().flatten(), delimiter=",")
