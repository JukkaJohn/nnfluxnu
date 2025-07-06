import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
import os
from form_loss_fct import complete_loss_fct, raw_loss_fct
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(
    os.path.join(
        os.getcwd(),
        "/data/theorie/jjohn/git/faser_nufluxes_ml/ML_fit_enu/src/ML_fit_neutrinos",
    )
)
# act_functions, num_nodes, num_layers, preproc = 1, 1, 1, True


class SimplePerceptron(torch.nn.Module):
    def __init__(self, act_functions, num_nodes, num_layers):
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
        layers.append(act_functions[1])
        self.layers = nn.Sequential(*layers)

        # def _initialize_weights(self):

    # for layer in self.layers:
    #    if isinstance(layer, nn.Linear):
    #       nn.init.uniform_(layer.weight, a=-10, b=10)  # adjust a, b as needed
    #      if layer.bias is not None:
    #         nn.init.uniform_(layer.bias, a=-10, b=10)

    def forward(self, x):
        return self.layers(x)


# if not preproc:


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
        self, alpha, beta, gamma, act_functions, num_nodes, num_layers, preproc
    ):
        super(PreprocessedMLP, self).__init__()

        self.preprocessing = CustomPreprocessing(alpha, beta, gamma, preproc)
        self.mlp = SimplePerceptron(act_functions, num_nodes, num_layers)
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
    def __init__(self, extended_loss):
        super(CustomLoss, self).__init__()
        self.extended_loss = extended_loss

    def forward(self, pred, data, cov_matrix, small_x_point1, small_x_point2, model):
        if self.extended_loss:
            complete_loss_fct(
                pred, data, cov_matrix, small_x_point1, small_x_point2, model
            )
        else:
            loss = raw_loss_fct(pred, data, cov_matrix)
        return loss


(
    neutrino_pdfs_mu,
    neutrino_pdfs_mub,
    N_event_pred_mu,
    N_event_pred_mub,
    arc_lenghts,
    chi_squares,
    int_penaltys,
    pos_penaltys,
    preproc_pdfs,
    nn_pdfs,
    chi_square_for_postfit,
    validation_losses,
    k_fold_losses,
) = [], [], [], [], [], [], [], [], [], [], [], [], []


def perform_fit(
    pred,
    REPLICAS,
    range_alpha,
    range_beta,
    range_gamma,
    lr,
    wd,
    max_counter,
    x_alphas,
    fk_tables_mu,
    fk_tables_mub,
    binwidths_mu,
    binwidths_mub,
    cov_matrix,
    extended_loss,
    act_functions,
    num_nodes,
    num_layers,
    x_vals,
    preproc,
    validation,
    seed,
    max_num_epochs,
):
    x_vals = torch.tensor(x_vals, dtype=torch.float32).view(-1, 1)

    orig_pred = pred[0].squeeze()
    # num_folds = int(pred[0] / 2)

    # N-1 to fit
    # 1 for loss in the stopping algorithm
    indices = np.arange(pred[0].shape[0])
    k = 3
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Store folds as a list of (train_index, test_index) tuples
    folds = []
    for train_index, test_index in kf.split(indices):
        folds.append((train_index, test_index))

    print("Fold 1:")
    print("Train indices:", folds[0][0])
    print("Test indices:", folds[0][1])

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
            alpha, beta, gamma, act_functions, num_nodes, num_layers, preproc=preproc
        )

        criterion = CustomLoss(extended_loss=extended_loss)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        losses = []

        model.train()
        best_loss = 1e13  # initial loss
        counter = 0
        num_epochs = 0

        nbatch = 40
        training_length = 0

        while counter < max_counter:
            # print(max_counter, counter)
            if max_num_epochs < training_length:
                break

            # if training_length % 1 == 0:
            #     perm = torch.randperm(x_alphas.size(0))
            #     x_alphas = x_alphas[perm]
            #     _, unperm = perm.sort()
            training_length += 1

            optimizer.zero_grad()
            y_pred = model(x_alphas)

            # y_pred[:, 0] = y_pred[:, 0][unperm]
            # y_pred[:, 1] = y_pred[:, 1][unperm]
            y_pred_mu = (
                torch.matmul(fk_tables_mu, y_pred[:, 0]) * binwidths_mu.flatten()
            )
            y_pred_mub = (
                torch.matmul(fk_tables_mub, y_pred[:, 1]) * binwidths_mub.flatten()
            )

            y_pred_mu = y_pred_mu.squeeze()
            y_pred_mub = y_pred_mub.squeeze()

            y_preds = torch.hstack((y_pred_mu, y_pred_mub))
            # y_preds = y_pred_mu + y_pred_mub  # torch hstack
            # y_preds = y_pred_mu

            # y_shuffle = y_preds[shuffle]
            # pred_shuffle = pred[i][shuffle]
            # cov_matrix_shuffle = cov_matrix[shuffle][:, shuffle]

            x_int = torch.tensor([1e-4, 1.0], dtype=torch.float32).view(-1, 1)
            y_int_mu = model(x_int)[:, 0]
            y_int_mub = model(x_int)[:, 1]

            y_train = y_preds[train_indices]
            pred_train = pred[0][train_indices]
            cov_matrix_train = cov_matrix[train_indices][:, train_indices]

            loss = criterion(
                y_train,
                pred_train,
                cov_matrix_train,
                y_int_mu,
                y_int_mub,
                x_int,
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
                x_int,
            )

            if training_length % 100 == 0:
                # print(f"training loss = {loss_batch.item()}", training_length)
                print(f" loss = {loss.item()}", training_length, best_loss.item())

                # print(f"val loss = {loss_val.item()}")
                chi_squares.append(loss.detach().numpy())

            losses.append(loss.detach().numpy())
            optimizer.step()

            if loss_val < best_loss:
                best_loss = loss_val
                counter = 0
            else:
                counter += 1

        print(best_loss)

        y_k_fold = y_preds[k_fold_indices]
        pred_k_fold = pred[0][k_fold_indices]
        cov_matrix_k_fold = cov_matrix[k_fold_indices][:, k_fold_indices]

        loss_k_fold = criterion(
            y_k_fold,
            pred_k_fold,
            cov_matrix_k_fold,
            y_int_mu,
            y_int_mub,
            x_int,
        )

        k_fold_loss = loss_k_fold.item()
        k_fold_losses.append(k_fold_loss)
        print("k_fold_losses")
        print(k_fold_losses)

        # unnorm_pred = y_preds.detach().numpy()

        # #     # unnorm_data = pred[i] / 1000 * (max_val - min_val) + min_val
        # unnorm_data = orig_pred
        # plt.plot(
        #     np.arange(len(y_preds.detach().numpy().flatten())),
        #     unnorm_pred,
        #     label="NN",
        # )
        # plt.plot(
        #     np.arange(len(unnorm_data.detach().numpy().flatten())),
        #     unnorm_data.detach().numpy().flatten(),
        #     label="pseudo data",
        # )
        # plt.legend()
        # plt.yscale("log")
        # plt.show()
        # # print(y_pred_mu[10:40])
        # # print(y_preds)
        # # print(pred[i])
        # # print(y_preds - pred[i])

        # # print(f"rep {i + 1} done out of {REPLICAS}")
        # print(f"reduced chi^2 level 2 = {loss}")

        # print(f"Constrained alpha: {(model.preprocessing.alpha.item())}")
        # print(f"Constrained beta: {(model.preprocessing.beta.item())}")
        # print(f"Constrained gamma: {model.preprocessing.gamma.item()}")

        # f_nu_mub = model(x_vals)[:, 1].detach().numpy().flatten()
        # f_nu_mu = model(x_vals)[:, 0].detach().numpy().flatten()

        # plt.plot(x_vals.detach().numpy().flatten(), f_nu_mu, label="mu")
        # plt.plot(x_vals.detach().numpy().flatten(), f_nu_mub, label="mub")
        # plt.xscale("log")
        # plt.yscale("log")
        # plt.xlim(0, 1)
        # plt.ylim(10**-1, 10**8)
        # plt.legend()
        # plt.show()

        # chi_square_for_postfit.append(loss.detach().numpy())

        # preproc_pdfs.append(model.preproces(x_vals).detach().numpy().flatten())
        # nn_pdf = model.neuralnet(x_vals)[:, 0].detach().numpy().flatten()
        # nn_pdfs.append(nn_pdf)

        # N_event_pred_mu.append(
        #     y_pred_mu.detach().numpy()
        #     # / 1000
        #     # * (
        #     #     max_val.detach().numpy().flatten()
        #     #     - min_val.detach().numpy().flatten()
        #     # )
        #     # + min_val.detach().numpy().flatten()
        # )
        # N_event_pred_mub.append(y_pred_mub.detach().numpy())

        # neutrino_pdfs_mu.append(
        #     f_nu_mu
        #     # / 1000
        #     # * (
        #     #     max_val.detach().numpy().flatten()
        #     #     - min_val.detach().numpy().flatten()
        #     # )
        #     # + min_val.detach().numpy().flatten()
        # )
        # neutrino_pdfs_mub.append(
        #     f_nu_mub
        #     # / 1000
        #     # * (
        #     #     max_val.detach().numpy().flatten()
        #     #     - min_val.detach().numpy().flatten()
        #     # )
        #     # + min_val.detach().numpy().flatten()
        # )
    return np.mean(k_fold_losses)
