import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.legend_handler import HandlerTuple
import pandas as pd

# from read_fk_table import get_fk_table
import sys
import os

parent_dir = os.path.abspath(
    os.path.join(
        os.getcwd(),
        "/Users/jukkajohn/Masterscriptie/faser_nufluxes_ml/ML_fit_enu/src/ML_fit_neutrinos/",
    )
)
sys.path.append(parent_dir)
from read_faserv_pdf import read_pdf

# from read_LHEF import read_LHEF_data
from data_errors import compute_errors
from MC_data_reps import generate_MC_replicas
from postfit_criteria import Postfit
from form_loss_fct import complete_loss_fct, raw_loss_fct
from postfit_measures import Measures
from logspace_grid import generate_grid
# from rebin_fk_data import rebin_fk

# Define global variables
num_obs = 1  # between 1 and 4
REPLICAS = 100

# HYPERPARAMETERS
preproc = 1


fit_level = 1

# lr = 0.03
max_counter = 200
max_Nepochs = 3500

# events = [50, 97, 71, 69, 48, 27]
# minev = [5.9, 4.3, 2.5, 2.2, 3.7,5.1]
events = [223.16, 368.27, 258.92, 205.8, 108.74, 77.845]
# events = [33.6, 59.5, 51.6, 84.1, 50.1, 19.6]
std_errev = [0, 0, 0, 0, 0, 0]
# events, max_events, min_events, xvals_per_obs, binwidths, xlabels, events_per_obs, = read_LHEF_data()

sig_sys = std_errev
sig_sys = np.array(sig_sys)
# sig_stat =np.sqrt(events)
sig_stat = [72.011, 78.987, 64.535, 41.695, 24.934, 29.098]
sig_stat = np.array(sig_stat)
xlabels = ["mu"]
data = np.array(events)

xvals_per_obs = [100.0, 300.0, 600.0, 1000.0, -300.0, -100.0]
xvals_per_obs = np.array(xvals_per_obs)
xvals_per_obs /= 0.8

# Get errors
# sig_sys,sig_tot, cov_matrix = compute_errors(data,data_min,data_max)
# xvals_per_obs = [100,300,600,-100,-300,1000]
# xvals_per_obs = [100,300,600,100,-300,-100]
# cov_matrix = np.array([
#     [9.2,-0.32,  0.08, -0.03,  0.00,  0.00],
#     [-0.32, 10.2, -0.43,  0.10, -0.01, -0.00],
#     [ 0.08, -0.43, 9.6, -0.31,  0.04, -0.02],
#     [-0.03,  0.10, -0.31, 22.3,-0.14,  0.01],
#     [ 0.00, -0.01,  0.04, -0.14, 12.1,-0.24],
#     [ 0.00, -0.00, -0.02,  0.01, -0.24,7.5]
# ])
cov_matrix = np.array(
    [
        [5186, -1623, 340, -69, 2, 5],
        [-1623, 6239, -1952, 281, -19, -4],
        [340, -1952, 4165, -734, 56, -27],
        [-69, 281, -734, 1738, -130, 15],
        [2, -19, 56, -130, 622, -147],
        [5, -4, -27, 15, -147, 847],
    ]
)

filename = "data_muon_sim_faser/FK_Enu_7TeV_nu_W.dat"
file_path = os.path.join(parent_dir, filename)
df = pd.read_csv(file_path, sep="\s+", header=None)
fk_table_mu = df.to_numpy()

x_alpha = fk_table_mu[0, :]
x_alpha = x_alpha.reshape(len(x_alpha), 1)

# strip first row to get fk table
fk_table_mu = fk_table_mu[1:, :]
fk_table_mu = fk_table_mu[:-1, :]
print(fk_table_mu.shape)
# fk_table_mu = np.transpose(fk_table_mu)
fk_table_mu = np.linalg.pinv(fk_table_mu)

# cov_matrix = np.diag(sig_sys**2 + sig_stat**2)
# sig_sys = np.array(sig_sys)
# sig_sys = np.sqrt(np.diag(cov_matrix))
# sig_stat = 0
# sig_stat = np.sqrt(data)
# np.fill_diagonal(cov_matrix, sig_stat**2 + sig_sys**2)
cov_matrix = np.linalg.inv(cov_matrix)
cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32, requires_grad=False)
# Generate MC replicas of data
level0, level1, level2 = generate_MC_replicas(REPLICAS, data, sig_sys, sig_stat, 1)
# Get faserv pdf
pdf = "faserv"
lowx = -8
n = 250
x_vals = generate_grid(lowx, n)
faser_pdf, x_faser = read_pdf(pdf, x_vals, 14)

if fit_level == 0:
    pred = level0
if fit_level == 1:
    pred = level1
if fit_level == 2:
    pred = level2


l1 = 4
l2 = 4
l3 = 4


class SimplePerceptron(torch.nn.Module):
    def __init__(self, l1, l2, l3):
        super(SimplePerceptron, self).__init__()
        self.linear = torch.nn.Linear(1, l1)
        self.hidden = torch.nn.Linear(l1, l2)
        self.hidden2 = torch.nn.Linear(l2, l3)
        self.hidden3 = torch.nn.Linear(l3, 1)
        # self.relu = torch.nn.ReLU()
        self.relu = torch.nn.ReLU()

    def forward(self, y):
        y = self.linear(y)
        y = self.relu(y)
        y = self.hidden(y)
        y = self.relu(y)
        y = self.hidden2(y)
        y = self.relu(y)
        y = self.hidden3(y)
        y = self.relu(y)
        # y = torch.nn.functional.softplus(y)

        return y


import torch.nn as nn

if preproc == 1:

    class CustomPreprocessing(nn.Module):
        def __init__(self, alpha, beta, gamma):
            super(CustomPreprocessing, self).__init__()

            # self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32, requires_grad=True))
            # self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32, requires_grad=True))
            # self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32, requires_grad=True))

            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
            self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32))
            self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

        def forward(self, x):
            # alpha = (1 - 0.2)/2 *torch.tanh(self.alpha) + (1 + 0.2)/2
            # beta = (1.9 - 0.1)/2 *torch.tanh(self.alpha) + (1.9 + 0.1)/2
            # beta = abs(self.beta)
            # alpha = 1 - torch.nn.functional.softplus(self.alpha)
            # beta = torch.nn.functional.softmax(self.beta)
            return self.gamma * (1 - x) ** self.beta * x ** (1 - self.alpha)
            # return 10 * (1 - x)**2 * x**(1.5)
            # return  self.gamma*(1 - x) ** self.beta * x**(1-self.alpha)

    class PreprocessedMLP(nn.Module):
        def __init__(self, alpha, beta, gamma, l1, l2, l3):
            super(PreprocessedMLP, self).__init__()
            self.preprocessing = CustomPreprocessing(alpha, beta, gamma)
            self.mlp = SimplePerceptron(l1, l2, l3)

        def forward(self, x):
            f_preproc = self.preprocessing(x)
            f_NN = self.mlp(x)
            f_nu = f_preproc * f_NN
            return f_NN

        def neuralnet(self, x):
            f_NN = self.mlp(x)
            return f_NN

        def preproc(self, x):
            f_preproc = self.preprocessing(x)
            return f_preproc


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, pred, data, cov_matrix):
        loss = raw_loss_fct(pred, data, cov_matrix)
        return loss


# Define variables for fit
# x_alpha_more_bins = np.logspace(-5,0,250)
# x_alpha_more_bins = torch.tensor(x_alpha_more_bins, dtype=torch.float32).view(-1,1)
lowx = -8
n = 250
x_vals = generate_grid(lowx, n)
x_vals = torch.tensor(x_vals, dtype=torch.float32).view(-1, 1)
xvals_per_obs = torch.tensor(xvals_per_obs, dtype=torch.float32).view(-1, 1)
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
) = [], [], [], [], [], [], [], [], []

# xvals_per_obs_mu = [-1 / 100, -1 / 300, -1 / 600, 1 / 1000, 1 / 300, 1 / 100]
xvals_per_obs_mu = [1, 2, 3, 10, 4, 5]
xvals_per_obs_mub = [100, 300, 1000]
xvals_per_obs_mu = np.array(xvals_per_obs_mu)
xvals_per_obs_mub = np.array(xvals_per_obs_mub)
xvals_per_obs_mub = torch.tensor(xvals_per_obs_mub, dtype=torch.float32).view(-1, 1)
xvals_per_obs_mu = torch.tensor(xvals_per_obs_mu, dtype=torch.float32).view(-1, 1)

# x_alphas *=2
lr = 0.001
max_counter = 100
max_ep = 10000


def perform_fit(pred, REPLICAS):
    if preproc == 1:
        alpha, beta, gamma = 1, 1, 10
        model_mu = PreprocessedMLP(alpha, beta, gamma, l1, l2, l3)
    for i in range(REPLICAS):
        # model_mub = PreprocessedMLP(alpha, beta, gamma, l1, l2, l3)

        # model.load_state_dict(model_params)

        criterion = CustomLoss()
        # criterion = torch.nn.MSELoss()
        optimizer_mu = torch.optim.Adam(model_mu.parameters(), lr=lr, weight_decay=1e-3)
        # optimizer_mub = torch.optim.Adam(
        #     model_mub.parameters(), lr=lr, weight_decay=1e-2
        # )
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.3)

        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        losses = []
        pred[i] = pred[i].squeeze()
        pred[i] = pred[i]

        model_mu.train()
        best_loss = 1e13  # initial loss
        counter = 0
        # num_epochs = 4000
        num_epochs = 0
        # for ep in range(num_epochs):
        while counter < max_counter:
            num_epochs += 1
            if num_epochs > max_ep:
                break

            # num_epochs+=1
            optimizer_mu.zero_grad()
            # optimizer_mub.zero_grad()

            y_preds_mu = model_mu(xvals_per_obs_mu) * 150
            # y_preds_mub = model_mub(xvals_per_obs_mub) * 1

            y_pred_mu = y_preds_mu.squeeze()
            y_preds = y_pred_mu
            # y_pred_mub = y_preds_mub.squeeze()

            # y_pred_mu[-1] = y_pred_mu[-1] + y_pred_mub[-1]

            # y_pred_mub = y_pred_mub[:-1]

            # y_pred_mub = torch.flip(y_pred_mub, dims=[0])

            # y_preds = torch.hstack((y_pred_mu, y_pred_mub))
            # y_pred =  model(x_alphas) * x_alphas

            # last_point = model(torch.tensor([5*0.1], dtype=torch.float32).view(-1,1))
            small_x_point1 = model_mu(
                torch.tensor([6 * 10**-4], dtype=torch.float32).view(-1, 1)
            )[:, 0]
            small_x_point2 = model_mu(
                torch.tensor([5 * 10**-4], dtype=torch.float32).view(-1, 1)
            )[:, 0]
            small_x_point3 = model_mu(
                torch.tensor([5 * 10**-4], dtype=torch.float32).view(-1, 1)
            )[:, 0]

            loss = criterion(y_preds, pred[i], cov_matrix)
            # loss = criterion(y_preds, pred[i],cov_matrix,small_x_point1,small_x_point2,small_x_point3)
            # loss = (criterion(y_preds, pred[i]))
            loss.backward()
            if num_epochs % 100 == 0:
                print(best_loss, num_epochs)

            losses.append(loss.detach().numpy())
            optimizer_mu.step()
            # optimizer_mub.step()

            # if ep % 100 == 0:
            #     print(loss.detach().numpy())
            if abs(loss) < 0.04:
                break
            if loss < best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1

        if loss < 5:
            print(f"reduced chi^2 level 2 = {loss}")
            # print(f"reduced chi^2 level 1 = {red_chi_square_level1}")
            # print(f"Constrained alpha: {(model.preprocessing.alpha.item())}")
            # print(f"Constrained beta: {(model.preprocessing.beta.item())}")
            # print(f"Constrained gamma: {model.preprocessing.gamma.item()}")

            # save outcome of fit and its measures for postfit selection criteria
            chi_squares.append(loss.detach().numpy())
            # preproc_pdf = ((1 - x_alpha_more_bins) ** (beta) * x_alpha_more_bins ** (1-alpha)).detach().numpy().flatten()

            N_event_pred.append(y_preds.detach().numpy())

            xvals_per_obsa = np.array(xvals_per_obs.detach().numpy().flatten())
            sorted_indices = np.argsort(xvals_per_obsa)
            sorted_x_vals = xvals_per_obsa[sorted_indices]
            sorted_y_vals = y_preds.detach().numpy().flatten()[sorted_indices]
            sorted_preds = pred[i][sorted_indices]
            sorted_level0 = level0[i][sorted_indices]

            # plt.plot(sorted_x_vals,sorted_y_vals,label = 'nn')
            # plt.plot(sorted_x_vals,sorted_preds,label = 'level1')
            # plt.plot(sorted_x_vals,sorted_level0,label = 'level0')
            # plt.legend()
            # plt.show()
            # cont_pred = model(x_alpha_more_bins).detach().numpy().flatten()

            # plt.yscale('log')
            # print(f'counter = {counter}')
            # plt.plot(range(1, len(losses) + 1), losses)
            # plt.xlabel("#epochs")
            # plt.ylabel("loss")
            # plt.title("level 0 closure test, no preprocessing")
            # plt.show()
            # plt.plot(x_alpha_more_bins,preproc_pdf)
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.show()
    # return arc_lenghts, chi_squares,pos_penaltys,int_penaltys,N_event_pred, neutrino_pdfs,model
    return chi_squares, N_event_pred, neutrino_pdfs_mu, neutrino_pdfs_mub, model_mu


chi_squares, N_event_pred, neutrino_pdfs_mu, neutrino_pdfs_mub, model = perform_fit(
    pred, REPLICAS
)

x_vals = x_vals.detach().numpy().flatten()
faser_pdf_mu, x_faser = read_pdf(pdf, x_vals, 14)
faser_pdf_mub, x_faser = read_pdf(pdf, x_vals, -14)
# mean_fnu = np.mean(neutrino_pdfs,axis=0) *x_vals
mean_fnu_mu = np.mean(neutrino_pdfs_mu, axis=0)
mean_fnu_mub = np.mean(neutrino_pdfs_mub, axis=0)

mean_pdf_preproc = np.mean(preproc_pdfs, axis=0)
mean_nn_pdf = np.mean(nn_pdfs, axis=0)

xvals_per_obs = [100.0, 300.0, 600.0, 1000.0, -300.0, -100.0]
xvals_per_obs = np.array(xvals_per_obs)
xvals_per_obs /= 0.8


fig = plt.figure(figsize=(8, 5), dpi=300)  # 2 rows, 2 columns
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
gs.update(left=0.09, right=0.95, top=0.93, hspace=0.18)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],  # Computer Modern
    }
)

axR = fig.add_subplot(gs[0, 0])
axrR = fig.add_subplot(gs[1, 0])


simulated_Enu = pred
# errors_enu = [5186, 6239, 4165, 1738, 622, 847]
# errors_enu = np.array(errors_enu)
errors_enu = np.sqrt(sig_sys**2 + sig_stat**2)
simulated_Enu = level0[0].detach().numpy().flatten()
preds_Enu = np.mean(N_event_pred, axis=0)
pred_stds_Enu = np.std(N_event_pred, axis=0)
xplot_Enumu = np.array(
    [-1 / 100, -1 / 300, -1 / 600, -1 / 1000, 1 / 1000, 1 / 300, 1 / 100]
)
xplot_ticks = np.array(
    [-1 / 100, -1 / 300, -1 / 600, -1 / 1000, 1 / 1000, 1 / 300, 1 / 100]
)
# xplot_stretched = np.array([-1.0, -0.7, -0.4, -0.1, 0.1, 0.4, 1.0])

ticks = np.linspace(0, 1, len(xplot_ticks))
ticks = np.array(
    [
        0,
        0.07407407407407407,
        0.18518518518518517,
        0.3333333333333333,
        0.6666666666666666,
        0.9259259259259259,
        1,
    ]
)
binwidths = [200, 300, 400, 900, 700, 200]

xplot_Enumu = np.interp(xplot_Enumu, xplot_ticks, ticks)

Enumu_centers = 0.5 * (xplot_Enumu[1:] + xplot_Enumu[:-1])
Enumu_errors = 0.5 * (xplot_Enumu[1:] - xplot_Enumu[:-1])

# Enumu_centers_plot = np.interp(Enumu_centers, xplot_ticks, xplot_stretched)
# Enumu_errors_plot = np.interp(Enumu_centers + Enumu_errors, xplot_ticks, xplot_stretched) - Enumu_centers_plot

Enumu_centers_plot = Enumu_centers
Enumu_errors_plot = Enumu_errors


# =========== TOP RIGHT (Rates Enu vs FK otimes f_NN)

axRpred = axR.errorbar(
    Enumu_centers_plot,
    preds_Enu / binwidths,  # y values (measurements)
    yerr=pred_stds_Enu / binwidths,  # vertical error bars
    xerr=Enumu_errors_plot,  # horizontal error bars (bin widths)
    fmt="o",  # circle marker, set markersize to 0 to hide if needed
    markersize=3,
    color="black",
    capsize=1,  # no caps
    elinewidth=0.5,
    markeredgewidth=0.5,
    label=r"DATA$E_\nu$",
    zorder=3,
)

axRsim = axR.bar(
    Enumu_centers_plot,
    (2 * errors_enu) / binwidths,
    width=2 * Enumu_errors_plot,
    bottom=(simulated_Enu - errors_enu) / binwidths,
    color="none",  # transparent fill
    #        edgecolor='gray',
    hatch="\\\\\\\\",  # diagonal hatch
    linewidth=0.8,
    edgecolor="tab:orange",
    alpha=0.4,
    zorder=1,  # behind other plots
)


# (axRpred,) = axR.plot(
#     xvals_per_obs,
#     preds_Enu,
#     color="green",
#     drawstyle="steps-post",
#     alpha=0.8,
#     label=r"$\mathrm{NN}(E_\nu)$",
# )
# axRprederr = axR.fill_between(
#     xvals_per_obs,
#     (preds_Enu + pred_stds_Enu),
#     (preds_Enu - pred_stds_Enu),
#     color="green",
#     alpha=0.2,
#     step="post",
#     label=r"$\pm 1\sigma$",
# )
axR.legend(
    [(axRsim), (axRpred)],
    [
        r"$\mathrm{DATA} \ E\nu$",
        r"$\mathrm{FK} \otimes f_{\mathrm{NN}}(x_\nu)$",
        # r"$\mathcal{A}_{\mathcal{fit}} \ \cdot \ N_{\mathrm{int}}(E_\nu)$",
    ],
    handler_map={tuple: HandlerTuple(ndivide=1)},
    loc="upper right",
).set_alpha(0.8)
# axR.set_xlim(0)
# axR.set_ylim(0)
axR.set_xlim(0, 1)
axR.set_ylim(0)
# axR.grid(color='grey', linestyle='-', linewidth=0.25)
axR.set_xticklabels([])
axR.set_xticks(ticks)
axR.axvline(
    x=np.interp(-1 / 1000, xplot_ticks, ticks),
    color="black",
    linestyle="-",
    linewidth=1,
    alpha=0.8,
)
axR.axvline(
    x=np.interp(1 / 1000, xplot_ticks, ticks),
    color="black",
    linestyle="-",
    linewidth=1,
    alpha=0.8,
)

axR.set_title(r"$\mathcal{L}_{\mathrm{pp}} = 65.6 \mathrm{fb}^{-1}$", loc="right")
axR.set_title(r"$\  \mathrm{FASER}\nu  \mathrm{Level\ 1}$", loc="left")
axR.text(-400, 30, r"$\nu_{\mu(\bar{\mu})} + W \rightarrow X_h+  \mu^{\pm} $")
# axR.set_title(r"$\mathcal{L}_{\mathrm{pp}} = 150 \mathrm{fb}^{-1}$", loc="right")
# axR.set_title(r"$\mathrm{FASER}\nu, \ \mathrm{Level\ 2}$", loc="left")
# axR.text(800, 400, r"$\nu_\mu W \rightarrow X_h \mu^- $")
axR.set_ylabel(r"$N_{\mathrm{int}}$")


ratio_center_pred = preds_Enu / simulated_Enu
ratio_lower_pred = (preds_Enu - pred_stds_Enu) / simulated_Enu
ratio_upper_pred = (preds_Enu + pred_stds_Enu) / simulated_Enu
ratio_upper_sim = (simulated_Enu + errors_enu) / simulated_Enu
ratio_lower_sim = (simulated_Enu + errors_enu) / simulated_Enu

axrRmeasmu = axrR.errorbar(
    Enumu_centers_plot,  # x values (bin centers)
    # np.ones_like(simulated_Enu),
    preds_Enu / simulated_Enu,  # y values (measurements)
    yerr=pred_stds_Enu / simulated_Enu,  # vertical error bars
    xerr=Enumu_errors_plot,  # horizontal error bars (bin widths)
    fmt="o",  # circle marker, set markersize to 0 to hide if needed
    markersize=3,
    color="black",
    capsize=1,  # no caps
    elinewidth=0.5,
    markeredgewidth=0.5,
    label=r"DATA$E_\nu$",
    zorder=3,
)

axrRsimerr = axrR.bar(
    Enumu_centers_plot,
    2 * errors_enu / simulated_Enu,  # full height (± error)
    width=2 * Enumu_errors_plot,
    bottom=(simulated_Enu - errors_enu) / simulated_Enu,  # center the bar at the value
    color="none",
    edgecolor="tab:orange",
    alpha=0.8,
    hatch="\\\\\\\\",
    linewidth=0.4,
    zorder=1,
)

axrR.set_ylabel(r"$\mathrm{Ratio}$")
axrR.set_xlabel(r"$E_\nu \ [\mathrm{GeV}]$")
# axrR.set_ylim(0.5, 1.5)
# axrR.set_xlim(0)
axrR.grid(color="grey", linestyle="-", linewidth=0.25)
tick_labels = [
    r"$-\frac{1}{100}$",
    r"$-\frac{1}{300}$",
    r"$-\frac{1}{600}$",
    r"$-\frac{1}{1000}$",
    r"$\frac{1}{1000}$",
    r"$\frac{1}{300}$",
    r"$\frac{1}{100}$",
]

axrR.set_ylabel(r"$\mathrm{Ratio}$")
axrR.set_xlabel(r"$q/E_\nu \ [\mathrm{1/GeV}]$")
axrR.set_ylim(0, 2)
axrR.set_xlim(0, 1)
axrR.set_xticks(ticks)
axrR.set_xticklabels(tick_labels)

plt.tight_layout()
plt.subplots_adjust(bottom=0.13)

plt.savefig("no_FK_ML_faserdat.pdf")
plt.show()
