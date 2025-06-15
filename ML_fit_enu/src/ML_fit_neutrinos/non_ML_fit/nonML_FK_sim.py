import sys

sys.path.append(
    "/Users/jukkajohn/Masterscriptie/faser_nufluxes_ml/ML_fit_enu/src/ML_fit_neutrinos"
)
import numpy as np
from matplotlib import gridspec
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar
from scipy.optimize import minimize
import pandas as pd
from data_errors import compute_errors
from read_faserv_pdf import read_pdf
from logspace_grid import generate_grid
import torch
from scipy.optimize import differential_evolution
import lhapdf

simcolor = "tab:red"
mucolor = "tab:blue"
mubcolor = "tab:blue"

lhapdf.setVerbosity(0)


def get_fk_table(filename):
    """This function reads the fk table for the neutrino flux and pads them for computational efficiency later on

    Returns:
        tuple: x_alphas(grid points) and the fk table in tensor to fit torch
    """

    file_path = f"data_muon_sim_faser/{filename}.dat"
    df = pd.read_csv(file_path, sep="\s+", header=None)
    fk_table = df.to_numpy()

    x_alpha = fk_table[0, :]
    x_alpha = x_alpha.reshape(len(x_alpha), 1)

    # strip first row to get fk table
    fk_table = fk_table[1:, :]

    x_alpha = torch.tensor(x_alpha, dtype=torch.float32).view(-1, 1)
    fk_table = torch.tensor(fk_table, dtype=torch.float32)

    return x_alpha, fk_table


filename = "FK_Enu_7TeV_nu_W"
x_alphas, fk_tables_mu = get_fk_table(filename=filename)
filename = "FK_Enu_7TeV_nub_W"
x_alphas, fk_tables_mub = get_fk_table(filename=filename)


xvals_per_obs_mu = [100, 300, 600, 1000]

xvals_per_obs = [100, 300, 600, 1000, -300, -100]
xvals_per_obs_mub = [-100, -300, -1000]

pdf = "FASERv_EPOS+POWHEG_7TeV"
binwidths_mu = [200, 300, 400, 900]
binwidths_mu = torch.tensor(binwidths_mu, dtype=torch.float32).view(-1, 1)
faser_bins_mu = [300, 600, 1000, 1900]
faser_pdf, x_faser = read_pdf(pdf, x_alphas.flatten(), 14)
faser_pdf = torch.tensor(faser_pdf, dtype=torch.float32).view(-1, 1)
data_mu = torch.matmul(fk_tables_mu, faser_pdf) * binwidths_mu * 65.6 / 150
data_mu = data_mu.detach().numpy().flatten()
data_max_mu = data_mu + data_mu / 20
data_min_mu = data_mu - data_mu / 20

binwidths_mub = [200, 700, 900]
faser_pdf, x_faser = read_pdf(pdf, x_alphas.flatten(), -14)
faser_pdf = torch.tensor(faser_pdf, dtype=torch.float32).view(-1, 1)
binwidths_mub = torch.tensor(binwidths_mub, dtype=torch.float32).view(-1, 1)
data_mub = torch.matmul(fk_tables_mub, faser_pdf) * binwidths_mub * 65.6 / 150
data_mub = data_mub.detach().numpy().flatten()
data_max_mub = data_mub + data_mub / 20
data_min_mub = data_mub - data_mub / 20
faser_bins_mub = [300, 1000, 1900]

xvals_per_obs_mu = [100, 300, 600, 1000]

xvals_per_obs = [100, 300, 600, 1000, -300, -100]
xvals_per_obs_mub = [-100, -300, -1000]
xlabels = ["Enu"]


xvals_per_obs_mu = [100, 300, 600, 1000]

xvals_per_obs = [100, 300, 600, 1000, -300, -100]
xvals_per_obs_mub = [-100, -300, -1000]


def combine_mu_mub(mu, mub):
    mu[-1] = mu[-1] + mub[-1]
    mub = mub[:-1]
    mub = mub[::-1]
    combined = np.hstack((mu, mub))
    return combined


sig_sys_mu, sig_tot_mu, cov_matrix_mu = compute_errors(
    data_mu, data_min_mu, data_max_mu
)
sig_sys_mub, sig_tot_mub, cov_matrix_mub = compute_errors(
    data_mub, data_min_mub, data_max_mub
)
# Combine data and errors
sig_stat_mu = np.sqrt(data_mu)
sig_stat_mub = np.sqrt(data_mub)
sig_stat = combine_mu_mub(sig_stat_mu, sig_stat_mub)

sig_sys = combine_mu_mub(sig_sys_mu, sig_sys_mub)

data = combine_mu_mub(data_mu, data_mub)

sig_tot = sig_stat**2 + sig_sys**2
cov_matrix = np.diag(sig_tot)
cov_matrix = np.linalg.inv(cov_matrix)


REPS = 100
xvals_per_obs = [100, 300, 600, 1000, -300, -100]
xvals_per_obs = np.array(xvals_per_obs)
pred = data
pred = np.array(pred)
cov = cov_matrix
cov = torch.tensor(cov, dtype=torch.float32, requires_grad=False)
params = []
f_mus = []
f_mubs = []
N_events = []
f_ref_mu, _ = read_pdf(pdf, x_alphas.flatten(), 14)
f_ref_mub, _ = read_pdf(pdf, x_alphas.flatten(), -14)
f_ref_mu = torch.tensor(f_ref_mu * 65.6 / 150, dtype=torch.float32).view(-1, 1)
f_ref_mub = torch.tensor(f_ref_mub * 65.6 / 150, dtype=torch.float32).view(-1, 1)

x = [100, 300, 600, 1000, 300, 100]
x = np.array(x)
x = x / 14000
x = x_alphas
# x_cont = generate_grid(-8, 250)
# x_cont = np.array(x_cont)
x_cont = np.logspace(-5, 0, 1000)
chi_squares = []
f_ref_mub_cont, _ = read_pdf(pdf, x_cont.flatten(), -14)
f_ref_mu_cont, _ = read_pdf(pdf, x_cont.flatten(), 14)
f_ref_mu_cont = f_ref_mu_cont * 65.6 / 150
f_ref_mub_cont = f_ref_mub_cont * 65.6 / 150
for _ in range(REPS):
    rng_level1 = np.random.default_rng(seed=42)
    r_sys_1 = rng_level1.normal(0, 1, len(data)) * sig_sys
    r_stat_1 = rng_level1.normal(0, 1, len(data)) * sig_stat
    level1 = pred + r_sys_1 + r_stat_1
    r_sys_2 = np.random.normal(0, 1, len(pred)) * sig_sys
    r_stat_2 = np.random.normal(0, 1, len(pred)) * sig_stat
    level2 = level1 + r_sys_2 + r_stat_2
    # level1 = pred
    # level2 = pred
    level2 = torch.tensor(level2, dtype=torch.float32).view(-1, 1)
    # level1 = pred

    def chi_square(parr):
        parr = torch.tensor(parr)
        # print(parr)
        # print(parr.shape)
        # print(parr[0])
        f_mu = parr[0] * x ** (parr[1] - 1) * ((1 - x) ** (parr[2] - 1)) * f_ref_mu
        f_mub = parr[0] * x ** (parr[1] - 1) * ((1 - x) ** (parr[2] - 1)) * f_ref_mub

        # print(f_mu.shape)

        # f_mu = torch.tensor(f_mu, dtype=torch.float32).view(-1, 1)
        # f_mub = torch.tensor(f_mub, dtype=torch.float32).view(-1, 1)

        y_pred_mu = torch.matmul(fk_tables_mu, f_mu) * binwidths_mu
        y_pred_mub = torch.matmul(fk_tables_mub, f_mub) * binwidths_mub

        # print(y_pred_mu.shape)

        y_pred_mu = y_pred_mu.squeeze()
        y_pred_mub = y_pred_mub.squeeze()

        y_pred_mu[-1] = y_pred_mu[-1] + y_pred_mub[-1]

        y_pred_mub = y_pred_mub[:-1]

        y_pred_mub = torch.flip(y_pred_mub, dims=[0])

        data = torch.hstack((y_pred_mu, y_pred_mub))

        # print(data.shape)

        # print(data)
        # print(level1)
        # # level1 = level1.squeeze()
        # print(level1.shape)
        # print(data.shape)
        diff = level2.squeeze() - data
        # print(diff.shape)
        loss = torch.matmul(cov, diff)
        # print(loss.shape)
        # loss = (1 / pred.size(0)) * torch.dot(diff.view(-1), loss.view(-1)) + last_point * 1
        loss = (1 / level2.size(0)) * torch.dot(diff.view(-1), loss.view(-1))
        # loss = sum(diff**2)
        # print(loss.shape)
        # print(loss)

        # print(level1_data.shape)
        # print(data.shape)
        # data = data.detach().numpy()
        # r = level1 - data
        # # print(r.shape)
        # # print(data)
        # r = r.reshape(len(r), 1)
        # # print(parr)
        # # print(cov.shape)
        # print(f"chi square = {np.dot(r.T, np.dot(cov, r))}")
        # diff = level1 - pred
        # print(diff**2)
        # return np.dot(r.T, np.matmul(cov, r))
        return loss

    x0 = np.random.uniform(0, 2, 3)
    bounds = [(0, 20), (-10, 10), (1, 10)]  # Set appropriate bounds
    # options = {
    #     "tol": 1e-6,  # Stop when function value change is below this
    #     "maxiter": 100,  # Stop after 100 iterations
    # }

    # result = minimize(chi_square, x0, method="BFGS", bounds=bounds)
    result = minimize(chi_square, x0, method="POWELL", bounds=bounds)
    # result = minimize(chi_square, x0)
    print("Optimized chi-square:", chi_square(result.x))
    print(result.x)
    # if 2 < sum(result.x) < 4:
    if chi_square(result.x) < 3.5:
        f_mu = (
            abs(result.x[0])
            * x_cont ** (result.x[1] - 1)
            * ((1 - x_cont) ** (result.x[2] - 1))
            * f_ref_mu_cont.flatten()
        )
        f_mub = (
            abs(result.x[0])
            * x_cont ** (result.x[1] - 1)
            * ((1 - x_cont) ** (result.x[2] - 1))
            * f_ref_mub_cont.flatten()
        )
        params.append((abs(result.x[0]), result.x[1], result.x[2]))

        f_mus.append(f_mu)
        f_mubs.append(f_mub)

        f_mu = (
            abs(result.x[0])
            * x ** (result.x[1] - 1)
            * ((1 - x) ** (result.x[2] - 1))
            * f_ref_mu
        )
        f_mub = (
            abs(result.x[0])
            * x ** (result.x[1] - 1)
            * ((1 - x) ** (result.x[2] - 1))
            * f_ref_mub
        )
        y_pred_mu = torch.matmul(fk_tables_mu, f_mu) * binwidths_mu
        y_pred_mub = torch.matmul(fk_tables_mub, f_mub) * binwidths_mub

        # print(y_pred_mu.shape)

        y_pred_mu = y_pred_mu.squeeze()
        y_pred_mub = y_pred_mub.squeeze()

        y_pred_mu[-1] = y_pred_mu[-1] + y_pred_mub[-1]

        y_pred_mub = y_pred_mub[:-1]

        y_pred_mub = torch.flip(y_pred_mub, dims=[0])

        data = torch.hstack((y_pred_mu, y_pred_mub))

        N_events.append(data)
        chi_squares.append(chi_square(result.x))
print(f"mean chi square = {np.mean(chi_squares)}")
mean_params = np.mean(params, axis=0)
std_params = np.std(params, axis=0)
print(f"mean = {np.mean(params, axis=0)}")
print(f" std = {np.std(params, axis=0)}")

mean_fnu = np.mean(f_mus, axis=0)
std_fmus = np.std(f_mus, axis=0)
error_fnu_max = mean_fnu + std_fmus
error_fnu_min = mean_fnu - std_fmus

mean_fnub = np.mean(f_mubs, axis=0)
std_fmubs = np.std(f_mubs, axis=0)
error_fnub_max = mean_fnub + std_fmubs
error_fnub_min = mean_fnub - std_fmubs
# error_fnub_max = error_fnu_max
preds_Enu = np.mean(N_events, axis=0)
pred_stds_Enu_max = np.std(N_events, axis=0) + preds_Enu
pred_stds_Enu_min = -np.std(N_events, axis=0) + preds_Enu

x_vals = x.flatten()

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],  # Computer Modern
    }
)
fig = plt.figure(figsize=(12.8, 4.0), dpi=300)  # 2 rows, 3 columns
gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])
gs.update(left=0.05, right=0.97, top=0.92, hspace=0.18, wspace=0.20)

axL = fig.add_subplot(gs[0, 0])
axM = fig.add_subplot(gs[0, 1])
axR = fig.add_subplot(gs[0, 2])
axrL = fig.add_subplot(gs[1, 0])
axrM = fig.add_subplot(gs[1, 1])
axrR = fig.add_subplot(gs[1, 2])

# ======== TOP LEFT (Main plot, f_NN vs f_FASERv ) =============


# mean_fnu = mean_fnu.detach().numpy().flatten()
# error_fnu_max = error_fnu_max.detach().numpy().flatten()
# error_fnu_min = error_fnu_min.detach().numpy().flatten()
# mean_fnub = mean_fnub.detach().numpy().flatten()
# error_fnub_max = error_fnub_max.detach().numpy().flatten()
# error_fnub_min = error_fnub_min.detach().numpy().flatten()


(axLsim,) = axL.plot(
    x_cont,
    f_ref_mu_cont,
    color=simcolor,
    linestyle="-",
    label=r"$f_{\mathrm{FASER}\nu}(x)$",
)

# (axLsim_mub,) = axL.plot(
#     x_cont,
#     f_ref_mub_cont,
#     color="blue",
#     linestyle="-",
#     label=r"$f_{\mathrm{FASER}\nu}(x)$",
# )

axLnnerr = axL.fill_between(
    x_cont,
    error_fnu_max,
    error_fnu_min,
    color=mucolor,
    alpha=0.2,
    label=r"$\pm 1\sigma$",
)

(axLnn,) = axL.plot(
    x_cont, mean_fnu, linestyle="-", color=mucolor, label=r"$f_{\mathrm{NN}}(x)$"
)

# axLnnerrmub = axL.fill_between(
#     x_cont,
#     error_fnub_max,
#     error_fnub_min,
#     color="green",
#     alpha=0.2,
#     label=r"$\pm 1\sigma$",
# )

# (axLnnmub,) = axL.plot(
#     x_cont, mean_fnub, linestyle="-", color="green", label=r"$f_{\mathrm{NN}}(x)$"
# )

axL.set_xlim(1e-4, 1)
axL.set_ylim(1e-2, 1e5)
axL.set_yscale("log")
axL.set_xscale("log")
# axL.set_title(
#     r"$f_{\mathrm{NN}}(x) = \mathcal{A} \ x^{1-\alpha}(1-x)^\beta \ \mathrm{NN}(x)$"
# )
axL.set_ylabel(r"$f_{\nu_\mu}(x_\nu)$")
axL.set_xticklabels([])
axL.grid(color="grey", linestyle="-", linewidth=0.25)
axL.legend(
    [(axLsim), (axLnn, axLnnerr)],
    [
        r"$f_{\mathrm{ref}\nu}(x_\nu)$",
        # r"$f_{\mathrm{FASER}\nu_{\bar{\mu}}}(x_\nu)$",
        r"$\mathcal{A} \cdot x^{\alpha-1} \cdot (1-x)^{\beta-1} \cdot f_{\mathrm{ref}\nu_\mu}(x_\nu)$",
        # r"$\mathcal{A} \cdot x^{\alpha-1} \cdot (1-x)^{\beta-1} \cdot f_{\mathrm{FASER}\nu_{\bar{\mu}}}(x_\nu)$",
    ],
    handler_map={tuple: HandlerTuple(ndivide=1)},
    loc="lower left",
).set_alpha(0.8)

# ======== TOP MIDDLE (Main plot, f_NN vs f_FASERv ) =============


(axMsim_mub,) = axM.plot(
    x_cont,
    f_ref_mub_cont,
    color=mubcolor,
    linestyle="-",
    label=r"$f_{\mathrm{FASER}\nu}(x)$",
)


axMnnerrmub = axM.fill_between(
    x_cont,
    error_fnub_max,
    error_fnub_min,
    color=mubcolor,
    alpha=0.2,
    label=r"$\pm 1\sigma$",
)

(axMnnmub,) = axM.plot(
    x_cont, mean_fnub, linestyle="-", color=simcolor, label=r"$f_{\mathrm{NN}}(x)$"
)

axM.set_xlim(1e-4, 1)
axM.set_ylim(1e-2, 1e5)
axM.set_yscale("log")
axM.set_xscale("log")
# axL.set_title(
#     r"$f_{\mathrm{NN}}(x) = \mathcal{A} \ x^{1-\alpha}(1-x)^\beta \ \mathrm{NN}(x)$"
# )
axM.set_ylabel(r"$f_{\nu_\mu}(x_\nu)$")
axM.set_xticklabels([])
axM.grid(color="grey", linestyle="-", linewidth=0.25)
axM.legend(
    [(axMsim_mub), (axMnnmub, axMnnerrmub)],
    [
        # r"$f_{\mathrm{ref}\nu}(x_\nu)$",
        r"$f_{\mathrm{ref}\nu_{\bar{\mu}}}(x_\nu)$",
        # r"$\mathcal{A} \cdot x^{\alpha-1} \cdot (1-x)^{\beta-1} \cdot f_{\mathrm{ref}\nu_\mu}(x_\nu)$",
        r"$\mathcal{A} \cdot x^{\alpha-1} \cdot (1-x)^{\beta-1} \cdot f_{\mathrm{ref}\nu_{\bar{\mu}}}(x_\nu)$",
    ],
    handler_map={tuple: HandlerTuple(ndivide=1)},
    loc="lower left",
).set_alpha(0.8)

# ========== BOTTOM LEFT (Ratio plot, f_NN vs f_FASERv )

ratio_center = mean_fnu / f_ref_mu_cont.flatten()
ratio_center_mub = mean_fnub / f_ref_mub_cont.flatten()
ratio_lower_mu = error_fnu_min / f_ref_mu_cont.flatten()
ratio_upper_mu = error_fnu_max / f_ref_mu_cont.flatten()
ratio_lower_mub = error_fnub_min / f_ref_mub_cont.flatten()
ratio_upper_mub = error_fnub_max / f_ref_mub_cont.flatten()

axrL.plot(x_cont, ratio_center, linestyle="-", color=mucolor)
axrL.fill_between(x_cont, ratio_upper_mu, ratio_lower_mu, color=mucolor, alpha=0.2)
axrL.plot(x_cont, np.ones(len(x_cont)), linestyle="-", color=simcolor)


# axrL.plot(x_cont, ratio_center_mub, linestyle="-", color="green")
# axrL.fill_between(x_cont, ratio_upper_mub, ratio_lower_mub, color="green", alpha=0.2)
# axrL.plot(x_cont, np.ones(len(x_cont)), linestyle="-", color="tab:blue")

axrL.set_xscale("log")
axrL.set_xlim(1e-5, 1)
axrL.set_ylim(0, 2)
axrL.grid(color="grey", linestyle="-", linewidth=0.25)
axrL.set_ylabel(r"$\mathrm{Ratio}$")
axrL.set_xlabel(r"$x_\nu$")

# ========== BOTTOM MIDDLE (Ratio plot, f_NN vs f_FASERv )


axrM.plot(x_cont, ratio_center_mub, linestyle="-", color=mubcolor)
axrM.fill_between(x_cont, ratio_upper_mub, ratio_lower_mub, color=mubcolor, alpha=0.2)
axrM.plot(x_cont, np.ones(len(x_cont)), linestyle="-", color=simcolor)

axrM.set_xscale("log")
axrM.set_xlim(1e-5, 1)
axrM.set_ylim(0, 2)
axrM.grid(color="grey", linestyle="-", linewidth=0.25)
axrM.set_ylabel(r"$\mathrm{Ratio}$")
axrM.set_xlabel(r"$x_\nu$")


x_vals = x.flatten()
simulated_Enu = pred
errors_enu = np.sqrt(sig_tot)
# errors_enu = [5186, 6239, 4165, 1738, 622, 847]
# errors_enu = np.array(errors_enu)
# errors_enu = np.sqrt(errors_enu)

xvals_per_obs = [-1500, -1100, -600, 0.0, 1200, 1900, 2300]
# sorted_indices = np.argsort(xvals_per_obs)
# xvals_per_obs = xvals_per_obs[sorted_indices]
# simulated_Enu = simulated_Enu[sorted_indices]
# errors_enu = errors_enu[sorted_indices]
# preds_Enu = preds_Enu[sorted_indices]
# pred_stds_Enu_min = pred_stds_Enu_min[sorted_indices]
# pred_stds_Enu_max = pred_stds_Enu_max[sorted_indices]

# print(len(sorted_indices))

# xvals_per_obs = np.insert(xvals_per_obs, -1, xvals_per_obs[-1])
# simulated_Enu = np.insert(simulated_Enu, -1, simulated_Enu[-1])
# preds_Enu = np.insert(preds_Enu, -1, preds_Enu[-1])
# pred_stds_Enu_min = np.insert(pred_stds_Enu_min, -1, pred_stds_Enu_min[-1])
# pred_stds_Enu_max = np.insert(pred_stds_Enu_max, -1, pred_stds_Enu_max[-1])
# errors_enu = np.insert(errors_enu, -1, errors_enu[-1])
# xvals_per_obs = [-300, -100, 100, 300, 600, 1000, 1900]
# xvals_per_obs = np.array(xvals_per_obs)

# =========== TOP RIGHT (Rates Enu vs FK otimes f_NN)

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

axRmeasmu = axR.errorbar(
    Enumu_centers_plot,  # x values (bin centers)
    preds_Enu / binwidths,  # y values (measurements)
    yerr=(pred_stds_Enu_max - preds_Enu) / binwidths,  # vertical error bars
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

axRpred = axR.bar(
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

axR.legend(
    [(axRmeasmu), (axRpred)],
    [
        r"$\mathrm{FK} \ \otimes \ \mathcal{A}_{\mathcal{fit}} f_{\nu_\mu, \mathrm{ref}}$",
        r"$\mathrm{FK} \otimes f_{\nu_\mu, \mathrm{ref}}$",
    ],
    handler_map={tuple: HandlerTuple(ndivide=1)},
    loc="upper right",
).set_alpha(0.8)

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
axR.set_title(r"$\mathrm{Pseudo \ \ Data }, \ \mathrm{Level\ 2}$", loc="left")
axR.text(-400, 30, r"$\nu_{\mu(\bar{\mu})} + W \rightarrow X_h+  \mu^{\pm} $")
# axR.set_title(r"$\mathcal{L}_{\mathrm{pp}} = 150 \mathrm{fb}^{-1}$", loc="right")
# axR.set_title(r"$\mathrm{FASER}\nu, \ \mathrm{Level\ 2}$", loc="left")
# axR.text(800, 400, r"$\nu_\mu W \rightarrow X_h \mu^- $")
axR.set_ylabel(r"$N_{\mathrm{int}}$")
pred_stds_Enu = pred_stds_Enu_max - preds_Enu
ratio_center_pred = preds_Enu / simulated_Enu
ratio_lower_pred = (preds_Enu - pred_stds_Enu) / simulated_Enu
ratio_upper_pred = (preds_Enu + pred_stds_Enu) / simulated_Enu
ratio_upper_sim = (simulated_Enu + errors_enu) / simulated_Enu
ratio_lower_sim = (simulated_Enu - errors_enu) / simulated_Enu

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

plt.savefig("nonMLFK_sim.pdf")
plt.show()

# fig.show()
