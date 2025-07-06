import numpy as np
from matplotlib import gridspec
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar
from scipy.optimize import minimize

REPS = 100
xvals_per_obs = [100, 300, 600, 1000, -300, -100]
xvals_per_obs = np.array(xvals_per_obs)
pred = [44.1, 92.7, 68.5, 66.8, 44.3, 21.9]
pred = np.array(pred)
cov = np.array(
    [
        [5186, -1623, 340, -69, 2, 5],
        [-1623, 6239, -1952, 281, -19, -4],
        [340, -1952, 4165, -734, 56, -27],
        [-69, 281, -734, 1738, -130, 15],
        [2, -19, 56, -130, 622, -147],
        [5, -4, -27, 15, -147, 847],
    ]
)

cov = np.linalg.inv(cov)
sig_stat = np.sqrt(pred)
sig_sys = [3.1, 2.5, 1.7, 1.6, 2.3, 2.9]
sig_sys = np.array(sig_sys)
cov = np.diag(sig_stat**2 + sig_sys**2)
cov = np.linalg.inv(cov)
params = []
# sig_sys = [5186, 6239, 4165, 1738, 622, 847]
# sig_sys = np.array(sig_sys)
x = [100, 300, 600, 1000, 300, 100]
x = np.array(x)
x = x / 14000
chi_squares = []
for _ in range(REPS):
    r_sys_1 = np.random.normal(0, 1, len(pred)) * sig_sys
    r_stat_1 = np.random.normal(0, 1, len(pred)) * sig_stat
    level1 = pred + r_sys_1 + r_stat_1
    # level1 = pred

    def chi_square(parr):
        data = parr[0] * x ** (parr[1] - 1) * ((1 - x) ** (parr[2] - 1)) * pred

        # print(level1_data.shape)
        # print(data.shape)
        r = level1 - data
        # print(r.shape)
        # print(data)
        r = r.reshape(len(r), 1)
        # print(parr)
        # print(cov.shape)
        # print(f"chi square = {np.dot(r.T, np.dot(cov, r))}")
        # diff = level1 - pred
        # print(diff**2)
        return np.dot(r.T, np.matmul(cov, r))

    x0 = np.random.uniform(0, 2, 3)
    # bounds = [(0, 2)] * len(x0)
    result = minimize(chi_square, x0, method="Powell")
    # result = minimize(chi_square, x0, method="BFGS")
    print("Optimized chi-square:", chi_square(result.x))
    print(result.x)
    # if 2 < sum(result.x) < 4:
    # if chi_square(result.x) < 4:
    params.append(result.x)
    chi_squares.append(chi_square(result.x))
print(f"mean chi square = {np.mean(chi_squares)}")

mean_params = np.mean(params, axis=0)
std_params = np.std(params, axis=0)
print(f"mean = {np.mean(params, axis=0)}")
print(f" std = {np.std(params, axis=0)}")

fig = plt.figure(figsize=(6.8, 3.4), dpi=300)  # 2 rows, 2 columns
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
gs.update(left=0.09, right=0.95, top=0.93, hspace=0.18)

axR = fig.add_subplot(gs[0, 0])
axrR = fig.add_subplot(gs[1, 0])


simulated_Enu = pred
# errors_enu = [5186, 6239, 4165, 1738, 622, 847]
# errors_enu = np.array(errors_enu)
# errors_enu = np.sqrt(errors_enu)
errors_enu = np.sqrt(sig_stat**2 + sig_sys**2)

fit_curves = np.array(
    [p[0] * x ** (p[1] - 1) * ((1 - x) ** (p[2] - 1)) * pred for p in params]
)

# Compute mean and percentiles for the uncertainty band
mean_fit = np.mean(fit_curves, axis=0)
pred_stds_Enu_max = np.std(fit_curves, axis=0) + mean_fit
pred_stds_Enu_min = -np.std(fit_curves, axis=0) + mean_fit

preds_Enu = mean_fit
# preds_Enu = (
#     mean_params[0]
#     * x ** (mean_params[1] - 1)
#     * ((1 - x) ** (mean_params[2] - 1))
#     * pred
# )
# pred_stds_Enu_max = (
#     (std_params[0] + mean_params[0])
#     * x ** ((std_params[1] + mean_params[1]) - 1)
#     * ((1 - x) ** ((std_params[2] + mean_params[2]) - 1))
#     * pred
# )

# pred_stds_Enu_min = (
#     (-std_params[0] + mean_params[0])
#     * x ** ((-std_params[1] + mean_params[1]) - 1)
#     * ((1 - x) ** ((-std_params[2] + mean_params[2]) - 1))
#     * pred
# )

# fit_curves = np.array(
#     [p[0] * x ** (p[1] - 1) * ((1 - x) ** (p[2] - 1)) * pred for p in params]
# )

# # Compute mean and percentiles for the uncertainty band
# mean_fit = np.mean(fit_curves, axis=0)
# pred_stds_Enu_min = np.percentile(fit_curves, 16, axis=0)  # 1σ lower bound
# pred_stds_Enu_max = np.percentile(fit_curves, 84, axis=0)  # 1σ upper bound

sorted_indices = np.argsort(xvals_per_obs)
xvals_per_obs = xvals_per_obs[sorted_indices]
simulated_Enu = simulated_Enu[sorted_indices]
errors_enu = errors_enu[sorted_indices]
preds_Enu = preds_Enu[sorted_indices]
pred_stds_Enu_max = pred_stds_Enu_max[sorted_indices]
pred_stds_Enu_min = pred_stds_Enu_min[sorted_indices]

print(len(sorted_indices))

# xvals_per_obs = np.insert(xvals_per_obs, -1, xvals_per_obs[-1])
simulated_Enu = np.insert(simulated_Enu, -1, simulated_Enu[-1])
preds_Enu = np.insert(preds_Enu, -1, preds_Enu[-1])
pred_stds_Enu_max = np.insert(pred_stds_Enu_max, -1, pred_stds_Enu_max[-1])
pred_stds_Enu_min = np.insert(pred_stds_Enu_min, -1, pred_stds_Enu_min[-1])
errors_enu = np.insert(errors_enu, -1, errors_enu[-1])
xvals_per_obs = [-300, -100, 100, 300, 600, 1000, 1900]
xvals_per_obs = np.array(xvals_per_obs)

# =========== TOP RIGHT (Rates Enu vs FK otimes f_NN)

(axRsim,) = axR.plot(
    xvals_per_obs,
    simulated_Enu,
    drawstyle="steps-post",
    color="tab:blue",
    alpha=0.8,
)
axRsimerr = axR.fill_between(
    xvals_per_obs,
    simulated_Enu + errors_enu,
    simulated_Enu - errors_enu,
    step="post",
    color="tab:blue",
    alpha=0.2,
    label=r"POWHEG $E_\nu$",
)

(axRpred,) = axR.plot(
    xvals_per_obs,
    preds_Enu,
    color="red",
    drawstyle="steps-post",
    alpha=0.8,
    label=r"$\mathrm{NN}(E_\nu)$",
)
axRprederr = axR.fill_between(
    xvals_per_obs,
    # (preds_Enu + pred_stds_Enu),
    # (preds_Enu - pred_stds_Enu),
    (pred_stds_Enu_max),
    (pred_stds_Enu_min),
    color="red",
    alpha=0.2,
    step="post",
    label=r"$\pm 1\sigma$",
)
axR.legend(
    [(axRsimerr, axRsim), (axRprederr, axRpred)],
    [
        r"$\mathrm{NLO+PS} \ E_\nu$",
        r"$\mathrm{FK} \ \otimes \ f_{\mathrm{NN}}(x_\nu)$",
    ],
    handler_map={tuple: HandlerTuple(ndivide=1)},
    loc="upper right",
).set_alpha(0.8)
# axR.set_xlim(0)
# axR.set_ylim(0)
axR.grid(color="grey", linestyle="-", linewidth=0.25)
axR.set_xticklabels([])
axR.set_title(r"$\mathcal{L}_{\mathrm{pp}} = 150 \mathrm{fb}^{-1}$", loc="right")
axR.set_title(r"$\mathrm{FASER}\nu, \ \mathrm{Level\ 2}$", loc="left")
axR.text(800, 400, r"$\nu_e W \rightarrow X_h e^- $")
axR.set_ylabel(r"$N_{\mathrm{int}}$")

ratio_center_pred = preds_Enu / simulated_Enu
ratio_lower_pred = (pred_stds_Enu_max) / simulated_Enu
ratio_upper_pred = (pred_stds_Enu_min) / simulated_Enu
ratio_upper_sim = (simulated_Enu + errors_enu) / simulated_Enu
ratio_lower_sim = (simulated_Enu - errors_enu) / simulated_Enu

axrR.fill_between(
    xvals_per_obs, ratio_upper_sim, ratio_lower_sim, step="post", alpha=0.2
)
axrR.plot(xvals_per_obs, np.ones(len(simulated_Enu)), drawstyle="steps-post", alpha=0.8)

axrR.fill_between(
    xvals_per_obs,
    ratio_upper_pred,
    ratio_lower_pred,
    step="post",
    alpha=0.2,
    color="red",
)
axrR.plot(
    xvals_per_obs, ratio_center_pred, drawstyle="steps-post", alpha=0.8, color="red"
)

axrR.set_ylabel(r"$\mathrm{Ratio}$")
axrR.set_xlabel(r"$E_\nu \ [\mathrm{GeV}]$")
axrR.set_ylim(0.5, 1.5)
# axrR.set_xlim(0)
axrR.grid(color="grey", linestyle="-", linewidth=0.25)
plt.savefig("nonMLfitfaserdata.pdf")
plt.show()

fig.show()


# import numpy as np

# import matplotlib.pyplot as plt
# from scipy.optimize import minimize
# from ML_fit_enu.src.ML_fit_neutrinos.non_ML_fit.read_data_with_fk import read_LHEF_data
# from ML_fit_enu.src.ML_fit_neutrinos.non_ML_fit.funcs_for_fit import (
#     compute_cov_matrix,
#     compute_hessian,
#     # determine_errors,
#     plot_error_bands,
#     func,
# )


# def get_data():
# pred, pred_max, pred_min, x, norm, _, _ = read_LHEF_data(0, 1)

#     x /= 14000

#     cov = compute_cov_matrix(pred, pred_max, pred_min)
#     f_ref = pred
#     return x, cov, f_ref, pred, norm


# def chi_square(parr):
#     x, cov, f_ref, pred, norm = get_data()
#     data = parr[0] ** 2 * x ** (parr[1] - 1) * ((1 - x) ** (parr[2] - 1)) * f_ref
#     r = pred - data
#     r = r.reshape(len(r), 1)
#     # chi_sq = np.dot(r.T, np.matmul(cov, r))
#     # print(chi_sq)
#     return np.dot(r.T, np.matmul(cov, r))


# def perform_fit():
#     x, cov, f_ref, pred, norm = get_data()
#     x0 = [0.9, 1.1, 0.95]
#     result = minimize(chi_square, x0, method="BFGS")

#     print("chi square")
#     print(chi_square(result.x))
#     print(f" a,b,c = {result.x}")
#     print(f"number of iter = {result.nit}")

#     hess_inv = result.hess_inv
#     p_min = result.x

#     hessian = compute_hessian(x, p_min, f_ref)

#     hess = np.linalg.inv(hess_inv)
#     print("my hessian")
#     print(hessian)
#     print("BFGS hessian")
#     print(hess)
#     print("EV BFGS")
#     val, vec = np.linalg.eig(hess)
#     print(val, vec)
#     print("EV analytical")
#     val, vec = np.linalg.eig(hessian)
#     print(val, vec)

#     fit = func(p_min, f_ref, x)

#     f_err_68 = plot_error_bands(1, chi_square, p_min, vec, f_ref, x, result)
#     f_err_68 = np.insert(f_err_68, -1, f_err_68[-1])
#     f_err_95 = plot_error_bands(4, chi_square, p_min, vec, f_ref, x, result)
#     f_err_95 = np.insert(f_err_95, -1, f_err_95[-1])
#     f_err_99 = plot_error_bands(9, chi_square, p_min, vec, f_ref, x, result)
#     f_err_99 = np.insert(f_err_99, -1, f_err_99[-1])

#     x = np.insert(x, 0, 0)
#     fit = np.insert(fit, -1, fit[-1])
#     pred = np.insert(pred, -1, pred[-1])

#     return x, fit, f_ref, norm, f_err_68, f_err_95, f_err_99, p_min, pred


# # x, fit, f_ref, norm, f_err_68, f_err_95, f_err_99, p_min, pred = perform_fit()

# # plt.grid(axis="both")
# # plt.plot(x, fit, drawstyle="steps-post", color="red", label="fit")
# # plt.plot(x, pred, drawstyle="steps-post", color="blue", alpha=0.5, label="pred")

# # # plt.xscale("log")
# # # plt.yscale("log")
# # plt.xlabel(r"$x_{\nu}$", fontsize=16)
# # plt.ylabel(r"$f(x_{\nu})$", fontsize=16)
# # plt.grid(axis="both")
# # plt.legend()
# # # plt.xlim(0, 1)
# # plt.show()
# # plt.plot(x, fit, drawstyle="steps-post", color="red", label="fit")


# # plt.fill_between(
# #     x,
# #     (f_err_68 + fit),
# #     (-f_err_68 + fit),
# #     color="blue",
# #     step="post",
# #     label=r"1$\sigma$ fit",
# # )

# # plt.fill_between(
# #     x,
# #     (f_err_95 + fit),
# #     (-f_err_95 + fit),
# #     color="green",
# #     alpha=0.5,
# #     step="post",
# #     label=r"1$\sigma$ fit",
# # )

# # plt.fill_between(
# #     x,
# #     (f_err_99 + fit),
# #     (-f_err_99 + fit),
# #     color="orange",
# #     alpha=0.5,
# #     step="post",
# #     label=r"1$\sigma$ fit",
# # )

# # # plt.xscale("log")
# # # plt.yscale("log")
# # plt.xlabel(r"$x_{\nu}$", fontsize=16)
# # plt.ylabel(r"$f(x_{\nu})$", fontsize=16)
# # plt.grid(axis="both")
# # plt.legend()
# # # plt.xlim(0, 1)
# # plt.show()
# # print(p_min)
# # print(norm)
# # # plt.plot(rawx, f_nu / f_nu, color="red", label="fit")
# # # plt.fill_between(
# # #     rawx,
# # #     (f_nu_68 + f_nu) / f_nu,
# # #     (-f_nu_68 + f_nu) / f_nu,
# # #     color="blue",
# # #     label=r"1$\sigma$ fit",
# # # )

# # # print("test")
# # # plt.xscale("log")
# # # plt.yscale("log")
# # # plt.xlabel(r"$x_{\nu}$", fontsize=16)
# # # plt.ylabel(r"$f(x_{\nu})$", fontsize=16)
# # # plt.grid(axis="both")
# # # plt.legend()
# # # # plt.xlim(0, 1)
# # # plt.show()
# # # print(p_min)


# # # plt.grid(axis="both")
# # # plt.plot(x, fit, drawstyle="steps-post", color="yellow", label="fit")
# # # plt.fill_between(
# # #     x, f_err_68 + fit, -f_err_68 + fit, step="post", color="blue", label=f"68%"
# # # )
# # # plt.fill_between(
# # #     x,
# # #     f_err_95 + fit,
# # #     -f_err_95 + fit,
# # #     step="post",
# # #     color="orange",
# # #     alpha=0.35,
# # #     label=f"95.4%",
# # # )
# # # plt.fill_between(
# # #     x,
# # #     f_err_99 + fit,
# # #     -f_err_99 + fit,
# # #     step="post",
# # #     color="green",
# # #     alpha=0.35,
# # #     label=f"99.7%",
# # # )

# # # # x = np.insert(x,0,0)

# # # # pred = np.insert(pred,-1,pred[-1])
# # # bin_width = np.diff(x)  # The width of each bin
# # # bin_centers = x[:-1] + bin_width / 2


# # # # # plt.fill_between(x,pred_min,pred_max,step="post",alpha=0.35,color="blue",linewidth=0.4,edgecolor="blue")
# # # # plt.xlabel("x_nu")
# # # # plt.ylabel("N_int")
# # # # plt.text(
# # # #     x=0.25, y=250, s=r"NLO FASER$\nu$, $\mathcal{L}_{\rm pp}=300 \, \mathrm{fb}^{-1}$"
# # # # )
# # # # plt.text(x=0.25, y=200, s=r"$\nu _e + \mathrm{W} \rightarrow e^{-} + X_h$")
# # # # plt.text(x=0.25, y=170, s=r"$\bar{\nu} _e + \mathrm{W} \rightarrow e^{+} + X_h$")
# # # # plt.title(
# # # #     r"fit: $f(x_{\nu}) = A \cdot x_{\nu}^{b-1} \cdot (1-x_{\nu})^{c-1} \cdot f_{ref}(x_{\nu})$"
# # # # )
# # # # plt.legend()
# # # # plt.savefig("fit_enu.pdf")
# # # # plt.show()
