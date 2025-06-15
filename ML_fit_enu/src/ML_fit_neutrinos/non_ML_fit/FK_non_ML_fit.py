import numpy as np

# from matplotlib import gridspec
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# from itertools import product
from read_faser_pdf import read_pdf
from ML_fit_enu.src.ML_fit_neutrinos.read_data_with_fk import (
    read_LHEF_data,
    get_fk_table,
)
from ML_fit_enu.src.ML_fit_neutrinos.non_ML_fit.funcs_fit_fk import (
    compute_cov_matrix,
    compute_hessian,
    # determine_errors,
    plot_error_bands,
    func,
)
from logspace_grid import generate_grid


def get_data():
    pred, pred_max, pred_min, x, norm, _, _ = read_LHEF_data(0, 1)
    x /= 14000
    x_alpha, fk_table = get_fk_table(0, 1)

    cov = compute_cov_matrix(pred, pred_max, pred_min)

    f_ref, _ = read_pdf("faserv", x_alpha)

    return pred, cov, f_ref, x, norm, x_alpha, fk_table


pred, cov, f_ref, x, norm, x_alpha, fk_table = get_data()


def chi_square(parr):
    data = (
        parr[0] ** 2
        * x_alpha ** (parr[1] - 1)
        * ((1 - x_alpha) ** (parr[2] - 1))
        * f_ref
    )
    r = pred - np.matmul(fk_table, data) * norm
    r = r.reshape(len(r), 1)
    # chi_sq = np.dot(r.T, np.matmul(cov, r))
    # print(chi_sq)
    return np.dot(r.T, np.matmul(cov, r))


def perform_fit_fk():
    pred, cov, f_ref, x, norm, x_alpha, fk_table = get_data()
    x0 = [0.9, 1.1, 0.95]
    result = minimize(chi_square, x0, method="BFGS")

    print("chi square")
    print(chi_square(result.x))
    print(f" a,b,c = {result.x}")
    print(f"number of iter = {result.nit}")

    hess_inv = result.hess_inv
    p_min = result.x
    hessian = compute_hessian(x_alpha, p_min, f_ref, fk_table, norm)

    hess = np.linalg.inv(hess_inv)
    print("my hessian")
    print(hessian)
    print("BFGS hessian")
    print(hess)
    print("EV BFGS")
    val, vec = np.linalg.eig(hess)
    print(val, vec)
    print("EV analytical")
    val, vec = np.linalg.eig(hessian)
    print(val, vec)

    event_fit = np.matmul(fk_table, func(p_min, f_ref, x_alpha)) * norm

    rawx = generate_grid(-8, 250)
    raw, _ = read_pdf("faserv", rawx)
    rawx = np.array(rawx)
    fit = func(p_min, raw, rawx)

    f_err_68_cont = plot_error_bands(1, chi_square, p_min, vec, raw, rawx, result)
    f_err_68_cont = np.insert(f_err_68_cont, -1, f_err_68_cont[-1])
    f_err_95_cont = plot_error_bands(4, chi_square, p_min, vec, raw, rawx, result)
    f_err_95_cont = np.insert(f_err_95_cont, -1, f_err_95_cont[-1])
    f_err_99_cont = plot_error_bands(9, chi_square, p_min, vec, raw, rawx, result)
    f_err_99_cont = np.insert(f_err_99_cont, -1, f_err_99_cont[-1])

    f_err_68 = plot_error_bands(1, chi_square, p_min, vec, f_ref, x_alpha, result)
    f_err_68 = np.matmul(fk_table, f_err_68) * norm
    f_err_68 = np.insert(f_err_68, -1, f_err_68[-1])
    f_err_95 = plot_error_bands(4, chi_square, p_min, vec, f_ref, x_alpha, result)
    f_err_95 = np.matmul(fk_table, f_err_95) * norm
    f_err_95 = np.insert(f_err_95, -1, f_err_95[-1])
    f_err_99 = plot_error_bands(9, chi_square, p_min, vec, f_ref, x_alpha, result)
    f_err_99 = np.matmul(fk_table, f_err_99) * norm
    f_err_99 = np.insert(f_err_99, -1, f_err_99[-1])

    x = np.insert(x, 0, 0)
    rawx = np.insert(rawx, 0, 0)
    fit = np.insert(fit, -1, fit[-1])
    pred = np.insert(pred, -1, pred[-1])
    event_fit = np.insert(event_fit, -1, event_fit[-1])

    return (
        rawx,
        fit,
        f_err_68,
        f_err_95,
        f_err_99,
        x,
        event_fit,
        pred,
        f_err_68_cont,
        f_err_95_cont,
        f_err_99_cont,
    )


# (
#     rawx,
#     fit,
#     f_err_68,
#     f_err_95,
#     f_err_99,
#     x,
#     event_fit,
#     pred,
#     f_err_68_cont,
#     f_err_95_cont,
#     f_err_99_cont,
# ) = perform_fit()


# plt.grid(axis="both")
# plt.plot(x, event_fit, drawstyle="steps-post", color="red", label="fit")
# plt.plot(x, pred, drawstyle="steps-post", color="blue", alpha=0.5, label="pred")

# # plt.xscale("log")
# # plt.yscale("log")
# plt.xlabel(r"$x_{\nu}$", fontsize=16)
# plt.ylabel(r"$f(x_{\nu})$", fontsize=16)
# plt.grid(axis="both")
# plt.legend()
# # plt.xlim(0, 1)
# plt.show()

# plt.plot(x, event_fit, drawstyle="steps-post", color="red", label="fit")

# plt.fill_between(
#     x,
#     (f_err_68 + event_fit),
#     (-f_err_68 + event_fit),
#     color="blue",
#     step="post",
#     label=r"1$\sigma$ fit",
# )

# plt.fill_between(
#     x,
#     (f_err_95 + event_fit),
#     (-f_err_95 + event_fit),
#     color="green",
#     alpha=0.5,
#     step="post",
#     label=r"1$\sigma$ fit",
# )

# plt.fill_between(
#     x,
#     (f_err_99 + event_fit),
#     (-f_err_99 + event_fit),
#     color="orange",
#     alpha=0.5,
#     step="post",
#     label=r"1$\sigma$ fit",
# )

# # plt.xscale("log")
# # plt.yscale("log")
# plt.xlabel(r"$x_{\nu}$", fontsize=16)
# plt.ylabel(r"$f(x_{\nu})$", fontsize=16)
# plt.grid(axis="both")
# plt.legend()
# # plt.xlim(0, 1)
# plt.show()

# ##############################################################################################################################################
# plt.plot(rawx, fit, drawstyle="steps-post", color="red", label="fit")


# plt.fill_between(
#     rawx,
#     (f_err_68 + fit),
#     (-f_err_68 + fit),
#     color="blue",
#     step="post",
#     label=r"1$\sigma$ fit",
# )

# plt.fill_between(
#     rawx,
#     (f_err_95 + fit),
#     (-f_err_95 + fit),
#     color="green",
#     alpha=0.5,
#     step="post",
#     label=r"1$\sigma$ fit",
# )

# plt.fill_between(
#     rawx,
#     (f_err_99 + fit),
#     (-f_err_99 + fit),
#     color="orange",
#     alpha=0.5,
#     step="post",
#     label=r"1$\sigma$ fit",
# )

# # plt.xscale("log")
# # plt.yscale("log")
# plt.xlabel(r"$x_{\nu}$", fontsize=16)
# plt.ylabel(r"$f(x_{\nu})$", fontsize=16)
# plt.grid(axis="both")
# plt.legend()
# # plt.xlim(0, 1)
# plt.show()


# plt.grid(axis="both")
# plt.plot(x, event_fit, color="red", label="fit")
# plt.plot(x, pred, color="blue", label="faser")

# plt.xlabel(r"$x_{\nu}$", fontsize=16)
# plt.ylabel(r"$f(x_{\nu})$", fontsize=16)
# plt.grid(axis="both")
# plt.legend()
# plt.title("events")
# # plt.xlim(0, 1)
# plt.show()
# plt.plot(rawx, fit, color="red", label="fit")
# plt.fill_between(
#     rawx,
#     (f_err_68 + fit),
#     (-f_err_68 + fit),
#     color="blue",
#     label=r"1$\sigma$ fit",
# )

# plt.fill_between(
#     rawx,
#     (f_err_95 + fit),
#     (-f_err_95 + fit),
#     color="green",
#     alpha=0.5,
#     label=r"1$\sigma$ fit",
# )

# plt.fill_between(
#     rawx,
#     (f_err_99 + fit),
#     (-f_err_99 + fit),
#     color="orange",
#     alpha=0.5,
#     label=r"1$\sigma$ fit",
# )

# print("test")
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel(r"$x_{\nu}$", fontsize=16)
# plt.ylabel(r"$f(x_{\nu})$", fontsize=16)
# plt.grid(axis="both")
# plt.legend()
# # plt.xlim(0, 1)
# plt.show()

# # plt.plot(rawx, f_nu / f_nu, color="red", label="fit")
# # plt.fill_between(
# #     rawx,
# #     (f_nu_68 + f_nu) / f_nu,
# #     (-f_nu_68 + f_nu) / f_nu,
# #     color="blue",
# #     label=r"1$\sigma$ fit",
# # )

# # print("test")
# # plt.xscale("log")
# # plt.yscale("log")
# # plt.xlabel(r"$x_{\nu}$", fontsize=16)
# # plt.ylabel(r"$f(x_{\nu})$", fontsize=16)
# # plt.grid(axis="both")
# # plt.legend()
# # # plt.xlim(0, 1)
# # plt.show()
# # print(p_min)


# # plt.grid(axis="both")
# # plt.plot(x, fit, drawstyle="steps-post", color="yellow", label="fit")
# # plt.fill_between(
# #     x, f_err_68 + fit, -f_err_68 + fit, step="post", color="blue", label=f"68%"
# # )
# # plt.fill_between(
# #     x,
# #     f_err_95 + fit,
# #     -f_err_95 + fit,
# #     step="post",
# #     color="orange",
# #     alpha=0.35,
# #     label=f"95.4%",
# # )
# # plt.fill_between(
# #     x,
# #     f_err_99 + fit,
# #     -f_err_99 + fit,
# #     step="post",
# #     color="green",
# #     alpha=0.35,
# #     label=f"99.7%",
# # )

# # # x = np.insert(x,0,0)

# # # pred = np.insert(pred,-1,pred[-1])
# # bin_width = np.diff(x)  # The width of each bin
# # bin_centers = x[:-1] + bin_width / 2


# # # # plt.fill_between(x,pred_min,pred_max,step="post",alpha=0.35,color="blue",linewidth=0.4,edgecolor="blue")
# # # plt.xlabel("x_nu")
# # # plt.ylabel("N_int")
# # # plt.text(
# # #     x=0.25, y=250, s=r"NLO FASER$\nu$, $\mathcal{L}_{\rm pp}=300 \, \mathrm{fb}^{-1}$"
# # # )
# # # plt.text(x=0.25, y=200, s=r"$\nu _e + \mathrm{W} \rightarrow e^{-} + X_h$")
# # # plt.text(x=0.25, y=170, s=r"$\bar{\nu} _e + \mathrm{W} \rightarrow e^{+} + X_h$")
# # # plt.title(
# # #     r"fit: $f(x_{\nu}) = A \cdot x_{\nu}^{b-1} \cdot (1-x_{\nu})^{c-1} \cdot f_{ref}(x_{\nu})$"
# # # )
# # # plt.legend()
# # # plt.savefig("fit_enu.pdf")
# # # plt.show()


# # conv = np.matmul(fk_table, f_ref) * norm

# # plt.plot(x, pred, label="data")
# # plt.plot(x, conv, label="faser")
# # # plt.plot(x, pred / f_ref, label="diff")
# # plt.legend()
# # plt.show()


# # x = x_alpha
