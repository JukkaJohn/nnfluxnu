import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from itertools import product
from data import get_reference_pdf
from ML_fit_enu.src.ML_fit_neutrinos.multiple_obs_fit.read_LHEF import read_LHEF_data
# return LO_low_binned, LO_val_binned

num_obs = 1
data, data_min, data_max, xvals_per_obs, binwidths, xlabels, events_per_obs = (
    read_LHEF_data(0, num_obs)
)

print("x,pred")
print(x, pred)
# def chi_square(parr)


delta_plus = pred_max - pred
delta_min = pred_min - pred
semi_diff = (delta_plus + delta_min) / 2
average = (delta_plus - delta_min) / 2
se_delta = semi_diff
sig_sys = np.sqrt(average * average + 2 * semi_diff * semi_diff)

r_sys = np.random.normal(0, sig_sys)
r_stat = np.random.normal(0, np.sqrt(pred))
# pred_stoch = pred + r_sys*sig_sys + r_stat*np.sqrt(pred)
pred_stoch = pred + r_sys + r_stat
sig_tot = sig_sys**2 + pred
cov = np.diag(sig_tot)
cov = np.linalg.inv(cov)


def chi_square(parr):
    data = parr[0] ** 2 * x ** (parr[1] - 1) * ((1 - x) ** (parr[2] - 1)) * pred

    r = pred - data

    r = r.reshape(len(r), 1)
    return np.dot(r.T, np.matmul(cov, r))


x0 = [0.9, 1.1, 0.95]
# bounds = [(0, 10),(None, None),(None, None)]
result = minimize(chi_square, x0, method="BFGS")
print("chi square")
print(chi_square(result.x))
print(f" a,b,c = {result.x}")
print(f"number of iter = {result.nit}")
hess_inv = result.hess_inv
cov_matrix = hess_inv
print("naive errors are:")
print(np.sqrt(np.diag(cov_matrix)))

p_min = result.x

hessian = np.empty([3, 3])

hessian[0, 0] = np.sum(
    2 * p_min[0] * x ** (p_min[1] - 1) * (1 - x) ** (p_min[2] - 1) * pred
)
hessian[1, 1] = np.sum(
    p_min[0] * x ** (p_min[1] - 1) * (np.log(x) ** 2) * (1 - x) ** (p_min[2] - 1) * pred
)
hessian[2, 2] = np.sum(
    p_min[0]
    * x ** (p_min[1] - 1)
    * (np.log(1 - x) ** 2)
    * (1 - x) ** (p_min[2] - 1)
    * pred
)
hessian[0, 1] = np.sum(
    x ** (p_min[1] - 1) * np.log(x) * (1 - x) ** (p_min[2] - 1) * pred
)
hessian[0, 2] = np.sum(
    x ** (p_min[1] - 1) * np.log(1 - x) * (1 - x) ** (p_min[2] - 1) * pred
)
hessian[1, 2] = np.sum(
    p_min[0]
    * x ** (p_min[1] - 1)
    * np.log(x)
    * np.log(1 - x)
    * (1 - x) ** (p_min[2] - 1)
    * pred
)
hessian[1, 0] = hessian[0, 1]
hessian[2, 0] = hessian[0, 2]
hessian[2, 1] = hessian[1, 2]


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
# print(vec[1,:],'vec')


def determine_errors(interval):
    p_hess_min = []
    p_hess_max = []
    for i in range(3):
        t = 0
        chi_68 = 0

        while chi_68 < interval + chi_square(p_min):
            chi_68 = chi_square(p_min + t * vec[i, :])
            # print(f'chi_68 = {chi_68}')
            t += 0.001
        # print('interval = ')
        # print(i,interval)
        p_hess_max.append(p_min + t * vec[i, :])
        t = 0

        chi_68 = 0
        while chi_68 < 1:
            chi_68 = chi_square(p_min + t * vec[i, :])
            # print(f'chi_68 = {chi_68}')
            t -= 0.001

        p_hess_min.append(p_min + t * vec[i, :])
    return p_hess_max, p_hess_min


theory = NLO_val
norm = (6000 - 25) / 120 * 2
theory *= norm


def func(parr, pred, x):
    return parr[0] ** 2 * x ** (parr[1] - 1) * ((1 - x) ** (parr[2] - 1)) * pred


def plot_error_bands(interval):
    p_hess_max, p_hess_min = determine_errors(interval)
    print("cont fit = ")
    # print(func(p_min, raw, rawx))
    print(func(p_min, pred, x))
    # plt.plot(x_fit,func(p_min),drawstyle = 'steps-post')
    f_err = 0
    err_param = 0
    for i in range(3):
        f_err += (func(p_hess_max[i], pred, x) - func(p_hess_min[i], pred, x)) ** 2
        err_param += (p_hess_max[i] - p_hess_min[i]) ** 2
    print("the errors are:")
    print(np.sqrt(err_param) * 0.5)
    hess_inv = result.hess_inv
    cov_matrix = hess_inv
    print("naive errors are:")
    print(np.sqrt(np.diag(cov_matrix)))
    f_err = np.sqrt(f_err) * 0.5

    return f_err


fit = func(p_min, pred, x)

plt.grid(axis="both")

f_err_68 = plot_error_bands(1)
f_err_68 = np.insert(f_err_68, -1, f_err_68[-1])
f_err_95 = plot_error_bands(4)
f_err_95 = np.insert(f_err_95, -1, f_err_95[-1])
f_err_99 = plot_error_bands(9)
f_err_99 = np.insert(f_err_99, -1, f_err_99[-1])

x = np.insert(x, 0, 0)
rawx = np.insert(rawx, 0, 0)
fit = np.insert(fit, -1, fit[-1])

# f_nu = fit / norm / 14000 * 2
# f_nu_68 = f_err_68 / norm / 14000 * 2
f_nu = fit / norm
f_nu_68 = f_err_68 / norm
plt.grid(axis="both")
plt.plot(x, f_nu, color="red", label="fit")
plt.fill_between(
    x,
    f_nu_68 + f_nu,
    -f_nu_68 + f_nu,
    color="blue",
    label=r"1$\sigma$ fit",
)
# plt.plot(rawx, f_nu, color="red", label="fit")
# plt.fill_between(
#     rawx,
#     f_nu_68 + f_nu,
#     -f_nu_68 + f_nu,
#     color="blue",
#     label=r"1$\sigma$ fit",
# )
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$x_{\nu}$", fontsize=16)
plt.ylabel(r"$f(x_{\nu})$", fontsize=16)
plt.grid(axis="both")
plt.legend()
plt.show()

plt.grid(axis="both")
plt.plot(x, fit, drawstyle="steps-post", color="yellow", label="fit")
plt.fill_between(
    x, f_err_68 + fit, -f_err_68 + fit, step="post", color="blue", label=f"68%"
)
plt.fill_between(
    x,
    f_err_95 + fit,
    -f_err_95 + fit,
    step="post",
    color="orange",
    alpha=0.35,
    label=f"95.4%",
)
plt.fill_between(
    x,
    f_err_99 + fit,
    -f_err_99 + fit,
    step="post",
    color="green",
    alpha=0.35,
    label=f"99.7%",
)

# x = np.insert(x,0,0)

# pred = np.insert(pred,-1,pred[-1])
bin_width = np.diff(x)  # The width of each bin
bin_centers = x[:-1] + bin_width / 2

pred_stoch = np.insert(pred_stoch, -1, pred_stoch[-1])
plt.plot(
    x,
    pred_stoch,
    drawstyle="steps-post",
    label="pseudo data + noise",
    color="red",
    linewidth=0.5,
)
plt.errorbar(
    bin_centers,
    pred_stoch[:-1],
    yerr=np.sqrt(pred_stoch[:-1]),
    fmt="none",
    ecolor="red",
    elinewidth=1.5,
    capsize=2,
    alpha=0.75,
)
# plt.fill_between(x,pred_min,pred_max,step="post",alpha=0.35,color="blue",linewidth=0.4,edgecolor="blue")
plt.xlabel("x_nu")
plt.ylabel("N_int")
plt.text(
    x=0.25, y=250, s=r"NLO FASER$\nu$, $\mathcal{L}_{\rm pp}=300 \, \mathrm{fb}^{-1}$"
)
plt.text(x=0.25, y=200, s=r"$\nu _e + \mathrm{W} \rightarrow e^{-} + X_h$")
plt.text(x=0.25, y=170, s=r"$\bar{\nu} _e + \mathrm{W} \rightarrow e^{+} + X_h$")
plt.title(
    r"fit: $f(x_{\nu}) = A \cdot x_{\nu}^{b-1} \cdot (1-x_{\nu})^{c-1} \cdot f_{ref}(x_{\nu})$"
)
plt.legend()
plt.savefig("fit_enu.pdf")
plt.show()
