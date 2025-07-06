import numpy as np
from matplotlib import gridspec
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar
from scipy.optimize import minimize

REPS = 1000
xvals_per_obs = [100, 300, 600, 1000, -300, -100]
xvals_per_obs = np.array(xvals_per_obs)
# pred = [44.1, 92.7, 68.5, 66.8, 44.3, 21.9]
pred = [223.16, 368.27, 258.92, 205.8, 108.74, 77.845]
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

std_errev = [0, 0, 0, 0, 0, 0]
# 39
sig_sys = std_errev
sig_sys = np.array(sig_sys)
# sig_stat =np.sqrt(events)
sig_stat = [72.011, 78.987, 64.535, 41.695, 24.934, 29.098]
sig_stat = np.array(sig_stat)

# sig_stat = np.sqrt(pred)
# sig_sys = [3.1, 2.5, 1.7, 1.6, 2.3, 2.9]
# sig_sys = np.array(sig_sys)
cov = np.diag(sig_stat**2 + sig_sys**2)
cov = np.linalg.inv(cov)
params = []
# sig_sys = [5186, 6239, 4165, 1738, 622, 847]
sig_sys = np.array(sig_sys)
chi_squares = []
for _ in range(REPS):
    r_sys_1 = np.random.normal(0, 1, len(pred)) * np.sqrt(sig_sys)
    r_stat_1 = np.random.normal(0, 1, len(pred)) * sig_stat
    level1 = pred + r_sys_1 + r_stat_1

    def chi_square(parr):
        data = pred * parr

        # print(level1_data.shape)
        # print(data.shape)
        r = level1 - data
        # print(r.shape)
        # print(data)
        r = r.reshape(len(r), 1)
        # print(parr)
        # print(cov.shape)
        print(f"chi square = {np.dot(r.T, np.dot(cov, r))}")
        diff = level1 - pred
        print(diff**2)
        return np.dot(r.T, np.matmul(cov, r))

    x0 = np.random.uniform(low=-1, high=3)

    result = minimize(chi_square, x0)
    print(result.x)
    params.append(result.x)
    chi_squares.append(chi_square(result.x))
print(f"mean chi square = {np.mean(chi_squares)}")
mean_norm = np.mean(params)
std_norm = np.std(params)
print(f"mean = {np.mean(params)}")
print(f" std = {np.std(params)}")


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],  # Computer Modern
    }
)
fig = plt.figure(figsize=(8, 5), dpi=300)  # 2 rows, 2 columns
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
gs.update(left=0.09, right=0.95, top=0.93, hspace=0.18)

axR = fig.add_subplot(gs[0, 0])
axrR = fig.add_subplot(gs[1, 0])


xvals_per_obs = [-1500, -1100, -600, 0.0, 1200, 1900, 2300]


simulated_Enu = pred
# errors_enu = [5186, 6239, 4165, 1738, 622, 847]
# errors_enu = np.array(errors_enu)
errors_enu = np.sqrt(sig_sys**2 + sig_stat**2)
preds_Enu = pred * mean_norm
pred_stds_Enu = pred * std_norm


# sorted_indices = np.argsort(xvals_per_obs)
# xvals_per_obs = xvals_per_obs[sorted_indices]
# simulated_Enu = simulated_Enu[sorted_indices]
# errors_enu = errors_enu[sorted_indices]
# preds_Enu = preds_Enu[sorted_indices]
# pred_stds_Enu = pred_stds_Enu[sorted_indices]

# print(len(sorted_indices))

# # xvals_per_obs = np.insert(xvals_per_obs, -1, xvals_per_obs[-1])
# simulated_Enu = np.insert(simulated_Enu, -1, simulated_Enu[-1])
# preds_Enu = np.insert(preds_Enu, -1, preds_Enu[-1])
# pred_stds_Enu = np.insert(pred_stds_Enu, -1, pred_stds_Enu[-1])
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

axRmeasmu = axR.errorbar(
    Enumu_centers_plot,  # x values (bin centers)
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
    [(axRpred), (axRmeasmu)],
    [
        r"$\mathrm{DATA} \ E_\nu$",
        r"$\mathcal{A}_{\mathcal{fit}} \ \cdot \ \mathrm{DATA} \ E_\nu$",
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
axR.set_title(
    r"$\ \mathrm{FASER}\nu \ \mathrm{Level\ 1},100 \mathrm{reps}$", loc="left"
)
axR.text(-400, 100, r"$\nu_{\mu(\bar{\mu})} + W \rightarrow X_h+  \mu^{\pm} $")
axR.set_ylabel(r"$N_{\mathrm{int}}$")

ratio_center_pred = preds_Enu / simulated_Enu
ratio_lower_pred = (preds_Enu - pred_stds_Enu) / simulated_Enu
ratio_upper_pred = (preds_Enu + pred_stds_Enu) / simulated_Enu
ratio_upper_sim = (simulated_Enu + errors_enu) / simulated_Enu
ratio_lower_sim = (simulated_Enu + errors_enu) / simulated_Enu

axrRmeasmu = axrR.errorbar(
    Enumu_centers_plot,  # x values (bin centers)
    np.ones_like(simulated_Enu),  # y values (measurements)
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

tick_labels = [
    r"$\frac{-1}{100}$",
    r"$\frac{-1}{300}$",
    r"$\frac{-1}{600}$",
    r"$\frac{-1}{1000}$",
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
# axrR.set_xlim(100,1000)
# axrR.grid(color='grey', linestyle='-', linewidth=0.25)

# time6 = time.time()

plt.tight_layout()
plt.subplots_adjust(bottom=0.13)
plt.savefig("norm_fit_faser.pdf")
plt.show()
