from matplotlib.legend_handler import HandlerTuple
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add the parent directory to sys.path
parent_dir = os.path.abspath(
    os.path.join(
        os.getcwd(),
        "/data/theorie/jjohn/git/faser_nufluxes_ml/ML_fit_enu/src/ML_fit_neutrinos/",
    )
)
sys.path.append(parent_dir)
from read_faserv_pdf import read_pdf

# Data for plot
import matplotlib.ticker as ticker


# Get number of reps from make runscripts
def plot(
    x_vals,
    neutrino_pdfs_mu,
    neutrino_pdfs_mub,
    data,
    N_event_pred,
    sig_tot,
    pid,
    simulated_data,
    err_sim,
):
    print(data)
    x_vals = np.array(x_vals)
    pdf = "FASERv_EPOS+POWHEG_7TeV"
    faser_pdf_mu, x_faser = read_pdf(pdf, x_vals, 14)
    faser_pdf_mub, x_faser = read_pdf(pdf, x_vals, -14)
    faser_pdf_mu = faser_pdf_mu * x_vals
    faser_pdf_mub = faser_pdf_mub * x_vals
    mean_fnu_mu = np.mean(neutrino_pdfs_mu, axis=0) * x_vals
    error_fnu_mu = np.std(neutrino_pdfs_mu, axis=0) * x_vals

    mean_fnu_mub = np.mean(neutrino_pdfs_mub, axis=0) * x_vals
    error_fnu_mub = np.std(neutrino_pdfs_mub, axis=0) * x_vals

    simulated_Enu = data

    preds_Enu = np.mean(N_event_pred, axis=0)
    print(preds_Enu.shape)
    pred_stds_Enu = np.std(N_event_pred, axis=0)
    # errors_enu = [5186, 6239, 4165, 1738, 622, 847]
    # errors_enu = np.array(errors_enu)
    # errors_enu = np.sqrt(level0[0])
    errors_enu = sig_tot

    simulated_Enu = np.append(simulated_Enu, simulated_Enu[-1])
    simulated_data = np.append(simulated_data, simulated_data[-1])
    err_sim = np.append(err_sim, err_sim[-1])
    print(simulated_Enu)
    print("pred enu")
    print(preds_Enu)
    preds_Enu = np.append(preds_Enu, preds_Enu[-1])

    pred_stds_Enu = np.append(pred_stds_Enu, pred_stds_Enu[-1])
    errors_enu = np.append(errors_enu, errors_enu[-1])
    # fig = plt.figure(figsize=(6.8, 3.4), dpi=300)  # 2 rows, 2 columns
    fig = plt.figure(figsize=(10, 5), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])
    gs.update(left=0.09, right=0.95, top=0.93, hspace=0.18)

    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    axrL = fig.add_subplot(gs[1, 0])
    axrR = fig.add_subplot(gs[1, 1])

    # ======== TOP LEFT (Main plot, f_NN vs f_FASERv ) =============
    (axLsim_mu,) = axL.plot(
        x_vals,
        faser_pdf_mu,
        linestyle="-",
        color="b",
        label=r"$f_{\mathrm{FASER}\nu_\mu}(x_\nu)$",
    )

    (axLsim_mub,) = axL.plot(
        x_vals,
        faser_pdf_mub,
        linestyle="--",
        color="b",
        label=r"$f_{\mathrm{FASER}\nu_{\bar{\mu}}}(x_\nu)$",
    )

    axLvert1 = axL.axvline(x=150 / 7000, color="red", label="axvline - full height")
    axLvert2 = axL.axvline(x=1850 / 7000, color="red", label="axvline - full height")
    axLnnerr_mu = axL.fill_between(
        x_vals,
        (mean_fnu_mu + error_fnu_mu),
        (mean_fnu_mu - error_fnu_mu),
        color="green",
        alpha=0.2,
        label=r"$\pm 1\sigma$",
    )

    axLnnerr_mub = axL.fill_between(
        x_vals,
        (mean_fnu_mub + error_fnu_mub),
        (mean_fnu_mub - error_fnu_mub),
        color="green",
        alpha=0.2,
        label=r"$\pm 1\sigma$",
    )

    (axLnn_mu,) = axL.plot(
        x_vals,
        mean_fnu_mu,
        linestyle="-",
        color="green",
        label=r"$f_{\mathrm{fit}\mu}(x)$",
    )

    (axLnn_mub,) = axL.plot(
        x_vals,
        mean_fnu_mub,
        linestyle="--",
        color="green",
        label=r"$f_{\mathrm{fit}\bar{\mu}}(x)$",
    )

    axL.set_xlim(1e-3, 1)
    axL.set_ylim(1e-2, 1e4)
    axL.set_yscale("log")
    axL.set_xscale("log")
    axL.set_title(
        r"$f_{\mathrm{NN}}(x) = \mathcal{A} \ x^{1-\alpha}(1-x)^\beta \ \mathrm{NN}(x)$"
        # r"$f_{\mathrm{NN}}(x) = \mathrm{NN_{2,2,2}}(x) -\mathrm{NN_{2,2,2}}(1)$"
    )
    axL.set_ylabel(r"$f_{\nu_\mu}(x_\nu)$")
    axL.set_xticklabels([])
    axL.grid(color="grey", linestyle="-", linewidth=0.25)
    axL.yaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    )
    axL.legend(
        [
            (axLsim_mu, axLsim_mub),
            (axLvert1, axLvert2),
            (axLsim_mub),
            (axLnn_mu, axLnnerr_mu),
            (axLnn_mub, axLnnerr_mub),
        ],
        [
            r"$f_{\mathrm{FASER}\nu_\mu}(x_\nu)$",
            r"$\mathrm{Data}\quad\mathrm{region}$",
            r"$f_{\mathrm{FASER}\nu_{\bar{\mu}}}(x_\nu)$",
            r"$f_{\mathrm{fit}_\mu}(x_\nu)$",
            r"$f_{\mathrm{fit}_{\bar{\mu}}}(x_\nu)$",
        ],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="lower left",
    ).set_alpha(0.8)

    # ========== BOTTOM LEFT (Ratio plot, f_NN vs f_FASERv )

    ratio_center = mean_fnu_mu / faser_pdf_mu
    ratio_lower = (mean_fnu_mu - error_fnu_mu) / faser_pdf_mu
    ratio_upper = (mean_fnu_mu + error_fnu_mu) / faser_pdf_mu

    axrL.plot(x_vals, ratio_center, linestyle="-", color="green")
    axrL.fill_between(x_vals, ratio_upper, ratio_lower, color="green", alpha=0.2)
    axrL.plot(x_vals, np.ones(len(x_vals)), linestyle="-", color="tab:blue")

    ratio_center = mean_fnu_mub / faser_pdf_mub
    ratio_lower = (mean_fnu_mub - error_fnu_mub) / faser_pdf_mub
    ratio_upper = (mean_fnu_mub + error_fnu_mub) / faser_pdf_mub

    axrL.plot(x_vals, ratio_center, linestyle="--", color="green")
    axrL.fill_between(x_vals, ratio_upper, ratio_lower, color="green", alpha=0.2)
    axrL.plot(x_vals, np.ones(len(x_vals)), linestyle="-", color="tab:blue")

    # axrL.axvline(x=200 / 7000, color="red", label="axvline - full height")
    # axrL.axvline(x=1450 / 7000, color="red", label="axvline - full height")

    axrL.set_xscale("log")
    axrL.set_xlim(1e-3, 1)
    axrL.set_ylim(0, 2)
    axrL.grid(color="grey", linestyle="-", linewidth=0.25)
    axrL.set_ylabel(r"$\mathrm{Ratio}$")
    axrL.set_xlabel(r"$x_\nu$")

    # =========== TOP RIGHT (Rates Enu vs FK otimes f_NN)

    xvals_per_obs = [-1500, -1100, -600, 0.0, 1200, 1900, 2300]

    # xplot_Enumu = 1 / xplot_Enumu
    # xplot_Enumu[-1] = -1 / 1000
    # xplot_ticks = np.array(
    #     [-1 / 100, -1 / 300, -1 / 600, -1 / 1000, 1 / 1000, 1 / 300, 1 / 100]
    # )
    # ticks = np.linspace(0, 1, len(xplot_ticks))
    # xplot_Enumu = np.interp(-xplot_Enumu, xplot_ticks, ticks)
    # xplot_Enumub = 1 / xplot_Enumub
    # xplot_Enumub[-1] = -1 / 1000
    # xplot_Enumub = np.interp(xplot_Enumub, xplot_ticks, ticks)

    # sorted_indices = np.argsort(xvals_per_obs)
    # xvals_per_obs = xvals_per_obs[sorted_indices]
    # simulated_Enu = simulated_Enu[sorted_indices]
    # errors_enu = errors_enu[sorted_indices]
    # preds_Enu = preds_Enu[sorted_indices]
    # pred_stds_Enu = pred_stds_Enu[sorted_indices]

    (axRdata,) = axR.plot(
        xvals_per_obs,
        simulated_Enu,
        drawstyle="steps-post",
        color="tab:red",
        alpha=0.8,
    )
    axRdataerr = axR.fill_between(
        xvals_per_obs,
        simulated_Enu + errors_enu,
        simulated_Enu - errors_enu,
        step="post",
        color="tab:red",
        alpha=0.2,
        label=r"POWHEG $E_\nu$",
    )

    (axRsim,) = axR.plot(
        xvals_per_obs,
        simulated_data,
        drawstyle="steps-post",
        color="tab:blue",
        alpha=0.8,
    )
    axRsimerr = axR.fill_between(
        xvals_per_obs,
        simulated_data + err_sim,
        simulated_data - err_sim,
        step="post",
        color="tab:blue",
        alpha=0.2,
        label=r"POWHEG $E_\nu$",
    )

    (axRpred,) = axR.plot(
        xvals_per_obs,
        preds_Enu,
        color="green",
        drawstyle="steps-post",
        alpha=0.8,
        label=r"$\mathrm{NN}(E_\nu)$",
    )
    axRprederr = axR.fill_between(
        xvals_per_obs,
        (preds_Enu + pred_stds_Enu),
        (preds_Enu - pred_stds_Enu),
        color="green",
        alpha=0.2,
        step="post",
        label=r"$\pm 1\sigma$",
    )
    axR.legend(
        [(axRsimerr, axRsim), (axRdataerr, axRdata), (axRprederr, axRpred)],
        [
            r"$\mathrm{FK} \ \otimes \ f_{\mathrm{input}}$",
            r"$\mathrm{FASER}\quad\mathrm{measurement}$",
            r"$\mathrm{FK} \ \otimes \ f_{\mathrm{fit}}$",
        ],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="upper right",
    ).set_alpha(0.8)
    # axR.set_xlim(0)
    # axR.set_ylim(0)
    tick_labels = [
        r"$-\frac{1}{100}$",
        r"$-\frac{1}{300}$",
        r"$-\frac{1}{600}$",
        r"$-\frac{1}{1000}$",
        r"$\frac{1}{1000}$",
        r"$\frac{1}{300}$",
        r"$\frac{1}{100}$",
    ]

    # Use `xvals_per_obs` positions for labels
    axR.set_xticks(xvals_per_obs)
    axR.set_xticklabels(tick_labels)

    axR.grid(color="grey", linestyle="-", linewidth=0.25)
    axR.set_xticklabels([])
    axR.set_title(r"$\mathcal{L}_{\mathrm{pp}} = 65.6 \mathrm{fb}^{-1}$", loc="right")
    axR.set_title(r"$\ \mathrm{Level\ 1},100 \mathrm{reps}$", loc="left")
    axR.text(-400, 100, r"$\nu_{\mu(\bar{\mu})} + W \rightarrow X_h+  \mu^{\pm} $")
    axR.set_ylabel(r"$N_{\mathrm{int}}$")

    ratio_center_pred = preds_Enu / simulated_Enu
    ratio_lower_pred = (preds_Enu - pred_stds_Enu) / simulated_Enu
    ratio_upper_pred = (preds_Enu + pred_stds_Enu) / simulated_Enu
    ratio_upper_sim = (simulated_Enu + errors_enu) / simulated_Enu
    ratio_lower_sim = (simulated_Enu + errors_enu) / simulated_Enu

    axrR.fill_between(
        xvals_per_obs, ratio_upper_sim, ratio_lower_sim, step="post", alpha=0.2
    )
    axrR.plot(
        xvals_per_obs, np.ones(len(simulated_Enu)), drawstyle="steps-post", alpha=0.8
    )

    axrR.fill_between(
        xvals_per_obs,
        ratio_upper_pred,
        ratio_lower_pred,
        step="post",
        alpha=0.2,
        color="green",
    )
    axrR.plot(
        xvals_per_obs,
        ratio_center_pred,
        drawstyle="steps-post",
        alpha=0.8,
        color="green",
    )

    axrR.set_ylabel(r"$\mathrm{Ratio}$")
    axrR.set_xlabel(r"$q/E_\nu \ [\mathrm{1/GeV}]$")
    axrR.set_ylim(0.5, 1.5)
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

    # Use `xvals_per_obs` positions for labels
    axrR.set_xticks(xvals_per_obs)
    axrR.set_xticklabels(tick_labels)

    axrR.set_xlabel(r"$q/E_\nu \ [\mathrm{1/GeV}]$")
    # plt.show()
    plt.savefig("fit_faser_data.pdf")
