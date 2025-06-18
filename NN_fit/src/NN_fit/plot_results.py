from matplotlib.legend_handler import HandlerTuple
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from read_faserv_pdf import read_pdf
# Data for plot


def plot(x_vals, neutrino_pdfs, level0, N_event_pred, sig_tot, xvals_per_obs, pid):
    pdf = "faserv"
    faser_pdf, x_faser = read_pdf(pdf, x_vals, pid)
    mean_fnu = np.mean(neutrino_pdfs, axis=0)
    error_fnu = np.std(neutrino_pdfs, axis=0)

    simulated_Enu = level0[0]
    preds_Enu = np.mean(N_event_pred, axis=0)
    pred_stds_Enu = np.std(N_event_pred, axis=0)
    # errors_enu = [5186, 6239, 4165, 1738, 622, 847]
    # errors_enu = np.array(errors_enu)
    # errors_enu = np.sqrt(level0[0])
    errors_enu = sig_tot

    xvals_per_obs = np.append(xvals_per_obs, 1900)
    simulated_Enu = np.append(simulated_Enu, simulated_Enu[-1])
    print(simulated_Enu)
    preds_Enu = np.append(preds_Enu, preds_Enu[-1])
    print(preds_Enu)
    pred_stds_Enu = np.append(pred_stds_Enu, pred_stds_Enu[-1])
    errors_enu = np.append(errors_enu, errors_enu[-1])
    fig = plt.figure(figsize=(6.8, 3.4), dpi=300)  # 2 rows, 2 columns
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])
    gs.update(left=0.09, right=0.95, top=0.93, hspace=0.18)

    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    axrL = fig.add_subplot(gs[1, 0])
    axrR = fig.add_subplot(gs[1, 1])

    # ======== TOP LEFT (Main plot, f_NN vs f_FASERv ) =============
    (axLsim,) = axL.plot(
        x_vals, faser_pdf, linestyle="-", label=r"$f_{\mathrm{FASER}\nu}(x)$"
    )

    axLvert1 = axL.axvline(
        x=abs(min(xvals_per_obs)) / 14000, color="b", label="axvline - full height"
    )
    axLvert2 = axL.axvline(
        x=xvals_per_obs[-2] / 14000, color="b", label="axvline - full height"
    )
    axLnnerr = axL.fill_between(
        x_vals,
        (mean_fnu + error_fnu),
        (mean_fnu - error_fnu),
        color="red",
        alpha=0.2,
        label=r"$\pm 1\sigma$",
    )

    (axLnn,) = axL.plot(
        x_vals, mean_fnu, linestyle="-", color="red", label=r"$f_{\mathrm{NN}}(x)$"
    )

    axL.set_xlim(5e-4, 1)
    axL.set_ylim(1e-2, 1e5)
    axL.set_yscale("log")
    axL.set_xscale("log")
    axL.set_title(
        r"$f_{\mathrm{NN}}(x) = \mathcal{A} \ x^{1-\alpha}(1-x)^\beta \ \mathrm{NN}(x)$"
    )
    axL.set_ylabel(r"$xf_{\nu_e}(x_\nu)$")
    axL.set_xticklabels([])
    axL.grid(color="grey", linestyle="-", linewidth=0.25)
    axL.legend(
        [(axLsim, axLvert1, axLvert2), (axLnn, axLnnerr)],
        [r"$f_{\mathrm{FASER}\nu}(x_\nu)$", r"$f_{\mathrm{NN}}(x_\nu)$"],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="lower left",
    ).set_alpha(0.8)

    # ========== BOTTOM LEFT (Ratio plot, f_NN vs f_FASERv )

    ratio_center = mean_fnu / faser_pdf
    ratio_lower = (mean_fnu - error_fnu) / faser_pdf
    ratio_upper = (mean_fnu + error_fnu) / faser_pdf

    axrL.plot(x_vals, ratio_center, linestyle="-", color="red")
    axrL.fill_between(x_vals, ratio_upper, ratio_lower, color="red", alpha=0.2)
    axrL.plot(x_vals, np.ones(len(x_vals)), linestyle="-", color="tab:blue")
    axrL.set_xscale("log")
    axrL.set_xlim(5e-4, 1)
    axrL.set_ylim(0.8, 1.2)
    axrL.grid(color="grey", linestyle="-", linewidth=0.25)
    axrL.set_ylabel(r"$\mathrm{Ratio}$")
    axrL.set_xlabel(r"$x_\nu$")

    # =========== TOP RIGHT (Rates Enu vs FK otimes f_NN)

    sorted_indices = np.argsort(xvals_per_obs)
    xvals_per_obs = xvals_per_obs[sorted_indices]
    simulated_Enu = simulated_Enu[sorted_indices]
    errors_enu = errors_enu[sorted_indices]
    preds_Enu = preds_Enu[sorted_indices]
    pred_stds_Enu = pred_stds_Enu[sorted_indices]

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
        (preds_Enu + pred_stds_Enu),
        (preds_Enu - pred_stds_Enu),
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
    # axR.text(800, 400, r"$\nu_e W \rightarrow X_h e^- $")
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
        color="red",
    )
    axrR.plot(
        xvals_per_obs, ratio_center_pred, drawstyle="steps-post", alpha=0.8, color="red"
    )

    axrR.set_ylabel(r"$\mathrm{Ratio}$")
    axrR.set_xlabel(r"$E_\nu \ [\mathrm{GeV}]$")
    axrR.set_ylim(0.5, 1.5)
    axrR.set_xlim(0)
    axrR.grid(color="grey", linestyle="-", linewidth=0.25)

    fig.show()
