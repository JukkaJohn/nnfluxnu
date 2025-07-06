from matplotlib.legend_handler import HandlerTuple
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import lhapdf

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from read_faserv_pdf import read_pdf

# Data for plot
import matplotlib.ticker as ticker


simcolor = "tab:red"
mucolor = "tab:blue"
mubcolor = "tab:blue"

lhapdf.setVerbosity(0)
mubpid = -14
mupid = 14

npt = 200


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
    # Get data ready to plot

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

    # simulated_Enu = np.append(simulated_Enu, simulated_Enu[-1])
    # simulated_data = np.append(simulated_data, simulated_data[-1])
    # err_sim = np.append(err_sim, err_sim[-1])
    print(simulated_Enu)
    print("pred enu")
    print(preds_Enu)
    # preds_Enu = np.append(preds_Enu, preds_Enu[-1])

    # pred_stds_Enu = np.append(pred_stds_Enu, pred_stds_Enu[-1])
    # errors_enu = np.append(errors_enu, errors_enu[-1])

    # initialise plot]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],  # Computer Modern
        }
    )
    fig = plt.figure(figsize=(10.2, 3.4), dpi=300)  # 2 rows, 3 columns
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.05, right=0.97, top=0.92, hspace=0.18, wspace=0.20)

    axL = fig.add_subplot(gs[0, 0])
    # axM = fig.add_subplot(gs[0, 1])
    # axR = fig.add_subplot(gs[0, 2])
    # axrL = fig.add_subplot(gs[1, 0])
    # axrM = fig.add_subplot(gs[1, 1])
    # axrR = fig.add_subplot(gs[1, 2])

    # TOP LEFT PLOT

    # (axLsim,) = axL.plot(
    #     x_vals,
    #     faser_pdf_mu,
    #     linestyle="-",
    #     label=r"$f_{\mathrm{FASER}\nu_\mu}(x)$",
    #     color=simcolor,
    # )
    axLnnerr = axL.fill_between(
        x_vals,
        mean_fnu_mu + error_fnu_mu,
        mean_fnu_mu - error_fnu_mu,
        color=mucolor,
        alpha=0.2,
        label=r"$\pm 1\sigma$",
    )

    (axLnn,) = axL.plot(
        x_vals,
        mean_fnu_mu,
        linestyle="-",
        color=mucolor,
        label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
    )

    # axL.legend(
    #     [axLsim, (axLnn, axLnnerr)],
    #     [
    #         r"$f_{\mathrm{EPOS+PWG}\nu_\mu}(x_\nu)$",
    #         r"$f_{\mathrm{NNflux}_{\nu_\mu}}(x_\nu)$",
    #     ],
    #     handler_map={tuple: HandlerTuple(ndivide=1)},
    #     loc="lower left",
    # ).set_alpha(0.8)

    axL.set_xlim(5e-4, 1)
    axL.set_ylim(1e-3, 1e3)
    axL.set_yscale("log")
    axL.set_xscale("log")

    # if preproc == 1:
    # title_str = rf"$f_{{\mathrm{{NN}}}}(x) = \mathcal{{A}} \ x^{{\alpha}}(1-x)^\beta \ \mathrm{{NN}}_{{{4, 4, 4}}}(x)$"
    # if preproc == 2:
    # title_str = rf"$f_{{\mathrm{{NN}}}}(x) = \mathrm{{NN}}_{{{layers}}}(x) -  \mathrm{{NN}}_{{{layers}}}(1)$"

    # axL.set_title(title_str)
    # fig.text(0.33, 0.94, title_str, ha="center", va="bottom", fontsize=10)
    # axL.set_ylabel(r"$xf_{\nu_\mu}(x_\nu)$")
    # axL.set_xticklabels([])
    # axL.grid(color="grey", linestyle="-", linewidth=0.25)

    # # ========== TOP MIDDLE (f_NN_mub vs f_faserv)

    # (axMsimb,) = axM.plot(
    #     x_vals,
    #     faser_pdf_mub,
    #     linestyle="--",
    #     label=r"$f_{\mathrm{FASER}\bar{\nu}_\mu}(x)$",
    #     color=simcolor,
    # )
    # axMnnberr = axM.fill_between(
    #     x_vals,
    #     (mean_fnu_mub + error_fnu_mub),
    #     (mean_fnu_mub - error_fnu_mub),
    #     color=mubcolor,
    #     alpha=0.2,
    #     label=r"$\pm 1\sigma$",
    # )
    # (axMnnb,) = axM.plot(
    #     x_vals,
    #     mean_fnu_mub,
    #     linestyle="--",
    #     color=mubcolor,
    #     label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
    # )

    # axM.legend(
    #     [axMsimb, (axMnnberr, axMnnb)],
    #     [
    #         r"$f_{\mathrm{EPOS+PWG}\bar{\nu}_\mu}(x_\nu)$",
    #         r"$f_{\mathrm{NNflux}_{\bar{\nu}_\mu}}(x_\nu)$",
    #     ],
    #     handler_map={tuple: HandlerTuple(ndivide=1)},
    #     loc="lower left",
    # ).set_alpha(0.8)

    # # axM.set_ylabel(r'$xf_{\bar{\nu}_\mu}(x_\nu)$')

    # axM.set_xlim(5e-4, 1)
    # axM.set_ylim(1e-3, 1e3)
    # axM.set_yscale("log")
    # axM.set_xscale("log")
    # axM.set_xticklabels([])
    # axM.grid(color="grey", linestyle="-", linewidth=0.25)

    # # ========== BOTTOM LEFT (Ratio plot, f_NN vs f_FASERv )

    # ratio_center = mean_fnu_mu / faser_pdf_mu
    # ratio_lower = (mean_fnu_mu - error_fnu_mu) / faser_pdf_mu
    # ratio_upper = (mean_fnu_mu + error_fnu_mu) / faser_pdf_mu

    # axrL.plot(x_vals, ratio_center, linestyle="-", color=mucolor)
    # axrL.fill_between(x_vals, ratio_upper, ratio_lower, color=mucolor, alpha=0.2)

    # axrL.plot(x_vals, np.ones(len(x_vals)), linestyle="-", color=simcolor)

    # axrL.set_xscale("log")
    # axrL.set_xlim(5e-4, 1)
    # axrL.set_ylim(0, 2)
    # axrL.grid(color="grey", linestyle="-", linewidth=0.25)
    # axrL.set_ylabel(r"$\mathrm{Ratio}$")
    # axrL.set_xlabel(r"$x_\nu$")

    # # ========== BOTTOM MIDDLE (ratio f_NN_mub)
    # mean_fnu_mub
    # error_fnu_mub
    # axrM.plot(x_vals, mean_fnu_mub / faser_pdf_mub, linestyle="--", color=mubcolor)
    # axrM.fill_between(
    #     x_vals,
    #     (mean_fnu_mub + error_fnu_mub) / faser_pdf_mub,
    #     (mean_fnu_mub - error_fnu_mub) / faser_pdf_mub,
    #     color=mubcolor,
    #     alpha=0.2,
    # )
    # axrM.plot(x_vals, np.ones(len(x_vals)), linestyle="--", color=simcolor)

    # axrM.set_xscale("log")
    # axrM.set_xlim(5e-4, 1)
    # axrM.set_ylim(0, 2)
    # axrM.grid(color="grey", linestyle="-", linewidth=0.25)
    # # axrM.set_ylabel(r"$\mathrm{Ratio}$")
    # axrM.set_xlabel(r"$x_\nu$")

    # =========== TOP RIGHT (Rates Enu vs FK otimes f_NN)

    # xplot_Enumu = np.array(
    #     [-1 / 100, -1 / 300, -1 / 600, -1 / 1000, 1 / 1000, 1 / 300, 1 / 100]
    # )
    # xplot_ticks = np.array(
    #     [-1 / 100, -1 / 300, -1 / 600, -1 / 1000, 1 / 1000, 1 / 300, 1 / 100]
    # )
    # # xplot_stretched = np.array([-1.0, -0.7, -0.4, -0.1, 0.1, 0.4, 1.0])

    # ticks = np.linspace(0, 1, len(xplot_ticks))
    # ticks = np.array(
    #     [
    #         0,
    #         0.07407407407407407,
    #         0.18518518518518517,
    #         0.3333333333333333,
    #         0.6666666666666666,
    #         0.9259259259259259,
    #         1,
    #     ]
    # )
    # binwidths = [200, 300, 400, 900, 700, 200]

    # xplot_Enumu = np.interp(xplot_Enumu, xplot_ticks, ticks)

    # Enumu_centers = 0.5 * (xplot_Enumu[1:] + xplot_Enumu[:-1])
    # Enumu_errors = 0.5 * (xplot_Enumu[1:] - xplot_Enumu[:-1])

    # # Enumu_centers_plot = np.interp(Enumu_centers, xplot_ticks, xplot_stretched)
    # # Enumu_errors_plot = np.interp(Enumu_centers + Enumu_errors, xplot_ticks, xplot_stretched) - Enumu_centers_plot

    # Enumu_centers_plot = Enumu_centers
    # Enumu_errors_plot = Enumu_errors

    # axRmeasmu = axR.errorbar(
    #     Enumu_centers_plot,  # x values (bin centers)
    #     simulated_Enu / binwidths,  # y values (measurements)
    #     yerr=errors_enu / binwidths,  # vertical error bars
    #     xerr=Enumu_errors_plot,  # horizontal error bars (bin widths)
    #     fmt="o",  # circle marker, set markersize to 0 to hide if needed
    #     markersize=3,
    #     color="black",
    #     capsize=1,  # no caps
    #     elinewidth=0.5,
    #     markeredgewidth=0.5,
    #     label=r"DATA$E_\nu$",
    #     zorder=3,
    # )

    # axRpred = axR.bar(
    #     Enumu_centers_plot,
    #     (2 * pred_stds_Enu) / binwidths,
    #     width=2 * Enumu_errors_plot,
    #     bottom=(preds_Enu - pred_stds_Enu) / binwidths,
    #     color="none",  # transparent fill
    #     #        edgecolor='gray',
    #     hatch="////",  # diagonal hatch
    #     linewidth=0.8,
    #     zorder=2,  # behind other plots
    # )

    # axRsimerr = axR.bar(
    #     Enumu_centers_plot,
    #     2 * err_sim / binwidths,  # full height (± error)
    #     width=2 * Enumu_errors_plot,
    #     bottom=(simulated_data - err_sim) / binwidths,  # center the bar at the value
    #     color="none",
    #     edgecolor="tab:orange",
    #     alpha=0.8,
    #     hatch="\\\\\\\\",
    #     linewidth=0.4,
    #     zorder=1,
    # )

    # axRsim = axR.bar(
    #     Enumu_centers_plot,
    #     simulated_data / binwidths,
    #     width=2 * Enumu_errors_plot,
    #     color="tab:red",
    #     alpha=1,
    #     zorder=0,
    # )

    # axR.legend(
    #     [(axRmeasmu), (axRpred), (axRsim, axRsimerr)],
    #     [
    #         r"$\mathrm{DATA} \ E_\nu$",
    #         r"$\mathrm{FK} \otimes  f_{\mathrm{NNflux}}$",
    #         r"$\mathrm{FK} \otimes f_\mathrm{EPOS+PWG}$",
    #     ],
    #     handler_map={tuple: HandlerTuple(ndivide=1)},
    #     loc="upper right",
    # ).set_alpha(0.8)

    # axR.set_xlim(0, 1)
    # axR.set_ylim(0)
    # # axR.grid(color='grey', linestyle='-', linewidth=0.25)
    # axR.set_xticklabels([])
    # axR.set_xticks(ticks)
    # axR.axvline(
    #     x=np.interp(-1 / 1000, xplot_ticks, ticks),
    #     color="black",
    #     linestyle="--",
    #     linewidth=1,
    #     alpha=0.8,
    # )
    # axR.axvline(
    #     x=np.interp(1 / 1000, xplot_ticks, ticks),
    #     color="black",
    #     linestyle="--",
    #     linewidth=1,
    #     alpha=0.8,
    # )
    # axR.set_title(r"$\mathcal{L}_{\mathrm{pp}} = 65.6 \mathrm{fb}^{-1}$", loc="right")
    # axR.set_title(r"$\mathrm{FASER}\nu, \ \mathrm{Level\ 1}$", loc="left")
    # # axR.text(np.interp(1/500, xplot_ticks,ticks), 170, r"$\bar{\nu}_\mu$", alpha=0.8)
    # # axR.text(np.interp(-1/400, xplot_ticks,ticks), 170, r"$\nu_\mu$", alpha=0.8)
    # # axR.text(np.interp(-1/1010, xplot_ticks,ticks), 170, r"$\nu_\mu + \bar{\nu}_\mu$", alpha=0.8)
    # axR.set_ylabel(
    #     r"$N_{\mathrm{int}} \ /  \mathrm{ \ bin \ width \ } [\mathrm{1/GeV}]$"
    # )
    # # axR.set_xlabel(r'$E_\nu$')

    # # ========= BOTTOM RIGHT (Ratio Rates Enu vs FK otimes f_NN)

    # # plot mu bins

    # axrRmeasmu = axrR.errorbar(
    #     Enumu_centers_plot,  # x values (bin centers)
    #     np.ones_like(simulated_Enu),  # y values (measurements)
    #     yerr=errors_enu / simulated_Enu,  # vertical error bars
    #     xerr=Enumu_errors_plot,  # horizontal error bars (bin widths)
    #     fmt="o",  # circle marker, set markersize to 0 to hide if needed
    #     markersize=3,
    #     color="black",
    #     capsize=1,  # no caps
    #     elinewidth=0.5,
    #     markeredgewidth=0.5,
    #     label=r"DATA$E_\nu$",
    #     zorder=3,
    # )

    # axrRpred = axrR.bar(
    #     Enumu_centers_plot,
    #     (2 * pred_stds_Enu) / simulated_Enu,
    #     width=2 * Enumu_errors_plot,
    #     bottom=(preds_Enu - pred_stds_Enu) / simulated_Enu,
    #     color="none",  # transparent fill
    #     #        edgecolor='gray',
    #     hatch="////",  # diagonal hatch
    #     linewidth=0.8,
    #     zorder=2,  # behind other plots
    # )

    # axrRsimerr = axrR.bar(
    #     Enumu_centers_plot,
    #     2 * err_sim / simulated_Enu,  # full height (± error)
    #     width=2 * Enumu_errors_plot,
    #     bottom=(simulated_data - err_sim)
    #     / simulated_Enu,  # center the bar at the value
    #     color="none",
    #     edgecolor="tab:orange",
    #     alpha=0.8,
    #     hatch="\\\\\\\\",
    #     linewidth=0.4,
    #     zorder=1,
    # )

    # axRsim = axrR.bar(
    #     Enumu_centers_plot,
    #     simulated_data / simulated_Enu,
    #     width=2 * Enumu_errors_plot,
    #     color="tab:red",
    #     alpha=1,
    #     zorder=0,
    # )

    # tick_labels = [
    #     r"$-\frac{1}{100}$",
    #     r"$-\frac{1}{300}$",
    #     r"$-\frac{1}{600}$",
    #     r"$-\frac{1}{1000}$",
    #     r"$\frac{1}{1000}$",
    #     r"$\frac{1}{300}$",
    #     r"$\frac{1}{100}$",
    # ]

    # axR.legend(
    #     [(axrRmeasmu), (axrRpred), (axRsim, axrRsimerr)],
    #     [
    #         r"$\mathrm{DATA} \ E_\nu$",
    #         r"$\mathrm{FK} \otimes  f_{\mathrm{NNflux}}$",
    #         r"$\mathrm{FK} \otimes f_\mathrm{EPOS+PWG}$",
    #     ],
    #     handler_map={tuple: HandlerTuple(ndivide=1)},
    #     loc="upper right",
    # ).set_alpha(0.8)

    # axrR.set_ylabel(r"$\mathrm{Ratio}$")
    # axrR.set_xlabel(r"$q/E_\nu \ [\mathrm{1/GeV}]$")
    # axrR.set_ylim(0, 1.5)
    # axrR.set_xlim(0, 1)
    # axrR.set_xticks(ticks)
    # axrR.set_xticklabels(tick_labels)
    # axrR.set_xlim(100,1000)
    # axrR.grid(color='grey', linestyle='-', linewidth=0.25)

    # time6 = time.time()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.13)

    # plt.show()
    plt.savefig("titlepage.pdf")
