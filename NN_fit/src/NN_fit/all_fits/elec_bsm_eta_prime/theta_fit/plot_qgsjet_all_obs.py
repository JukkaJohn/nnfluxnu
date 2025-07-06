from matplotlib.legend_handler import HandlerTuple
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import lhapdf

# Add the parent directory to sys.path
parent_dir = os.path.abspath(
    os.path.join(
        os.getcwd(),
        "/Users/jukkajohn/Masterscriptie/faser_nufluxes_ml/ML_fit_enu/src/ML_fit_neutrinos/",
    )
)
sys.path.append(parent_dir)
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

# DPMJET all obs
# neutrino_pdfs_mu,
# neutrino_pdfs_mub,
# faser data projection
# neutrino_pdfs_mu_faser = np.loadtxt("fit_sim_faser_data/mu_pdf.txt", delimiter=",")
# neutrino_pdfs_mub_faser = np.loadtxt("fit_sim_faser_data/mub_pdf.txt", delimiter=",")

neutrino_pdfs_mu_dpm_Eh = np.loadtxt("fit_qgsjet/Eh_fit/mu_pdf.txt", delimiter=",")
neutrino_pdfs_mub_dpm_Eh = np.loadtxt("fit_qgsjet/Eh_fit/mub_pdf.txt", delimiter=",")

neutrino_pdfs_mu_dpm_El = np.loadtxt("fit_qgsjet/El_fit/mu_pdf.txt", delimiter=",")
neutrino_pdfs_mub_dpm_El = np.loadtxt("fit_qgsjet/El_fit/mub_pdf.txt", delimiter=",")

neutrino_pdfs_mu_dpm_theta = np.loadtxt(
    "fit_qgsjet/theta_fit/mu_pdf.txt", delimiter=","
)
neutrino_pdfs_mub_dpm_theta = np.loadtxt(
    "fit_qgsjet/theta_fit/mub_pdf.txt", delimiter=","
)


# Get number of reps from make runscripts
def plot(
    x_vals,
):
    x_vals = np.array(x_vals)
    pdf = "FASERv_Run3_QGSJET+POWHEG_7TeV"
    faser_pdf_mu, x_faser = read_pdf(pdf, x_vals, 14)
    faser_pdf_mub, x_faser = read_pdf(pdf, x_vals, -14)
    faser_pdf_mu = faser_pdf_mu * x_vals * 1.16186e-09
    faser_pdf_mub = faser_pdf_mub * x_vals * 1.16186e-09

    # mean_mu_faser = np.mean(neutrino_pdfs_mu_faser, axis=0) * x_vals * 150 / 65.6
    # error_mu_faser = np.std(neutrino_pdfs_mu_faser, axis=0) * x_vals * 150 / 65.6
    # mean_mub_faser = np.mean(neutrino_pdfs_mub_faser, axis=0) * x_vals * 150 / 65.6
    # error_mub_faser = np.std(neutrino_pdfs_mub_faser, axis=0) * x_vals * 150 / 65.6

    mean_mu_dpm_Eh = np.mean(neutrino_pdfs_mu_dpm_Eh, axis=0) * x_vals
    error_mu_dpm_Eh = np.std(neutrino_pdfs_mu_dpm_Eh, axis=0) * x_vals
    mean_mub_dpm_Eh = np.mean(neutrino_pdfs_mub_dpm_Eh, axis=0) * x_vals
    error_mub_dpm_Eh = np.std(neutrino_pdfs_mub_dpm_Eh, axis=0) * x_vals

    mean_mu_dpm_El = np.mean(neutrino_pdfs_mu_dpm_El, axis=0) * x_vals
    error_mu_dpm_El = np.std(neutrino_pdfs_mu_dpm_El, axis=0) * x_vals
    mean_mub_dpm_El = np.mean(neutrino_pdfs_mub_dpm_El, axis=0) * x_vals
    error_mub_dpm_El = np.std(neutrino_pdfs_mub_dpm_El, axis=0) * x_vals

    mean_mu_dpm_theta = np.mean(neutrino_pdfs_mu_dpm_theta, axis=0) * x_vals
    error_mu_dpm_theta = np.std(neutrino_pdfs_mu_dpm_theta, axis=0) * x_vals
    mean_mub_dpm_theta = np.mean(neutrino_pdfs_mub_dpm_theta, axis=0) * x_vals
    error_mub_dpm_theta = np.std(neutrino_pdfs_mub_dpm_theta, axis=0) * x_vals

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],  # Computer Modern
        }
    )
    fig = plt.figure(figsize=(12.8, 4.0), dpi=300)  # 2 rows, 3 columns
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])
    gs.update(left=0.05, right=0.97, top=0.92, hspace=0.18, wspace=0.20)

    axL = fig.add_subplot(gs[0, 0])
    axM = fig.add_subplot(gs[0, 1])
    # axR = fig.add_subplot(gs[0, 2])
    axrL = fig.add_subplot(gs[1, 0])
    axrM = fig.add_subplot(gs[1, 1])
    # axrR = fig.add_subplot(gs[1, 2])

    # TOP LEFT PLOT

    (axLsim,) = axL.plot(
        x_vals,
        faser_pdf_mu,
        linestyle="-",
        label=r"$f_{\mathrm{EPOS+POWHEG}\nu_\mu}(x)$",
        color=simcolor,
    )
    # axLfasererr = axL.fill_between(
    #     x_vals,
    #     mean_mu_faser + error_mu_faser,
    #     mean_mu_faser - error_mu_faser,
    #     color=mucolor,
    #     alpha=0.2,
    #     label=r"$\pm 1\sigma$",
    # )

    # (axLfaser,) = axL.plot(
    #     x_vals,
    #     mean_mu_faser,
    #     linestyle="-",
    #     color=mucolor,
    #     label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
    # )

    axLdpm_Eh_err = axL.fill_between(
        x_vals,
        mean_mu_dpm_Eh + error_mu_dpm_Eh,
        mean_mu_dpm_Eh - error_mu_dpm_Eh,
        color="green",
        alpha=0.2,
        label=r"$\pm 1\sigma$",
    )

    (axLdpm_Eh,) = axL.plot(
        x_vals,
        mean_mu_dpm_Eh,
        linestyle="-",
        color="green",
        label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
    )

    axLdpm_El_err = axL.fill_between(
        x_vals,
        mean_mu_dpm_El + error_mu_dpm_El,
        mean_mu_dpm_El - error_mu_dpm_El,
        color="black",
        alpha=0.2,
        label=r"$\pm 1\sigma$",
    )

    (axLdpm_El,) = axL.plot(
        x_vals,
        mean_mu_dpm_El,
        linestyle="-",
        color="black",
        label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
    )

    axLdpm_theta_err = axL.fill_between(
        x_vals,
        mean_mu_dpm_theta + error_mu_dpm_theta,
        mean_mu_dpm_theta - error_mu_dpm_theta,
        color="yellow",
        alpha=0.2,
        label=r"$\pm 1\sigma$",
    )

    (axLdpm_theta,) = axL.plot(
        x_vals,
        mean_mu_dpm_theta,
        linestyle="-",
        color="yellow",
        label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
    )

    # axLvert1 = axL.axvline(
    #     x=100 / 7000,
    #     color="grey",
    #     linestyle="--",
    #     linewidth=1.5,
    #     label="axvline - full height",
    # )
    # axLvert2 = axL.axvline(
    #     x=1000 / 7000,
    #     color="grey",
    #     linestyle="--",
    #     linewidth=1.5,
    #     label="axvline - full height",
    # )

    axL.legend(
        # [axLsim, (axLnn, axLnnerr), (axLvert1, axLvert2)],
        [
            axLsim,
            # (axLfaser, axLfasererr),
            (axLdpm_Eh_err, axLdpm_Eh),
            (axLdpm_El_err, axLdpm_El),
            (axLdpm_theta_err, axLdpm_theta),
        ],
        [
            r"$f_{\nu_\mu, \mathrm{ref \ (QGS+PWG)}}$",
            # r"$f_{\nu_\mu, \mathrm{EPOS+POWHEG} \ E_\nu}$",
            r"$f_{\nu_\mu, \mathrm{QGS+PWG}\ E_h}$",
            r"$f_{\nu_\mu, \mathrm{QGS+PWG}\ E_l}$",
            r"$f_{\nu_\mu, \mathrm{QGS+PWG}\ \theta}$",
        ],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="lower left",
    ).set_alpha(0.8)

    axL.set_xlim(5e-4, 1)
    axL.set_ylim(1e-3, 1e5)
    axL.set_yscale("log")
    axL.set_xscale("log")

    # if preproc == 1:
    title_str = r"$\mathrm{QGSJET} \ + \ \mathrm{PWG} $"
    # if preproc == 2:
    # title_str = rf"$f_{{\mathrm{{NN}}}}(x) = \mathrm{{NN}}_{{{layers}}}(x) -  \mathrm{{NN}}_{{{layers}}}(1)$"

    axL.set_title(title_str)
    # fig.text(0.33, 0.94, title_str, ha="center", va="bottom", fontsize=10)
    axL.set_ylabel(r"$xf_{\nu_\mu}(x_\nu)$")
    axL.set_xticklabels([])
    axL.grid(color="grey", linestyle="-", linewidth=0.25)

    # ========== TOP MIDDLE (f_NN_mub vs f_faserv)

    (axMsimb,) = axM.plot(
        x_vals,
        faser_pdf_mub,
        linestyle="-",
        label=r"$f_{\mathrm{EPOS+POWHEG}\bar{\nu}_\mu}(x)$",
        color=simcolor,
    )
    # axMfasererr = axM.fill_between(
    #     x_vals,
    #     (mean_mub_faser + error_mub_faser),
    #     (mean_mub_faser - error_mub_faser),
    #     color=mubcolor,
    #     alpha=0.2,
    #     label=r"$\pm 1\sigma$",
    # )
    # (axMfaser,) = axM.plot(
    #     x_vals,
    #     mean_mub_faser,
    #     linestyle="-",
    #     color=mubcolor,
    #     label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
    # )

    axMdpm_Eh_err = axM.fill_between(
        x_vals,
        (mean_mub_dpm_Eh + error_mub_dpm_Eh),
        (mean_mub_dpm_Eh - error_mub_dpm_Eh),
        color="green",
        alpha=0.2,
        label=r"$\pm 1\sigma$",
    )
    (axMdpm_Eh,) = axM.plot(
        x_vals,
        mean_mub_dpm_Eh,
        linestyle="-",
        color="green",
        label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
    )

    axMdpm_El_err = axM.fill_between(
        x_vals,
        (mean_mub_dpm_El + error_mub_dpm_El),
        (mean_mub_dpm_El - error_mub_dpm_El),
        color="black",
        alpha=0.2,
        label=r"$\pm 1\sigma$",
    )
    (axMdpm_El,) = axM.plot(
        x_vals,
        mean_mub_dpm_El,
        linestyle="-",
        color="black",
        label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
    )

    axMdpm_theta_err = axM.fill_between(
        x_vals,
        (mean_mub_dpm_theta + error_mub_dpm_theta),
        (mean_mub_dpm_theta - error_mub_dpm_theta),
        color="yellow",
        alpha=0.2,
        label=r"$\pm 1\sigma$",
    )
    (axMdpm_theta,) = axM.plot(
        x_vals,
        mean_mub_dpm_theta,
        linestyle="-",
        color="yellow",
        label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
    )
    # axMvert1 = axM.axvline(
    #     x=100 / 7000,
    #     color="grey",
    #     linestyle="--",
    #     linewidth=1.5,
    #     label="axvline - full height",
    # )
    # axMvert2 = axM.axvline(
    #     x=1000 / 7000,
    #     color="grey",
    #     linestyle="--",
    #     linewidth=1.5,
    #     label="axvline - full height",
    # )

    axM.legend(
        # [axMsimb, (axMnnberr, axMnnb), (axMvert1, axMvert2)],
        [
            axMsimb,
            # (axMfasererr, axMfaser),
            (axMdpm_Eh_err, axMdpm_Eh),
            (axMdpm_El_err, axMdpm_El),
            (axMdpm_theta_err, axMdpm_theta),
        ],
        [
            r"$f_{\bar{\nu}_\mu, \mathrm{ref} \ (QGS+PWG)}$",
            # r"$f_{\bar{\nu}_\mu, \mathrm{EPOS+POWHEG} \ E_\nu}$",
            r"$f_{\bar{\nu}_\mu, \mathrm{QGS+PWG}\ E_h}$",
            r"$f_{\bar{\nu}_\mu, \mathrm{QGS+PWG} \ E_l}$",
            r"$f_{\bar{\nu}_\mu, \mathrm{QGS+PWG} \ \theta}$",
        ],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="lower left",
    ).set_alpha(0.8)

    title_str = r"$\mathcal{L}_{\mathrm{pp}} = 65.6 \mathrm{fb}^{-1}, \ \mathrm{FASER} \ \mathrm{Run} \ 3$"
    # if preproc == 2:
    # title_str = rf"$f_{{\mathrm{{NN}}}}(x) = \mathrm{{NN}}_{{{layers}}}(x) -  \mathrm{{NN}}_{{{layers}}}(1)$"

    axM.set_title(title_str)

    # axM.set_ylabel(r'$xf_{\bar{\nu}_\mu}(x_\nu)$')

    axM.set_xlim(5e-4, 1)
    axM.set_ylim(1e-3, 1e5)
    axM.set_yscale("log")
    axM.set_xscale("log")
    axM.set_xticklabels([])
    axM.grid(color="grey", linestyle="-", linewidth=0.25)

    # ========== BOTTOM LEFT (Ratio plot, f_NN vs f_FASERv )

    # ratio_center = mean_fnu_mu / faser_pdf_mu
    # ratio_lower = (mean_fnu_mu - error_fnu_mu) / faser_pdf_mu
    # ratio_upper = (mean_fnu_mu + error_fnu_mu) / faser_pdf_mu

    # axrL.plot(x_vals, ratio_center, linestyle="-", color=mucolor)
    # axrL.fill_between(x_vals, ratio_upper, ratio_lower, color=mucolor, alpha=0.2)

    axrL.plot(x_vals, np.ones(len(x_vals)), linestyle="-", color=simcolor)

    axrL.set_xscale("log")
    axrL.set_xlim(5e-4, 1)
    axrL.set_ylim(0, 2)
    axrL.grid(color="grey", linestyle="-", linewidth=0.25)
    axrL.set_ylabel(r"$\mathrm{Ratio}$")
    axrL.set_xlabel(r"$x_\nu$")

    # ratio_center = mean_mu_faser / faser_pdf_mu
    # ratio_lower = (mean_mu_faser - error_mu_faser) / faser_pdf_mu
    # ratio_upper = (mean_mu_faser + error_mu_faser) / faser_pdf_mu

    # axrL.plot(x_vals, ratio_center, linestyle="-", color=mucolor)
    # axrL.fill_between(x_vals, ratio_upper, ratio_lower, color=mucolor, alpha=0.2)

    ratio_center = mean_mu_dpm_Eh / faser_pdf_mu
    ratio_lower = (mean_mu_dpm_Eh - error_mu_dpm_Eh) / faser_pdf_mu
    ratio_upper = (mean_mu_dpm_Eh + error_mu_dpm_Eh) / faser_pdf_mu

    axrL.plot(x_vals, ratio_center, linestyle="-", color="green")
    axrL.fill_between(x_vals, ratio_upper, ratio_lower, color="green", alpha=0.2)

    ratio_center = mean_mu_dpm_El / faser_pdf_mu
    ratio_lower = (mean_mu_dpm_El - error_mu_dpm_El) / faser_pdf_mu
    ratio_upper = (mean_mu_dpm_El + error_mu_dpm_El) / faser_pdf_mu

    axrL.plot(x_vals, ratio_center, linestyle="-", color="black")
    axrL.fill_between(x_vals, ratio_upper, ratio_lower, color="black", alpha=0.2)

    ratio_center = mean_mu_dpm_theta / faser_pdf_mu
    ratio_lower = (mean_mu_dpm_theta - error_mu_dpm_theta) / faser_pdf_mu
    ratio_upper = (mean_mu_dpm_theta + error_mu_dpm_theta) / faser_pdf_mu

    axrL.plot(x_vals, ratio_center, linestyle="-", color="yellow")
    axrL.fill_between(x_vals, ratio_upper, ratio_lower, color="yellow", alpha=0.2)

    # axrL.plot(x_vals, np.ones(len(x_vals)), linestyle="-", color=simcolor)

    # axrL.axvline(300 / 7000, color="red", label="axvline - full height")
    # axrL.axvline(1450 / 7000, color="red")
    # axrL.axvline(
    #     x=100 / 7000,
    #     color="grey",
    #     linestyle="--",
    #     linewidth=1.5,
    #     label="axvline - full height",
    # )
    # axrL.axvline(
    #     x=1000 / 7000,
    #     color="grey",
    #     linestyle="--",
    #     linewidth=1.5,
    #     label="axvline - full height",
    # )

    axrL.set_xscale("log")
    axrL.set_xlim(5e-4, 1)
    axrL.set_ylim(0, 2)
    axrL.grid(color="grey", linestyle="-", linewidth=0.25)
    axrL.set_ylabel(r"$\mathrm{Ratio}$")
    axrL.set_xlabel(r"$x_\nu$")

    # ========== BOTTOM MIDDLE (ratio f_NN_mub)

    # axrM.plot(x_vals, mean_mub_faser / faser_pdf_mub, linestyle="-", color=mubcolor)
    # axrM.fill_between(
    #     x_vals,
    #     (mean_mub_faser + error_mub_faser) / faser_pdf_mub,
    #     (mean_mub_faser - error_mub_faser) / faser_pdf_mub,
    #     color=mubcolor,
    #     alpha=0.2,
    # )
    axrM.plot(x_vals, np.ones(len(x_vals)), linestyle="-", color=simcolor)

    axrM.plot(x_vals, mean_mub_dpm_Eh / faser_pdf_mub, linestyle="-", color="green")
    axrM.fill_between(
        x_vals,
        (mean_mub_dpm_Eh + error_mub_dpm_Eh) / faser_pdf_mub,
        (mean_mub_dpm_Eh - error_mub_dpm_Eh) / faser_pdf_mub,
        color="green",
        alpha=0.2,
    )

    axrM.plot(x_vals, mean_mub_dpm_El / faser_pdf_mub, linestyle="-", color="black")
    axrM.fill_between(
        x_vals,
        (mean_mub_dpm_El + error_mub_dpm_El) / faser_pdf_mub,
        (mean_mub_dpm_El - error_mub_dpm_El) / faser_pdf_mub,
        color="black",
        alpha=0.2,
    )

    axrM.plot(x_vals, mean_mub_dpm_theta / faser_pdf_mub, linestyle="-", color="yellow")
    axrM.fill_between(
        x_vals,
        (mean_mub_dpm_theta + error_mub_dpm_theta) / faser_pdf_mub,
        (mean_mub_dpm_theta - error_mub_dpm_theta) / faser_pdf_mub,
        color="yellow",
        alpha=0.2,
    )

    # axrM.axvline(300 / 7000, color="red", label="axvline - full height")
    # axrM.axvline(1450 / 7000, color="red")
    # axrM.axvline(
    #     x=100 / 7000,
    #     color="grey",
    #     linestyle="--",
    #     linewidth=1.5,
    #     label="axvline - full height",
    # )
    # axrM.axvline(
    #     x=1000 / 7000,
    #     color="grey",
    #     linestyle="--",
    #     linewidth=1.5,
    #     label="axvline - full height",
    # )
    axrM.set_xscale("log")
    axrM.set_xlim(5e-4, 1)
    axrM.set_ylim(0, 2)
    axrM.grid(color="grey", linestyle="-", linewidth=0.25)
    # axrM.set_ylabel(r"$\mathrm{Ratio}$")
    axrM.set_xlabel(r"$x_\nu$")

    # # =========== TOP RIGHT (Rates Enu vs FK otimes f_NN)

    # x_vals_per_obs = low_bin
    # x_vals_per_obs = np.append(x_vals_per_obs, high_bin[-1])

    # pred_stds_Enu = np.append(pred_stds_Enu, pred_stds_Enu[-1])
    # preds_Enu = np.append(preds_Enu, preds_Enu[-1])
    # simulated_Enu = np.append(simulated_Enu, simulated_Enu[-1])
    # errors_enu = np.append(errors_enu, errors_enu[-1])

    # pred_stds_Enub = np.append(pred_stds_Enub, pred_stds_Enub[-1])
    # preds_Enub = np.append(preds_Enub, preds_Enub[-1])
    # simulated_Enub = np.append(simulated_Enub, simulated_Enub[-1])
    # errors_enub = np.append(errors_enub, errors_enub[-1])

    # x_vals_per_obs_mub = low_bin_mub
    # x_vals_per_obs_mub = np.append(x_vals_per_obs_mub, high_bin_mub[-1])

    # print("x_vals_per_obs_mub")
    # print(x_vals_per_obs_mub)

    # print("x_vals_per_obs_mu")
    # print(x_vals_per_obs)

    # axRmeasmu = axR.fill_between(
    #     x_vals_per_obs,
    #     # np.arange(len(preds_Enu)),
    #     preds_Enu + pred_stds_Enu,
    #     preds_Enu - pred_stds_Enu,
    #     step="post",
    #     color="tab:blue",
    #     alpha=0.6,
    #     label=r"POWHEG $E_\nu$",
    # )

    # axRmeasmub = axR.fill_between(
    #     x_vals_per_obs_mub,
    #     # np.arange(len(preds_Enu)),
    #     preds_Enub + pred_stds_Enub,
    #     preds_Enub - pred_stds_Enub,
    #     step="post",
    #     color="tab:red",
    #     alpha=0.6,
    #     label=r"POWHEG $E_\nu$",
    # )

    # axRpred = axR.fill_between(
    #     x_vals_per_obs,
    #     # np.arange(len(simulated_Enu)),
    #     simulated_Enu + errors_enu,
    #     simulated_Enu - errors_enu,
    #     step="post",
    #     color="tab:orange",
    #     alpha=0.6,
    #     label=r"POWHEG $E_\nu$",
    # )

    # axRpredb = axR.fill_between(
    #     x_vals_per_obs_mub,
    #     # np.arange(len(simulated_Enu)),
    #     simulated_Enub + errors_enub,
    #     simulated_Enub - errors_enub,
    #     step="post",
    #     color="tab:green",
    #     alpha=0.6,
    #     label=r"POWHEG $E_\nu$",
    # )

    # axR.legend(
    #     [(axRmeasmu), (axRpred), (axRmeasmub), (axRpredb)],
    #     [
    #         # r"$\mathrm{DATA} \ E_\nu$",
    #         r"$\mathrm{FK} \otimes  f_{\nu_{\mu}, NN}$",
    #         r"$\mathrm{PSEUDO} \ \mathrm{DATA} \ \mu $",
    #         r"$\mathrm{FK} \otimes  f_{\nu_{\bar{\mu}}, NN}$",
    #         r"$\mathrm{PSEUDO} \ \mathrm{DATA} \ \bar{\mu}$",
    #     ],
    #     handler_map={tuple: HandlerTuple(ndivide=1)},
    #     loc="lower left",
    # ).set_alpha(0.8)

    # # axR.set_xlim(0, 1)
    # # axR.set_ylim(0)
    # axR.set_yscale("log")
    # axR.set_xscale("log")
    # # axR.grid(color='grey', linestyle='-', linewidth=0.25)
    # # axR.set_xticklabels([])
    # # axR.set_xticks(ticks)
    # # axR.axvline(
    # #     x=np.interp(-1 / 1000, xplot_ticks, ticks),
    # #     color="black",
    # #     linestyle="-",
    # #     linewidth=1,
    # #     alpha=0.8,
    # # )
    # # axR.axvline(
    # #     x=np.interp(1 / 1000, xplot_ticks, ticks),
    # #     color="black",
    # #     linestyle="-",
    # #     linewidth=1,
    # #     alpha=0.8,
    # # )
    # axR.set_title(r"$\mathcal{L}_{\mathrm{pp}} = 150 \mathrm{fb}^{-1}$", loc="right")
    # axR.set_title(r"$\mathrm{Pseudo \ \ Data }, \ \mathrm{Level\ 2}$", loc="left")
    # # axR.text(np.interp(1/500, xplot_ticks,ticks), 170, r"$\bar{\nu}_\mu$", alpha=0.8)
    # # axR.text(np.interp(-1/400, xplot_ticks,ticks), 170, r"$\nu_\mu$", alpha=0.8)
    # # axR.text(np.interp(-1/1010, xplot_ticks,ticks), 170, r"$\nu_\mu + \bar{\nu}_\mu$", alpha=0.8)
    # axR.set_ylabel(r"$N_{\mathrm{int}} \  [\mathrm{GeV}]$")
    # axR.set_xlabel(r"$E_h$")

    # # ========= BOTTOM RIGHT (Ratio Rates Enu vs FK otimes f_NN)

    # # plot mu bins

    # axrRmeasmu = axrR.fill_between(
    #     x_vals_per_obs,
    #     # np.arange(len(preds_Enu)),
    #     (preds_Enu + pred_stds_Enu) / simulated_Enu,
    #     (preds_Enu - pred_stds_Enu) / simulated_Enu,
    #     step="post",
    #     color="tab:blue",
    #     alpha=0.8,
    #     label=r"POWHEG $E_\nu$",
    # )

    # axrRpred = axrR.fill_between(
    #     x_vals_per_obs,
    #     # np.arange(len(simulated_Enu)),
    #     (simulated_Enu + errors_enu) / simulated_Enu,
    #     (simulated_Enu - errors_enu) / simulated_Enu,
    #     step="post",
    #     color="tab:orange",
    #     alpha=0.8,
    #     label=r"POWHEG $E_\nu$",
    # )

    # # axrR.legend(
    # #     [
    # #         (axrRmeasmu),
    # #         (axrRpred),
    # #     ],
    # #     [
    # #         # r"$\mathrm{DATA} \ E_\nu$",
    # #         r"$\mathrm{FK} \otimes  f_{\nu_{\bar{\mu}}, NN}$",
    # #         r"$\mathrm{FK} \otimes f_{\nu_\mu, \mathrm{ref}}$",
    # #     ],
    # #     handler_map={tuple: HandlerTuple(ndivide=1)},
    # #     loc="upper right",
    # # ).set_alpha(0.8)

    # axrR.set_ylabel(r"$\mathrm{Ratio}$")
    # axrR.set_xlabel(r"$E_h$")
    # axrR.set_ylim(0, 2)
    # axrR.set_xscale("log")
    # # axrR.set_xlim(0, 1)
    # # axrR.set_xticks(ticks)
    # # axrR.set_xticklabels(tick_labels)
    # # axrR.set_xlim(100,1000)
    # axrR.grid(color="grey", linestyle="-", linewidth=0.25)

    # time6 = time.time()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.13)
    plt.savefig("qgs_pwg_fits_obs.pdf")
    plt.show()


x_vals = np.logspace(-5, 0, 1000)
plot(x_vals)
