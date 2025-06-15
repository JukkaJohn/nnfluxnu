from matplotlib.legend_handler import HandlerTuple
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import lhapdf
from data_for_plot_elec import data_needed_for_fit

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from read_faserv_pdf import read_pdf

# Data for plot
import matplotlib.ticker as ticker

simcolor = "tab:red"
mucolor = "tab:blue"
mubcolor = "tab:blue"

lhapdf.setVerbosity(0)
mubpid = -12
mupid = 12

npt = 200


# Get number of reps from make runscripts
def plot(
    x_vals,
    # data,
    # low_bin,
    # high_bin,
    # low_bin_mub,
    # high_bin_mub,
):
    geometries = [
        # "new_run_3_gens",
        # "new_2024faser",
        # "new_faserv2",
        "new_high_lumi",
    ]
    print(f"geometries = {geometries}")

    generators = [
        # "dpmjet",
        # "epos",
        "qgsjet",
        "sibyll",
    ]
    observables = [
        # "Eh",
        "El",
        # "Enu",
        # "theta",
    ]
    for geometry in geometries:
        for generator in generators:
            for obs in observables:
                # if geometry == "new_run_3_gens" and obs == "theta" and generator == "epos":
                #     break
                # if geometry == "new_run_3_gens" and obs == "Enu" and generator == "qgjset":
                #     break
                print("now plotting")
                print(geometry, generator, obs)
                if geometry == "new_run_3_gens":
                    formal_geometry = "FASERv_Run3"
                    factor = 1
                if geometry == "new_2024faser":
                    formal_geometry = "FASER_2412.03186"
                    factor = 1
                if geometry == "new_faserv2":
                    formal_geometry = "FASERv2"
                    factor = 1
                if geometry == "new_high_lumi":
                    formal_geometry = "FASERv_Run3"
                    factor = 20
                if generator == "dpmjet":
                    formal_gen = "DPMJET+DPMJET"
                if generator == "epos":
                    formal_gen = "EPOS+POWHEG"
                if generator == "qgsjet":
                    formal_gen = "QGSJET+POWHEG"
                if generator == "sibyll":
                    formal_gen = "SIBYLL+SIBYLL"

                if geometry == "new_high_lumi" or geometry == "new_faserv2":
                    filename_fk_mub_n = (
                        f"../../../FKtables/data/fastkernel/FK_{obs}_fine_nubmu_n.dat"
                    )
                    filename_fk_mub_p = (
                        f"../../../FKtables/data/fastkernel/FK_{obs}_fine_nubmu_p.dat"
                    )
                    filename_fk_mu_n = (
                        f"../../../FKtables/data/fastkernel/FK_{obs}_fine_numu_n.dat"
                    )
                    filename_fk_mu_p = (
                        f"../../../FKtables/data/fastkernel/FK_{obs}_fine_numu_p.dat"
                    )
                    filename_binsize = (
                        f"../../../FKtables/data/binning/FK_{obs}_fine_binsize.dat"
                    )
                else:
                    if obs != "Enu":
                        filename_fk_mub_n = f"../../../FKtables/data/fastkernel/FK_{obs}_final_nubmu_n.dat"
                        filename_fk_mub_p = f"../../../FKtables/data/fastkernel/FK_{obs}_final_nubmu_p.dat"
                        filename_fk_mu_n = f"../../../FKtables/data/fastkernel/FK_{obs}_final_numu_n.dat"
                        filename_fk_mu_p = f"../../../FKtables/data/fastkernel/FK_{obs}_final_numu_p.dat"
                        filename_binsize = (
                            f"../../../FKtables/data/binning/FK_{obs}_binsize.dat"
                        )
                        print(filename_fk_mub_n, "yesss")
                    else:
                        filename_fk_mub_n = f"../../../FKtables/data/fastkernel/FK_{obs}_fine_nubmu_n.dat"
                        filename_fk_mub_p = f"../../../FKtables/data/fastkernel/FK_{obs}_fine_nubmu_p.dat"
                        filename_fk_mu_n = f"../../../FKtables/data/fastkernel/FK_{obs}_fine_numu_n.dat"
                        filename_fk_mu_p = f"../../../FKtables/data/fastkernel/FK_{obs}_fine_numu_p.dat"
                        filename_binsize = (
                            f"../../../FKtables/data/binning/FK_{obs}_fine_binsize.dat"
                        )
                        print(filename_fk_mub_n, "yesss")

                pdf = f"{formal_geometry}_{formal_gen}_7TeV"
                print(pdf)
                (
                    data,
                    low_bin,
                    high_bin,
                ) = data_needed_for_fit(
                    filename_fk_mub_n,
                    filename_fk_mub_p,
                    filename_fk_mu_n,
                    filename_fk_mu_p,
                    filename_binsize,
                    pdf,
                )
                x_vals = np.array(x_vals)

                faser_pdf_mu, x_faser = read_pdf(pdf, x_vals, 12)
                faser_pdf_mub, x_faser = read_pdf(pdf, x_vals, -12)
                faser_pdf_mu = faser_pdf_mu * x_vals * factor
                faser_pdf_mub = faser_pdf_mub * x_vals * factor
                faser_pdf_mu += faser_pdf_mub

                neutrino_pdfs_mu = np.loadtxt(
                    f"{geometry}/fit_{generator}/{obs}_fit/pdf.txt", delimiter=","
                )
                neutrino_pdfs_mu *= 2
                # neutrino_pdfs_mub = np.loadtxt(
                #     f"{geometry}/fit_{generator}/{obs}_fit/mub_pdf.txt", delimiter=","
                # )
                N_event_pred_mu = np.loadtxt(
                    f"{geometry}/fit_{generator}/{obs}_fit/events.txt", delimiter=","
                )
                # N_event_pred_mub = np.loadtxt(
                #     f"{geometry}/fit_{generator}/{obs}_fit/events_mub.txt",
                #     delimiter=",",
                # )

                mean_fnu_mu = np.mean(neutrino_pdfs_mu, axis=0) * x_vals
                error_fnu_mu = np.std(neutrino_pdfs_mu, axis=0) * x_vals
                # mean_fnu_mub = np.mean(neutrino_pdfs_mub, axis=0) * x_vals
                # error_fnu_mub = np.std(neutrino_pdfs_mub, axis=0) * x_vals

                preds_Enu = np.mean(N_event_pred_mu, axis=0)

                pred_stds_Enu = np.std(N_event_pred_mu, axis=0)

                # preds_Enub = np.mean(N_event_pred_mub, axis=0)
                # pred_stds_Enub = np.std(N_event_pred_mub, axis=0)

                simulated_Enu = data

                errors_enu = np.sqrt(data)

                # simulated_Enub = data_mub * factor
                # errors_enub = np.sqrt(data_mub * factor)

                plt.rcParams["text.usetex"] = True
                plt.rcParams.update(
                    {
                        "font.size": 10,
                    }
                )
                fig = plt.figure(figsize=(12.8, 4.0), dpi=300)  # 2 rows, 3 columns
                gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1])
                gs.update(left=0.08, right=0.97, top=0.92, hspace=0.18, wspace=0.20)

                axL = fig.add_subplot(gs[0, 0])
                # axM = fig.add_subplot(gs[0, 1])
                axR = fig.add_subplot(gs[0, 1])
                axrL = fig.add_subplot(gs[1, 0])
                # axrM = fig.add_subplot(gs[1, 1])
                axrR = fig.add_subplot(gs[1, 1])

                # TOP LEFT PLOT

                (axLsim,) = axL.plot(
                    x_vals,
                    faser_pdf_mu,
                    linestyle="-",
                    label=r"$f_{\mathrm{FASER}\nu_\mu}(x)$",
                    color=simcolor,
                )
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

                axL.legend(
                    # [axLsim, (axLnn, axLnnerr), (axLvert1, axLvert2)],
                    [
                        axLsim,
                        (axLnn, axLnnerr),
                    ],
                    [
                        r"$\mathrm{baseline (\nu_e + \bar{\nu}_e)}$",
                        # r"$f_{\mathrm{EPOS+PWG}\nu_\mu}(x_\nu)$",
                        # r"$f_{\mathrm{NNflux}_{\nu_\mu}}(x_\nu)$",
                        r"$\mathrm{NN}$",
                    ],
                    handler_map={tuple: HandlerTuple(ndivide=1)},
                    loc="lower left",
                ).set_alpha(0.8)

                axL.set_xlim(5e-4, 1)
                axL.set_ylim(1e-0, 1e6)
                axL.set_yscale("log")
                axL.set_xscale("log")

                # if preproc == 1:
                title_str = rf"$ {generator.replace('_', '')} \ {obs.replace('_', '')} \ {geometry.replace('_', '')}$"

                fig.suptitle(title_str, fontsize=10)
                # if preproc == 2:
                # title_str = rf"$f_{{\mathrm{{NN}}}}(x) = \mathrm{{NN}}_{{{layers}}}(x) -  \mathrm{{NN}}_{{{layers}}}(1)$"

                # axL.set_title(title_str)
                # fig.text(0.33, 0.94, title_str, ha="center", va="bottom", fontsize=10)
                axL.set_ylabel(r"$xf_{\nu_\mu}(x_\nu) + xf_{\bar{\nu}_\mu}(x_\nu)$")
                axL.set_xticklabels([])
                axL.grid(color="grey", linestyle="-", linewidth=0.25)

                # ========== TOP MIDDLE (f_NN_mub vs f_faserv)

                # (axMsimb,) = axM.plot(
                #     x_vals,
                #     faser_pdf_mub,
                #     linestyle="-",
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
                #     linestyle="-",
                #     color=mubcolor,
                #     label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
                # )
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

                # axM.legend(
                #     # [axMsimb, (axMnnberr, axMnnb), (axMvert1, axMvert2)],
                #     [
                #         axMsimb,
                #         (axMnnberr, axMnnb),
                #     ],
                #     [
                #         r"$\mathrm{baseline}$",
                #         # r"$f_{\mathrm{EPOS+PWG}\nu_\mu}(x_\nu)$",
                #         # r"$f_{\mathrm{NNflux}_{\nu_\mu}}(x_\nu)$",
                #         r"$\mathrm{NN}$",
                #     ],
                #     handler_map={tuple: HandlerTuple(ndivide=1)},
                #     loc="lower left",
                # ).set_alpha(0.8)

                # # axM.set_ylabel(r'$xf_{\bar{\nu}_\mu}(x_\nu)$')

                # axM.set_xlim(5e-4, 1)
                # axM.set_ylim(1e-0, 1e6)
                # axM.set_yscale("log")
                # axM.set_xscale("log")
                # axM.set_xticklabels([])
                # axM.grid(color="grey", linestyle="-", linewidth=0.25)

                # ========== BOTTOM LEFT (Ratio plot, f_NN vs f_FASERv )

                ratio_center = mean_fnu_mu / faser_pdf_mu
                ratio_lower = (mean_fnu_mu - error_fnu_mu) / faser_pdf_mu
                ratio_upper = (mean_fnu_mu + error_fnu_mu) / faser_pdf_mu

                axrL.plot(x_vals, ratio_center, linestyle="-", color=mucolor)
                axrL.fill_between(
                    x_vals, ratio_upper, ratio_lower, color=mucolor, alpha=0.2
                )

                axrL.plot(x_vals, np.ones(len(x_vals)), linestyle="-", color=simcolor)

                axrL.set_xscale("log")
                axrL.set_xlim(5e-4, 1)
                axrL.set_ylim(0, 2)
                axrL.grid(color="grey", linestyle="-", linewidth=0.25)
                axrL.set_ylabel(r"$\mathrm{Ratio}$")
                axrL.set_xlabel(r"$x_\nu$")

                # ========== BOTTOM LEFT (Ratio plot, f_NN vs f_FASERv )

                ratio_center = mean_fnu_mu / faser_pdf_mu
                ratio_lower = (mean_fnu_mu - error_fnu_mu) / faser_pdf_mu
                ratio_upper = (mean_fnu_mu + error_fnu_mu) / faser_pdf_mu

                axrL.plot(x_vals, ratio_center, linestyle="-", color=mucolor)
                axrL.fill_between(
                    x_vals, ratio_upper, ratio_lower, color=mucolor, alpha=0.2
                )

                axrL.plot(x_vals, np.ones(len(x_vals)), linestyle="-", color=simcolor)

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
                # mean_fnu_mub
                # error_fnu_mub
                # axrM.plot(
                #     x_vals, mean_fnu_mub / faser_pdf_mub, linestyle="-", color=mubcolor
                # )
                # axrM.fill_between(
                #     x_vals,
                #     (mean_fnu_mub + error_fnu_mub) / faser_pdf_mub,
                #     (mean_fnu_mub - error_fnu_mub) / faser_pdf_mub,
                #     color=mubcolor,
                #     alpha=0.2,
                # )
                # axrM.plot(x_vals, np.ones(len(x_vals)), linestyle="-", color=simcolor)
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
                # axrM.set_xscale("log")
                # axrM.set_xlim(5e-4, 1)
                # axrM.set_ylim(0, 2)
                # axrM.grid(color="grey", linestyle="-", linewidth=0.25)
                # # axrM.set_ylabel(r"$\mathrm{Ratio}$")
                # axrM.set_xlabel(r"$x_\nu$")

                # =========== TOP RIGHT (Rates Enu vs FK otimes f_NN)

                x_vals_per_obs = low_bin
                x_vals_per_obs = np.append(x_vals_per_obs, high_bin[-1])

                pred_stds_Enu = np.append(pred_stds_Enu, pred_stds_Enu[-1])
                preds_Enu = np.append(preds_Enu, preds_Enu[-1])
                simulated_Enu = np.append(simulated_Enu, simulated_Enu[-1])
                errors_enu = np.append(errors_enu, errors_enu[-1])

                # pred_stds_Enub = np.append(pred_stds_Enub, pred_stds_Enub[-1])
                # preds_Enub = np.append(preds_Enub, preds_Enub[-1])
                # simulated_Enub = np.append(simulated_Enub, simulated_Enub[-1])
                # errors_enub = np.append(errors_enub, errors_enub[-1])

                # x_vals_per_obs_mub = low_bin_mub
                # x_vals_per_obs_mub = np.append(x_vals_per_obs_mub, high_bin_mub[-1])

                print("x_vals_per_obs_mub")
                # print(len(x_vals_per_obs_mub))
                # print(len(preds_Enub))

                print("x_vals_per_obs_mu")
                print(len(x_vals_per_obs))
                print(len(preds_Enu))
                print(len(pred_stds_Enu))

                if abs(len(x_vals_per_obs) - len(pred_stds_Enu)) > 0:
                    continue
                # if abs(len(x_vals_per_obs_mub) - len(pred_stds_Enub)) > 0:
                #     continue

                if len(x_vals_per_obs) - len(pred_stds_Enu) == 1:
                    x_vals_per_obs = x_vals_per_obs[:-1]
                    print("loopt de soep in")
                    print(obs, geometry, generator)
                if len(x_vals_per_obs) - len(pred_stds_Enu) == -1:
                    pred_stds_Enu = pred_stds_Enu[:-1]
                    preds_Enu = preds_Enu[:-1]
                    print("loopt de soep in")
                    print(obs, geometry, generator)

                # if len(x_vals_per_obs_mub) - len(pred_stds_Enub) == 1:
                #     x_vals_per_obs_mub = x_vals_per_obs_mub[:-1]
                #     print("loopt de soep in")
                #     print(obs, geometry, generator)
                # if len(x_vals_per_obs) - len(pred_stds_Enub) == -1:
                #     pred_stds_Enub = pred_stds_Enub[:-1]
                #     preds_Enub = preds_Enub[:-1]
                #     print("loopt de soep in")
                #     print(obs, geometry, generator)

                axRpred = axR.fill_between(
                    x_vals_per_obs,
                    # np.arange(len(simulated_Enu)),
                    simulated_Enu + errors_enu,
                    simulated_Enu - errors_enu,
                    step="post",
                    color="tab:orange",
                    alpha=0.6,
                    label=r"POWHEG $E_\nu$",
                )

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

                axRmeasmu = axR.fill_between(
                    x_vals_per_obs,
                    # np.arange(len(preds_Enu)),
                    preds_Enu + pred_stds_Enu,
                    preds_Enu - pred_stds_Enu,
                    step="post",
                    color="tab:blue",
                    alpha=0.6,
                    label=r"POWHEG $E_\nu$",
                )

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

                axR.legend(
                    [(axRmeasmu), (axRpred)],
                    [
                        # r"$\mathrm{DATA} \ E_\nu$",
                        r"$\mathrm{NN \ \mu}$",
                        r"$\mathrm{Input \ \mu} $",
                        # r"$\mathrm{NN \ \bar{\mu}}$",
                        # r"$\mathrm{Input \ \bar{\mu}} $",
                    ],
                    handler_map={tuple: HandlerTuple(ndivide=1)},
                    loc="upper left",
                ).set_alpha(0.8)

                # axR.set_xlim(0, 1)
                # axR.set_ylim(0)
                axR.set_yscale("log")
                axR.set_xscale("log")

                # axR.set_title(
                #     r"$\mathcal{L}_{\mathrm{pp}} = 150 \mathrm{fb}^{-1}$", loc="right"
                # )
                # axR.set_title(
                #     r"$\mathrm{Pseudo \ \ Data }, \ \mathrm{Level\ 2}$", loc="left"
                # )
                # axR.text(np.interp(1/500, xplot_ticks,ticks), 170, r"$\bar{\nu}_\mu$", alpha=0.8)
                # axR.text(np.interp(-1/400, xplot_ticks,ticks), 170, r"$\nu_\mu$", alpha=0.8)
                # axR.text(np.interp(-1/1010, xplot_ticks,ticks), 170, r"$\nu_\mu + \bar{\nu}_\mu$", alpha=0.8)
                axR.set_ylabel(r"$N_{\mathrm{int}} \  [\mathrm{GeV}]$")
                axR.set_xlabel(r"$E_h$")

                # ========= BOTTOM RIGHT (Ratio Rates Enu vs FK otimes f_NN)

                # plot mu bins

                axrRmeasmu = axrR.fill_between(
                    x_vals_per_obs,
                    # np.arange(len(preds_Enu)),
                    (preds_Enu + pred_stds_Enu) / simulated_Enu,
                    (preds_Enu - pred_stds_Enu) / simulated_Enu,
                    step="post",
                    color="tab:blue",
                    alpha=0.8,
                    label=r"POWHEG $E_\nu$",
                )

                axrRpred = axrR.fill_between(
                    x_vals_per_obs,
                    # np.arange(len(simulated_Enu)),
                    (simulated_Enu + errors_enu) / simulated_Enu,
                    (simulated_Enu - errors_enu) / simulated_Enu,
                    step="post",
                    color="tab:orange",
                    alpha=0.8,
                    label=r"POWHEG $E_\nu$",
                )

                # axrR.legend(
                #     [
                #         (axrRmeasmu),
                #         (axrRpred),
                #     ],
                #     [
                #         # r"$\mathrm{DATA} \ E_\nu$",
                #         r"$\mathrm{FK} \otimes  f_{\nu_{\bar{\mu}}, NN}$",
                #         r"$\mathrm{FK} \otimes f_{\nu_\mu, \mathrm{ref}}$",
                #     ],
                #     handler_map={tuple: HandlerTuple(ndivide=1)},
                #     loc="upper right",
                # ).set_alpha(0.8)

                axrR.set_ylabel(r"$\mathrm{Ratio}$")
                axrR.set_xlabel(r"$E_l$")
                axrR.set_ylim(0, 2)
                axrR.set_xscale("log")

                axrR.grid(color="grey", linestyle="-", linewidth=0.25)

                plt.tight_layout()
                plt.subplots_adjust(bottom=0.13)

                plt.savefig(f"{geometry}_{generator}_{obs}_l2_elec.pdf")
                plt.show()


x_vals = np.logspace(-5, 0, 1000)
plot(x_vals)
