from matplotlib.legend_handler import HandlerTuple
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import lhapdf
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

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


# Get number of reps from make runscripts
def plot(
    x_vals,
):
    observables = [
        "Eh",
        "El",
        "Enu",
        "theta",
    ]
    gens = [
        # "dpmjet",
        # "epos",
        "qgsjet",
        "sibyll",
    ]
    for obs in observables:
        for gen in gens:
            print(f"now plotting {obs} and {gen}")
            # obs = "Eh"
            # gen = "sibyll"

            neutrino_pdfs_mu_run3 = np.loadtxt(
                f"new_run_3_gens/fit_{gen}/{obs}_fit/pdf.txt", delimiter=","
            )
            neutrino_pdfs_mu_run3 *= 2
            # neutrino_pdfs_mub_run3 = np.loadtxt(
            #     f"new_run_3_gens/fit_{gen}/{obs}_fit/mub_pdf.txt", delimiter=","
            # )

            neutrino_pdfs_mu_2024faser = np.loadtxt(
                f"new_2024faser/fit_{gen}/{obs}_fit/pdf.txt", delimiter=","
            )
            # neutrino_pdfs_mub_2024faser = np.loadtxt(
            #     f"new_2024faser/fit_{gen}/{obs}_fit/mub_pdf.txt", delimiter=","
            # )

            neutrino_pdfs_mu_2024faser *= 2

            neutrino_pdfs_mu_faserv2 = np.loadtxt(
                f"new_faserv2/fit_{gen}/{obs}_fit/pdf.txt", delimiter=","
            )
            neutrino_pdfs_mu_faserv2 *= 2
            # neutrino_pdfs_mub_faserv2 = np.loadtxt(
            #     f"new_faserv2/fit_{gen}/{obs}_fit/mub_pdf.txt", delimiter=","
            # )

            neutrino_pdfs_mu_high_lumi = np.loadtxt(
                f"new_high_lumi/fit_{gen}/{obs}_fit/pdf.txt", delimiter=","
            )
            neutrino_pdfs_mu_high_lumi *= 2
            # neutrino_pdfs_mub_high_lumi = np.loadtxt(
            #     f"new_high_lumi/fit_{gen}/{obs}_fit/mub_pdf.txt", delimiter=","
            # )
            x_vals = np.array(x_vals)

            # same lumi
            if gen == "dpmjet":
                formal_gen = "DPMJET+DPMJET"

            if gen == "epos":
                formal_gen = "EPOS+POWHEG"
            if gen == "qgsjet":
                formal_gen = "QGSJET+POWHEG"
            if gen == "sibyll":
                formal_gen = "SIBYLL+SIBYLL"
            pdf_run3 = f"FASERv_Run3_{formal_gen}_7TeV"
            faser_pdf_mu_run3, x_faser = read_pdf(pdf_run3, x_vals, 12)
            faser_pdf_mub_run3, x_faser = read_pdf(pdf_run3, x_vals, -12)
            faser_pdf_mu_run3 = faser_pdf_mu_run3 * x_vals
            faser_pdf_mub_run3 = faser_pdf_mub_run3 * x_vals
            faser_pdf_mu_run3 += faser_pdf_mub_run3

            pdf_2024faser = f"FASER_2412.03186_{formal_gen}_7TeV"
            faser_pdf_mu_2024faser, x_faser = read_pdf(pdf_2024faser, x_vals, 12)
            faser_pdf_mub_2024faser, x_faser = read_pdf(pdf_2024faser, x_vals, -12)
            faser_pdf_mu_2024faser = faser_pdf_mu_2024faser * x_vals * 1.16186e-09
            faser_pdf_mub_2024faser = faser_pdf_mub_2024faser * x_vals * 1.16186e-09

            faser_pdf_mu_2024faser += faser_pdf_mub_2024faser

            pdf_faserv2 = f"FASERv2_{formal_gen}_7TeV"
            faser_pdf_mu_faserv2, x_faser = read_pdf(pdf_faserv2, x_vals, 12)
            faser_pdf_mub_faserv2, x_faser = read_pdf(pdf_faserv2, x_vals, -12)
            faser_pdf_mu_faserv2 = faser_pdf_mu_faserv2 * x_vals
            faser_pdf_mub_faserv2 = faser_pdf_mub_faserv2 * x_vals
            faser_pdf_mu_faserv2 += faser_pdf_mub_faserv2

            pdf_high_lumi = f"FASERv_Run3_{formal_gen}_7TeV"
            faser_pdf_mu_high_lumi, x_faser = read_pdf(pdf_high_lumi, x_vals, 12)
            faser_pdf_mub_high_lumi, x_faser = read_pdf(pdf_high_lumi, x_vals, -12)
            faser_pdf_mu_high_lumi = faser_pdf_mu_high_lumi * x_vals * 20
            faser_pdf_mub_high_lumi = faser_pdf_mub_high_lumi * x_vals * 20
            faser_pdf_mu_high_lumi += faser_pdf_mub_high_lumi

            # faser_pdf_mu = gaussian_filter1d(faser_pdf_mu, sigma=3)
            # faser_pdf_mub = gaussian_filter1d(faser_pdf_mub, sigma=3)

            mean_mu_run3 = np.mean(neutrino_pdfs_mu_run3, axis=0) * x_vals
            error_mu_run3 = np.std(neutrino_pdfs_mu_run3, axis=0) * x_vals
            # mean_mub_run3 = np.mean(neutrino_pdfs_mub_run3, axis=0) * x_vals
            # error_mub_run3 = np.std(neutrino_pdfs_mub_run3, axis=0) * x_vals

            mean_mu_2024faser = np.mean(neutrino_pdfs_mu_2024faser, axis=0) * x_vals
            error_mu_2024faser = np.std(neutrino_pdfs_mu_2024faser, axis=0) * x_vals
            # mean_mub_2024faser = np.mean(neutrino_pdfs_mub_2024faser, axis=0) * x_vals
            # error_mub_2024faser = np.std(neutrino_pdfs_mub_2024faser, axis=0) * x_vals

            mean_mu_faserv2 = np.mean(neutrino_pdfs_mu_faserv2, axis=0) * x_vals
            error_mu_faserv2 = np.std(neutrino_pdfs_mu_faserv2, axis=0) * x_vals
            # mean_mub_faserv2 = np.mean(neutrino_pdfs_mub_faserv2, axis=0) * x_vals
            # error_mub_faserv2 = np.std(neutrino_pdfs_mub_faserv2, axis=0) * x_vals

            mean_mu_high_lumi = np.mean(neutrino_pdfs_mu_high_lumi, axis=0) * x_vals
            error_mu_high_lumi = np.std(neutrino_pdfs_mu_high_lumi, axis=0) * x_vals
            # mean_mub_high_lumi = np.mean(neutrino_pdfs_mub_high_lumi, axis=0) * x_vals
            # error_mub_high_lumi = np.std(neutrino_pdfs_mub_high_lumi, axis=0) * x_vals

            plt.rcParams["text.usetex"] = True
            plt.rcParams.update(
                {
                    # "font.family": "serif",
                    # "font.serif": ["cmr10"],  # Computer Modern]
                    "font.size": 10,
                }
            )
            fig = plt.figure(figsize=(8.636, 9.0), dpi=300)  # 2 rows, 3 columns
            gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 1])

            gs.update(
                left=0.07, right=0.97, top=0.92, bottom=0.08, hspace=0.15, wspace=0.15
            )

            axL = fig.add_subplot(gs[0, 0])
            # axM = fig.add_subplot(gs[0, 1])
            # axR = fig.add_subplot(gs[0, 2])
            axrL = fig.add_subplot(gs[1, 0])
            # axrM = fig.add_subplot(gs[1, 1])
            axLsig = fig.add_subplot(gs[2, 0])
            # axMsig = fig.add_subplot(gs[2, 1])
            # axrR = fig.add_subplot(gs[1, 2])

            # TOP LEFT PLOT

            # (axLsim,) = axL.plot(
            #     x_vals,
            #     faser_pdf_mu,
            #     linestyle="-",
            #     label=r"$f_{\mathrm{EPOS+POWHEG}\nu_\mu}(x)$",
            #     color=simcolor,
            # )

            axLrun3_err = axL.fill_between(
                x_vals,
                mean_mu_run3 + error_mu_run3,
                mean_mu_run3 - error_mu_run3,
                color="#648fff",
                alpha=0.2,
                label=r"$\pm 1\sigma$",
            )

            (axLrun3,) = axL.plot(
                x_vals,
                mean_mu_run3,
                linestyle="-.",
                color="#648fff",
                label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            )

            axL2024_err = axL.fill_between(
                x_vals,
                mean_mu_2024faser + error_mu_2024faser,
                mean_mu_2024faser - error_mu_2024faser,
                color="#fe6100",
                alpha=0.2,
                label=r"$\pm 1\sigma$",
            )

            (axL2024,) = axL.plot(
                x_vals,
                mean_mu_2024faser,
                linestyle="--",
                color="#fe6100",
                label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            )

            axLfaserv2_err = axL.fill_between(
                x_vals,
                mean_mu_faserv2 + error_mu_faserv2,
                mean_mu_faserv2 - error_mu_faserv2,
                color="#dc267f",
                alpha=0.2,
                label=r"$\pm 1\sigma$",
            )

            (axLfaserv2,) = axL.plot(
                x_vals,
                mean_mu_faserv2,
                linestyle=":",
                color="#dc267f",
                label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            )

            axLhigh_lumi_err = axL.fill_between(
                x_vals,
                mean_mu_high_lumi + error_mu_high_lumi,
                mean_mu_high_lumi - error_mu_high_lumi,
                color="#000000",
                alpha=0.2,
                label=r"$\pm 1\sigma$",
            )

            (axLhigh_lumi,) = axL.plot(
                x_vals,
                mean_mu_high_lumi,
                linestyle="-",
                markersize=0.4,
                color="#000000",
                label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            )

            axL.legend(
                [
                    # axLsim,
                    (axLrun3_err, axLrun3),
                    (axL2024_err, axL2024),
                    (axLfaserv2_err, axLfaserv2),
                    (axLhigh_lumi_err, axLhigh_lumi),
                ],
                [
                    # r"${ \mathrm{baseline}}$",
                    r"$ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$",
                    r"$ \mathrm{FASER} 2412.03186$",
                    r"${ \mathrm{FASER}\nu2}$",
                    r"${ \mathrm{FASER}\nu \ \mathrm{Run} \ 4+5}$",
                ],
                handler_map={tuple: HandlerTuple(ndivide=1)},
                loc="lower left",
                ncols=1,
                # fontsize="small",
                handlelength=1.2,
                handletextpad=0.4,
                borderpad=0.3,
                labelspacing=0.3,
            ).set_alpha(0.8)

            axL.set_xlim(0.02, 1)
            axL.set_ylim(1e-0, 1e6)
            axL.set_yscale("log")
            axL.set_xscale("log")

            if obs == "Eh" and gen == "dpmjet":
                title_str = r"$\mathrm{Comparison} \ \mathrm{Detector} \ \mathrm{Geometries} \ E_h, \mathrm{DPMJET}(\pi,K,c)$"
            if obs == "Eh" and gen == "epos":
                title_str = r"$\mathrm{{Comparison}} \ \mathrm{{Detector}} \ \mathrm{{Geometries}} \ E_h, \mathrm{EPOS(\pi,K)+POWHEG(c)}$"
            if obs == "Eh" and gen == "qgsjet":
                title_str = r"$\mathrm{{Comparison}} \ \mathrm{{Detector}} \ \mathrm{{Geometries}} \ E_h, \mathrm{QGSJET(\pi,K)+POWHEG(c)}$"
            if obs == "Eh" and gen == "sibyll":
                title_str = r"$\mathrm{{Comparison}} \ \mathrm{{Detector}} \ \mathrm{{Geometries}} \ E_h, \mathrm{SIBYLL(\pi,K,c)}$"

            if obs == "El" and gen == "dpmjet":
                title_str = r"$\mathrm{{Comparison}} \ \mathrm{{Detector}} \ \mathrm{{Geometries}} \ E_\ell, \mathrm{DPMJET}(\pi,K,c)$"
            if obs == "El" and gen == "epos":
                title_str = r"$\mathrm{{Comparison}} \ \mathrm{{Detector}} \ \mathrm{{Geometries}} \ E_\ell, \mathrm{EPOS(\pi,K)+POWHEG(c)}$"
            if obs == "El" and gen == "qgsjet":
                title_str = r"$\mathrm{{Comparison}} \ \mathrm{{Detector}} \ \mathrm{{Geometries}} \ E_\ell, \mathrm{QGSJET(\pi,K)+POWHEG(c)}$"
            if obs == "El" and gen == "sibyll":
                title_str = r"$\mathrm{{Comparison}} \ \mathrm{{Detector}} \ \mathrm{{Geometries}} \ E_\ell, \mathrm{SIBYLL(\pi,K,c)}$"

            if obs == "Enu" and gen == "dpmjet":
                title_str = r"$\mathrm{{Comparison}} \ \mathrm{{Detector}} \ \mathrm{{Geometries}} \ E_\nu, \mathrm{DPMJET}(\pi,K,c)$"
            if obs == "Enu" and gen == "epos":
                title_str = r"$\mathrm{{Comparison}} \ \mathrm{{Detector}} \ \mathrm{{Geometries}} \ E_\nu, \mathrm{EPOS(\pi,K)+POWHEG(c)}$"
            if obs == "Enu" and gen == "qgsjet":
                title_str = r"$\mathrm{{Comparison}} \ \mathrm{{Detector}} \ \mathrm{{Geometries}} \ E_\nu, \mathrm{QGSJET(\pi,K)+POWHEG(c)}$"
            if obs == "Enu" and gen == "sibyll":
                title_str = r"$\mathrm{{Comparison}} \ \mathrm{{Detector}} \ \mathrm{{Geometries}} \ E_\nu, \mathrm{SIBYLL(\pi,K,c)}$"

            if obs == "theta" and gen == "dpmjet":
                title_str = r"$\mathrm{{Comparison}} \ \mathrm{{Detector}} \ \mathrm{{Geometries}} \ \theta, \mathrm{DPMJET}(\pi,K,c)$"
            if obs == "theta" and gen == "epos":
                title_str = r"$\mathrm{{Comparison}} \ \mathrm{{Detector}} \ \mathrm{{Geometries}} \ \theta, \mathrm{EPOS(\pi,K)+POWHEG(c)}$"
            if obs == "theta" and gen == "qgsjet":
                title_str = r"$\mathrm{{Comparison}} \ \mathrm{{Detector}} \ \mathrm{{Geometries}} \ \theta, \mathrm{QGSJET(\pi,K)+POWHEG(c)}$"
            if obs == "theta" and gen == "sibyll":
                title_str = r"$\mathrm{{Comparison}} \ \mathrm{{Detector}} \ \mathrm{{Geometries}} \ \theta, \mathrm{SIBYLL(\pi,K,c)}$"

            fig.suptitle(title_str, fontsize=14)

            # axL.set_ylabel(r"$xf_{\nu_e}(x_\nu)$", fontsize=10)
            axL.set_ylabel(
                r"$xf_{\nu_e}(x_\nu) + xf_{\bar{\nu}_e}(x_\nu)$", labelpad=10
            )

            axL.set_xticklabels([])
            axL.grid(color="grey", linestyle="-", linewidth=0.25)

            # ========== TOP MIDDLE (f_NN_mub vs f_faserv)

            # (axMsimb,) = axM.plot(
            #     x_vals,
            #     faser_pdf_mub,
            #     linestyle="-",
            #     label=r"$f_{\mathrm{EPOS+POWHEG}\bar{\nu}_\mu}(x)$",
            #     color=simcolor,
            # )

            # axMrun3_err = axM.fill_between(
            #     x_vals,
            #     (mean_mub_run3 + error_mub_run3),
            #     (mean_mub_run3 - error_mub_run3),
            #     color="#648fff",
            #     alpha=0.2,
            #     label=r"$\pm 1\sigma$",
            # )
            # (axMrun3,) = axM.plot(
            #     x_vals,
            #     mean_mub_run3,
            #     linestyle="-.",
            #     color="#648fff",
            #     label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            # )

            # # axM2024_err = axM.fill_between(
            # #     x_vals,
            # #     (mean_mub_2024faser + error_mub_2024faser),
            # #     (mean_mub_2024faser - error_mub_2024faser),
            # #     color="#fe6100",
            # #     alpha=0.2,
            # #     label=r"$\pm 1\sigma$",
            # # )
            # # (axM2024,) = axM.plot(
            # #     x_vals,
            # #     mean_mub_2024faser,
            # #     linestyle="--",
            # #     color="#fe6100",
            # #     label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            # # )

            # axMfaserv2_err = axM.fill_between(
            #     x_vals,
            #     (mean_mub_faserv2 + error_mub_faserv2),
            #     (mean_mub_faserv2 - error_mub_faserv2),
            #     color="#dc267f",
            #     alpha=0.2,
            #     label=r"$\pm 1\sigma$",
            # )
            # (axMfaserv2,) = axM.plot(
            #     x_vals,
            #     mean_mub_faserv2,
            #     linestyle=":",
            #     color="#dc267f",
            #     label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            # )

            # axMhigh_lumi_err = axM.fill_between(
            #     x_vals,
            #     (mean_mub_high_lumi + error_mub_high_lumi),
            #     (mean_mub_high_lumi - error_mub_high_lumi),
            #     color="#000000",
            #     alpha=0.2,
            #     label=r"$\pm 1\sigma$",
            # )
            # (axMhigh_lumi,) = axM.plot(
            #     x_vals,
            #     mean_mub_high_lumi,
            #     linestyle="-",
            #     color="#000000",
            #     label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            # )

            # axM.legend(
            #     # [axMsimb, (axMnnberr, axMnnb), (axMvert1, axMvert2)],
            #     [
            #         # axMsimb,
            #         # (axMfasererr, axMfaser),
            #         (axMrun3_err, axMrun3),
            #         # (axM2024_err, axM2024),
            #         (axMfaserv2_err, axMfaserv2),
            #         (axMhigh_lumi_err, axMhigh_lumi),
            #     ],
            #     [
            #         r"${ \mathrm{FASER}\nu \ \mathrm{Run} \ 3}$",
            #         # r"${ \mathrm{FASER} 2412.03186}$",
            #         r"${ \mathrm{FASER}\nu2}$",
            #         r"${ \mathrm{FASER}\nu \ \mathrm{Run} \ 4+5}$",
            #     ],
            #     handler_map={tuple: HandlerTuple(ndivide=1)},
            #     loc="lower left",
            #     ncols=1,
            #     # fontsize="small",
            #     handlelength=1.2,
            #     handletextpad=0.4,
            #     borderpad=0.3,
            #     labelspacing=0.3,
            # ).set_alpha(0.8)

            # # title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \ \mathrm{EPOS} + \ \mathrm{POWHEG}, \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"

            # axM.set_ylabel(r"$xf_{\bar{\nu}_e}(x_\nu)$", fontsize=10)

            # axM.set_xlim(1e-2, 1)
            # axM.set_ylim(1e-0, 1e6)
            # axM.set_yscale("log")
            # axM.set_xscale("log")
            # axM.set_xticklabels([])
            # axM.grid(color="grey", linestyle="-", linewidth=0.25)

            # ========== BOTTOM LEFT (Ratio plot, f_NN vs f_FASERv )

            axrL.plot(
                x_vals,
                np.ones(len(x_vals)),
                linestyle="-",
                color=simcolor,
                label=r"$\mathrm{baseline (\nu_e + \bar{\nu}_e)}$",
            )

            axrL.set_xscale("log")
            # axrL.set_xlim(1e-2, 1)
            axrL.set_ylim(0.8, 1.2)
            axrL.grid(color="grey", linestyle="-", linewidth=0.25)
            axrL.set_ylabel(r"$\mathrm{Ratio}$")
            # axrL.set_xlabel(r"$x_\nu$")

            ratio_center = mean_mu_run3 / faser_pdf_mu_run3
            ratio_lower = (mean_mu_run3 - error_mu_run3) / faser_pdf_mu_run3
            ratio_upper = (mean_mu_run3 + error_mu_run3) / faser_pdf_mu_run3

            axrL.plot(x_vals, ratio_center, linestyle="-.", color="#648fff")
            axrL.fill_between(
                x_vals, ratio_upper, ratio_lower, color="#648fff", alpha=0.2
            )

            ratio_center = mean_mu_2024faser / faser_pdf_mu_2024faser
            ratio_lower = (
                mean_mu_2024faser - error_mu_2024faser
            ) / faser_pdf_mu_2024faser
            ratio_upper = (
                mean_mu_2024faser + error_mu_2024faser
            ) / faser_pdf_mu_2024faser

            axrL.plot(x_vals, ratio_center, linestyle="--", color="#fe6100")
            axrL.fill_between(
                x_vals, ratio_upper, ratio_lower, color="#fe6100", alpha=0.2
            )

            ratio_center = mean_mu_faserv2 / faser_pdf_mu_faserv2
            ratio_lower = (mean_mu_faserv2 - error_mu_faserv2) / faser_pdf_mu_faserv2
            ratio_upper = (mean_mu_faserv2 + error_mu_faserv2) / faser_pdf_mu_faserv2

            axrL.plot(x_vals, ratio_center, linestyle=":", color="#dc267f")
            axrL.fill_between(
                x_vals, ratio_upper, ratio_lower, color="#dc267f", alpha=0.2
            )

            ratio_center = mean_mu_high_lumi / faser_pdf_mu_high_lumi
            ratio_lower = (
                mean_mu_high_lumi - error_mu_high_lumi
            ) / faser_pdf_mu_high_lumi
            ratio_upper = (
                mean_mu_high_lumi + error_mu_high_lumi
            ) / faser_pdf_mu_high_lumi

            axrL.plot(x_vals, ratio_center, linestyle="-", color="#000000")
            axrL.fill_between(
                x_vals, ratio_upper, ratio_lower, color="#000000", alpha=0.2
            )

            axrL.set_xscale("log")
            axrL.set_xlim(0.02, 1)
            axrL.set_ylim(0.5, 1.5)
            axrL.grid(color="grey", linestyle="-", linewidth=0.25)
            axrL.set_ylabel(r"$\mathrm{Ratio}$")
            # axrL.set_xlabel(r"$x_\nu$")
            axrL.tick_params(labelbottom=False)

            axrL.legend()

            # ========== BOTTOM MIDDLE (ratio f_NN_mub)

            # axrM.plot(
            #     x_vals,
            #     np.ones(len(x_vals)),
            #     linestyle="-",
            #     color=simcolor,
            #     label=r"$\mathrm{baseline}$",
            # )

            # axrM.plot(
            #     x_vals,
            #     mean_mub_run3 / faser_pdf_mub_run3,
            #     linestyle="-.",
            #     color="#648fff",
            # )
            # axrM.fill_between(
            #     x_vals,
            #     (mean_mub_run3 + error_mub_run3) / faser_pdf_mub_run3,
            #     (mean_mub_run3 - error_mub_run3) / faser_pdf_mub_run3,
            #     color="#648fff",
            #     alpha=0.2,
            # )

            # axrM.plot(
            #     x_vals,
            #     mean_mub_2024faser / faser_pdf_mub_2024faser,
            #     linestyle="--",
            #     color="#fe6100",
            # )
            # axrM.fill_between(
            #     x_vals,
            #     (mean_mub_2024faser + error_mub_2024faser) / faser_pdf_mub_2024faser,
            #     (mean_mub_2024faser - error_mub_2024faser) / faser_pdf_mub_2024faser,
            #     color="#fe6100",
            #     alpha=0.2,
            # )

            # axrM.plot(
            #     x_vals,
            #     mean_mub_faserv2 / faser_pdf_mub_faserv2,
            #     linestyle=":",
            #     color="#dc267f",
            # )
            # axrM.fill_between(
            #     x_vals,
            #     (mean_mub_faserv2 + error_mub_faserv2) / faser_pdf_mub_faserv2,
            #     (mean_mub_faserv2 - error_mub_faserv2) / faser_pdf_mub_faserv2,
            #     color="#dc267f",
            #     alpha=0.2,
            # )

            # axrM.plot(
            #     x_vals,
            #     mean_mub_high_lumi / faser_pdf_mub_high_lumi,
            #     linestyle="-",
            #     color="#000000",
            # )
            # axrM.fill_between(
            #     x_vals,
            #     (mean_mub_high_lumi + error_mub_high_lumi) / faser_pdf_mub_high_lumi,
            #     (mean_mub_high_lumi - error_mub_high_lumi) / faser_pdf_mub_high_lumi,
            #     color="#000000",
            #     alpha=0.2,
            # )

            # axrM.set_xscale("log")
            # axrM.set_xlim(1e-2, 1)
            # axrM.set_ylim(0.5, 1.5)
            # axrM.grid(color="grey", linestyle="-", linewidth=0.25)
            # axrM.set_ylabel(r"$\mathrm{Ratio}$")
            # # axrM.set_xlabel(r"$x_\nu$")
            # axrM.legend()
            # axrM.tick_params(labelbottom=False)

            # # 1 sigma error bands

            axLsig.plot(
                x_vals,
                (mean_mu_run3 + error_mu_run3) / mean_mu_run3 - 1,
                linestyle="-.",
                color="#648fff",
            )
            axLsig.fill_between(
                x_vals,
                (mean_mu_run3 + error_mu_run3) / mean_mu_run3,
                (mean_mu_run3 - error_mu_run3) / mean_mu_run3,
                color="#648fff",
                alpha=0.2,
            )

            axLsig.plot(
                x_vals,
                (mean_mu_2024faser + error_mu_2024faser) / mean_mu_2024faser - 1,
                linestyle="--",
                color="#fe6100",
            )

            axLsig.plot(
                x_vals,
                (mean_mu_high_lumi + error_mu_high_lumi) / mean_mu_high_lumi - 1,
                linestyle="-",
                color="#000000",
            )

            axLsig.plot(
                x_vals,
                (mean_mu_faserv2 + error_mu_faserv2) / mean_mu_faserv2 - 1,
                linestyle=":",
                color="#dc267f",
            )

            axLsig.set_xscale("log")
            axLsig.set_xlim(0.02, 1)
            axLsig.set_ylim(0, 0.5)
            axLsig.grid(color="grey", linestyle="-", linewidth=0.25)
            axLsig.set_ylabel(r"$68\% \ \mathrm{CL} \ \mathrm{error}$")
            axLsig.set_xlabel(r"$x_\nu$", fontsize=10)

            # axMsig.plot(
            #     x_vals,
            #     (mean_mub_run3 + error_mub_run3) / mean_mub_run3 - 1,
            #     linestyle="-.",
            #     color="#648fff",
            # )

            # # axMsig.plot(
            # #     x_vals,
            # #     (mean_mub_2024faser + error_mub_2024faser) / mean_mub_2024faser - 1,
            # #     linestyle="--",
            # #     color="#fe6100",
            # # )

            # axMsig.plot(
            #     x_vals,
            #     (mean_mub_high_lumi + error_mub_high_lumi) / mean_mub_high_lumi - 1,
            #     linestyle="-",
            #     color="#000000",
            # )

            # axMsig.plot(
            #     x_vals,
            #     (mean_mub_faserv2 + error_mub_faserv2) / mean_mub_faserv2 - 1,
            #     linestyle=":",
            #     color="#dc267f",
            # )

            # axMsig.set_xscale("log")
            # axMsig.set_xlim(1e-2, 1)
            # axMsig.set_ylim(0, 0.5)
            # axMsig.grid(color="grey", linestyle="-", linewidth=0.25)
            # axMsig.set_ylabel(r"$68\% \ \mathrm{CL} \ \mathrm{error}$")
            # axMsig.set_xlabel(r"$x_\nu$", fontsize=10)

            # if geometry == "new_high_lumi" or geometry == "new_faserv2":
            axL.text(
                0.1,
                5 * 10**5,
                r"$(\nu_e + \bar{\nu}_e) W \rightarrow (e+e^{+}) X_h$",
                fontsize=12,
                color="red",
            )

            # else:
            #     axL.text(
            #         0.1,
            #         10**3,
            #         r"$(nu_e + \bar{nu}_e) W \rightarrow (e+e^{+}) X_h$",
            #         fontsize=12,
            #         color="red",
            #     )

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.13)
            plt.savefig(f"{gen}_{obs}_compare_geoms_elec.pdf")
            # plt.show()
    # plt.show()


x_vals = np.logspace(-5, 0, 1000)
plot(x_vals)
