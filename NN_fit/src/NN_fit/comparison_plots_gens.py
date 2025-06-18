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
    obs = "Eh"
    # geometry = "run_3_gens"
    observables = [
        "Eh",
        "El",
        "Enu",
        "theta",
    ]
    geometries = [
        "run_3_gens",
        # "2024_faser",
        # "faserv2",
        "high_lumi_faserv",
    ]
    for obs in observables:
        for geometry in geometries:
            print(f"now plotting {obs} and {geometry}")
            neutrino_pdfs_mu_dpm = np.loadtxt(
                f"{geometry}/fit_dpmjet/{obs}_fit/mu_pdf.txt", delimiter=","
            )
            neutrino_pdfs_mub_dpm = np.loadtxt(
                f"{geometry}/fit_dpmjet/{obs}_fit/mub_pdf.txt", delimiter=","
            )

            neutrino_pdfs_mu_epos = np.loadtxt(
                f"{geometry}/fit_epos/{obs}_fit/mu_pdf.txt", delimiter=","
            )
            neutrino_pdfs_mub_epos = np.loadtxt(
                f"{geometry}/fit_epos/{obs}_fit/mub_pdf.txt", delimiter=","
            )

            neutrino_pdfs_mu_qgsjet = np.loadtxt(
                f"{geometry}/fit_qgsjet/{obs}_fit/mu_pdf.txt", delimiter=","
            )
            neutrino_pdfs_mub_qgsjet = np.loadtxt(
                f"{geometry}/fit_qgsjet/{obs}_fit/mub_pdf.txt", delimiter=","
            )

            neutrino_pdfs_mu_sibyll = np.loadtxt(
                f"{geometry}/fit_sibyll/{obs}_fit/mu_pdf.txt", delimiter=","
            )
            neutrino_pdfs_mub_sibyll = np.loadtxt(
                f"{geometry}/fit_sibyll/{obs}_fit/mub_pdf.txt", delimiter=","
            )

            x_vals = np.array(x_vals)

            if geometry == "run_3_gens":
                formal_geometry = "FASERv_Run3"
                factor = 1
            if geometry == "2024_faser":
                formal_geometry = "FASER_2412.03186"
                factor = 1.16186e-09
            if geometry == "faserv2":
                formal_geometry = "FASERv2"
                factor = 1
            if geometry == "high_lumi_faserv":
                formal_geometry = "FASERv_Run3"
                factor = 20

            pdf_dpm = f"{formal_geometry}_DPMJET+DPMJET_7TeV"
            faser_pdf_mu_dpm, x_faser = read_pdf(pdf_dpm, x_vals, 14)
            faser_pdf_mub_dpm, x_faser = read_pdf(pdf_dpm, x_vals, -14)
            faser_pdf_mu_dpm = faser_pdf_mu_dpm * x_vals * factor
            faser_pdf_mub_dpm = faser_pdf_mub_dpm * x_vals * factor

            pdf_epos = f"{formal_geometry}_EPOS+POWHEG_7TeV"
            faser_pdf_mu_epos, x_faser = read_pdf(pdf_epos, x_vals, 14)
            faser_pdf_mub_epos, x_faser = read_pdf(pdf_epos, x_vals, -14)
            faser_pdf_mu_epos = faser_pdf_mu_epos * x_vals * factor
            faser_pdf_mub_epos = faser_pdf_mub_epos * x_vals * factor

            pdf_qgs = f"{formal_geometry}_QGSJET+POWHEG_7TeV"
            faser_pdf_mu_qgs, x_faser = read_pdf(pdf_qgs, x_vals, 14)
            faser_pdf_mub_qgs, x_faser = read_pdf(pdf_qgs, x_vals, -14)
            faser_pdf_mu_qgs = faser_pdf_mu_qgs * x_vals * factor
            faser_pdf_mub_qgs = faser_pdf_mub_qgs * x_vals * factor

            pdf_sib = f"{formal_geometry}_SIBYLL+SIBYLL_7TeV"
            faser_pdf_mu_sib, x_faser = read_pdf(pdf_sib, x_vals, 14)
            faser_pdf_mub_sib, x_faser = read_pdf(pdf_sib, x_vals, -14)
            faser_pdf_mu_sib = faser_pdf_mu_sib * x_vals * factor
            faser_pdf_mub_sib = faser_pdf_mub_sib * x_vals * factor

            # faser_pdf_mu = gaussian_filter1d(faser_pdf_mu, sigma=3)
            # faser_pdf_mub = gaussian_filter1d(faser_pdf_mub, sigma=3)

            mean_mu_dpm = np.mean(neutrino_pdfs_mu_dpm, axis=0) * x_vals
            error_mu_dpm = np.std(neutrino_pdfs_mu_dpm, axis=0) * x_vals
            mean_mub_dpm = np.mean(neutrino_pdfs_mub_dpm, axis=0) * x_vals
            error_mub_dpm = np.std(neutrino_pdfs_mub_dpm, axis=0) * x_vals

            mean_mu_epos = np.mean(neutrino_pdfs_mu_epos, axis=0) * x_vals
            error_mu_epos = np.std(neutrino_pdfs_mu_epos, axis=0) * x_vals
            mean_mub_epos = np.mean(neutrino_pdfs_mub_epos, axis=0) * x_vals
            error_mub_epos = np.std(neutrino_pdfs_mub_epos, axis=0) * x_vals

            mean_mu_dpm_qgsjet = np.mean(neutrino_pdfs_mu_qgsjet, axis=0) * x_vals
            error_mu_dpm_qgsjet = np.std(neutrino_pdfs_mu_qgsjet, axis=0) * x_vals
            mean_mub_dpm_qgsjet = np.mean(neutrino_pdfs_mub_qgsjet, axis=0) * x_vals
            error_mub_dpm_qgsjet = np.std(neutrino_pdfs_mub_qgsjet, axis=0) * x_vals

            mean_mu_sibyll = np.mean(neutrino_pdfs_mu_sibyll, axis=0) * x_vals
            error_mu_sibyll = np.std(neutrino_pdfs_mu_sibyll, axis=0) * x_vals
            mean_mub_sibyll = np.mean(neutrino_pdfs_mub_sibyll, axis=0) * x_vals
            error_mub_sibyll = np.std(neutrino_pdfs_mub_sibyll, axis=0) * x_vals

            plt.rcParams["text.usetex"] = True
            plt.rcParams.update(
                {
                    # "font.family": "serif",
                    # "font.serif": ["cmr10"],  # Computer Modern]
                    "font.size": 10,
                }
            )
            fig = plt.figure(figsize=(8.636, 9.0), dpi=300)  # 2 rows, 3 columns
            gs = gridspec.GridSpec(3, 2, height_ratios=[3, 2, 1])

            gs.update(
                left=0.12, right=0.97, top=0.92, bottom=0.08, hspace=0.15, wspace=0.15
            )

            axL = fig.add_subplot(gs[0, 0])
            axM = fig.add_subplot(gs[0, 1])
            # axR = fig.add_subplot(gs[0, 2])
            axrL = fig.add_subplot(gs[1, 0])
            axrM = fig.add_subplot(gs[1, 1])
            axLsig = fig.add_subplot(gs[2, 0])
            axMsig = fig.add_subplot(gs[2, 1])
            # axrR = fig.add_subplot(gs[1, 2])

            # TOP LEFT PLOT

            # (axLsim,) = axL.plot(
            #     x_vals,
            #     faser_pdf_mu,
            #     linestyle="-",
            #     label=r"$f_{\mathrm{EPOS+POWHEG}\nu_\mu}(x)$",
            #     color=simcolor,
            # )

            axLdpm_err = axL.fill_between(
                x_vals,
                mean_mu_dpm + error_mu_dpm,
                mean_mu_dpm - error_mu_dpm,
                color="#648fff",
                alpha=0.2,
                label=r"$\pm 1\sigma$",
            )

            (axLdpm,) = axL.plot(
                x_vals,
                mean_mu_dpm,
                linestyle="-.",
                color="#648fff",
                label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            )

            axLepos_err = axL.fill_between(
                x_vals,
                mean_mu_epos + error_mu_epos,
                mean_mu_epos - error_mu_epos,
                color="#fe6100",
                alpha=0.2,
                label=r"$\pm 1\sigma$",
            )

            (axLepos,) = axL.plot(
                x_vals,
                mean_mu_epos,
                linestyle="--",
                color="#fe6100",
                label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            )

            axLdpm_qgsjet_err = axL.fill_between(
                x_vals,
                mean_mu_dpm_qgsjet + error_mu_dpm_qgsjet,
                mean_mu_dpm_qgsjet - error_mu_dpm_qgsjet,
                color="purple",
                alpha=0.2,
                label=r"$\pm 1\sigma$",
            )

            (axLdpm_qgsjet,) = axL.plot(
                x_vals,
                mean_mu_dpm_qgsjet,
                linestyle=":",
                color="purple",
                label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            )

            axLsibyll_err = axL.fill_between(
                x_vals,
                mean_mu_sibyll + error_mu_sibyll,
                mean_mu_sibyll - error_mu_sibyll,
                color="#000000",
                alpha=0.2,
                label=r"$\pm 1\sigma$",
            )

            (axLsibyll,) = axL.plot(
                x_vals,
                mean_mu_sibyll,
                linestyle="-",
                markersize=0.4,
                color="#000000",
                label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            )

            axL.legend(
                [
                    # axLsim,
                    (axLdpm_err, axLdpm),
                    (axLepos_err, axLepos),
                    (axLsibyll_err, axLsibyll),
                    (axLdpm_qgsjet_err, axLdpm_qgsjet),
                ],
                [
                    # r"${ \mathrm{baseline}}$",
                    r"$\mathrm{DPMJET(\pi,K,c)}$",
                    r"$ \mathrm{EPOS(\pi,K)+POWHEG(c)}$",
                    r"$ \mathrm{SIBYLL(\pi,K,c)}$",
                    r"$ \mathrm{QGSJET(\pi,K)+POWHEG(c)}$",
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
            if geometry == "high_lumi_faserv" or geometry == "faserv2":
                axL.set_ylim(1e-0, 1e6)
            else:
                axL.set_ylim(1e-1, 1e4)
            axL.set_yscale("log")
            axL.set_xscale("log")
            # geometries = [
            #     "run_3_gens",
            #     "2024_faser",
            #     "faserv2",
            #     "high_lumi_faserv",
            # ]
            if obs == "Eh" and geometry == "run_3_gens":
                title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_h, \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"
            if obs == "Eh" and geometry == "2024_faser":
                title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_h, \ \mathcal{L}_{\mathrm{pp}} = 65.6 \ \mathrm{fb}^{-1} \ \mathrm{FASER} \ 2412.03186$"
            if obs == "Eh" and geometry == "faserv2":
                title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_h, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu2 $"
            if obs == "Eh" and geometry == "high_lumi_faserv":
                title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_h, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 4+5$"

            if obs == "El" and geometry == "run_3_gens":
                title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_\ell, \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"
            if obs == "El" and geometry == "2024_faser":
                title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_\ell, \ \mathcal{L}_{\mathrm{pp}} = 65.6 \ \mathrm{fb}^{-1} \ \mathrm{FASER} \ 2412.03186$"
            if obs == "El" and geometry == "faserv2":
                title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_\ell, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu2 $"
            if obs == "El" and geometry == "high_lumi_faserv":
                title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_\ell, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 4+5$"

            if obs == "Enu" and geometry == "run_3_gens":
                title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_\nu, \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"
            if obs == "Enu" and geometry == "2024_faser":
                title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_\nu, \ \mathcal{L}_{\mathrm{pp}} = 65.6 \ \mathrm{fb}^{-1} \ \mathrm{FASER} \ 2412.03186$"
            if obs == "Enu" and geometry == "faserv2":
                title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_\nu, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu2 $"
            if obs == "Enu" and geometry == "high_lumi_faserv":
                title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_\nu, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 4+5$"

            if obs == "theta" and geometry == "run_3_gens":
                title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ \theta, \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"
            if obs == "theta" and geometry == "2024_faser":
                title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ \theta, \ \mathcal{L}_{\mathrm{pp}} = 65.6 \ \mathrm{fb}^{-1} \ \mathrm{FASER} \ 2412.03186$"
            if obs == "theta" and geometry == "faserv2":
                title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ \theta, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu2 $"
            if obs == "theta" and geometry == "high_lumi_faserv":
                title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ \theta, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 4+5$"

            fig.suptitle(title_str, fontsize=18)

            axL.set_ylabel(r"$xf_{\nu_\mu}(x_\nu)$", fontsize=10)

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

            axMdpm_err = axM.fill_between(
                x_vals,
                (mean_mub_dpm + error_mub_dpm),
                (mean_mub_dpm - error_mub_dpm),
                color="#648fff",
                alpha=0.2,
                label=r"$\pm 1\sigma$",
            )
            (axMdpm,) = axM.plot(
                x_vals,
                mean_mub_dpm,
                linestyle="-.",
                color="#648fff",
                label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            )

            axMepos_err = axM.fill_between(
                x_vals,
                (mean_mub_epos + error_mub_epos),
                (mean_mub_epos - error_mub_epos),
                color="#fe6100",
                alpha=0.2,
                label=r"$\pm 1\sigma$",
            )
            (axMepos,) = axM.plot(
                x_vals,
                mean_mub_epos,
                linestyle="--",
                color="#fe6100",
                label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            )

            axMdpm_qgsjet_err = axM.fill_between(
                x_vals,
                (mean_mub_dpm_qgsjet + error_mub_dpm_qgsjet),
                (mean_mub_dpm_qgsjet - error_mub_dpm_qgsjet),
                color="purple",
                alpha=0.2,
                label=r"$\pm 1\sigma$",
            )
            (axMdpm_qgsjet,) = axM.plot(
                x_vals,
                mean_mub_dpm_qgsjet,
                linestyle=":",
                color="purple",
                label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            )

            axMsibyll_err = axM.fill_between(
                x_vals,
                (mean_mub_sibyll + error_mub_sibyll),
                (mean_mub_sibyll - error_mub_sibyll),
                color="#000000",
                alpha=0.2,
                label=r"$\pm 1\sigma$",
            )
            (axMsibyll,) = axM.plot(
                x_vals,
                mean_mub_sibyll,
                linestyle="-",
                color="#000000",
                label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            )

            axM.legend(
                # [axMsimb, (axMnnberr, axMnnb), (axMvert1, axMvert2)],
                [
                    # axMsimb,
                    # (axMfasererr, axMfaser),
                    (axMdpm_err, axMdpm),
                    (axMepos_err, axMepos),
                    (axMsibyll_err, axMsibyll),
                    (axMdpm_qgsjet_err, axMdpm_qgsjet),
                ],
                [
                    r"$\mathrm{DPMJET(\pi,K,c)}$",
                    r"$ \mathrm{EPOS(\pi,K)+POWHEG(c)}$",
                    r"$ \mathrm{SIBYLL(\pi,K,c)}$",
                    r"$ \mathrm{QGSJET(\pi,K)+POWHEG(c)}$",
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

            title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \ \mathrm{EPOS} + \ \mathrm{POWHEG}, \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"

            axM.set_ylabel(r"$xf_{\bar{\nu}_\mu}(x_\nu)$", fontsize=10)

            axM.set_xlim(0.02, 1)
            if geometry == "high_lumi_faserv" or geometry == "faserv2":
                axM.set_ylim(1e-0, 1e6)
            else:
                axM.set_ylim(1e-1, 1e4)
            axM.set_yscale("log")
            axM.set_xscale("log")
            axM.set_xticklabels([])
            axM.grid(color="grey", linestyle="-", linewidth=0.25)

            # ========== BOTTOM LEFT (Ratio plot, f_NN vs f_FASERv )

            axrL.plot(
                x_vals,
                np.ones(len(x_vals)),
                linestyle="-",
                color=simcolor,
                label=r"$\mathrm{baseline}$",
            )

            axrL.set_xscale("log")
            # axrL.set_xlim(1e-2, 1)
            axrL.set_ylim(0.8, 1.2)
            axrL.grid(color="grey", linestyle="-", linewidth=0.25)
            axrL.set_ylabel(r"$\mathrm{Ratio}$")
            # axrL.set_xlabel(r"$x_\nu$")

            ratio_center = mean_mu_dpm / faser_pdf_mu_dpm
            ratio_lower = (mean_mu_dpm - error_mu_dpm) / faser_pdf_mu_dpm
            ratio_upper = (mean_mu_dpm + error_mu_dpm) / faser_pdf_mu_dpm

            axrL.plot(x_vals, ratio_center, linestyle="-.", color="#648fff")
            axrL.fill_between(
                x_vals, ratio_upper, ratio_lower, color="#648fff", alpha=0.2
            )

            ratio_center = mean_mu_epos / faser_pdf_mu_epos
            ratio_lower = (mean_mu_epos - error_mu_epos) / faser_pdf_mu_epos
            ratio_upper = (mean_mu_epos + error_mu_epos) / faser_pdf_mu_epos

            axrL.plot(x_vals, ratio_center, linestyle="--", color="#fe6100")
            axrL.fill_between(
                x_vals, ratio_upper, ratio_lower, color="#fe6100", alpha=0.2
            )

            ratio_center = mean_mu_dpm_qgsjet / faser_pdf_mu_qgs
            ratio_lower = (mean_mu_dpm_qgsjet - error_mu_dpm_qgsjet) / faser_pdf_mu_qgs
            ratio_upper = (mean_mu_dpm_qgsjet + error_mu_dpm_qgsjet) / faser_pdf_mu_qgs

            axrL.plot(x_vals, ratio_center, linestyle=":", color="purple")
            axrL.fill_between(
                x_vals, ratio_upper, ratio_lower, color="purple", alpha=0.2
            )

            ratio_center = mean_mu_sibyll / faser_pdf_mu_sib
            ratio_lower = (mean_mu_sibyll - error_mu_sibyll) / faser_pdf_mu_sib
            ratio_upper = (mean_mu_sibyll + error_mu_sibyll) / faser_pdf_mu_sib

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

            axrM.plot(
                x_vals,
                np.ones(len(x_vals)),
                linestyle="-",
                color=simcolor,
                label=r"$\mathrm{baseline}$",
            )

            axrM.plot(
                x_vals,
                mean_mub_dpm / faser_pdf_mub_dpm,
                linestyle="-.",
                color="#648fff",
            )
            axrM.fill_between(
                x_vals,
                (mean_mub_dpm + error_mub_dpm) / faser_pdf_mub_dpm,
                (mean_mub_dpm - error_mub_dpm) / faser_pdf_mub_dpm,
                color="#648fff",
                alpha=0.2,
            )

            axrM.plot(
                x_vals,
                mean_mub_epos / faser_pdf_mub_epos,
                linestyle="--",
                color="#fe6100",
            )
            axrM.fill_between(
                x_vals,
                (mean_mub_epos + error_mub_epos) / faser_pdf_mub_epos,
                (mean_mub_epos - error_mub_epos) / faser_pdf_mub_epos,
                color="#fe6100",
                alpha=0.2,
            )

            axrM.plot(
                x_vals,
                mean_mub_dpm_qgsjet / faser_pdf_mub_qgs,
                linestyle=":",
                color="purple",
            )
            axrM.fill_between(
                x_vals,
                (mean_mub_dpm_qgsjet + error_mub_dpm_qgsjet) / faser_pdf_mub_qgs,
                (mean_mub_dpm_qgsjet - error_mub_dpm_qgsjet) / faser_pdf_mub_qgs,
                color="purple",
                alpha=0.2,
            )

            axrM.plot(
                x_vals,
                mean_mub_sibyll / faser_pdf_mub_sib,
                linestyle="-",
                color="#000000",
            )
            axrM.fill_between(
                x_vals,
                (mean_mub_sibyll + error_mub_sibyll) / faser_pdf_mub_sib,
                (mean_mub_sibyll - error_mub_sibyll) / faser_pdf_mub_sib,
                color="#000000",
                alpha=0.2,
            )

            axrM.set_xscale("log")
            axrM.set_xlim(0.02, 1)
            axrM.set_ylim(0.5, 1.5)
            axrM.grid(color="grey", linestyle="-", linewidth=0.25)
            axrM.set_ylabel(r"$\mathrm{Ratio}$")
            # axrM.set_xlabel(r"$x_\nu$")
            axrM.legend()
            axrM.tick_params(labelbottom=False)

            # 1 sigma error bands

            axLsig.plot(
                x_vals,
                (mean_mu_dpm + error_mu_dpm) / mean_mu_dpm - 1,
                linestyle="-.",
                color="#648fff",
            )
            # axLsig.fill_between(
            #     x_vals,
            #     (mean_mu_dpm + error_mu_dpm) / mean_mu_dpm,
            #     (mean_mu_dpm - error_mu_dpm) / mean_mu_dpm,
            #     color="#648fff",
            #     alpha=0.2,
            # )

            axLsig.plot(
                x_vals,
                (mean_mu_epos + error_mu_epos) / mean_mu_epos - 1,
                linestyle="--",
                color="#fe6100",
            )

            axLsig.plot(
                x_vals,
                (mean_mu_sibyll + error_mu_sibyll) / mean_mu_sibyll - 1,
                linestyle="-",
                color="#000000",
            )

            axLsig.plot(
                x_vals,
                (mean_mu_dpm_qgsjet + error_mu_dpm_qgsjet) / mean_mu_dpm_qgsjet - 1,
                linestyle=":",
                color="purple",
            )

            axLsig.set_xscale("log")
            axLsig.set_xlim(0.02, 1)
            axLsig.set_ylim(0, 0.5)
            axLsig.grid(color="grey", linestyle="-", linewidth=0.25)
            axLsig.set_ylabel(r"$68\% \ \mathrm{CL} \ \mathrm{error}$")
            axLsig.set_xlabel(r"$x_\nu$", fontsize=10)

            axMsig.plot(
                x_vals,
                (mean_mub_dpm + error_mub_dpm) / mean_mub_dpm - 1,
                linestyle="-.",
                color="#648fff",
            )

            axMsig.plot(
                x_vals,
                (mean_mub_epos + error_mub_epos) / mean_mub_epos - 1,
                linestyle="--",
                color="#fe6100",
            )

            axMsig.plot(
                x_vals,
                (mean_mub_sibyll + error_mub_sibyll) / mean_mub_sibyll - 1,
                linestyle="-",
                color="#000000",
            )

            axMsig.plot(
                x_vals,
                (mean_mub_dpm_qgsjet + error_mub_dpm_qgsjet) / mean_mub_dpm_qgsjet - 1,
                linestyle=":",
                color="purple",
            )

            axMsig.set_xscale("log")
            axMsig.set_xlim(0.02, 1)
            axMsig.set_ylim(0, 0.5)
            axMsig.grid(color="grey", linestyle="-", linewidth=0.25)
            axMsig.set_ylabel(r"$68\% \ \mathrm{CL} \ \mathrm{error}$")
            axMsig.set_xlabel(r"$x_\nu$", fontsize=10)

            if geometry == "high_lumi_faserv" or geometry == "faserv2":
                axL.text(
                    0.2,
                    5 * 10**5,
                    r"$\nu_\mu W \rightarrow \mu X_h$",
                    fontsize=12,
                    color="red",
                )

                axM.text(
                    0.2,
                    5 * 10**5,
                    r"$ \bar{\nu}_\mu W \rightarrow \mu^{+} X_h$",
                    fontsize=12,
                    color="red",
                )

            else:
                axL.text(
                    0.2,
                    10**3,
                    r"$\nu_\mu W \rightarrow \mu X_h$",
                    fontsize=12,
                    color="red",
                )

                axM.text(
                    0.2,
                    10**3,
                    r"$\bar{\nu}_\mu W \rightarrow \mu X_h$",
                    fontsize=12,
                    color="red",
                )

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.13)
            plt.savefig(f"{geometry}_{obs}_compare_gens.pdf")
            # plt.show()


x_vals = np.logspace(-5, 0, 1000)
plot(x_vals)
