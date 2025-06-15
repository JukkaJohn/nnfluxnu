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
mubpid = -12
mupid = 12

npt = 200


# Get number of reps from make runscripts
def plot(
    x_vals,
):
    # generator = "epos"
    # geometry = "run_3_gens"
    geometries = [
        "new_run_3_gens",
        # "new_2024faser",
        # "new_faserv2",
        "new_high_lumi",
    ]

    generators = [
        # "dpmjet",
        # "epos",
        "qgsjet",
        "sibyll",
    ]
    for generator in generators:
        for geometry in geometries:
            neutrino_pdfs_mu_dpm_Eh = np.loadtxt(
                f"{geometry}/fit_{generator}/Eh_fit/pdf.txt", delimiter=","
            )
            # neutrino_pdfs_mub_dpm_Eh = np.loadtxt(
            #     f"{geometry}/fit_{generator}/Eh_fit/mub_pdf.txt", delimiter=","
            # )

            neutrino_pdfs_mu_dpm_Eh *= 2
            neutrino_pdfs_mu_dpm_El = np.loadtxt(
                f"{geometry}/fit_{generator}/El_fit/pdf.txt", delimiter=","
            )
            # neutrino_pdfs_mub_dpm_El = np.loadtxt(
            #     f"{geometry}/fit_{generator}/El_fit/mub_pdf.txt", delimiter=","
            # )
            neutrino_pdfs_mu_dpm_El *= 2
            neutrino_pdfs_mu_dpm_Enu = np.loadtxt(
                f"{geometry}/fit_{generator}/Enu_fit/pdf.txt", delimiter=","
            )
            # neutrino_pdfs_mub_dpm_Enu = np.loadtxt(
            #     f"{geometry}/fit_{generator}/Enu_fit/mub_pdf.txt", delimiter=","
            # )
            neutrino_pdfs_mu_dpm_Enu *= 2

            neutrino_pdfs_mu_dpm_theta = np.loadtxt(
                f"{geometry}/fit_{generator}/theta_fit/pdf.txt", delimiter=","
            )
            neutrino_pdfs_mu_dpm_theta *= 2
            # neutrino_pdfs_mub_dpm_theta = np.loadtxt(
            #     f"{geometry}/fit_{generator}/theta_fit/mub_pdf.txt", delimiter=","
            # )

            x_vals = np.array(x_vals)
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

            pdf = f"{formal_geometry}_{formal_gen}_7TeV"
            faser_pdf_mu, x_faser = read_pdf(pdf, x_vals, 12)
            faser_pdf_mub, x_faser = read_pdf(pdf, x_vals, -12)
            faser_pdf_mu = faser_pdf_mu * x_vals * factor
            faser_pdf_mub = faser_pdf_mub * x_vals * factor
            faser_pdf_mu += faser_pdf_mub

            mean_mu_dpm_Eh = np.mean(neutrino_pdfs_mu_dpm_Eh, axis=0) * x_vals
            error_mu_dpm_Eh = np.std(neutrino_pdfs_mu_dpm_Eh, axis=0) * x_vals
            # mean_mub_dpm_Eh = np.mean(neutrino_pdfs_mub_dpm_Eh, axis=0) * x_vals
            # error_mub_dpm_Eh = np.std(neutrino_pdfs_mub_dpm_Eh, axis=0) * x_vals

            mean_mu_dpm_El = np.mean(neutrino_pdfs_mu_dpm_El, axis=0) * x_vals
            error_mu_dpm_El = np.std(neutrino_pdfs_mu_dpm_El, axis=0) * x_vals
            # mean_mub_dpm_El = np.mean(neutrino_pdfs_mub_dpm_El, axis=0) * x_vals
            # error_mub_dpm_El = np.std(neutrino_pdfs_mub_dpm_El, axis=0) * x_vals

            mean_mu_dpm_theta = np.mean(neutrino_pdfs_mu_dpm_theta, axis=0) * x_vals
            error_mu_dpm_theta = np.std(neutrino_pdfs_mu_dpm_theta, axis=0) * x_vals
            # mean_mub_dpm_theta = np.mean(neutrino_pdfs_mub_dpm_theta, axis=0) * x_vals
            # error_mub_dpm_theta = np.std(neutrino_pdfs_mub_dpm_theta, axis=0) * x_vals

            mean_mu_dpm_Enu = np.mean(neutrino_pdfs_mu_dpm_Enu, axis=0) * x_vals
            error_mu_dpm_Enu = np.std(neutrino_pdfs_mu_dpm_Enu, axis=0) * x_vals
            # mean_mub_dpm_Enu = np.mean(neutrino_pdfs_mub_dpm_Enu, axis=0) * x_vals
            # error_mub_dpm_Enu = np.std(neutrino_pdfs_mub_dpm_Enu, axis=0) * x_vals

            plt.rcParams["text.usetex"] = True
            plt.rcParams.update(
                {
                    # "font.family": "serif",
                    # "font.serif": ["cmr10"],  # Computer Modern]
                    "font.size": 15,
                }
            )
            fig = plt.figure(figsize=(8.636, 9.0), dpi=300)  # 2 rows, 3 columns
            gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 1])

            gs.update(
                left=0.12, right=0.97, top=0.92, bottom=0.08, hspace=0.15, wspace=0.15
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

            (axLsim,) = axL.plot(
                x_vals,
                faser_pdf_mu,
                linestyle="-",
                label=r"$f_{\mathrm{EPOS+POWHEG}\nu_\mu}(x)$",
                color=simcolor,
            )

            axLdpm_Eh_err = axL.fill_between(
                x_vals,
                mean_mu_dpm_Eh + error_mu_dpm_Eh,
                mean_mu_dpm_Eh - error_mu_dpm_Eh,
                color="#648fff",
                alpha=0.2,
                label=r"$\pm 1\sigma$",
            )

            (axLdpm_Eh,) = axL.plot(
                x_vals,
                mean_mu_dpm_Eh,
                linestyle="-.",
                color="#648fff",
                label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            )

            axLdpm_El_err = axL.fill_between(
                x_vals,
                mean_mu_dpm_El + error_mu_dpm_El,
                mean_mu_dpm_El - error_mu_dpm_El,
                color="#fe6100",
                alpha=0.2,
                label=r"$\pm 1\sigma$",
            )

            (axLdpm_El,) = axL.plot(
                x_vals,
                mean_mu_dpm_El,
                linestyle="--",
                color="#fe6100",
                label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            )

            axLdpm_theta_err = axL.fill_between(
                x_vals,
                mean_mu_dpm_theta + error_mu_dpm_theta,
                mean_mu_dpm_theta - error_mu_dpm_theta,
                color="#dc267f",
                alpha=0.2,
                label=r"$\pm 1\sigma$",
            )

            (axLdpm_theta,) = axL.plot(
                x_vals,
                mean_mu_dpm_theta,
                linestyle=":",
                color="#dc267f",
                label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            )

            axLdpm_Enu_err = axL.fill_between(
                x_vals,
                mean_mu_dpm_Enu + error_mu_dpm_Enu,
                mean_mu_dpm_Enu - error_mu_dpm_Enu,
                color="#000000",
                alpha=0.2,
                label=r"$\pm 1\sigma$",
            )

            (axLdpm_Enu,) = axL.plot(
                x_vals,
                mean_mu_dpm_Enu,
                linestyle="-",
                markersize=0.4,
                color="#000000",
                label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            )

            axL.legend(
                [
                    axLsim,
                    (axLdpm_Eh_err, axLdpm_Eh),
                    (axLdpm_El_err, axLdpm_El),
                    (axLdpm_Enu_err, axLdpm_Enu),
                    (axLdpm_theta_err, axLdpm_theta),
                ],
                [
                    r"$\mathrm{baseline (\nu_e + \bar{\nu}_e)}$",
                    r"${\mathrm{fit} \ \mathrm{input:} \ E_h}$",
                    r"${\mathrm{fit} \ \mathrm{input:} \ E_\ell}$",
                    r"${\mathrm{fit} \ \mathrm{input:} \ E_\nu}$",
                    r"${\mathrm{fit} \ \mathrm{input:} \ \theta}$",
                ],
                handler_map={tuple: HandlerTuple(ndivide=1)},
                loc="lower left",
                ncols=2,
                # fontsize="small",
                handlelength=1.2,
                handletextpad=0.4,
                borderpad=0.3,
                labelspacing=0.3,
            ).set_alpha(0.8)

            axL.set_xlim(0.02, 1)
            if geometry == "new_high_lumi" or geometry == "new_faserv2":
                axL.set_ylim(1e-0, 1e6)
            else:
                axL.set_ylim(1e-1, 1e4)
            axL.set_yscale("log")
            axL.set_xscale("log")

            title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \ \mathrm{EPOS}+\mathrm{POWHEG}, \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"

            #  generators = ["dpmjet", "epos", "qgsjet", "sibyll"]

            if generator == "dpmjet" and geometry == "new_run_3_gens":
                title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \ \mathrm{DPMJET}(\pi,K,c), \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"
            if generator == "epos" and geometry == "new_run_3_gens":
                title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \ \mathrm{EPOS(\pi,K)+POWHEG(c)}, \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"
            if generator == "qgsjet" and geometry == "new_run_3_gens":
                title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \  \mathrm{QGSJET(\pi,K)+POWHEG(c)}, \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"
            if generator == "sibyll" and geometry == "new_run_3_gens":
                title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \ \mathrm{SIBYLL(\pi,K,c)}, \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"

            if generator == "dpmjet" and geometry == "new_2024faser":
                title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \ \mathrm{DPMJET}(\pi,K,c), \ \mathcal{L}_{\mathrm{pp}} = 65.6 \ \mathrm{fb}^{-1} \ \mathrm{FASER} \ 2412.03186$"
            if generator == "epos" and geometry == "new_2024faser":
                title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \ \mathrm{EPOS(\pi,K)+POWHEG(c)}, \ \mathcal{L}_{\mathrm{pp}} = 65.6 \ \mathrm{fb}^{-1} \ \mathrm{FASER} \ 2412.03186$"
            if generator == "qgsjet" and geometry == "new_2024faser":
                title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \  \mathrm{QGSJET(\pi,K)+POWHEG(c)}, \ \mathcal{L}_{\mathrm{pp}} = 65.6 \ \mathrm{fb}^{-1} \ \mathrm{FASER} \ 2412.03186$"
            if generator == "sibyll" and geometry == "new_2024faser":
                title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \ \mathrm{SIBYLL(\pi,K,c)}, \ \mathcal{L}_{\mathrm{pp}} = 65.6 \ \mathrm{fb}^{-1} \ \mathrm{FASER} \ 2412.03186$"

            if generator == "dpmjet" and geometry == "new_faserv2":
                title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \ \mathrm{DPMJET}(\pi,K,c), \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu2 $"
            if generator == "epos" and geometry == "new_faserv2":
                title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \ \mathrm{EPOS(\pi,K)+POWHEG(c)}, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu2 $"
            if generator == "qgsjet" and geometry == "new_faserv2":
                title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \  \mathrm{QGSJET(\pi,K)+POWHEG(c)}, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu2 $"
            if generator == "sibyll" and geometry == "new_faserv2":
                title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \ \mathrm{SIBYLL(\pi,K,c)}, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu2 $"

            if generator == "dpmjet" and geometry == "new_high_lumi":
                title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \ \mathrm{DPMJET}(\pi,K,c), \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 4+5$"
            if generator == "epos" and geometry == "new_high_lumi":
                title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \ \mathrm{EPOS(\pi,K)+POWHEG(c)}, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 4+5$"
            if generator == "qgsjet" and geometry == "new_high_lumi":
                title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \  \mathrm{QGSJET(\pi,K)+POWHEG(c)}, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 4+5$"
            if generator == "sibyll" and geometry == "new_high_lumi":
                title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \ \mathrm{SIBYLL(\pi,K,c)}, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 4+5$"

            # if generator == "El" and geometry == "run_3_gens":
            #     title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_\ell, \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"
            # if generator == "El" and geometry == "2024_faser":
            #     title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_\ell, \ \mathcal{L}_{\mathrm{pp}} = 65.6 \ \mathrm{fb}^{-1} \ \mathrm{FASER} \ 2412.03186$"
            # if generator == "El" and geometry == "faserv2":
            #     title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_\ell, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu2 $"
            # if generator == "El" and geometry == "high_lumi_faserv":
            #     title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_\ell, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 4+5$"

            # if generator == "Enu" and geometry == "run_3_gens":
            #     title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_\nu, \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"
            # if generator == "Enu" and geometry == "2024_faser":
            #     title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_\nu, \ \mathcal{L}_{\mathrm{pp}} = 65.6 \ \mathrm{fb}^{-1} \ \mathrm{FASER} \ 2412.03186$"
            # if generator == "Enu" and geometry == "faserv2":
            #     title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_\nu, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu2 $"
            # if generator == "Enu" and geometry == "high_lumi_faserv":
            #     title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ E_\nu, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 4+5$"

            # if generator == "theta" and geometry == "run_3_gens":
            #     title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ \theta, \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"
            # if generator == "theta" and geometry == "2024_faser":
            #     title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ \theta, \ \mathcal{L}_{\mathrm{pp}} = 65.6 \ \mathrm{fb}^{-1} \ \mathrm{FASER} \ 2412.03186$"
            # if generator == "theta" and geometry == "faserv2":
            #     title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ \theta, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu2 $"
            # if generator == "theta" and geometry == "high_lumi_faserv":
            #     title_str = r"$\mathrm{Comparison} \ \mathrm{MC} \ \mathrm{Generators} \ \theta, \ \mathcal{L}_{\mathrm{pp}} = 3 \ \mathrm{ab}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 4+5$"

            fig.suptitle(title_str, fontsize=14)

            # axL.set_ylabel(r"$xf_{\nu_e}(x_\nu)$", fontsize=18)
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

            # axMdpm_Eh_err = axM.fill_between(
            #     x_vals,
            #     (mean_mub_dpm_Eh + error_mub_dpm_Eh),
            #     (mean_mub_dpm_Eh - error_mub_dpm_Eh),
            #     color="#648fff",
            #     alpha=0.2,
            #     label=r"$\pm 1\sigma$",
            # )
            # (axMdpm_Eh,) = axM.plot(
            #     x_vals,
            #     mean_mub_dpm_Eh,
            #     linestyle="-.",
            #     color="#648fff",
            #     label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            # )

            # axMdpm_El_err = axM.fill_between(
            #     x_vals,
            #     (mean_mub_dpm_El + error_mub_dpm_El),
            #     (mean_mub_dpm_El - error_mub_dpm_El),
            #     color="#fe6100",
            #     alpha=0.2,
            #     label=r"$\pm 1\sigma$",
            # )
            # (axMdpm_El,) = axM.plot(
            #     x_vals,
            #     mean_mub_dpm_El,
            #     linestyle="--",
            #     color="#fe6100",
            #     label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            # )

            # axMdpm_theta_err = axM.fill_between(
            #     x_vals,
            #     (mean_mub_dpm_theta + error_mub_dpm_theta),
            #     (mean_mub_dpm_theta - error_mub_dpm_theta),
            #     color="#dc267f",
            #     alpha=0.2,
            #     label=r"$\pm 1\sigma$",
            # )
            # (axMdpm_theta,) = axM.plot(
            #     x_vals,
            #     mean_mub_dpm_theta,
            #     linestyle=":",
            #     color="#dc267f",
            #     label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            # )

            # axMdpm_Enu_err = axM.fill_between(
            #     x_vals,
            #     (mean_mub_dpm_Enu + error_mub_dpm_Enu),
            #     (mean_mub_dpm_Enu - error_mub_dpm_Enu),
            #     color="#000000",
            #     alpha=0.2,
            #     label=r"$\pm 1\sigma$",
            # )
            # (axMdpm_Enu,) = axM.plot(
            #     x_vals,
            #     mean_mub_dpm_Enu,
            #     linestyle="-",
            #     color="#000000",
            #     label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
            # )

            # axM.legend(
            #     # [axMsimb, (axMnnberr, axMnnb), (axMvert1, axMvert2)],
            #     [
            #         axMsimb,
            #         # (axMfasererr, axMfaser),
            #         (axMdpm_Eh_err, axMdpm_Eh),
            #         (axMdpm_El_err, axMdpm_El),
            #         (axMdpm_Enu_err, axMdpm_Enu),
            #         (axMdpm_theta_err, axMdpm_theta),
            #     ],
            #     [
            #         r"${ \mathrm{baseline}}$",
            #         # r"$f_{\nu_\mu, \mathrm{EPOS+POWHEG} \ E_\nu}$",
            #         r"${\mathrm{fit} \ \mathrm{input:} \ E_h}$",
            #         r"${\mathrm{fit} \ \mathrm{input:} \ E_\ell}$",
            #         r"${\mathrm{fit} \ \mathrm{input:} \ E_\nu}$",
            #         r"${\mathrm{fit} \ \mathrm{input:} \ \theta}$",
            #     ],
            #     handler_map={tuple: HandlerTuple(ndivide=1)},
            #     loc="lower left",
            #     ncols=2,
            #     # fontsize="small",
            #     handlelength=1.2,
            #     handletextpad=0.4,
            #     borderpad=0.3,
            #     labelspacing=0.3,
            # ).set_alpha(0.8)

            # title_str = r"$\mathrm{baseline} \ \mathrm{fluxes:} \ \mathrm{EPOS} + \ \mathrm{POWHEG}, \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"

            # axM.set_ylabel(r"$xf_{\bar{\nu}_e}(x_\nu)$", fontsize=18)

            # axM.set_xlim(1e-2, 1)
            # if geometry == "new_high_lumi" or geometry == "new_faserv2":
            #     axM.set_ylim(1e-0, 1e6)
            # else:
            #     axM.set_ylim(1e-1, 1e4)
            # axM.set_yscale("log")
            # axM.set_xscale("log")
            # axM.set_xticklabels([])
            # axM.grid(color="grey", linestyle="-", linewidth=0.25)

            # ========== BOTTOM LEFT (Ratio plot, f_NN vs f_FASERv )

            axrL.plot(x_vals, np.ones(len(x_vals)), linestyle="-", color=simcolor)

            axrL.set_xscale("log")
            # axrL.set_xlim(1e-2, 1)
            axrL.set_ylim(0.8, 1.2)
            axrL.grid(color="grey", linestyle="-", linewidth=0.25)
            axrL.set_ylabel(r"$\mathrm{Ratio}$")
            # axrL.set_xlabel(r"$x_\nu$")

            ratio_center = mean_mu_dpm_Eh / faser_pdf_mu
            ratio_lower = (mean_mu_dpm_Eh - error_mu_dpm_Eh) / faser_pdf_mu
            ratio_upper = (mean_mu_dpm_Eh + error_mu_dpm_Eh) / faser_pdf_mu

            axrL.plot(x_vals, ratio_center, linestyle="-.", color="#648fff")
            axrL.fill_between(
                x_vals, ratio_upper, ratio_lower, color="#648fff", alpha=0.2
            )

            ratio_center = mean_mu_dpm_El / faser_pdf_mu
            ratio_lower = (mean_mu_dpm_El - error_mu_dpm_El) / faser_pdf_mu
            ratio_upper = (mean_mu_dpm_El + error_mu_dpm_El) / faser_pdf_mu

            axrL.plot(x_vals, ratio_center, linestyle="--", color="#fe6100")
            axrL.fill_between(
                x_vals, ratio_upper, ratio_lower, color="#fe6100", alpha=0.2
            )

            ratio_center = mean_mu_dpm_theta / faser_pdf_mu
            ratio_lower = (mean_mu_dpm_theta - error_mu_dpm_theta) / faser_pdf_mu
            ratio_upper = (mean_mu_dpm_theta + error_mu_dpm_theta) / faser_pdf_mu

            axrL.plot(x_vals, ratio_center, linestyle=":", color="#dc267f")
            axrL.fill_between(
                x_vals, ratio_upper, ratio_lower, color="#dc267f", alpha=0.2
            )

            ratio_center = mean_mu_dpm_Enu / faser_pdf_mu
            ratio_lower = (mean_mu_dpm_Enu - error_mu_dpm_Enu) / faser_pdf_mu
            ratio_upper = (mean_mu_dpm_Enu + error_mu_dpm_Enu) / faser_pdf_mu

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

            # ========== BOTTOM MIDDLE (ratio f_NN_mub)

            # axrM.plot(x_vals, np.ones(len(x_vals)), linestyle="-", color=simcolor)

            # axrM.plot(
            #     x_vals, mean_mub_dpm_Eh / faser_pdf_mub, linestyle="-.", color="#648fff"
            # )
            # axrM.fill_between(
            #     x_vals,
            #     (mean_mub_dpm_Eh + error_mub_dpm_Eh) / faser_pdf_mub,
            #     (mean_mub_dpm_Eh - error_mub_dpm_Eh) / faser_pdf_mub,
            #     color="#648fff",
            #     alpha=0.2,
            # )

            # axrM.plot(
            #     x_vals, mean_mub_dpm_El / faser_pdf_mub, linestyle="--", color="#fe6100"
            # )
            # axrM.fill_between(
            #     x_vals,
            #     (mean_mub_dpm_El + error_mub_dpm_El) / faser_pdf_mub,
            #     (mean_mub_dpm_El - error_mub_dpm_El) / faser_pdf_mub,
            #     color="#fe6100",
            #     alpha=0.2,
            # )

            # axrM.plot(
            #     x_vals,
            #     mean_mub_dpm_theta / faser_pdf_mub,
            #     linestyle=":",
            #     color="#dc267f",
            # )
            # axrM.fill_between(
            #     x_vals,
            #     (mean_mub_dpm_theta + error_mub_dpm_theta) / faser_pdf_mub,
            #     (mean_mub_dpm_theta - error_mub_dpm_theta) / faser_pdf_mub,
            #     color="#dc267f",
            #     alpha=0.2,
            # )

            # axrM.plot(
            #     x_vals, mean_mub_dpm_Enu / faser_pdf_mub, linestyle="-", color="#000000"
            # )
            # axrM.fill_between(
            #     x_vals,
            #     (mean_mub_dpm_Enu + error_mub_dpm_Enu) / faser_pdf_mub,
            #     (mean_mub_dpm_Enu - error_mub_dpm_Enu) / faser_pdf_mub,
            #     color="#000000",
            #     alpha=0.2,
            # )

            # axrM.set_xscale("log")
            # axrM.set_xlim(1e-2, 1)
            # axrM.set_ylim(0.5, 1.5)
            # axrM.grid(color="grey", linestyle="-", linewidth=0.25)
            # axrM.set_ylabel(r"$\mathrm{Ratio}$")
            # # axrM.set_xlabel(r"$x_\nu$")
            # axrM.tick_params(labelbottom=False)

            # 1 sigma error bands

            axLsig.plot(
                x_vals,
                (mean_mu_dpm_Eh + error_mu_dpm_Eh) / mean_mu_dpm_Eh - 1,
                linestyle="-.",
                color="#648fff",
            )
            # axLsig.fill_between(
            #     x_vals,
            #     (mean_mu_dpm_Eh + error_mu_dpm_Eh) / mean_mu_dpm_Eh,
            #     (mean_mu_dpm_Eh - error_mu_dpm_Eh) / mean_mu_dpm_Eh,
            #     color="#648fff",
            #     alpha=0.2,
            # )

            axLsig.plot(
                x_vals,
                (mean_mu_dpm_El + error_mu_dpm_El) / mean_mu_dpm_El - 1,
                linestyle="--",
                color="#fe6100",
            )

            axLsig.plot(
                x_vals,
                (mean_mu_dpm_Enu + error_mu_dpm_Enu) / mean_mu_dpm_Enu - 1,
                linestyle="-",
                color="#000000",
            )

            axLsig.plot(
                x_vals,
                (mean_mu_dpm_theta + error_mu_dpm_theta) / mean_mu_dpm_theta - 1,
                linestyle=":",
                color="#dc267f",
            )

            axLsig.set_xscale("log")
            axLsig.set_xlim(0.02, 1)
            axLsig.set_ylim(0, 0.5)
            axLsig.grid(color="grey", linestyle="-", linewidth=0.25)
            axLsig.set_ylabel(r"$68\% \ \mathrm{CL} \ \mathrm{error}$")
            axLsig.set_xlabel(r"$x_\nu$", fontsize=18)

            # axMsig.plot(
            #     x_vals,
            #     (mean_mub_dpm_Eh + error_mub_dpm_Eh) / mean_mub_dpm_Eh - 1,
            #     linestyle="-.",
            #     color="#648fff",
            # )

            # axMsig.plot(
            #     x_vals,
            #     (mean_mub_dpm_El + error_mub_dpm_El) / mean_mub_dpm_El - 1,
            #     linestyle="--",
            #     color="#fe6100",
            # )

            # axMsig.plot(
            #     x_vals,
            #     (mean_mub_dpm_Enu + error_mub_dpm_Enu) / mean_mub_dpm_Enu - 1,
            #     linestyle="-",
            #     color="#000000",
            # )

            # axMsig.plot(
            #     x_vals,
            #     (mean_mub_dpm_theta + error_mub_dpm_theta) / mean_mub_dpm_theta - 1,
            #     linestyle=":",
            #     color="#dc267f",
            # )

            # axMsig.set_xscale("log")
            # axMsig.set_xlim(1e-2, 1)
            # axMsig.set_ylim(0, 0.5)
            # axMsig.grid(color="grey", linestyle="-", linewidth=0.25)
            # axMsig.set_ylabel(r"$68\% \ \mathrm{CL} \ \mathrm{error}$")
            # axMsig.set_xlabel(r"$x_\nu$", fontsize=18)

            if geometry == "new_high_lumi" or geometry == "new_faserv2":
                axL.text(
                    0.1,
                    5 * 10**5,
                    r"$(\nu_e + \bar{\nu}_e) W \rightarrow (e+e^{+}) X_h$",
                    fontsize=12,
                    color="red",
                )

            else:
                axL.text(
                    0.1,
                    10**3,
                    r"$(\nu_e + \bar{\nu}_e) W \rightarrow (e+e^{+}) X_h$",
                    fontsize=12,
                    color="red",
                )

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.13)
            plt.savefig(f"{geometry}_{generator}_compare_obs_elec.pdf")
            # plt.show()


x_vals = np.logspace(-5, 0, 1000)
plot(x_vals)
