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
# from read_faserv_pdf import read_pdf

# Data for plot
import matplotlib.ticker as ticker

simcolor = "tab:red"
mucolor = "tab:blue"
mubcolor = "tab:blue"

lhapdf.setVerbosity(0)
mubpid = -14
mupid = 14

npt = 200


def read_pdf(pdf, x_vals, particle, set):
    pid = particle
    Q2 = 10
    pdf = lhapdf.mkPDF(pdf, set)
    pdf_vals = [pdf.xfxQ2(pid, x, Q2) for x in x_vals]
    pdf_vals = np.array(pdf_vals)
    pdf_vals /= x_vals
    return pdf_vals, x_vals


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

    for obs in observables:
        # neutrino_pdfs_mu = np.loadtxt(
        #     f"run_3_gens/fit_epos/{obs}_fit/mu_pdf.txt", delimiter=","
        # )
        # neutrino_pdfs_mub = np.loadtxt(
        #     f"run_3_gens/fit_epos/{obs}_fit/mub_pdf.txt", delimiter=","
        # )

        neutrino_pdfs_mu_bsm_eta = np.loadtxt(
            f"fit_bsm_eta/{obs}_fit/mu_pdf.txt", delimiter=","
        )
        neutrino_pdfs_mub_bsm_eta = np.loadtxt(
            f"fit_bsm_eta/{obs}_fit/mub_pdf.txt", delimiter=","
        )

        neutrino_pdfs_mu_bsm_eta_prime = np.loadtxt(
            f"fit_bsm_eta_prima/{obs}_fit/mu_pdf.txt", delimiter=","
        )
        neutrino_pdfs_mub_bsm_eta_prime = np.loadtxt(
            f"fit_bsm_eta_prima/{obs}_fit/mub_pdf.txt", delimiter=","
        )

        neutrino_pdfs_mu_bsm_pi = np.loadtxt(
            f"fit_bsm_pi/{obs}_fit/mu_pdf.txt", delimiter=","
        )
        neutrino_pdfs_mub_bsm_pi = np.loadtxt(
            f"fit_bsm_pi/{obs}_fit/mub_pdf.txt", delimiter=","
        )

        neutrino_pdfs_mu = np.loadtxt(
            f"run_3_gens/fit_epos/{obs}_fit/mu_pdf.txt", delimiter=","
        )

        neutrino_pdfs_mub = np.loadtxt(
            f"run_3_gens/fit_epos/{obs}_fit/mub_pdf.txt", delimiter=","
        )
        # neutrino_pdfs_mub_bsm_pi = np.loadtxt(
        #     f"elec_fnu_bsm_pi/{obs}_fit/mub_pdf.txt", delimiter=","
        # )

        x_vals = np.array(x_vals)

        factor = 1
        pdf = "FASERv_Run3_EPOS+POWHEG_7TeV"
        faser_pdf_mu_epos, x_faser = read_pdf(pdf, x_vals, 14, 2)
        faser_pdf_mub_epos, x_faser = read_pdf(pdf, x_vals, -14, 2)
        faser_pdf_mu_epos = faser_pdf_mu_epos * x_vals * factor
        faser_pdf_mub_epos = faser_pdf_mub_epos * x_vals * factor

        pdf_bsm_eta = "FASERv_Run3_BSM_eta_7TeV"
        faser_pdf_mu_bsm_eta, x_faser = read_pdf(pdf_bsm_eta, x_vals, 14, 1)
        faser_pdf_mub_bsm_eta, x_faser = read_pdf(pdf_bsm_eta, x_vals, -14, 1)
        faser_pdf_mu_bsm_eta = faser_pdf_mu_bsm_eta * x_vals * factor
        faser_pdf_mub_bsm_eta = faser_pdf_mub_bsm_eta * x_vals * factor

        pdf_bsm_eta_prime = "FASERv_Run3_BSM_eta_prime_7TeV"
        faser_pdf_mu_bsm_eta_prime, x_faser = read_pdf(pdf_bsm_eta_prime, x_vals, 14, 1)
        faser_pdf_mub_bsm_eta_prime, x_faser = read_pdf(
            pdf_bsm_eta_prime, x_vals, -14, 1
        )
        faser_pdf_mu_bsm_eta_prime = faser_pdf_mu_bsm_eta_prime * x_vals * factor
        faser_pdf_mub_bsm_eta_prime = faser_pdf_mub_bsm_eta_prime * x_vals * factor

        pdf_bsm_pi = "FASERv_Run3_BSM_pi0_7TeV"
        faser_pdf_mu_bsm_pi, x_faser = read_pdf(pdf_bsm_pi, x_vals, 14, 1)
        faser_pdf_mub_bsm_pi, x_faser = read_pdf(pdf_bsm_pi, x_vals, -14, 1)
        faser_pdf_mu_bsm_pi = faser_pdf_mu_bsm_pi * x_vals * factor
        faser_pdf_mub_bsm_pi = faser_pdf_mub_bsm_pi * x_vals * factor

        # mean_mu = np.mean(neutrino_pdfs_mu, axis=0) * x_vals
        # error_mu = np.std(neutrino_pdfs_mu, axis=0) * x_vals
        # mean_mub = np.mean(neutrino_pdfs_mub, axis=0) * x_vals
        # error_mub = np.std(neutrino_pdfs_mub, axis=0) * x_vals

        mean_mu_bsm_eta = np.mean(neutrino_pdfs_mu_bsm_eta, axis=0) * x_vals
        error_mu_bsm_eta = np.std(neutrino_pdfs_mu_bsm_eta, axis=0) * x_vals
        mean_mub_bsm_eta = np.mean(neutrino_pdfs_mub_bsm_eta, axis=0) * x_vals
        error_mub_bsm_eta = np.std(neutrino_pdfs_mub_bsm_eta, axis=0) * x_vals

        mean_mu_bsm_eta_prime = np.mean(neutrino_pdfs_mu_bsm_eta_prime, axis=0) * x_vals
        error_mu_bsm_eta_prime = np.std(neutrino_pdfs_mu_bsm_eta_prime, axis=0) * x_vals
        mean_mub_bsm_eta_prime = (
            np.mean(neutrino_pdfs_mub_bsm_eta_prime, axis=0) * x_vals
        )
        error_mub_bsm_eta_prime = (
            np.std(neutrino_pdfs_mub_bsm_eta_prime, axis=0) * x_vals
        )

        mean_mu_bsm_pi = np.mean(neutrino_pdfs_mu_bsm_pi, axis=0) * x_vals
        error_mu_bsm_pi = np.std(neutrino_pdfs_mu_bsm_pi, axis=0) * x_vals
        mean_mub_bsm_pi = np.mean(neutrino_pdfs_mub_bsm_pi, axis=0) * x_vals
        error_mub_bsm_pi = np.std(neutrino_pdfs_mub_bsm_pi, axis=0) * x_vals

        error_mu = np.std(neutrino_pdfs_mu, axis=0) * x_vals
        error_mub = np.std(neutrino_pdfs_mub, axis=0) * x_vals

        plt.rcParams["text.usetex"] = True
        plt.rcParams.update(
            {
                # "font.family": "serif",
                # "font.serif": ["cmr10"],  # Computer Modern]
                "font.size": 10,
            }
        )
        fig = plt.figure(figsize=(3.4 * 2, 3.4 * 2), dpi=300)  # 2 rows, 3 columns

        gs = gridspec.GridSpec(3, 2, height_ratios=[3, 2, 1])

        gs.update(
            left=0.04, right=0.97, top=0.92, bottom=0.08, hspace=0.15, wspace=0.15
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

        (axLsim,) = axL.plot(
            x_vals,
            faser_pdf_mu_epos,
            linestyle="-",
            label=r"$f_{\mathrm{EPOS+POWHEG}\nu_\mu}(x)$",
            color="red",
        )

        # axL_err = axL.fill_between(
        #     x_vals,
        #     mean_mu + error_mu,
        #     mean_mu - error_mu,
        #     color="#648fff",
        #     alpha=0.2,
        #     label=r"$\pm 1\sigma$",
        # )

        # axL_mu = axL.plot(
        #     x_vals,
        #     mean_mu,
        #     linestyle="-.",
        #     color="#648fff",
        #     label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
        # )

        (axLbsm_eta_err) = axL.fill_between(
            x_vals,
            mean_mu_bsm_eta + error_mu_bsm_eta,
            mean_mu_bsm_eta - error_mu_bsm_eta,
            color="#fe6100",
            alpha=0.2,
            label=r"$\pm 1\sigma$",
        )

        (axLbsm_eta,) = axL.plot(
            x_vals,
            mean_mu_bsm_eta,
            linestyle="--",
            color="#fe6100",
            label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
        )

        (axLbsm_eta_prime_err) = axL.fill_between(
            x_vals,
            mean_mu_bsm_eta_prime + error_mu_bsm_eta_prime,
            mean_mu_bsm_eta_prime - error_mu_bsm_eta_prime,
            color="#dc267f",
            alpha=0.2,
            label=r"$\pm 1\sigma$",
        )

        (axLbsm_eta_prime,) = axL.plot(
            x_vals,
            mean_mu_bsm_eta_prime,
            linestyle="-",
            color="#dc267f",
            label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
        )

        (axLbsm_pi_err) = axL.fill_between(
            x_vals,
            mean_mu_bsm_pi + error_mu_bsm_pi,
            mean_mu_bsm_pi - error_mu_bsm_pi,
            color="#000000",
            alpha=0.2,
            label=r"$\pm 1\sigma$",
        )

        (axLbsm_pi,) = axL.plot(
            x_vals,
            mean_mu_bsm_pi,
            linestyle=":",
            color="#000000",
            label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
        )

        axL.legend(
            [
                axLsim,
                # (axL_mu, axL_err),
                (axLbsm_eta_err, axLbsm_eta),
                (axLbsm_eta_prime_err, axLbsm_eta_prime),
                (axLbsm_pi_err, axLbsm_pi),
            ],
            [
                # r"${ \mathrm{baseline}}$",
                r"$\mathrm{SM}  \ (f_{\nu_\mu} +f_{\bar{\nu}_\mu})$",
                r"$\mathrm{BSM}(\eta \rightarrow \nu_\mu + \bar{\nu}_\mu)$",
                r"$\mathrm{BSM}(\eta^{'} \rightarrow \nu_\mu + \bar{\nu}_\mu)$",
                r"$\mathrm{BSM}(\pi \rightarrow \nu_\mu + \bar{\nu}_\mu)$",
                # r"$ \mathrm{SIBYLL(\pi,K,c)}$",
                # r"$ \mathrm{QGSJET(\pi,K)+POWHEG(c)}$",
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

        axL.set_ylim(1e0, 1e4)
        axL.set_yscale("log")
        axL.set_xscale("log")

        if obs == "Eh":
            title_str = r"$\mathrm{Comparison} \ \mathrm{BSM} \ \mathrm{Decays} \ E_h, \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"

        if obs == "El":
            title_str = r"$\mathrm{Comparison} \ \mathrm{BSM} \ \mathrm{Decays} \ E_\ell, \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"

        if obs == "Enu":
            title_str = r"$\mathrm{Comparison} \ \mathrm{BSM} \ \mathrm{Decays} \ E_\nu, \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"

        if obs == "theta":
            title_str = r"$\mathrm{Comparison} \ \mathrm{BSM} \ \mathrm{Decays} \ \theta, \ \mathcal{L}_{\mathrm{pp}} = 150 \ \mathrm{fb}^{-1} \ \mathrm{FASER}\nu \ \mathrm{Run} \ 3$"

        fig.suptitle(title_str, fontsize=18)

        axL.set_ylabel(r"$x_\nu f_{\nu_\mu}$")

        axL.set_xticklabels([])
        axL.grid(color="grey", linestyle="-", linewidth=0.25)

        # ========== TOP MIDDLE (f_NN_mub vs f_faserv)

        (axMsimb,) = axM.plot(
            x_vals,
            faser_pdf_mub_epos,
            linestyle="-",
            label=r"$f_{\mathrm{EPOS+POWHEG}\bar{\nu}_\mu}(x)$",
            color="red",
        )

        # axM_err = axM.fill_between(
        #     x_vals,
        #     mean_mub + error_mub,
        #     mean_mub - error_mub,
        #     color="#648fff",
        #     alpha=0.2,
        #     label=r"$\pm 1\sigma$",
        # )

        # axM_mu = axM.plot(
        #     x_vals,
        #     mean_mub,
        #     linestyle="-.",
        #     color="#648fff",
        #     label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
        # )

        (axMbsm_eta_err) = axM.fill_between(
            x_vals,
            mean_mub_bsm_eta + error_mub_bsm_eta,
            mean_mub_bsm_eta - error_mub_bsm_eta,
            color="#fe6100",
            alpha=0.2,
            label=r"$\pm 1\sigma$",
        )

        (axMbsm_eta,) = axM.plot(
            x_vals,
            mean_mub_bsm_eta,
            linestyle="--",
            color="#fe6100",
            label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
        )

        (axMbsm_eta_prime_err) = axM.fill_between(
            x_vals,
            mean_mub_bsm_eta_prime + error_mub_bsm_eta_prime,
            mean_mub_bsm_eta_prime - error_mub_bsm_eta_prime,
            color="#dc267f",
            alpha=0.2,
            label=r"$\pm 1\sigma$",
        )

        (axMbsm_eta_prime,) = axM.plot(
            x_vals,
            mean_mub_bsm_eta_prime,
            linestyle="-",
            color="#dc267f",
            label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
        )

        (axMbsm_pi_err) = axM.fill_between(
            x_vals,
            mean_mub_bsm_pi + error_mub_bsm_pi,
            mean_mub_bsm_pi - error_mub_bsm_pi,
            color="#000000",
            alpha=0.2,
            label=r"$\pm 1\sigma$",
        )

        (axMbsm_pi,) = axM.plot(
            x_vals,
            mean_mub_bsm_pi,
            linestyle=":",
            color="#000000",
            label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
        )

        axM.legend(
            [
                axLsim,
                # (axM_mu, axM_err),
                (axMbsm_eta_err, axMbsm_eta),
                (axMbsm_eta_prime_err, axMbsm_eta_prime),
                (axMbsm_pi_err, axMbsm_pi),
            ],
            [
                # r"${ \mathrm{baseline}}$",
                r"$\mathrm{SM}  \ (f_{\nu_\mu} +f_{\bar{\nu}_\mu})$",
                r"$\mathrm{BSM}(\eta \rightarrow \nu_\mu + \bar{\nu}_\mu)$",
                r"$\mathrm{BSM}(\eta^{'} \rightarrow \nu_\mu + \bar{\nu}_\mu)$",
                r"$\mathrm{BSM}(\pi \rightarrow \nu_\mu + \bar{\nu}_\mu)$",
                # r"$ \mathrm{SIBYLL(\pi,K,c)}$",
                # r"$ \mathrm{QGSJET(\pi,K)+POWHEG(c)}$",
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

        axM.set_ylabel(r"$x_\nu f_{\bar{\nu}_\mu}$")

        axM.set_xlim(0.02, 1)

        axM.set_ylim(1e0, 1e4)
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
            label=r"$\mathrm{EPOS}$",
        )

        axrL.set_xscale("log")
        # axrL.set_xlim(.02, 1)
        axrL.set_ylim(0.8, 1.2)
        axrL.grid(color="grey", linestyle="-", linewidth=0.25)
        axrL.set_ylabel(r"$\mathrm{Ratio}$")
        # axrL.set_xlabel(r"$x_\nu$")

        # ratio_center = mean_mu / faser_pdf_mu_epos
        # ratio_lower = (mean_mu - error_mu) / faser_pdf_mu_epos
        # ratio_upper = (mean_mu + error_mu) / faser_pdf_mu_epos

        # axrL.plot(x_vals, ratio_center, linestyle="-.", color="#648fff")
        # axrL.fill_between(x_vals, ratio_upper, ratio_lower, color="#648fff", alpha=0.2)

        ratio_center = mean_mu_bsm_eta / faser_pdf_mu_bsm_eta
        ratio_lower = (mean_mu_bsm_eta - error_mu_bsm_eta) / faser_pdf_mu_bsm_eta
        ratio_upper = (mean_mu_bsm_eta + error_mu_bsm_eta) / faser_pdf_mu_bsm_eta

        axrL.plot(x_vals, ratio_center, linestyle="--", color="#fe6100")
        axrL.fill_between(x_vals, ratio_upper, ratio_lower, color="#fe6100", alpha=0.2)

        ratio_center = mean_mu_bsm_eta_prime / faser_pdf_mu_bsm_eta_prime
        ratio_lower = (
            mean_mu_bsm_eta_prime - error_mu_bsm_eta_prime
        ) / faser_pdf_mu_bsm_eta_prime
        ratio_upper = (
            mean_mu_bsm_eta_prime + error_mu_bsm_eta_prime
        ) / faser_pdf_mu_bsm_eta_prime

        axrL.plot(x_vals, ratio_center, linestyle="-", color="#dc267f")
        axrL.fill_between(x_vals, ratio_upper, ratio_lower, color="#dc267f", alpha=0.2)

        ratio_center = mean_mu_bsm_pi / faser_pdf_mu_bsm_pi
        ratio_lower = (mean_mu_bsm_pi - error_mu_bsm_pi) / faser_pdf_mu_bsm_pi
        ratio_upper = (mean_mu_bsm_pi + error_mu_bsm_pi) / faser_pdf_mu_bsm_pi

        axrL.plot(x_vals, ratio_center, linestyle=":", color="#000000")
        axrL.fill_between(x_vals, ratio_upper, ratio_lower, color="#000000", alpha=0.2)

        # ratio_center = mean_mu_dpm_qgsjet / faser_pdf_mu_qgs
        # ratio_lower = (mean_mu_dpm_qgsjet - error_mu_dpm_qgsjet) / faser_pdf_mu_qgs
        # ratio_upper = (mean_mu_dpm_qgsjet + error_mu_dpm_qgsjet) / faser_pdf_mu_qgs

        # axrL.plot(x_vals, ratio_center, linestyle=":", color="purple")
        # axrL.fill_between(x_vals, ratio_upper, ratio_lower, color="purple", alpha=0.2)

        # ratio_center = mean_mu_sibyll / faser_pdf_mu_sib
        # ratio_lower = (mean_mu_sibyll - error_mu_sibyll) / faser_pdf_mu_sib
        # ratio_upper = (mean_mu_sibyll + error_mu_sibyll) / faser_pdf_mu_sib

        # axrL.plot(x_vals, ratio_center, linestyle="-", color="#000000")
        # axrL.fill_between(x_vals, ratio_upper, ratio_lower, color="#000000", alpha=0.2)

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
            label=r"$\mathrm{EPOS}$",
        )

        # axrM.plot(
        #     x_vals,
        #     mean_mub / faser_pdf_mub_dpm,
        #     linestyle="-.",
        #     color="#648fff",
        # )
        # axrM.fill_between(
        #     x_vals,
        #     (mean_mub + error_mub) / faser_pdf_mub_dpm,
        #     (mean_mub - error_mub) / faser_pdf_mub_dpm,
        #     color="#648fff",
        #     alpha=0.2,
        # )

        axrM.plot(
            x_vals,
            mean_mub_bsm_eta / faser_pdf_mub_bsm_eta,
            linestyle="--",
            color="#fe6100",
        )
        axrM.fill_between(
            x_vals,
            (mean_mub_bsm_eta + error_mub_bsm_eta) / faser_pdf_mub_bsm_eta,
            (mean_mub_bsm_eta - error_mub_bsm_eta) / faser_pdf_mub_bsm_eta,
            color="#fe6100",
            alpha=0.2,
        )

        axrM.plot(
            x_vals,
            mean_mub_bsm_eta_prime / faser_pdf_mub_bsm_eta,
            linestyle="-",
            color="#dc267f",
        )
        axrM.fill_between(
            x_vals,
            (mean_mub_bsm_eta_prime + error_mub_bsm_eta_prime)
            / faser_pdf_mub_bsm_eta_prime,
            (mean_mub_bsm_eta_prime - error_mub_bsm_eta_prime)
            / faser_pdf_mub_bsm_eta_prime,
            color="#dc267f",
            alpha=0.2,
        )

        axrM.plot(
            x_vals,
            mean_mub_bsm_pi / faser_pdf_mub_bsm_pi,
            linestyle=":",
            color="#000000",
        )
        axrM.fill_between(
            x_vals,
            (mean_mub_bsm_pi + error_mub_bsm_pi) / faser_pdf_mub_bsm_pi,
            (mean_mub_bsm_pi - error_mub_bsm_pi) / faser_pdf_mub_bsm_pi,
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

        # axLsig.plot(
        #     x_vals,
        #     faser_pdf_mu_epos / faser_pdf_mu_epos,
        #     linestyle="-",
        #     color="red",
        # )
        # axLsig.fill_between(
        #     x_vals,
        #     (mean_mu + error_mu) / mean_mu,
        #     (mean_mu - error_mu) / mean_mu,
        #     color="#648fff",
        #     alpha=0.2,
        # )

        axLsig.plot(
            x_vals,
            np.abs(mean_mu_bsm_eta - faser_pdf_mu_epos)
            / (np.sqrt(error_mu_bsm_eta**2 + error_mu**2)),
            linestyle="--",
            color="#fe6100",
        )
        # axLsig.fill_between(
        #     x_vals,
        #     (mean_mu_bsm_eta + error_mu_bsm_eta) / faser_pdf_mu_epos,
        #     (mean_mu_bsm_eta - error_mu_bsm_eta) / faser_pdf_mu_epos,
        #     color="#fe6100",
        #     alpha=0.2,
        # )

        axLsig.plot(
            x_vals,
            np.abs(mean_mu_bsm_eta_prime - faser_pdf_mu_epos)
            / (np.sqrt(error_mu_bsm_eta_prime**2 + error_mu**2)),
            linestyle="-",
            color="#dc267f",
        )
        # axLsig.fill_between(
        #     x_vals,
        #     (mean_mu_bsm_eta_prime + error_mu_bsm_eta_prime) / faser_pdf_mu_epos,
        #     (mean_mu_bsm_eta_prime - error_mu_bsm_eta_prime) / faser_pdf_mu_epos,
        #     color="#dc267f",
        #     alpha=0.2,
        # )

        axLsig.plot(
            x_vals,
            np.abs(mean_mu_bsm_pi - faser_pdf_mu_epos)
            / (np.sqrt(error_mu_bsm_pi**2 + error_mu**2)),
            linestyle=":",
            color="#000000",
        )
        # axLsig.fill_between(
        #     x_vals,
        #     (mean_mu_bsm_pi + error_mu_bsm_pi) / faser_pdf_mu_epos,
        #     (mean_mu_bsm_pi - error_mu_bsm_pi) / faser_pdf_mu_epos,
        #     color="#000000",
        #     alpha=0.2,
        # )

        axLsig.set_xscale("log")
        axLsig.set_xlim(0.02, 1)
        axLsig.set_ylim(0, 5)
        axLsig.grid(color="grey", linestyle="-", linewidth=0.25)
        axLsig.set_ylabel(r"$\mathrm{Deviation} \ \sigma$")
        axLsig.set_xlabel(r"$x_\nu$")

        # axMsig.plot(
        #     x_vals,
        #     faser_pdf_mub_epos / faser_pdf_mub_epos,
        #     linestyle="-",
        #     color="red",
        # )
        # axMsig.fill_between(
        #     x_vals,
        #     (mean_mub + error_mu) / mean_mub,
        #     (mean_mub - error_mu) / mean_mub,
        #     color="#648fff",
        #     alpha=0.2,
        # )

        axMsig.plot(
            x_vals,
            np.abs(mean_mub_bsm_eta - faser_pdf_mub_epos)
            / (np.sqrt(error_mub_bsm_eta**2 + error_mub**2)),
            linestyle="--",
            color="#fe6100",
        )
        # axMsig.fill_between(
        #     x_vals,
        #     (mean_mub_bsm_eta + error_mub_bsm_eta) / faser_pdf_mub_epos,
        #     (mean_mub_bsm_eta - error_mub_bsm_eta) / faser_pdf_mub_epos,
        #     color="#fe6100",
        #     alpha=0.2,
        # )

        axMsig.plot(
            x_vals,
            np.abs(mean_mub_bsm_eta_prime - faser_pdf_mub_epos)
            / (np.sqrt(error_mub_bsm_eta_prime**2 + error_mub**2)),
            linestyle="-",
            color="#dc267f",
        )
        # axMsig.fill_between(
        #     x_vals,
        #     (mean_mub_bsm_eta_prime + error_mub_bsm_eta_prime) / faser_pdf_mub_epos,
        #     (mean_mub_bsm_eta_prime - error_mub_bsm_eta_prime) / faser_pdf_mub_epos,
        #     color="#dc267f",
        #     alpha=0.2,
        # )

        axMsig.plot(
            x_vals,
            np.abs(mean_mub_bsm_pi - faser_pdf_mub_epos)
            / (np.sqrt(error_mub_bsm_pi**2 + error_mub**2)),
            linestyle=":",
            color="#000000",
        )
        # axMsig.fill_between(
        #     x_vals,
        #     (mean_mub_bsm_pi + error_mub_bsm_pi) / faser_pdf_mub_epos,
        #     (mean_mub_bsm_pi - error_mub_bsm_pi) / faser_pdf_mub_epos,
        #     color="#000000",
        #     alpha=0.2,
        # )

        axMsig.set_xscale("log")
        axMsig.set_xlim(0.02, 1)
        axMsig.set_ylim(0, 5)
        axMsig.grid(color="grey", linestyle="-", linewidth=0.25)
        axMsig.set_ylabel(r"$\mathrm{Deviation} \ \sigma$")
        axMsig.set_xlabel(r"$x_\nu$")

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.13)
        plt.savefig(f"{obs}_compare_bsm_eta_mu.pdf")
        plt.show()


x_vals = np.logspace(-5, 0, 1000)
plot(x_vals)
