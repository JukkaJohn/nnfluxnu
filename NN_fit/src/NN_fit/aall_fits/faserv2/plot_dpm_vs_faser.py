from matplotlib.legend_handler import HandlerTuple
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import lhapdf

# Add the parent directory to sys.path
# parent_dir = os.path.abspath(
#     os.path.join(
#         os.getcwd(),
#         "/Users/jukkajohn/Masterscriptie/faser_nufluxes_ml/ML_fit_enu/src/ML_fit_neutrinos/",
#     )
# )
# sys.path.append(parent_dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from read_faserv_pdf import read_pdf

# Data for plot


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
neutrino_pdfs_mu_faser = np.loadtxt(
    "/Users/jukkajohn/Masterscriptie/faser_nufluxes_ml/ML_fit_enu/src/ML_fit_neutrinos/fit_faser_data/mu_pdf.txt",
    delimiter=",",
)
neutrino_pdfs_mub_faser = np.loadtxt(
    "/Users/jukkajohn/Masterscriptie/faser_nufluxes_ml/ML_fit_enu/src/ML_fit_neutrinos/fit_faser_data/mub_pdf.txt",
    delimiter=",",
)

# neutrino_pdfs_mu_dpm_Eh = np.loadtxt("fit_dpmjet/Eh_fit/mu_pdf.txt", delimiter=",")
# neutrino_pdfs_mub_dpm_Eh = np.loadtxt("fit_dpmjet/Eh_fit/mub_pdf.txt", delimiter=",")

neutrino_pdfs_mu_dpm_El = np.loadtxt(
    "/Users/jukkajohn/Masterscriptie/faser_nufluxes_ml/ML_fit_enu/src/ML_fit_neutrinos/2024_faser/fit_dpmjet/El_fit/mu_pdf.txt",
    delimiter=",",
)
neutrino_pdfs_mub_dpm_El = np.loadtxt(
    "/Users/jukkajohn/Masterscriptie/faser_nufluxes_ml/ML_fit_enu/src/ML_fit_neutrinos/2024_faser/fit_dpmjet/El_fit/mub_pdf.txt",
    delimiter=",",
)

# neutrino_pdfs_mu_dpm_theta = np.loadtxt(
#     "fit_dpmjet/theta_fit/mu_pdf.txt", delimiter=","
# )
# neutrino_pdfs_mub_dpm_theta = np.loadtxt(
#     "fit_dpmjet/theta_fit/mub_pdf.txt", delimiter=","
# )


# Get number of reps from make runscripts
def plot(
    x_vals,
):
    x_vals = np.array(x_vals)
    pdf = "FASER_2412.03186_DPMJET+DPMJET_7TeV"
    faser_pdf_mu, x_faser = read_pdf(pdf, x_vals, 14)
    faser_pdf_mub, x_faser = read_pdf(pdf, x_vals, -14)
    faser_pdf_mu = faser_pdf_mu * x_vals * 1.16186e-09
    faser_pdf_mub = faser_pdf_mub * x_vals * 1.16186e-09

    mean_mu_faser = np.mean(neutrino_pdfs_mu_faser, axis=0) * x_vals
    error_mu_faser = np.std(neutrino_pdfs_mu_faser, axis=0) * x_vals
    mean_mub_faser = np.mean(neutrino_pdfs_mub_faser, axis=0) * x_vals
    error_mub_faser = np.std(neutrino_pdfs_mub_faser, axis=0) * x_vals

    mean_mu_dpm_El = np.mean(neutrino_pdfs_mu_dpm_El, axis=0) * x_vals * 65.6 / 150
    error_mu_dpm_El = np.std(neutrino_pdfs_mu_dpm_El, axis=0) * x_vals * 65.6 / 150
    mean_mub_dpm_El = np.mean(neutrino_pdfs_mub_dpm_El, axis=0) * x_vals * 65.6 / 150
    error_mub_dpm_El = np.std(neutrino_pdfs_mub_dpm_El, axis=0) * x_vals * 65.6 / 150

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

    axLfasererr2 = axL.fill_between(
        x_vals,
        mean_mu_faser + error_mu_faser * 2,
        mean_mu_faser - error_mu_faser * 2,
        color=mucolor,
        alpha=0.4,
        label=r"$\pm 1\sigma$",
    )

    axLfasererr4 = axL.fill_between(
        x_vals,
        mean_mu_faser + error_mu_faser * 4,
        mean_mu_faser - error_mu_faser * 4,
        color="green",
        alpha=0.2,
        label=r"$\pm 1\sigma$",
    )

    (axLfaser,) = axL.plot(
        x_vals,
        mean_mu_faser,
        linestyle="-",
        color=mucolor,
        label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
    )

    # axLdpm_El_err2 = axL.fill_between(
    #     x_vals,
    #     mean_mu_dpm_El + error_mu_dpm_El * 2,
    #     mean_mu_dpm_El - error_mu_dpm_El * 2,
    #     color="red",
    #     alpha=0.4,
    #     label=r"$\pm 1\sigma$",
    # )

    # axLdpm_El_err4 = axL.fill_between(
    #     x_vals,
    #     mean_mu_dpm_El + error_mu_dpm_El * 4,
    #     mean_mu_dpm_El - error_mu_dpm_El * 4,
    #     color="black",
    #     alpha=0.2,
    #     label=r"$\pm 1\sigma$",
    # )

    # (axLdpm_El,) = axL.plot(
    #     x_vals,
    #     mean_mu_dpm_El,
    #     linestyle="-",
    #     color="red",
    #     label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
    # )

    (axLdpm_El,) = axL.plot(
        x_vals,
        faser_pdf_mu,
        linestyle="-",
        color="red",
        label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
    )

    axL.legend(
        # [axLsim, (axLnn, axLnnerr), (axLvert1, axLvert2)],
        [
            # axLsim,
            (
                axLfaser,
                axLfasererr2,
            ),
            (axLfaser, axLfasererr4),
            # (axLdpm_Eh_err, axLdpm_Eh),
            # (axLdpm_El_err2, axLdpm_El),
            # (axLdpm_El_err4, axLdpm_El),
            (axLdpm_El),
            # (axLdpm_theta_err, axLdpm_theta),
        ],
        [
            # r"$f_{\nu_\mu, \mathrm{ref}}$",
            r"$f_{\nu_\mu, \mathrm{FASER} \ E_\nu \ 2 \sigma}$",
            r"$f_{\nu_\mu, \mathrm{FASER} \ E_\nu \ 4 \sigma}$",
            # r"$f_{\nu_\mu, \mathrm{DPM}\ E_h}$",
            r"$f_{\nu_\mu, \mathrm{DPMJET}}$",
            # r"$f_{\nu_\mu, \mathrm{DPMJET}\ E_l \ 4 \sigma}$",
            # r"$f_{\nu_\mu, \mathrm{DPM}\ \theta}$",
        ],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="lower left",
    ).set_alpha(0.8)

    axL.set_xlim(5e-4, 1)
    axL.set_ylim(1e-3, 1e5)
    axL.set_yscale("log")
    axL.set_xscale("log")

    # if preproc == 1:
    title_str = r"$\mathrm{DPMJET} \ \mathrm{vs} \ \mathrm{FASER} \ E_\nu$"
    # if preproc == 2:
    # title_str = rf"$f_{{\mathrm{{NN}}}}(x) = \mathrm{{NN}}_{{{layers}}}(x) -  \mathrm{{NN}}_{{{layers}}}(1)$"

    axL.set_title(title_str)
    # fig.text(0.33, 0.94, title_str, ha="center", va="bottom", fontsize=10)
    axL.set_ylabel(r"$xf_{\nu_\mu}(x_\nu)$")
    axL.set_xticklabels([])
    axL.grid(color="grey", linestyle="-", linewidth=0.25)

    # ========== TOP MIDDLE (f_NN_mub vs f_faserv)

    axMfasererr2 = axM.fill_between(
        x_vals,
        (mean_mub_faser + error_mub_faser * 2),
        (mean_mub_faser - error_mub_faser * 2),
        color=mubcolor,
        alpha=0.4,
        label=r"$\pm 1\sigma$",
    )

    axMfasererr4 = axM.fill_between(
        x_vals,
        (mean_mub_faser + error_mub_faser * 4),
        (mean_mub_faser - error_mub_faser * 4),
        color="green",
        alpha=0.2,
        label=r"$\pm 1\sigma$",
    )
    (axMfaser,) = axM.plot(
        x_vals,
        mean_mub_faser,
        linestyle="-",
        color=mubcolor,
        label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
    )

    # axMdpm_El_err2 = axM.fill_between(
    #     x_vals,
    #     (mean_mub_dpm_El + error_mub_dpm_El * 2),
    #     (mean_mub_dpm_El - error_mub_dpm_El * 2),
    #     color="black",
    #     alpha=0.4,
    #     label=r"$\pm 1\sigma$",
    # )

    # axMdpm_El_err4 = axM.fill_between(
    #     x_vals,
    #     (mean_mub_dpm_El + error_mub_dpm_El * 4),
    #     (mean_mub_dpm_El - error_mub_dpm_El * 4),
    #     color="red",
    #     alpha=0.4,
    #     label=r"$\pm 1\sigma$",
    # )

    # (axMdpm_El,) = axM.plot(
    #     x_vals,
    #     mean_mub_dpm_El,
    #     linestyle="-",
    #     color="red",
    #     label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
    # )

    (axMdpm_El,) = axM.plot(
        x_vals,
        faser_pdf_mub,
        linestyle="-",
        color="red",
        label=r"$f_{\mathrm{NNflux}_\nu}(x)$",
    )

    axM.legend(
        # [axMsimb, (axMnnberr, axMnnb), (axMvert1, axMvert2)],
        [
            # axMsimb,
            (axMfasererr2, axMfaser),
            (axMfasererr4, axMfaser),
            # (axMdpm_Eh_err, axMdpm_Eh),
            # (axMdpm_El_err2, axMdpm_El),
            # (axMdpm_El_err4, axMdpm_El),
            (axMdpm_El),
            # (axMdpm_theta_err, axMdpm_theta),
        ],
        [
            # r"$f_{\nu_{\bar{\mu}}, \mathrm{ref}}$",
            r"$f_{\bar{\nu}_\mu, \mathrm{FASER} \ E_\nu \ 2 \sigma}$",
            r"$f_{\bar{\nu}_\mu, \mathrm{EPOS+FASER} \ E_\nu \ 4 \sigma}$",
            # r"$f_{\nu_{\bar{\mu}}, \mathrm{DPM}\ E_h}$",
            r"$f_{\bar{\nu}_\mu, \mathrm{DPMJET}}$",
            # r"$f_{\bar{\nu}_\mu, \mathrm{DPMJET} \ E_l \ 2 \sigma}$",
            # r"$f_{\bar{\nu}_\mu, \mathrm{DPMJET} \ E_l \ 4 \sigma}$",
            # r"$f_{\nu_{\bar{\mu}}, \mathrm{DPM} \ \theta}$",
        ],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="lower left",
    ).set_alpha(0.8)

    title_str = r"$\mathcal{L}_{\mathrm{pp}} = 65.6 \mathrm{fb}^{-1}$"
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

    axrL.plot(x_vals, np.ones(len(x_vals)), linestyle="-", color=simcolor)

    axrL.set_xscale("log")
    axrL.set_xlim(5e-4, 1)
    axrL.set_ylim(0, 2)
    axrL.grid(color="grey", linestyle="-", linewidth=0.25)
    axrL.set_ylabel(r"$\mathrm{Ratio}$")
    axrL.set_xlabel(r"$x_\nu$")

    ratio_center = mean_mu_faser / mean_mu_faser
    ratio_lower = (mean_mu_faser - error_mu_faser * 2) / mean_mu_faser
    ratio_upper = (mean_mu_faser + error_mu_faser * 2) / mean_mu_faser

    axrL.plot(x_vals, ratio_center, linestyle="-", color=mucolor)
    axrL.fill_between(x_vals, ratio_upper, ratio_lower, color=mucolor, alpha=0.4)

    ratio_center = mean_mu_faser / mean_mu_faser
    ratio_lower = (mean_mu_faser - error_mu_faser * 4) / mean_mu_faser
    ratio_upper = (mean_mu_faser + error_mu_faser * 4) / mean_mu_faser
    axrL.fill_between(x_vals, ratio_upper, ratio_lower, color="green", alpha=0.2)

    ratio_center = faser_pdf_mu / mean_mu_faser
    ratio_lower = (mean_mu_dpm_El - error_mu_dpm_El * 2) / mean_mu_faser
    ratio_upper = (mean_mu_dpm_El + error_mu_dpm_El * 2) / mean_mu_faser

    axrL.plot(x_vals, ratio_center, linestyle="-", color="red")
    # axrL.fill_between(x_vals, ratio_upper, ratio_lower, color="black", alpha=0.4)

    ratio_center = mean_mu_dpm_El / mean_mu_faser
    ratio_lower = (mean_mu_dpm_El - error_mu_dpm_El * 4) / mean_mu_faser
    ratio_upper = (mean_mu_dpm_El + error_mu_dpm_El * 4) / mean_mu_faser

    # axrL.fill_between(x_vals, ratio_upper, ratio_lower, color="red", alpha=0.4)

    axrL.set_xscale("log")
    axrL.set_xlim(5e-4, 1)
    axrL.set_ylim(0, 3)
    axrL.grid(color="grey", linestyle="-", linewidth=0.25)
    axrL.set_ylabel(r"$\mathrm{Ratio}$")
    axrL.set_xlabel(r"$x_\nu$")

    # ========== BOTTOM MIDDLE (ratio f_NN_mub)

    axrM.plot(x_vals, mean_mub_faser / mean_mub_faser, linestyle="-", color=mubcolor)
    axrM.fill_between(
        x_vals,
        (mean_mub_faser + error_mub_faser * 2) / mean_mub_faser,
        (mean_mub_faser - error_mub_faser * 2) / mean_mub_faser,
        color=mubcolor,
        alpha=0.4,
    )

    axrM.fill_between(
        x_vals,
        (mean_mub_faser + error_mub_faser * 4) / mean_mub_faser,
        (mean_mub_faser - error_mub_faser * 4) / mean_mub_faser,
        color="green",
        alpha=0.2,
    )

    axrM.plot(x_vals, faser_pdf_mu / mean_mub_faser, linestyle="-", color="red")
    # axrM.fill_between(
    #     x_vals,
    #     (mean_mub_dpm_El + error_mub_dpm_El * 2) / mean_mub_faser,
    #     (mean_mub_dpm_El - error_mub_dpm_El * 2) / mean_mub_faser,
    #     color="red",
    #     alpha=0.4,
    # )

    # axrM.fill_between(
    #     x_vals,
    #     (mean_mub_dpm_El + error_mub_dpm_El * 4) / mean_mub_faser,
    #     (mean_mub_dpm_El - error_mub_dpm_El * 4) / mean_mub_faser,
    #     color="black",
    #     alpha=0.4,
    # )

    axrM.set_xscale("log")
    axrM.set_xlim(5e-4, 1)
    axrM.set_ylim(0, 3)
    axrM.grid(color="grey", linestyle="-", linewidth=0.25)
    # axrM.set_ylabel(r"$\mathrm{Ratio}$")
    axrM.set_xlabel(r"$x_\nu$")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.13)
    # plt.savefig("dpm_vs_faser.pdf")
    plt.show()


x_vals = np.logspace(-5, 0, 1000)
plot(x_vals)
