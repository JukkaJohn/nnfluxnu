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


# Get number of reps from make runscripts
def plot(
    x_vals,
):
    x_vals = np.array(x_vals)
    pdf = "FASERv_Run3_DPMJET+DPMJET_7TeV"
    faser_pdf_mu, x_faser = read_pdf(pdf, x_vals, 14)
    faser_pdf_mub, x_faser = read_pdf(pdf, x_vals, -14)
    faser_pdf_mu = faser_pdf_mu * x_vals * 1.16186e-09
    faser_pdf_mub = faser_pdf_mub * x_vals * 1.16186e-09

    plt.plot(x_vals, faser_pdf_mu, label="dpmjet")
    pdf = "FASERv_Run3_EPOS+POWHEG_7TeV"
    faser_pdf_mu, x_faser = read_pdf(pdf, x_vals, 14)
    faser_pdf_mub, x_faser = read_pdf(pdf, x_vals, -14)
    faser_pdf_mu = faser_pdf_mu * x_vals * 1.16186e-09
    faser_pdf_mub = faser_pdf_mub * x_vals * 1.16186e-09
    plt.plot(x_vals, faser_pdf_mu, label="epos + powheg")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel(r"$xf_{\nu_\mu}(x_\nu)$")
    plt.xlabel(r"$x_\nu$")
    plt.legend()
    plt.show()


plot(x_vals=np.logspace(-5, 0, 1000))
