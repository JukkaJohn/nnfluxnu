import lhapdf
import numpy as np
from read_faserv_pdf import read_pdf
import os

import matplotlib.pyplot as plt
import sys
import pandas as pd

lhapdf.setVerbosity(0)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

plt.rcParams["text.usetex"] = True
plt.rcParams.update(
    {
        # "font.family": "serif",
        # "font.serif": ["cmr10"],  # Computer Modern]
        "font.size": 14,
    }
)
file_path = os.path.join(parent_dir, "x_alpha.dat")
df = pd.read_csv(file_path, sep="\s+", header=None)
x_vals = df.to_numpy()
x_vals = x_vals.flatten()
# x_vals = np.logspace(-5, 0, 1000)
pdf = "FASER_2412.03186_EPOS+POWHEG_7TeV"
neutrino, x = read_pdf(pdf, x_vals, 14)
plt.plot(x_vals, neutrino, linestyle="-", label="FASER 2412.03186")

pdf = "FASERv_Run3_EPOS+POWHEG_7TeV"
neutrino_mu, x = read_pdf(pdf, x_vals, 14)
plt.plot(x_vals, neutrino, linestyle="--", label="FASER run 3")

pdf = "FASERv_Run3_EPOS+POWHEG_7TeV"
neutrino, x = read_pdf(pdf, x_vals, -14)
plt.plot(x_vals, neutrino, linestyle="--", label="FASER run 3")
print(neutrino_mu / neutrino)

# pdf = "FASERv2_EPOS+POWHEG_7TeV"
# neutrino, x = read_pdf(pdf, x_vals, -12)
# plt.plot(x_vals, neutrino, linestyle=":", label="FASERv2")

# pdf = "FASERv_Run3_EPOS+POWHEG_7TeV"
# neutrino, x = read_pdf(pdf, x_vals, -12)
# plt.plot(x_vals, neutrino * 20, linestyle="-.", label="FASER high lumi")
# plt.legend()
# plt.text(
#     0.2,
#     5 * 10**5,
#     "electron neutrinos",
#     fontsize=14,
#     color="black",
# )
# plt.xlabel(r"$x_\nu$")
# plt.ylabel(r"$f_{{\nu}_e}(x_\nu)$")
# plt.xlim(1e-2, 1)
# plt.ylim(1e-0, 1e6)
# plt.xscale("log")
# plt.yscale("log")
# plt.savefig("elec_pdfs_sim.pdf")
# plt.show()
