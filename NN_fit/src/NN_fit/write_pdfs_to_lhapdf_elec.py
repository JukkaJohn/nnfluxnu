import numpy as np
from write_all_pdfs_to_lhapdf import write_lhapdf_grid, customize_info_file
import os

import lhapdf

lhapdf.setVerbosity(0)


def is_numeric_row(row):
    try:
        [float(x) for x in row]
        return True
    except ValueError:
        return False


geometries_mu = [
    # "elec_bsm_eta_prime",
    # "elec_bsm_pi",
    # "elec_fnu_bsm_eta_prime",
    # "2024_faser",
    # "faserv2",
    # "high_lumi_faserv",
    # "run_3_gens",
    "IC_fit",
    "IC_faserv2",
]

geometries_el = [
    # "elec_bsm_eta_prime",
    # "elec_bsm_pi",
    # "elec_fnu_bsm_eta_prime",
    # "new_2024faser",
    # "new_faserv2",
    # "new_high_lumi",
    # "new_run_3_gens",
    "elec_IC",
    "elec_IC_faserv2",
]
x_vals = np.logspace(-5, 0, 1000)
generators = ["fit_dpmjet", "fit_epos", "fit_qgsjet", "fit_sibyll"]
observables = ["Eh_fit", "El_fit", "Enu_fit", "theta_fit"]
template_path = "/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/template_.info"
set_index = 10000000
num_members = 2
for geometry_mu, geometry_el in zip(geometries_mu, geometries_el):
    for generator in generators:
        for observable in observables:
            print(geometries_mu, generator, observable)
            pdf_dict_central = {}
            pdf_dict_error = {}
            dir_for_lhapdf = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{geometry_mu}_{generator}_{observable}"

            os.makedirs(dir_for_lhapdf, exist_ok=True)
            path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{geometry_mu}_{generator}_{observable}/{geometry_mu}_{generator}_{observable}.info"
            set_index += 1
            customize_info_file(
                template_path, path, set_index, "12,-14,14", num_members
            )
            with open(
                f"/Users/jukkajohn/Masterscriptie/faser_nufluxes_ml/ML_fit_enu/src/ML_fit_neutrinos/{geometry_el}/{generator}/{observable}/pdf.txt"
            ) as f:
                lines = [line.strip().split(",") for line in f if line.strip()]
                numeric_data = [
                    list(map(float, row)) for row in lines if is_numeric_row(row)
                ]

            neutrino_pdfs = np.array(numeric_data)
            pdf_dict_central[12] = np.mean(neutrino_pdfs, axis=0)
            pdf_dict_error[12] = np.std(neutrino_pdfs, axis=0)

            mean_pdf = np.mean(neutrino_pdfs, axis=0)
            path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{geometry_mu}_{generator}_{observable}/{geometry_mu}_{generator}_{observable}_0000.dat"
            # write_lhapdf_grid(x_vals, mean_pdf, path, 12)
            err_pdf = np.std(neutrino_pdfs, axis=0)
            path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{geometry_mu}_{generator}_{observable}/{geometry_mu}_{generator}_{observable}_0001.dat"
            # write_lhapdf_grid(x_vals, err_pdf, path, 12)

            with open(
                f"/Users/jukkajohn/Masterscriptie/faser_nufluxes_ml/ML_fit_enu/src/ML_fit_neutrinos/{geometry_mu}/{generator}/{observable}/mu_pdf.txt"
            ) as f:
                lines = [line.strip().split(",") for line in f if line.strip()]
                numeric_data = [
                    list(map(float, row)) for row in lines if is_numeric_row(row)
                ]

            neutrino_pdfs = np.array(numeric_data)
            mean_pdf = np.mean(neutrino_pdfs, axis=0)
            pdf_dict_central[14] = np.mean(neutrino_pdfs, axis=0)
            pdf_dict_error[14] = np.std(neutrino_pdfs, axis=0)

            path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{geometry_mu}_{generator}_{observable}/{geometry_mu}_{generator}_{observable}_0000.dat"
            # write_lhapdf_grid(x_vals, mean_pdf, path, 14)
            err_pdf = np.std(neutrino_pdfs, axis=0)
            path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{geometry_mu}_{generator}_{observable}/{geometry_mu}_{generator}_{observable}_0001.dat"
            # write_lhapdf_grid(x_vals, err_pdf, path, 14)

            with open(
                f"/Users/jukkajohn/Masterscriptie/faser_nufluxes_ml/ML_fit_enu/src/ML_fit_neutrinos/{geometry_mu}/{generator}/{observable}/mub_pdf.txt"
            ) as f:
                lines = [line.strip().split(",") for line in f if line.strip()]
                numeric_data = [
                    list(map(float, row)) for row in lines if is_numeric_row(row)
                ]

            neutrino_pdfs = np.array(numeric_data)
            mean_pdf = np.mean(neutrino_pdfs, axis=0)
            pdf_dict_central[-14] = np.mean(neutrino_pdfs, axis=0)
            pdf_dict_error[-14] = np.std(neutrino_pdfs, axis=0)

            path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{geometry_mu}_{generator}_{observable}/{geometry_mu}_{generator}_{observable}_0000.dat"
            # write_lhapdf_grid(x_vals, mean_pdf, path, -14)
            err_pdf = np.std(neutrino_pdfs, axis=0)
            path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{geometry_mu}_{generator}_{observable}/{geometry_mu}_{generator}_{observable}_0001.dat"
            # write_lhapdf_grid(x_vals, err_pdf, path, -14)
            # exit()
            central_path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{geometry_mu}_{generator}_{observable}/{geometry_mu}_{generator}_{observable}_0000.dat"
            error_path = f"/opt/anaconda3/envs/test_lhapdf/share/LHAPDF/{geometry_mu}_{generator}_{observable}/{geometry_mu}_{generator}_{observable}_0001.dat"
            write_lhapdf_grid(x_vals, pdf_dict_central, central_path)
            write_lhapdf_grid(x_vals, pdf_dict_error, error_path)

            # if set_index == 10000004:
            #     exit()

            # pdf, x = read_pdf(f"{geometry_mu}_{generator}_{observable}", x_vals, 14, 0)
            # sig, x = read_pdf(f"{geometry_mu}_{generator}_{observable}", x_vals, 14, 1)
            # plt.plot(x, pdf)
            # plt.plot(x, pdf + sig)
            # plt.plot(x, pdf - sig)
            # pdf, x = read_pdf(f"{geometry_mu}_{generator}_{observable}", x_vals, -14, 0)
            # sig, x = read_pdf(f"{geometry_mu}_{generator}_{observable}", x_vals, -14, 1)
            # plt.plot(x, pdf)
            # plt.plot(x, pdf + sig)
            # plt.plot(x, pdf - sig)
            # plt.xlim(0.02, 1)
            # plt.ylim(1e-1, 1e4)
            # plt.xscale("log")
            # plt.yscale("log")
            # plt.show()
            # exit()
