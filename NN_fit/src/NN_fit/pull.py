import numpy as np


def compute_pull(mean_pdf1, mean_pdf2, error_pdf1, error_pdf2):
    pull = np.abs(mean_pdf1 - mean_pdf2) / np.sqrt(error_pdf1**2 - error_pdf2**2)
    return pull
