import numpy as np


def compute_pull(mean_pdf1, mean_pdf2, error_pdf1, error_pdf2):
    """Computes pull between pdfs

    Args:
        mean_pdf1 (np array): neutrino pdf 1
        mean_pdf2 (np array): neutrino pdf 2
        error_pdf1 (np array): std neutrino pdf 1
        error_pdf2 (np array): std neutrino pdf 2

    Returns:
        list: pull
    """
    pull = np.abs(mean_pdf1 - mean_pdf2) / np.sqrt(error_pdf1**2 - error_pdf2**2)
    return pull
