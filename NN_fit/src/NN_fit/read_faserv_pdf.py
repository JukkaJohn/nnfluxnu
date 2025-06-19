import lhapdf
import numpy as np
from typing import Tuple


def read_pdf(
    pdf: str, x_vals: np.ndarray, particle: int, set: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the parton distribution function (PDF) values for a given particle
    at specified momentum fractions and energy scale using LHAPDF.

    Parameters
    ----------
    pdf : str
        Name of the PDF set to load.
    x_vals : np.ndarray
        Array of momentum fraction values (x) at which to evaluate the PDF.
    particle : int
        Particle ID (PDG code) for which the PDF is evaluated.
    set : int
        Specific member or set number within the PDF.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - pdf_vals: np.ndarray of PDF values normalized by x_vals.
        - x_vals: The input array of momentum fractions.
    """
    pid = particle
    Q2 = 10
    pdf = lhapdf.mkPDF(pdf, set)
    pdf_vals = [pdf.xfxQ2(pid, x, Q2) for x in x_vals]
    pdf_vals = np.array(pdf_vals)
    pdf_vals /= x_vals
    return pdf_vals, x_vals
