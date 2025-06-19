# Author: Jukka John
# This file writes neutrino pdfs to lhapdf grids
from typing import Dict, Union, List
import numpy as np


def write_lhapdf_grid(
    xgrid: Union[np.ndarray, List[float]],
    pdf_dict: Dict[int, Union[np.ndarray, List[float]]],
    path: str,
) -> None:
    """
    Writes a set of neutrino PDFs to a file in LHAPDF grid format (lhagrid1).

    This function formats and saves PDF data into a file readable by LHAPDF tools,
    using the specified x-grid and dictionary of parton distribution functions.

    Parameters:
    -----------
    xgrid : array-like
        Array of Bjorken-x values at which PDFs are evaluated.
    pdf_dict : dict
        Dictionary mapping particle IDs (PDG codes) to arrays of PDF values.
        Each value should be an array of length equal to the length of `xgrid`.
    path : str
        Path to the output `.dat` file where the grid will be written.

    Notes:
    ------
    - The output format follows LHAPDF's `lhagrid1` specification.
    - Each PDF line is duplicated, as required by the LHAPDF grid format.
    - The order of flavors is sorted by PDG code.
    """
    with open(path, "w") as f:
        f.write("PdfType: replica\n")
        f.write("Format: lhagrid1\n")
        f.write("---\n")

        f.write("  " + "  ".join(f"{val:.8e}" for val in xgrid) + "\n")

        f.write("0.1E+001 0.1E+007\n")

        pids = sorted(pdf_dict.keys())
        f.write(" ".join(str(pid) for pid in pids) + "\n")

        num_x = len(xgrid)
        for i in range(num_x):
            line = " ".join(f"{pdf_dict[pid][i]:.14e}" for pid in pids)
            f.write(line + "\n")
            f.write(line + "\n")
        f.write("---\n")


def customize_info_file(
    template_path: str, output_path: str, set_index: int, flavor: str, num_members: int
) -> None:
    """
    Creates a customized LHAPDF `.info` file from a template by replacing placeholders.

    Parameters:
    -----------
    template_path : str
        Path to the `.info` template file containing placeholders (e.g., SETINDEX, FLAVOR).
    output_path : str
        Path to the output `.info` file to be generated.
    set_index : int
        Unique identifier for the PDF set (used to replace "SETINDEX" in the template).
    flavor : str
        Flavor content to be listed in the info file (used to replace "FLAVOR").
        Can be a single PDG ID (e.g., "12") or a comma-separated list (e.g., "14, -14").
    num_members : int
        Number of PDF members or replicas (used to replace the "NumMembers" field).

    Notes:
    ------
    - The function assumes the template has default "NumMembers: 1000" and replaces that value.
    - All replaced content is written to the specified output path.
    """
    with open(template_path, "r") as file:
        content = file.read()

    content = content.replace("SETINDEX", str(set_index))
    content = content.replace("FLAVOR", str(flavor))
    content = content.replace("NumMembers: 1000", f"NumMembers: {str(num_members)}")

    with open(output_path, "w") as file:
        file.write(content)
