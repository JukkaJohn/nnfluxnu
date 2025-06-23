import numpy as np
from typing import List, Tuple


class Postfit:
    def __init__(self) -> None:
        pass

    # def compute_arc_length(self, model):
    #     npoints = 199  # 200 intervals
    #     seg_min = [1e-6, 1e-4, 1e-2]
    #     seg_max = [1e-4, 1e-2, 1.0]
    #     res = 0
    #     for a, b in zip(seg_min, seg_max):
    #         eps = (b - a) / npoints
    #         ixgrid = np.linspace(a, b, npoints, endpoint=False)
    #         ixgrid = torch.tensor(ixgrid, dtype=torch.float32).view(-1, 1)

    #         pdf_vals_grid = model(ixgrid)
    #         pdf_vals_grid = pdf_vals_grid.detach().numpy().flatten()
    #         ixgrid = ixgrid.detach().numpy().flatten()

    #         fdiff = np.diff(pdf_vals_grid) / eps
    #         res += integrate.simpson(np.sqrt(1 + np.square(fdiff)), x=ixgrid[1:])
    #     return res

    def apply_postfit_criteria(
        self,
        chi_squares: List[float],
        N_event_pred: np.ndarray,
        neutrino_pdfs: np.ndarray,
        pred: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies post-fit criteria to filter out replicas with chi-squared values
        that deviate significantly from the mean (more than 4 standard deviations).

        Parameters
        ----------
        chi_squares : List[float]
            List of chi-squared values for each replica.
        N_event_pred : np.ndarray
            Array of predicted event yields for each replica.
        neutrino_pdfs : np.ndarray
            Array of predicted neutrino PDFs for each replica.
        pred : np.ndarray
            Array of original pseudo-data predictions for each replica.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Filtered arrays of neutrino_pdfs, N_event_pred, and pred with outlier replicas removed.

        Notes
        -----
        Replicas whose chi-squared differ from the mean by more than 4 standard deviations
        are considered outliers and removed.
        """
        sig_chi_squares = np.std(chi_squares)
        mean_chi_squares = np.mean(chi_squares)

        indices_to_remove = []
        for i in range(len(chi_squares)):
            diff_chi_squares = abs(chi_squares[i] - mean_chi_squares)
            if diff_chi_squares > 4 * sig_chi_squares:
                indices_to_remove.append(i)

        if len(indices_to_remove) > 0:
            indices_to_remove = np.array(indices_to_remove)
            N_event_pred = np.delete(N_event_pred, indices_to_remove)
            neutrino_pdfs = np.delete(neutrino_pdfs, indices_to_remove)
            pred = np.delete(pred, indices_to_remove)

        return neutrino_pdfs, N_event_pred, pred
