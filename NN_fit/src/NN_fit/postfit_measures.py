import torch
import numpy as np
from read_faserv_pdf import read_pdf
from typing import Union, Tuple, List


class Measures:
    def __init__(
        self,
        cov_matrix: torch.Tensor,
        pdf: np.ndarray,
        N_event_pred: np.ndarray,
    ) -> None:
        self.cov_matrix = cov_matrix
        self.pdf = pdf
        self.N_event_pred = N_event_pred
        """
        Initialize Measures class with covariance matrix, PDF, and predicted events.

        Parameters
        ----------
        cov_matrix : torch.Tensor
            Covariance matrix used in chi-squared calculations.
        pdf : np.ndarray
            Reference PDF array.
        N_event_pred : np.ndarray
            Predicted event yields array, shape (num_replicas, num_bins).
        """

    def compute_delta_chi(
        self,
        level0: Union[np.ndarray, torch.Tensor],
        N_event_pred: np.ndarray,
        data_level1: torch.Tensor,
        x_vals: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the relative change in chi-squared between a baseline theory prediction
        and the fit prediction.

        Parameters
        ----------
        level0 : Union[np.ndarray, torch.Tensor]
            Baseline theory prediction.
        N_event_pred : np.ndarray
            Predicted events for all replicas.
        data_level1 : torch.Tensor
            Observed data level 1 tensor, shape (num_bins, 1).
        x_vals : Union[np.ndarray, torch.Tensor]
            x-values (unused in this function).

        Returns
        -------
        torch.Tensor
            Relative change in chi-squared (delta chi).
        """
        level0 = torch.tensor(level0, dtype=torch.float32)
        data_level1 = data_level1.view(-1, 1)
        diff = level0 - data_level1

        diffcov = torch.matmul(self.cov_matrix, diff)
        chi_theory = torch.dot(diff.view(-1), diffcov.view(-1))

        mean_N_events = np.mean(N_event_pred, axis=0)
        mean_N_events = torch.tensor(mean_N_events, dtype=torch.float32)
        diff = mean_N_events - data_level1[0]

        diffcov = torch.matmul(self.cov_matrix, diff)
        chi_fit = torch.dot(diff.view(-1), diffcov.view(-1))

        delta_chi = (chi_fit - chi_theory) / chi_theory

        return delta_chi

    def compute_phi(
        self,
        data: Union[np.ndarray, torch.Tensor],
        chi_squares: List[float],
    ) -> float:
        """
        Compute the phi metric as the difference between average chi-square over replicas
        and chi-square of the mean prediction.

        Parameters
        ----------
        data : Union[np.ndarray, torch.Tensor]
            Observed data points.
        chi_squares : List[float]
            List of chi-square losses per replica.

        Returns
        -------
        float
            The phi metric.
        """
        num_reps = self.N_event_pred.shape[0]
        data = torch.tensor(data, dtype=torch.float32)
        chis = []

        for i in range(num_reps):
            N_event_rep = torch.tensor(self.N_event_pred[i], dtype=torch.float32)

            diff = N_event_rep - data
            diffcov = torch.matmul(self.cov_matrix, diff)
            loss = torch.dot(diff.view(-1), diffcov.view(-1))
            chis.append(loss)
        mean_N_event_fits = np.mean(self.N_event_pred, axis=0)

        # data = torch.tensor(data, dtype=torch.float32)
        mean_N_event_fits = torch.tensor(mean_N_event_fits, dtype=torch.float32)

        diff = mean_N_event_fits - data
        diffcov = torch.matmul(self.cov_matrix, diff)
        mean = torch.dot(diff.view(-1), diffcov.view(-1))

        phi_chi = np.mean(chis) - mean
        return phi_chi

    def compute_accuracy(
        self,
        x_alphas: np.ndarray,
        neutrino_pdf: np.ndarray,
        pdf: str,
        n: float,
        pdf_set: str,
        pid: int,
    ) -> float:
        """
        Compute the accuracy metric as the fraction of bins where the predicted neutrino PDF
        agrees with the reference PDF within n standard deviations.

        Parameters
        ----------
        x_alphas : np.ndarray
            Input x-values (not used directly in this function).
        neutrino_pdf : np.ndarray
            Array of predicted neutrino PDFs (shape: replicas x bins).
        pdf : str
            Path or identifier for the reference PDF file.
        n : float
            Number of standard deviations for the acceptance criterion.
        pdf_set : str
            Identifier of the PDF set.
        pid : int
            Particle ID for PDF retrieval.

        Returns
        -------
        float
            Fraction of bins where predicted PDF agrees within n std deviations.
        """
        mean_neutrino_pdf = np.mean(neutrino_pdf, axis=0)
        std_pdfs = np.std(neutrino_pdf, axis=0)
        distances = []

        arr = np.logspace(-5, 0, 1000)

        log_vals = np.logspace(np.log10(0.02), np.log10(0.8), 100)
        lin_vals = np.linspace(0.1, 0.8, 10)
        log_indices = np.searchsorted(arr, log_vals)
        lin_indices = np.searchsorted(arr, lin_vals)
        indices = np.concatenate((log_indices, lin_indices))

        faser_pdf, x_faser = read_pdf(pdf, arr, pid, pdf_set)
        faser_pdf = faser_pdf.flatten() * 1.16186e-09
        faser_pdf = faser_pdf[indices]
        mean_neutrino_pdf = mean_neutrino_pdf[indices]

        std_pdfs = std_pdfs[indices]

        for i in range(len(indices)):
            if abs(mean_neutrino_pdf[i] - faser_pdf[i]) < n * std_pdfs[i]:
                distances.append(1)

        distance = len(distances) / len(indices)

        return distance

    def compute_bias_to_variance(
        self,
        level0: Union[np.ndarray, torch.Tensor],
        level2: np.ndarray,
        N_event_pred: np.ndarray,
        REPLICAS: int,
    ) -> torch.Tensor:
        """
        Compute the ratio of bias to variance in the PDF fits.

        Parameters
        ----------
        level0 : Union[np.ndarray, torch.Tensor]
            Baseline theory prediction.
        level2 : np.ndarray
            Array of predictions at level 2, shape (REPLICAS, num_bins).
        N_event_pred : np.ndarray
            Predicted events for all replicas.
        REPLICAS : int
            Number of replicas.

        Returns
        -------
        torch.Tensor
            Ratio of bias to variance.
        """
        mean_N_events = np.mean(N_event_pred, axis=0)
        mean_N_events = torch.tensor(mean_N_events, dtype=torch.float32).view(-1)

        level0 = torch.tensor(level0, dtype=torch.float32).view(-1)

        diff = mean_N_events - level0
        diffcov = torch.matmul(self.cov_matrix, diff)
        bias = torch.dot(diff.view(-1), diffcov.view(-1))

        chi_square_level2 = 0

        for j in range(REPLICAS):
            diff = mean_N_events - level2[j]
            diff = diff.float()
            diffcov = torch.matmul(self.cov_matrix, diff)
            chi_square = torch.dot(diff.view(-1), diffcov.view(-1))

            chi_square_level2 += chi_square

        variance = chi_square_level2 / REPLICAS

        ratio = bias / variance
        return ratio
