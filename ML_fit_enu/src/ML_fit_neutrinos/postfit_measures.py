import torch
import numpy as np
from read_faserv_pdf import read_pdf


class Measures:
    def __init__(self, cov_matrix, pdf, N_event_pred):
        self.cov_matrix = cov_matrix
        self.pdf = pdf
        self.N_event_pred = N_event_pred

    def compute_delta_chi(
        self,
        level0,
        N_event_pred,
        data_level1,
        x_vals,
    ):
        # we need a list of level 1 instances
        # we need a list of mean N events for several level 1 instances
        # delta_chis = []

        # for rep in data_level1:
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
        # delta_chis.append(delta_chi)

        return delta_chi

    def compute_phi(self, data, chi_squares):
        num_reps = self.N_event_pred.shape[0]
        data = torch.tensor(data, dtype=torch.float32)
        chis = []
        print("num reps")
        print(num_reps)
        for i in range(num_reps):
            N_event_rep = torch.tensor(self.N_event_pred[i], dtype=torch.float32)
            print(N_event_rep.shape)
            print(data.shape)
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

        phi_chi = (
            np.mean(chis) - mean
        )  # has to be compared to level 0 data, not level 2 data
        return phi_chi

    def compute_accuracy(self, x_alphas, neutrino_pdf, pdf, n):
        # we need several mean_neutrino pdfs
        # we need several std_neutrino pdfs
        mean_neutrino_pdf = np.mean(neutrino_pdf, axis=0)
        std_pdfs = np.std(neutrino_pdf, axis=0)
        distances = []

        arr = np.logspace(-5, 0, 1000)

        log_vals = np.logspace(np.log10(0.02), np.log10(0.8), 100)
        lin_vals = np.linspace(0.1, 0.8, 10)
        log_indices = np.searchsorted(arr, log_vals)
        lin_indices = np.searchsorted(arr, lin_vals)
        indices = np.concatenate((log_indices, lin_indices))

        faser_pdf, x_faser = read_pdf(pdf, arr, 14)
        faser_pdf = faser_pdf.flatten() * 1.16186e-09
        faser_pdf = faser_pdf[indices]
        mean_neutrino_pdf = mean_neutrino_pdf[indices]

        std_pdfs = std_pdfs[indices]
        print(mean_neutrino_pdf.shape)
        print(faser_pdf.shape)
        print(std_pdfs.shape)
        for i in range(len(indices)):
            print(mean_neutrino_pdf[i], faser_pdf[i], std_pdfs[i])
            if abs(mean_neutrino_pdf[i] - faser_pdf[i]) < n * std_pdfs[i]:
                distances.append(1)
                print("yes")

        distance = len(distances) / len(indices)

        return distance

    def compute_bias_to_variance(
        self,
        level0,
        level2,
        N_event_pred,
        REPLICAS,
    ):
        # we need several level2 lists for every level 1 instance
        # we need several mean_N_events for every instance
        # so convenient way to run all these fits and also convenient way to store and access results: perhaps different folder with specific files and postfit measures
        # 100 job submits for example ( or just try w/ 10 which I think should be enough)

        num_diff_level1_shifts = (
            10  # (10 level 1 closure tetst with 100 reps????) # diff level 1 shifts
        )

        for _ in range(num_diff_level1_shifts):
            pass  # should be the same: different level 1 fits yield same result probabaly
        # bias: 1 level1 closure fit is enough for the bias I think
        # variance:

        mean_N_events = np.mean(N_event_pred, axis=0)
        mean_N_events = torch.tensor(mean_N_events, dtype=torch.float32).view(-1)

        level0 = torch.tensor(level0, dtype=torch.float32).view(-1)

        diff = mean_N_events - level0
        diffcov = torch.matmul(self.cov_matrix, diff)
        bias = torch.dot(diff.view(-1), diffcov.view(-1))

        # So keep track of all level1s, level2s somewhere

        chi_square_level2 = 0

        # REPLICAS = 2
        for j in range(REPLICAS):
            diff = mean_N_events - level2[j]
            diff = diff.float()
            diffcov = torch.matmul(self.cov_matrix, diff)
            chi_square = torch.dot(diff.view(-1), diffcov.view(-1))

            chi_square_level2 += chi_square

        variance = chi_square_level2 / REPLICAS

        ratio = bias / variance
        return ratio
