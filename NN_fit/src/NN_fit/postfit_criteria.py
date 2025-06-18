import numpy as np


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
        # arc_lenghts,
        chi_squares,
        # pos_penaltys,
        # int_penaltys,
        N_event_pred,
        neutrino_pdfs,
        pred,
    ):
        # sig_arc_length = np.std(arc_lenghts)
        # mean_arc_length = np.mean(arc_lenghts)

        sig_chi_squares = np.std(chi_squares)
        mean_chi_squares = np.mean(chi_squares)
        # print(pos_penaltys)
        # print(int_penaltys)

        indices_to_remove = []
        for i in range(len(chi_squares)):
            print(i)
            diff_chi_squares = abs(chi_squares[i] - mean_chi_squares)
            # diff_arc_length = abs(arc_lenghts[i] - mean_arc_length)
            # print(diff_arc_length, sig_arc_length)
            # print(diff_chi_squares, sig_chi_squares)
            if (
                # pos_penaltys[i] > 10**-6
                # or int_penaltys[i] > 0.5
                # or diff_arc_length > 4 * sig_arc_length
                diff_chi_squares > 4 * sig_chi_squares
                # or chi_squares[i] > 1
            ):
                indices_to_remove.append(i)
                print(f"replica {i} is going to be removed")

        if len(indices_to_remove) > 0:
            indices_to_remove = np.array(
                indices_to_remove
            )  # make sure it's a NumPy array
            N_event_pred = np.delete(N_event_pred, indices_to_remove)
            neutrino_pdfs = np.delete(neutrino_pdfs, indices_to_remove)
            pred = np.delete(pred, indices_to_remove)

        # for i in sorted(indices_to_remove, reverse=True):
        #     N_event_pred = np.delete(N_event_pred, i)
        #     neutrino_pdfs = np.delete(neutrino_pdfs, i)
        #     pred = np.delete(pred, i)
        #     # N_event_pred.pop(i)
        #     # neutrino_pdfs.pop(i)
        #     # pred.pop(i)
        return neutrino_pdfs, N_event_pred, pred


# save for later
# def compute_effective_coeff(model):
#     y_pred_min_alpha = model(torch.tensor([[10**-6]], dtype=torch.float32) ).detach().numpy().flatten()
#     y_pred_max_alpha = model(torch.tensor([[10**-3]], dtype=torch.float32)).detach().numpy().flatten()
#     y_pred_min_beta = model(torch.tensor([[0.65]], dtype=torch.float32)).detach().numpy().flatten()
#     y_pred_max_beta = model(torch.tensor([[0.5]], dtype=torch.float32)).detach().numpy().flatten()

#     max_alpha = torch.max(np.log(y_pred_max_alpha)/np.log(1/x_alphas[0]))
#     max_beta = torch.max(np.log(y_pred_max_beta)/np.log(1-x_alphas[0]))

#     min_alpha = torch.min(np.log(y_pred_min_alpha)/np.log(1/x_alphas[0]))
#     min_beta = torch.min(np.log(y_pred_min_beta)/np.log(1-x_alphas[0]))

#     return max_alpha,min_alpha,max_beta,min_beta
