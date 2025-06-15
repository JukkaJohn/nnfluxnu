import numpy as np


def compute_cov_matrix(pred, pred_max, pred_min):
    pred = np.where(pred == 0, 1, pred)
    delta_plus = pred_max - pred
    delta_min = pred_min - pred
    semi_diff = (delta_plus + delta_min) / 2
    average = (delta_plus - delta_min) / 2
    # se_delta = semi_diff
    sig_sys = np.sqrt(average * average + 2 * semi_diff * semi_diff)

    sig_tot = sig_sys**2 + pred
    cov = np.diag(sig_tot)
    cov = np.linalg.inv(cov)
    return cov


def compute_hessian(x, p_min, f_ref, fk_table, norm):
    print(x.shape)
    # print(pred.shape)
    pred = f_ref
    hessian = np.empty([3, 3])

    hessian[0, 0] = np.sum(
        np.matmul(
            fk_table,
            2 * p_min[0] * x ** (p_min[1] - 1) * (1 - x) ** (p_min[2] - 1) * f_ref,
        )
        * norm
    )
    hessian[1, 1] = np.sum(
        np.matmul(
            fk_table,
            p_min[0]
            * x ** (p_min[1] - 1)
            * (np.log(x) ** 2)
            * (1 - x) ** (p_min[2] - 1)
            * pred,
        )
        * norm
    )
    hessian[2, 2] = np.sum(
        np.matmul(
            fk_table,
            p_min[0]
            * x ** (p_min[1] - 1)
            * (np.log(1 - x) ** 2)
            * (1 - x) ** (p_min[2] - 1)
            * pred,
        )
        * norm
    )
    hessian[0, 1] = np.sum(
        np.matmul(
            fk_table, x ** (p_min[1] - 1) * np.log(x) * (1 - x) ** (p_min[2] - 1) * pred
        )
        * norm
    )
    hessian[0, 2] = np.sum(
        np.matmul(
            fk_table,
            x ** (p_min[1] - 1) * np.log(1 - x) * (1 - x) ** (p_min[2] - 1) * pred,
        )
        * norm
    )
    hessian[1, 2] = np.sum(
        np.matmul(
            fk_table,
            p_min[0]
            * x ** (p_min[1] - 1)
            * np.log(x)
            * np.log(1 - x)
            * (1 - x) ** (p_min[2] - 1)
            * pred,
        )
        * norm
    )
    hessian[1, 0] = hessian[0, 1]
    hessian[2, 0] = hessian[0, 2]
    hessian[2, 1] = hessian[1, 2]

    return hessian


def determine_errors(interval, chi_square, p_min, vec):
    p_hess_min = []
    p_hess_max = []
    for i in range(3):
        t = 0
        chi_68 = 0

        while chi_68 < interval + chi_square(p_min):
            chi_68 = chi_square(p_min + t * vec[i, :])
            # print(f'chi_68 = {chi_68}')
            t += 0.001
        # print('interval = ')
        # print(i,interval)
        p_hess_max.append(p_min + t * vec[i, :])
        t = 0

        chi_68 = 0
        print(chi_square(p_min))
        while chi_68 < (interval + chi_square(p_min)):
            # print(t)
            chi_68 = chi_square(p_min + t * vec[i, :])
            # print(f'chi_68 = {chi_68}')
            t -= 0.001

        p_hess_min.append(p_min + t * vec[i, :])
    return p_hess_max, p_hess_min


def plot_error_bands(interval, chi_square, p_min, vec, raw, rawx, result):
    p_hess_max, p_hess_min = determine_errors(interval, chi_square, p_min, vec)
    print("cont fit = ")
    print(func(p_min, raw, rawx))
    # print(func(p_min, pred, x))
    # plt.plot(x_fit,func(p_min),drawstyle = 'steps-post')
    f_err = 0
    err_param = 0
    for i in range(3):
        # f_err += (func(p_hess_max[i], pred, x) - func(p_hess_min[i], pred, x)) ** 2
        f_err += (func(p_hess_max[i], raw, rawx) - func(p_hess_min[i], raw, rawx)) ** 2
        err_param += (p_hess_max[i] - p_hess_min[i]) ** 2
    print("the errors are:")
    print(np.sqrt(err_param) * 0.5)
    hess_inv = result.hess_inv
    cov_matrix = hess_inv
    print("naive errors are:")
    print(np.sqrt(np.diag(cov_matrix)))
    f_err = np.sqrt(f_err) * 0.5

    return f_err


def func(parr, pred, x):
    return parr[0] ** 2 * x ** (parr[1] - 1) * ((1 - x) ** (parr[2] - 1)) * pred
