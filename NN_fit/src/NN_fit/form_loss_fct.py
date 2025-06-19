import torch
import numpy as np


# def compute_integrability(x_integrability, y_integrability, lag_mult_int):
#     int_penalty = torch.sum((y_integrability) ** 2) * lag_mult_int
#     return int_penalty


# def compute_positivity(pred, lag_mult_pos):
#     alpha = 1e-7
#     neg_mask = pred < 0
#     pos_mask = pred > 0
#     penalty = lag_mult_pos * (
#         alpha * sum((torch.exp(-pred[pos_mask])) - 1) - sum(pred[neg_mask])
#     )
#     penalty = torch.tensor(penalty, dtype=torch.float32)
#     pos_penalty = torch.sum(penalty)
#     return pos_penalty


def complete_loss_fct_nu_nub(
    pred,
    data,
    cov_matrix,
    f,
    int_point_nu,
    int_point_nub,
    x_int,
    lag_mult_pos,
    lag_mult_int,
):
    pos_mult = 2
    int_mult = 0.001
    diff = pred - data
    diffcov = torch.matmul(cov_matrix, diff)
    # fnu = fnu.detach().numpy()
    # print(fnu[:, 0])
    # print(fnu[:, 1])
    fnu = f[:, 0]
    fnub = f[:, 1]
    torch.where(fnu > 0, 0, fnu)
    torch.where(fnub > 0, 0, fnub)
    # np.where(fnu[:, 1] > 0, 0, fnu[:, 1])
    # loss = (1 / pred.size(0)) * torch.dot(diff.view(-1), diffcov.view(-1))
    # print(loss)
    loss = (
        (1 / pred.size(0)) * torch.dot(diff.view(-1), diffcov.view(-1))
        + abs(torch.sum(fnu)) * lag_mult_pos
        + abs(torch.sum(fnub)) * lag_mult_pos
        + abs(torch.sum(int_point_nu * x_int)) * lag_mult_int
        + abs(torch.sum(int_point_nub * x_int)) * lag_mult_int
    )

    return loss


def complete_loss_fct_comb(
    pred,
    data,
    cov_matrix,
    f,
    int_point_nu,
    x_int,
    lag_mult_pos,
    lag_mult_int,
):
    pos_mult = 2
    int_mult = 0.001
    diff = pred - data
    diffcov = torch.matmul(cov_matrix, diff)
    # fnu = fnu.detach().numpy()
    # print(fnu[:, 0])
    # print(fnu[:, 1])
    torch.where(f > 0, 0, f)
    # np.where(fnu[:, 1] > 0, 0, fnu[:, 1])
    # loss = (1 / pred.size(0)) * torch.dot(diff.view(-1), diffcov.view(-1))
    # print(loss)
    loss = (
        (1 / pred.size(0)) * torch.dot(diff.view(-1), diffcov.view(-1))
        + abs(torch.sum(f)) * lag_mult_pos
        + abs(torch.sum(int_point_nu * x_int)) * lag_mult_int
    )

    return loss

    # loss += abs(small_x_point1 - 0.01) * 0.1
    #     # + abs(point2 - 10**3) * lag_mult

    # if positivity and integrability:
    #     int_penalty, pos_penalty = (
    #         compute_integrability(x_integrability, y_integrability, lag_mult_int),
    #         compute_positivity(pred, lag_mult_pos),
    #     )
    #     loss = loss + pos_penalty + int_penalty
    #     return loss, pos_penalty, int_penalty

    # if positivity and not integrability:
    #     pos_penalty = compute_positivity(pred, lag_mult_pos)
    #     loss = loss + pos_penalty
    #     return loss, pos_penalty

    # if integrability and not positivity:
    #     int_penalty = compute_integrability(
    #         x_integrability, y_integrability, lag_mult_int
    #     )
    #     loss = loss + int_penalty
    #     return loss, int_penalty

    # if not integrability and not positivity:
    #     return loss


def raw_loss_fct(pred, data, cov_matrix):
    # lag_mult = 0.1

    diff = pred - data

    # loss = (1 / pred.size(0)) * torch.sum(diff**2) + torch.sum(abs(last_point)) * 0.0
    diffcov = torch.matmul(cov_matrix, diff)

    # loss = (1 / pred.size(0)) * torch.dot(diff.view(-1), diffcov.view(-1))
    # print(point1)
    # print(point2)

    # print(target.shape)
    # print(point1.shape)
    # print(point2.shape)
    # print(diff[5])
    # print((model * point2).shape)
    # print((torch.sum((model * point2) ** 2) * 10).shape)
    # print(model.shape)
    # print(point1.shape)
    # print((model * point2) ** 2)
    # print(torch.sum((model * point2) ** 2) * 10)
    # print(pred.size(0))
    # print(pred)
    # print(data)
    # print(cov_matrix)

    loss = (1 / pred.size(0)) * torch.dot(diff.view(-1), diffcov.view(-1))
    # if loss < 5:
    #     print(abs(model - 1) * 0.1)
    #     loss = loss + abs(model - 1) * 0.1
    # loss = (
    #     torch.dot(diff.view(-1), diffcov.view(-1))
    #     # Exponential decay function
    #     # + 0.1 * torch.mean((point1 - target) ** 2)
    #     # + 0.1 * torch.mean((point2 - target) ** 2)
    #     # + diff[5] ** 2 * 0.1
    #     # + diff[0] ** 2 * 0.1
    #     # + diff[1] ** 2 * 0.1
    #     # + diff[2] ** 2 * 0.1
    #     + abs(model - 1) * 0.1
    # + abs(point1 - 0.01) * 0.0
    #     # + abs(point2) * 0.01

    #     # + diff[4] ** 2 * 0.1
    # )
    # loss = (
    #     torch.dot(diff.view(-1), diffcov.view(-1)) + torch.abs(point1 - 0.1) * 0.0
    #     # + diff[0] ** 2
    #     # + abs(point2 - 0.01) * 0.1
    #     # + abs(point2 - 10**3) * lag_mult
    # )

    # + torch.sum(
    #     abs(last_point)
    # ) * 0.0
    # # print(sum(integravility_points.shape))
    # loss = (
    #     (1 / pred.size(0)) * torch.dot(diff.view(-1), loss.view(-1))
    #     # + sum(abs(last_point)) * 0.0
    #     # + sum(abs(pred[5:30] - data[5:30])) * 0.0
    # )

    # + 0.01 * sum(
    #     abs(integravility_points) - 1e-2
    # )

    # 0 * sum(
    #     torch.norm(param, p=2) ** 2 for param in network_params
    # )
    return loss
