# HYPERPARAMETERS
import torch


def hyperparams():
    # REPLICAS = 20
    postfit_criteria = True
    postfit_measures = True
    preproc = True
    extended_loss = False
    lr = 0.03
    fit_level = 2
    wd = 1e-2
    range_alpha, range_beta, range_gamma = 5, 20, 100
    max_counter = 400
    max_num_epochs = 6000
    # l1, l2, l3 = 2, 2, 2
    num_layers = 3
    # layers = [2,2,2]
    num_nodes = 4
    act_function = torch.nn.Softplus()
    act_functions = []
    for _ in range(num_layers + 1):
        act_functions.append(act_function)

    validation = 0.0
    return (
        # REPLICAS,
        preproc,
        lr,
        fit_level,
        max_counter,
        num_nodes,
        num_layers,
        act_functions,
        postfit_criteria,
        postfit_measures,
        wd,
        range_alpha,
        range_beta,
        range_gamma,
        extended_loss,
        validation,
        max_num_epochs,
    )
