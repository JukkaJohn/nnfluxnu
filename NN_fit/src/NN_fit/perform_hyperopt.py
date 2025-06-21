# Author: Jukka John
# This files performs a hyperparam optimization

import yaml
import sys
import os
import numpy as np
import torch
from MC_data_reps import generate_MC_replicas
from help_read_files import safe_loadtxt
from bayes_opt import BayesianOptimization


if len(sys.argv) < 2:
    print("Usage: python script.py <input_file>")
    sys.exit(1)

config_path = sys.argv[1]


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

config = load_config(config_path)

# hidden_layers = config["model"]["hidden_layers"]
# activation_function = config["model"]["activation_function"]
preproc = config["model"]["preproc"]
extended_loss = False
num_output_layers = config["model"]["num_output_layers"]
num_input_layers = config["model"]["num_input_layers"]

fit_level = config["closure_test"]["fit_level"]

range_alpha = config["training"]["range_alpha"]
range_beta = config["training"]["range_beta"]
range_gamma = config["training"]["range_gamma"]
optimizer = config["training"]["optimizer"]

lag_mult_pos = config["training"]["lag_mult_pos"]
lag_mult_int = config["training"]["lag_mult_int"]
x_int = config["training"]["x_int"]

observable = config["dataset"]["observable"]
filename_data = config["dataset"]["filename_data"]
grid_node = config["dataset"]["grid_node"]
filename_stat_error = config["dataset"]["filename_stat_error"]
filename_sys_error = config["dataset"]["filename_sys_error"]
filename_cov_matrix = config["dataset"]["filename_cov_matrix"]
filename_binning = config["dataset"]["filename_binning"]

particle_id_nu = config["postfit"]["particle_id_nu"]
particle_id_nub = config["postfit"]["particle_id_nub"]
act_functions = config["hyperopt_params"]["act_functions"]
lower_max_num_epochs = config["hyperopt_params"]["lower_max_num_epochs"]
upper_max_num_epochs = config["hyperopt_params"]["upper_max_num_epochs"]
min_num_layers = config["hyperopt_params"]["min_num_layers"]
max_num_layers = config["hyperopt_params"]["max_num_layers"]
min_num_nodes = config["hyperopt_params"]["min_num_nodes"]
max_num_nodes = config["hyperopt_params"]["max_num_nodes"]
min_wd = config["hyperopt_params"]["min_wd"]
max_wd = config["hyperopt_params"]["max_wd"]
min_lr = config["hyperopt_params"]["min_lr"]
max_lr = config["hyperopt_params"]["max_lr"]
min_patience = config["hyperopt_params"]["min_patience"]
max_patience = config["hyperopt_params"]["max_patience"]
num_folds = config["hyperopt_params"]["num_folds"]


# Electron neutrino fit
if num_output_layers == 1:
    from hyperopt_comb import perform_fit

    low_bin, high_bin, binwidths_mu = safe_loadtxt(
        f"../../../Data/binning/{filename_binning}_{particle_id_nu}", unpack=True
    )
    fk_tables = safe_loadtxt(
        f"../../../Data/fastkernel/FK_{observable}_comb_min_20_events_{particle_id_nu}",
    )
    binwidths_mu = torch.tensor(binwidths_mu, dtype=torch.float32).view(-1, 1)
    fk_tables = torch.tensor(fk_tables, dtype=torch.float32)

    data = safe_loadtxt(
        f"../../../Data/data/{filename_data}_{particle_id_nu}", delimiter=None
    )
    stat_error = safe_loadtxt(
        f"../../../Data/uncertainties/{filename_stat_error}_{particle_id_nu}",
        delimiter=None,
    )
    sys_error = safe_loadtxt(
        f"../../../Data/uncertainties/{filename_sys_error}_{particle_id_nu}",
        delimiter=None,
    )
    cov_matrix = safe_loadtxt(
        f"../../../Data/uncertainties/{filename_cov_matrix}_{particle_id_nu}",
        delimiter=None,
    )
    cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32, requires_grad=False)

# Muon neutrino fit
elif num_output_layers == 2:
    from hyperopt_nu_nub import perform_fit

    low_bin_mu, high_bin_mu, binwidths_mu = safe_loadtxt(
        f"../../../Data/binning/FK_{observable}_binsize_mu_min_20_events_{particle_id_nu}",
        unpack=True,
    )
    low_bin_mub, high_bin_mub, binwidths_mub = safe_loadtxt(
        f"../../../Data/binning/FK_{observable}_binsize_mub_min_20_events_{particle_id_nub}",
        unpack=True,
    )
    fk_tables_nu = safe_loadtxt(
        f"../../../Data/fastkernel/FK_{observable}_mu_min_20_events_{particle_id_nu}",
    )
    fk_tables_nub = safe_loadtxt(
        f"../../../Data/fastkernel/FK_{observable}_mub_min_20_events_{particle_id_nub}",
    )
    data = safe_loadtxt(
        f"../../../Data/data/{filename_data}_{particle_id_nu}", delimiter=None
    )
    stat_error = safe_loadtxt(
        f"../../../Data/uncertainties/{filename_stat_error}_{particle_id_nu}",
        delimiter=None,
    )
    sys_error = safe_loadtxt(
        f"../../../Data/uncertainties/{filename_sys_error}_{particle_id_nu}",
        delimiter=None,
    )
    cov_matrix = safe_loadtxt(
        f"../../../Data/uncertainties/{filename_cov_matrix}_{particle_id_nu}",
        delimiter=None,
    )
    cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32, requires_grad=False)

    binwidths_mu = torch.tensor(binwidths_mu, dtype=torch.float32).view(-1, 1)
    binwidths_mub = torch.tensor(binwidths_mub, dtype=torch.float32).view(-1, 1)
    fk_tables_nu = torch.tensor(fk_tables_nu, dtype=torch.float32)
    fk_tables_nub = torch.tensor(fk_tables_nub, dtype=torch.float32)
    cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32, requires_grad=False)
else:
    print("please choose a number of layers between 1 and 2")
    exit()

x_alphas = np.loadtxt("../../../Data/gridnodes/x_alpha.dat", unpack=True)
x_alphas = torch.tensor(x_alphas, dtype=torch.float32).view(-1, 1)
x_vals = np.logspace(-5, 0, 1000)


(
    mean_pdf_all_fits_mu,
    mean_pdf_all_fits_mub,
    total_std_mu,
    total_std_mub,
    total_preds_Enu,
    total_std_preds_Enu,
    total_preds_Enu_mub,
    total_std_preds_Enu_mub,
) = 0, 0, 0, 0, 0, 0, 0, 0

seed = int(np.random.rand() * 100)

level0, level1, level2 = generate_MC_replicas(1, data, sys_error, stat_error, seed)

if fit_level == 0:
    pred = level0
elif fit_level == 1:
    pred = level1
elif fit_level == 2:
    pred = level2
else:
    print("please select 0,1 or 2 for fit level")


act_function_map = act_functions


def objective(
    lr, max_counter, num_nodes, num_layers, act_fn_idx, log_wd, max_num_epoch
):
    lr = float(lr)
    max_counter = int(max_counter)
    num_nodes = int(num_nodes)
    num_layers = int(num_layers)
    hidden_layers = []
    for _ in range(num_layers):
        hidden_layers.append(num_nodes)

    act_function = act_function_map[int(act_fn_idx)]
    wd = 10**log_wd  # Since we’ll search in log10 scale
    max_num_epoch = int(max_num_epoch)

    act_functions = [act_function] * (num_layers + 1)

    loss = perform_fit(
        pred,
        range_alpha,
        range_beta,
        range_gamma,
        lr,
        wd,
        max_counter,
        x_alphas,
        fk_tables_nu,
        fk_tables_nub,
        binwidths_mu,
        binwidths_mub,
        cov_matrix,
        extended_loss,
        act_functions,
        num_input_layers,
        num_output_layers,
        hidden_layers,
        x_vals,
        preproc,
        max_num_epoch,
        lag_mult_pos,
        lag_mult_int,
        x_int,
        num_folds,
    )

    print(loss)
    return loss


pbounds = {
    "lr": (min_lr, max_lr),
    "max_counter": (min_patience, max_patience),
    "num_nodes": (min_num_nodes, max_num_nodes),
    "num_layers": (min_num_layers, max_num_layers),
    "act_fn_idx": (0, 3.999),
    "log_wd": (min_wd, max_wd),
    "max_num_epoch": (lower_max_num_epochs, upper_max_num_epochs),
}

optimizer = BayesianOptimization(
    f=objective, pbounds=pbounds, random_state=42, verbose=2
)

optimizer.maximize(
    init_points=1,
    n_iter=1,
)

print("Best parameters:")
print(optimizer.max)
params = np.array(optimizer.max)
np.savetxt(
    "hyperparams.txt",
    params,
)
