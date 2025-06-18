import pandas as pd
import torch
import torch.nn.functional


def get_fk_table(start_index, num_obs):
    """This function reads the fk table for the neutrino flux and pads them for computational efficiency later on

    Returns:
        tuple: x_alphas(grid points) and the fk table in tensor to fit torch
    """

    filenames = ["FK_Enu", "FK_El.dat", "FK_Eh.dat", "FK_theta.dat"]

    fk_tables, x_alphas = [], []

    for i in range(start_index, num_obs):
        # file_path = f"data/{filenames[i]}"
        # df = pd.read_csv(file_path, sep="\s+", header=None)
        # fk_table = df.to_numpy()

        file_path_n = f"data/{filenames[i]}_n.dat"
        df_n = pd.read_csv(file_path_n, sep="\s+", header=None)
        fk_table_n = df_n.to_numpy()

        file_path_p = f"data/{filenames[i]}_p.dat"
        df_p = pd.read_csv(file_path_p, sep="\s+", header=None)
        fk_table_p = df_p.to_numpy()

        fk_table = 74 / 183 * fk_table_n + (183 - 74) / 183 * fk_table_p

        x_alpha = fk_table[0, :]
        x_alpha = x_alpha.reshape(len(x_alpha), 1)

        # strip first row to get fk table
        fk_table = fk_table[1:, :]

        x_alpha = torch.tensor(x_alpha, dtype=torch.float32).view(-1, 1)
        fk_table = torch.tensor(fk_table, dtype=torch.float32)

        fk_tables.append(fk_table)
        x_alphas.append(x_alpha)

    shapes = torch.tensor([tensor.shape for tensor in fk_tables])
    max_rows, max_cols = torch.max(shapes, dim=0).values
    padded_fk_tables = []
    mask = torch.zeros(len(fk_tables), max_rows, dtype=torch.bool)

    for i, tensor in enumerate(fk_tables):
        rows = tensor.size(0)
        cols = tensor.size(1)
        rows_to_pad = max_rows - rows
        cols_to_pad = max_cols - cols

        padded_fk_table = torch.nn.functional.pad(
            tensor, (0, cols_to_pad, 0, rows_to_pad), mode="constant", value=0
        )
        padded_fk_tables.append(padded_fk_table)

        mask[i, :rows] = True
    mask = mask.unsqueeze(-1)
    stack_fk_tables = torch.stack(padded_fk_tables, dim=0)

    return x_alphas, stack_fk_tables, mask
