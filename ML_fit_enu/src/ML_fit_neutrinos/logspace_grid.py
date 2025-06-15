import numpy as np


def generate_grid(lowx, n):
    incexp = lowx / n
    x_vals = []
    for i in range(n):
        ri = i
        x_vals.append(np.exp(lowx - ri * incexp))
    return x_vals
