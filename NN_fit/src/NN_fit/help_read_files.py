# Author: Jukka John
# This file helps to try except reading files
import numpy as np


def safe_loadtxt(path, **kwargs):
    try:
        return np.loadtxt(path, **kwargs)
    except OSError as e:
        print(f"Failed to load {path}: {e}")
        return None
