import os
import random
import numpy as np
import torch

def add_suffix(filename, sample_size):
    if (sample_size / 1000) % 1 != 0:
        size_str = f"{sample_size / 1000:.1f}k"
    else:
        size_str = f"{sample_size // 1000}k"
    return f"{filename}_{size_str}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
