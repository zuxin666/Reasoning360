import os
import random
import numpy as np
import torch

def get_output_dir_name(output_dir, sample_size=None):
    output_dir_base = os.path.dirname(output_dir)
    output_dir_name = os.path.basename(output_dir)

    if (sample_size / 1000) % 1 != 0:
        size_str = f"{sample_size / 1000:.1f}k"
    else:
        size_str = f"{sample_size // 1000}k"

    if sample_size is not None:
        if "_" in output_dir_name:
            name_parts = output_dir_name.split("_", 1)
            output_dir_name = f"{name_parts[0]}{size_str}_{name_parts[1]}"
        else:
            output_dir_name = f"{output_dir_name}{size_str}"

    return os.path.join(output_dir_base, output_dir_name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
