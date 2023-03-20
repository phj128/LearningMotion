import numpy as np
import os.path as osp


def load_norm_data(data_dir, split="train"):
    # Always use norm of training data
    input_norm_data = np.float32(np.loadtxt(osp.join(data_dir, split, "InputNorm.txt")))
    input_mean = input_norm_data[0]
    input_std = input_norm_data[1]
    for i in range(input_std.size):
        if input_std[i] == 0:
            input_std[i] = 1

    if not osp.exists(osp.join(data_dir, split, "OutputNorm.txt")):
        return input_mean, input_std, None, None

    output_norm_data = np.float32(
        np.loadtxt(osp.join(data_dir, split, "OutputNorm.txt"))
    )
    output_mean = output_norm_data[0]
    output_std = output_norm_data[1]
    for i in range(output_std.size):
        if output_std[i] == 0:
            output_std[i] = 1
    return input_mean, input_std, output_mean, output_std


def load_norm_data_prefix(data_dir, split="train", prefix="Input"):
    # Always use norm of training data
    if not osp.exists(osp.join(data_dir, split, f"{prefix}Norm.txt")):
        print(osp.join(data_dir, split, f"{prefix}Norm.txt") + " does not exists!")
        return 0, 1

    input_norm_data = np.float32(
        np.loadtxt(osp.join(data_dir, split, f"{prefix}Norm.txt"))
    )
    input_mean = input_norm_data[0]
    input_std = input_norm_data[1]
    for i in range(input_std.size):
        if input_std[i] == 0:
            input_std[i] = 1

    return input_mean, input_std
