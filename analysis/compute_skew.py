# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:06:41 2026
"""

import os
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")


def tensor_to_1d_numpy(x):
    """Convert a torch tensor (or array-like) to a flattened NumPy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


def compute_skewness(x):
    """
    Compute Fisher-Pearson skewness:
        skew = E[(X - mu)^3] / sigma^3
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)

    if x.size == 0:
        return np.nan

    mu = np.mean(x)
    sigma = np.std(x)

    if sigma == 0:
        return 0.0

    return np.mean(((x - mu) / sigma) ** 3)


def main():
    # ---- settings ----
    model_name = "DALSTM"
    datasets = [
        "P2P",
        "Sepsis",
        "BPIC_2017_W",
        "HelpDesk",
        "BPIC15_1",
        "BPIC15_2",
        "BPIC15_3",
        "BPIC15_4",
        "BPIC15_5",
        "BPIC20ID",
        "BPIC20DD",
        "BPIC20PTC",
        "BPIC20TPD",
        "BPIC20RFP",
    ]

    # ---- paths ----
    root_path = os.getcwd()
    temp_path = os.path.join(root_path, "temp", model_name)

    print(f"{'Dataset':<12} {'Skewness':>10}")
    print("-" * 24)

    for dataset in datasets:
        temp_dir = os.path.join(temp_path, dataset)

        y_train_path = os.path.join(temp_dir, f"{model_name}_y_train_{dataset}.pt")
        y_val_path   = os.path.join(temp_dir, f"{model_name}_y_val_{dataset}.pt")
        y_test_path  = os.path.join(temp_dir, f"{model_name}_y_test_{dataset}.pt")

        missing_files = [
            p for p in [y_train_path, y_val_path, y_test_path] if not os.path.exists(p)
        ]
        if missing_files:
            print(f"{dataset:<12} {'MISSING':>10}")
            for mf in missing_files:
                print(f"  Missing file: {mf}")
            continue

        # Load tensors
        y_train = torch.load(y_train_path)
        y_val = torch.load(y_val_path)
        y_test = torch.load(y_test_path)

        # Convert to 1D numpy arrays
        y_train = tensor_to_1d_numpy(y_train)
        y_val = tensor_to_1d_numpy(y_val)
        y_test = tensor_to_1d_numpy(y_test)

        # Concatenate full target distribution
        y_all = np.concatenate([y_train, y_val, y_test], axis=0)

        # Compute skewness
        skewness = compute_skewness(y_all)

        print(f"{dataset:<12} {skewness:>10.4f}")


if __name__ == "__main__":
    main()