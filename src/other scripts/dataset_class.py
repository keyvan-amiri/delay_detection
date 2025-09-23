# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 12:33:48 2025
@author: Keyvan Amiri Elyasi
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import convolve1d

from src.utils.LDS import get_lds_kernel_window


class DALSTM_dataset(Dataset):
    def __init__(self, X, y, args):
        """
        X: torch.Tensor [N, ...]
        y: torch.Tensor [N]
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.weights = self._prepare_weights(
            reweight=args.reweight,
            lds=args.LDS,
            lds_kernel=args.lds_kernel,
            lds_ks=args.lds_ks,
            lds_sigma=args.lds_sigma
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.weights is not None:
            w = torch.tensor(self.weights[idx], dtype=torch.float32)
        else:
            w = torch.tensor(1.0, dtype=torch.float32)            
        return x, y, w

    def _prepare_weights(self, reweight, lds=False,
                         lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert (not lds) or (reweight != 'none'), \
            "Use 'inverse' or 'sqrt_inv' with LDS"

        labels = self.y.cpu().numpy().tolist()
        max_target = int(max(labels)) + 1  # infer upper bound
        value_dict = {x: 0 for x in range(max_target)}
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1

        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}

        num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
        if not len(num_per_label) or reweight == 'none':
            return None

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]),
                weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[min(max_target - 1, int(label))]
                             for label in labels]

        weights = np.array([1.0 / x for x in num_per_label], dtype=np.float32)
        scaling = len(weights) / weights.sum()
        return weights * scaling