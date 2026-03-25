# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 12:33:48 2025
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import convolve1d

from src.utils.LDS import get_lds_kernel_window

class DALSTM_dataset(Dataset):
    def __init__(self, X, y, args=None, weights=None,
                 bin_edges=None, trainval_bin_weights=None):
        """
        weights: precomputed weights (train/val)
        For test:
          bin_edges: array of bin edges used for train+val
          trainval_bin_weights: per-bin weight array (len = n_bins)
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        if weights is not None:
            self.weights = weights
        elif bin_edges is not None and trainval_bin_weights is not None:
            # Test set: map by bin index (NOT nearest neighbor label)
            y_np = self.y.detach().cpu().numpy()
            bin_idx = np.digitize(y_np, bin_edges[1:-1], right=False)  # -> [0..n_bins-1]
            w = trainval_bin_weights[bin_idx].astype(np.float32)
            w *= len(w) / w.sum()
            self.weights = w
        else:
            # FIRST handle the case where args was not provided
            if args is None or getattr(args, "reweight", None) is None:
                # default / BMSE-like branch: no weighting
                n = int(self.y.shape[0])
                self.weights = np.ones(n, dtype=np.float32)
                self.bin_edges = None
                self.bin_weights = None
            else:
                # Train/val: compute weights and keep binning artifacts
                w, bin_edges, bin_weights = self._prepare_weights(
                    reweight=args.reweight,
                    lds=args.LDS,
                    lds_kernel=args.lds_kernel,
                    lds_ks=args.lds_ks,
                    lds_sigma=args.lds_sigma,
                    n_bins=getattr(args, "n_bins", 20),
                    binning=getattr(args, "binning", "quantile")
                    )
                self.weights = w
                self.bin_edges = bin_edges
                self.bin_weights = bin_weights

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        w = torch.tensor(self.weights[idx], dtype=torch.float32) if self.weights is not None \
            else torch.tensor(1.0, dtype=torch.float32)
        return x, y, w

    def _prepare_weights(self, reweight, lds=False,
                         lds_kernel='gaussian', lds_ks=5, lds_sigma=2,
                         n_bins=20, binning="quantile"):
        assert reweight in {'none', 'inverse', 'sqrt_inv'}
        assert (not lds) or (reweight != 'none'), "Use 'inverse' or 'sqrt_inv' with LDS"
        y = self.y.detach().cpu().numpy().astype(np.float32)
        if reweight == "none":
            w = np.ones_like(y, dtype=np.float32)
            return w, None, None
        # define bins
        y_min, y_max = float(y.min()), float(y.max())
        if binning == "quantile":
            # quantile edges; unique to avoid duplicates if many ties
            qs = np.linspace(0, 1, n_bins + 1)
            edges = np.quantile(y, qs)
            edges = np.unique(edges)
            # if too many duplicates, fall back to uniform
            if len(edges) < 3:
                edges = np.linspace(y_min, y_max + 1e-6, n_bins + 1)
        elif binning == "uniform":
            edges = np.linspace(y_min, y_max + 1e-6, n_bins + 1)
        else:
            raise ValueError("binning must be 'quantile' or 'uniform'")
        # digitize into [0..n_bins-1]
        bin_idx = np.digitize(y, edges[1:-1], right=False)
        # counts per bin
        n_bins_eff = len(edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins_eff - 1)
        counts = np.bincount(bin_idx, minlength=n_bins_eff).astype(np.float32)
        # avoid zero counts (can happen with weird edges); keep them as 1 for stability
        counts = np.clip(counts, 1.0, None)
        # LDS on counts (correct order)
        if lds:
            kernel = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            counts_smooth = convolve1d(counts, weights=kernel, mode='constant')
            counts_smooth = np.clip(counts_smooth, 1e-6, None)
        else:
            counts_smooth = counts
        # reweighting from (smoothed) counts
        if reweight == "inverse":
            denom = counts_smooth
        else:  # sqrt_inv
            denom = np.sqrt(counts_smooth)
        bin_weights = 1.0 / denom  # per-bin weight
        # clip extreme weights rather than clipping counts
        # (prevents a few singleton-ish bins from dominating)
        w_max = np.percentile(bin_weights, 99.5)
        bin_weights = np.clip(bin_weights, 0.0, w_max)
        # map to samples
        w = bin_weights[bin_idx].astype(np.float32)
        # normalize to mean 1
        w *= len(w) / w.sum()
        return w, edges, bin_weights