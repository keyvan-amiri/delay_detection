#import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
import torch
import torch.nn as nn
import torch.nn.functional as F

#print = logging.info

def calibrate_mean_var(
    matrix, m1, v1, m2, v2,
    clip_min=0.1, clip_max=10.0,
    eps=1e-6
    ):
    # If variance is basically zero everywhere, do nothing
    if torch.sum(v1) < 1e-10:
        return matrix
    # Ensure same dtype/device behavior
    v1_safe = torch.clamp(v1, min=eps)
    v2_safe = torch.clamp(v2, min=eps)
    # Compute per-dimension scale
    factor = torch.clamp(v2_safe / v1_safe, clip_min, clip_max)
    scale = torch.sqrt(factor)
    # If some dims originally had exactly zero variance, keep them unchanged
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        # Build output without in-place assignment
        transformed = (matrix - m1) * scale + m2
        # Keep invalid (zero-var) dims as original
        valid_mask = valid.unsqueeze(0).expand_as(matrix)
        return torch.where(valid_mask, transformed, matrix)
    # Normal case
    return (matrix - m1) * scale + m2

def add_bin_edges(model, train_loader, val_loader, fds_config, device):
    # Collect continuous labels from train + val
    y_all = []
    for _, y, _ in train_loader:
        y_all.append(y.detach().cpu().float().view(-1))
    for _, y, _ in val_loader:
        y_all.append(y.detach().cpu().float().view(-1))
    y_all = torch.cat(y_all, dim=0)  # (N,)
    bucket_start = fds_config["bucket_start"]
    bucket_num = fds_config["bucket_num"]
    num_bins = bucket_num - bucket_start
    q = torch.linspace(0, 1, steps=num_bins + 1)
    bin_edges = torch.quantile(y_all, q)  # (num_bins+1,)
    # Ensure strictly increasing edges (handles duplicate quantiles)
    eps = 1e-6
    for i in range(1, bin_edges.numel()):
        if bin_edges[i] <= bin_edges[i - 1]:
            bin_edges[i] = bin_edges[i - 1] + eps
    model.set_fds_bin_edges(bin_edges.to(device))
    return model

class FDS(nn.Module):

    def __init__(self, feature_dim, bucket_num=100, bucket_start=3, start_update=0, start_smooth=1,
                 kernel='gaussian', ks=5, sigma=2, momentum=0.9):
        super(FDS, self).__init__()
        self.feature_dim = feature_dim
        self.bucket_num = bucket_num
        self.bucket_start = bucket_start
        self.kernel_window = self._get_kernel_window(kernel, ks, sigma)
        self.half_ks = (ks - 1) // 2
        self.momentum = momentum
        self.start_update = start_update
        self.start_smooth = start_smooth

        self.register_buffer('epoch', torch.zeros(1).fill_(start_update))
        self.register_buffer('running_mean', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('num_samples_tracked', torch.zeros(bucket_num - bucket_start))

    @staticmethod
    def _get_kernel_window(kernel, ks, sigma):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            base_kernel = np.array(base_kernel, dtype=np.float32)
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / sum(gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks) / sum(triang(ks))
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / sum(map(laplace, np.arange(-half_ks, half_ks + 1)))

        print(f'Using FDS: [{kernel.upper()}] ({ks}/{sigma})')
        #return torch.tensor(kernel_window, dtype=torch.float32).cuda()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.tensor(kernel_window, dtype=torch.float32, device=device)

    def _update_last_epoch_stats(self):
        self.running_mean_last_epoch = self.running_mean
        self.running_var_last_epoch = self.running_var

        self.smoothed_mean_last_epoch = F.conv1d(
            input=F.pad(self.running_mean_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)
        self.smoothed_var_last_epoch = F.conv1d(
            input=F.pad(self.running_var_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)

    def reset(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.running_mean_last_epoch.zero_()
        self.running_var_last_epoch.fill_(1)
        self.smoothed_mean_last_epoch.zero_()
        self.smoothed_var_last_epoch.fill_(1)
        self.num_samples_tracked.zero_()

    def update_last_epoch_stats(self, epoch):
        if epoch == self.epoch + 1:
            self.epoch += 1
            self._update_last_epoch_stats()
            print(f"Updated smoothed statistics on Epoch [{epoch}]!")

    def update_running_stats(self, features, labels, epoch):
        if epoch < self.epoch:
            return
        if labels.dim() == 2:
            labels = labels.squeeze(1)  # <-- FIX: make labels shape (N,)

        assert self.feature_dim == features.size(1), "Input feature dimension is not aligned!"
        assert features.size(0) == labels.size(0), "Dimensions of features and labels are not aligned!"

        for label in torch.unique(labels):
            if label > self.bucket_num - 1 or label < self.bucket_start:
                continue
            elif label == self.bucket_start:
                curr_feats = features[labels <= label]
            elif label == self.bucket_num - 1:
                curr_feats = features[labels >= label]
            else:
                curr_feats = features[labels == label]
            curr_num_sample = curr_feats.size(0)
            curr_mean = torch.mean(curr_feats, 0)
            curr_var = torch.var(curr_feats, 0, unbiased=True if curr_feats.size(0) != 1 else False)

            self.num_samples_tracked[int(label - self.bucket_start)] += curr_num_sample
            factor = self.momentum if self.momentum is not None else \
                (1 - curr_num_sample / float(self.num_samples_tracked[int(label - self.bucket_start)]))
            factor = 0 if epoch == self.start_update else factor
            self.running_mean[int(label - self.bucket_start)] = \
                (1 - factor) * curr_mean + factor * self.running_mean[int(label - self.bucket_start)]
            self.running_var[int(label - self.bucket_start)] = \
                (1 - factor) * curr_var + factor * self.running_var[int(label - self.bucket_start)]

        print(f"Updated running statistics with Epoch [{epoch}] features!")

    def smooth(self, features, labels, epoch):
        """
        Autograd-safe, out-of-place smoothing.
        Avoids: features[mask] = ...
        """
        if epoch < self.start_smooth:
            return features

        # labels expected shape: (B,) or (B,1)
        if labels.dim() == 2:
            labels = labels.squeeze(1)

        out = features.clone()
        unique_labels = torch.unique(labels)
        for lab in unique_labels:
            lab_i = int(lab.item())  # safe Python int for indexing
            if lab_i > self.bucket_num - 1 or lab_i < self.bucket_start:
                continue
            bucket_idx = lab_i - self.bucket_start
            # Skip buckets with too few samples tracked (unstable stats)
            min_count = 20
            if self.num_samples_tracked[bucket_idx] < min_count:
                continue
            # Build mask for the three boundary cases (same as your original logic)
            if lab_i == self.bucket_start:
                mask = (labels <= lab_i)
            elif lab_i == self.bucket_num - 1:
                mask = (labels >= lab_i)
            else:
                mask = (labels == lab_i)

            if not mask.any():
                continue            
            # Calibrate ALL rows, then select only masked rows via torch.where
            # (This avoids needing to slice-assign into out.)
            calibrated = calibrate_mean_var(
                out,
                self.running_mean_last_epoch[bucket_idx],
                self.running_var_last_epoch[bucket_idx],
                self.smoothed_mean_last_epoch[bucket_idx],
                self.smoothed_var_last_epoch[bucket_idx],
                )
            out = torch.where(mask.unsqueeze(1), calibrated, out)
        return out