# -*- coding: utf-8 -*-
"""
SMOGN (Synthetic Minority Over-sampling for Regression with Gaussian Noise)
adapted for temporal / prefix-based process mining data.

Key adaptation: samples are grouped by prefix length before oversampling so
that interpolation only happens between sequences of the same temporal extent.
This prevents mixing real event features with zero-padding artefacts.

References
----------
Branco, P., Torgo, L., & Ribeiro, R. P. (2017).
    SMOGN: a pre-processing approach for imbalanced regression.
    Proceedings of Machine Learning Research, 74, 36-50.
"""
import os
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from src.utils.relevance_scores import phi_control, phi
from src.utils.GMM import fit_label_gmm, train_lstm_and_predict_test_components


# ---------------------------------------------------------------------------
# Helper: derive actual prefix lengths from left-padded 3D tensor
# ---------------------------------------------------------------------------

def get_prefix_lengths(X: np.ndarray) -> np.ndarray:
    """Return an array of actual (non-padded) prefix lengths for each sample.

    Parameters
    ----------
    X : np.ndarray, shape (N, T, F)
        Left-padded feature tensor. Padding rows are all-zero.

    Returns
    -------
    lengths : np.ndarray, shape (N,), dtype int
        Actual prefix length for every sample (>= 1).
    """
    # A row is padding iff every feature is exactly zero
    # row_nonzero: (N, T) boolean -- True where the timestep is real
    row_nonzero = np.any(X != 0, axis=2)  # (N, T)
    # prefix length = total number of non-zero rows (they sit at the end)
    lengths = row_nonzero.sum(axis=1).astype(int)
    # safety: at least 1 to avoid degenerate samples
    lengths = np.clip(lengths, 1, X.shape[1])
    return lengths


# ---------------------------------------------------------------------------
# k-NN search within a group
# ---------------------------------------------------------------------------

def _knn_within_group(X_group: np.ndarray, prefix_len: int, k: int):
    """Find k nearest neighbours for every sample in a same-length group.

    Only the *non-padded* trailing ``prefix_len`` timesteps are used.
    Features are flattened before computing Euclidean distance.

    Returns
    -------
    distances : np.ndarray (n, k)
    indices   : np.ndarray (n, k)
    """
    T = X_group.shape[1]
    start = T - prefix_len
    # flatten the real part: (n, prefix_len * F)
    flat = X_group[:, start:, :].reshape(X_group.shape[0], -1)
    n = flat.shape[0]
    eff_k = min(k + 1, n)  # +1 because query point is its own neighbour
    nn = NearestNeighbors(n_neighbors=eff_k, metric="euclidean", algorithm="auto")
    nn.fit(flat)
    distances, indices = nn.kneighbors(flat)
    # drop self-match (first column)
    return distances[:, 1:], indices[:, 1:]


# ---------------------------------------------------------------------------
# Synthetic sample generation
# ---------------------------------------------------------------------------

def _smoter_interpolate(x1: np.ndarray, y1: float,
                        x2: np.ndarray, y2: float,
                        prefix_len: int, T: int, rng: np.random.Generator):
    """Generate one synthetic sample via SMOTER interpolation.

    Interpolation is applied *event-by-event* only on the non-padded tail.
    """
    lam = rng.uniform(0.0, 1.0)
    x_new = x1.copy()
    start = T - prefix_len
    x_new[start:] = x1[start:] + lam * (x2[start:] - x1[start:])
    y_new = y1 + lam * (y2 - y1)
    return x_new, y_new


def _gaussian_noise(x: np.ndarray, y: float,
                    prefix_len: int, T: int,
                    feat_std: np.ndarray, y_std: float,
                    rng: np.random.Generator,
                    noise_scale: float = 0.1):
    """Generate one synthetic sample by adding Gaussian noise.

    Noise is scaled by per-feature standard deviation so that it is
    proportional to the actual data spread.
    """
    x_new = x.copy()
    start = T - prefix_len
    noise = rng.normal(0.0, noise_scale, size=x[start:].shape) * feat_std
    x_new[start:] = x[start:] + noise
    y_noise = rng.normal(0.0, noise_scale) * y_std
    y_new = y + y_noise
    return x_new, max(y_new, 0.0)  # remaining time is non-negative


# ---------------------------------------------------------------------------
# Core SMOGN routine (operates on numpy arrays)
# ---------------------------------------------------------------------------

def apply_smogn(X: np.ndarray,
                y: np.ndarray,
                *,
                k: int = 5,
                rel_thres: float = 0.5,
                over_ratio: float = 0.8,
                under_ratio: float = 0.5,
                smoter_prob: float = 0.5,
                noise_scale: float = 0.1,
                extr_type: str = "high",
                asym: bool = True,
                seed: int = 42):
    """Apply prefix-length-aware SMOGN to training data.

    Parameters
    ----------
    X : np.ndarray, shape (N, T, F)
        Left-padded feature tensor.
    y : np.ndarray, shape (N,)
        Continuous target (remaining time in days).
    k : int
        Number of nearest neighbours for SMOTER interpolation.
    rel_thres : float in (0, 1)
        Relevance threshold: samples with phi(y) >= rel_thres are "rare".
    over_ratio : float
        Fraction of new synthetic samples to generate relative to the
        number of rare samples (1.0 = generate as many synthetic as rare).
    under_ratio : float
        Fraction of *normal* samples to keep (1.0 = keep all, 0.5 = drop half).
    smoter_prob : float
        Probability of using SMOTER interpolation vs. Gaussian noise for
        each synthetic sample.
    noise_scale : float
        Scale factor for Gaussian noise (relative to feature std).
    extr_type : str
        Passed to ``phi_control`` (``"high"``, ``"low"``, or ``"both"``).
    asym : bool
        Passed to ``phi_control``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_aug : np.ndarray, shape (N', T, F)
    y_aug : np.ndarray, shape (N',)
    """
    rng = np.random.default_rng(seed)
    N, T, F = X.shape

    # --- 1. Compute relevance for all samples ---
    ph = phi_control(y, extr_type=extr_type, asym=asym)
    relevance = phi(y, ph)  # (N,)

    rare_mask = relevance >= rel_thres
    normal_mask = ~rare_mask
    n_rare = int(rare_mask.sum())
    n_normal = int(normal_mask.sum())

    if n_rare == 0:
        print("[SMOGN] No rare samples found with rel_thres={:.2f}. "
              "Returning original data.".format(rel_thres))
        return X.copy(), y.copy()

    # --- 2. Derive prefix lengths & group by length ---
    lengths = get_prefix_lengths(X)
    unique_lengths = np.unique(lengths)

    synth_X_list = []
    synth_y_list = []
    kept_normal_idx = []

    for pl in unique_lengths:
        grp_mask = (lengths == pl)
        grp_idx = np.where(grp_mask)[0]
        grp_X = X[grp_idx]
        grp_y = y[grp_idx]
        grp_rel = relevance[grp_idx]

        rare_in_grp = grp_rel >= rel_thres
        normal_in_grp = ~rare_in_grp
        n_rare_grp = int(rare_in_grp.sum())
        n_normal_grp = int(normal_in_grp.sum())

        # keep (possibly down-sampled) normal samples from this group
        normal_grp_idx = grp_idx[normal_in_grp]
        if n_normal_grp > 0 and under_ratio < 1.0:
            n_keep = max(1, int(round(n_normal_grp * under_ratio)))
            keep = rng.choice(normal_grp_idx, size=n_keep, replace=False)
            kept_normal_idx.extend(keep.tolist())
        else:
            kept_normal_idx.extend(normal_grp_idx.tolist())

        if n_rare_grp == 0:
            continue  # no rare samples in this prefix-length group

        # number of synthetics to generate for this group
        n_synth = max(1, int(round(n_rare_grp * over_ratio)))

        # k-NN within the group (uses only real timesteps)
        eff_k = min(k, grp_X.shape[0] - 1)
        if eff_k < 1:
            # group too small to find any neighbour
            continue
        _, nn_idx = _knn_within_group(grp_X, pl, eff_k)

        # feature std for Gaussian noise (computed on non-padded part)
        start = T - pl
        feat_std = grp_X[:, start:, :].std(axis=0)  # (pl, F)
        feat_std = np.clip(feat_std, 1e-8, None)
        y_std = max(grp_y.std(), 1e-8)

        # indices of rare samples within the group
        rare_grp_local = np.where(rare_in_grp)[0]

        for _ in range(n_synth):
            # pick a random rare sample from the group
            i = rng.choice(rare_grp_local)
            # pick a random neighbour
            j = nn_idx[i, rng.integers(0, nn_idx.shape[1])]

            if rng.random() < smoter_prob:
                x_new, y_new = _smoter_interpolate(
                    grp_X[i], grp_y[i], grp_X[j], grp_y[j], pl, T, rng)
            else:
                x_new, y_new = _gaussian_noise(
                    grp_X[i], grp_y[i], pl, T, feat_std, y_std, rng,
                    noise_scale=noise_scale)

            synth_X_list.append(x_new)
            synth_y_list.append(y_new)

    # --- 3. Combine: kept-normal + all-rare + synthetics ---
    rare_idx = np.where(rare_mask)[0]
    keep_idx = np.array(sorted(set(kept_normal_idx) | set(rare_idx.tolist())),
                        dtype=int)
    X_kept = X[keep_idx]
    y_kept = y[keep_idx]

    if synth_X_list:
        X_synth = np.stack(synth_X_list, axis=0)
        y_synth = np.array(synth_y_list, dtype=y.dtype)
        X_aug = np.concatenate([X_kept, X_synth], axis=0)
        y_aug = np.concatenate([y_kept, y_synth], axis=0)
    else:
        X_aug = X_kept
        y_aug = y_kept

    print(f"[SMOGN] Original: {N} samples ({n_rare} rare, {n_normal} normal)")
    print(f"[SMOGN] Augmented: {X_aug.shape[0]} samples "
          f"(kept {X_kept.shape[0]}, synthetic {len(synth_X_list)})")
    return X_aug, y_aug


# ---------------------------------------------------------------------------
# Orchestrator: load, augment, re-label, save
# ---------------------------------------------------------------------------

def check_smogn_tensors(args):
    """Return True if all SMOGN-augmented tensor files already exist."""
    paths = [args.X_train_smogn_path,
             args.y_train_smogn_path,
             args.z_train_smogn_path,
             args.z_test_smogn_path]
    return all(os.path.exists(p) for p in paths)


def smogn_augment_and_save(args, overwrite: bool = False):
    """Load preprocessed training tensors, apply SMOGN, re-run GMM, save.

    Validation and test data are **not** modified. Only the training tensors
    (X_train, y_train) are augmented and saved under SMOGN-specific paths.
    The GMM component labels (z_train) are recomputed on the augmented data.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain the standard DALSTM tensor paths as well as the
        SMOGN-specific paths (``X_train_smogn_path``, etc.).
    overwrite : bool
        If False, skip augmentation when output files already exist.
    """
    if not overwrite and check_smogn_tensors(args):
        print(f"[SMOGN] Augmented tensors for '{args.dataset}' already exist. "
              "Skipping. Use --overwrite to regenerate.")
        return

    print("[SMOGN] Loading preprocessed training tensors ...")
    # Load original (non-augmented) tensors
    X_train = torch.load(args.X_train_path, weights_only=True)
    y_train = torch.load(args.y_train_path, weights_only=True)

    # Convert to numpy for SMOGN processing
    # .float() handles bfloat16 tensors (used for large datasets to save disk)
    X_np = X_train.float().numpy()
    y_np = y_train.float().numpy()

    print("[SMOGN] Applying prefix-length-aware SMOGN ...")
    X_aug, y_aug = apply_smogn(X_np, y_np)

    # Convert back to tensors (preserve dtype of originals)
    X_aug_t = torch.tensor(X_aug, dtype=X_train.dtype)
    y_aug_t = torch.tensor(y_aug, dtype=y_train.dtype)

    # Save augmented X_train and y_train
    torch.save(X_aug_t, args.X_train_smogn_path)
    torch.save(y_aug_t, args.y_train_smogn_path)
    print(f"[SMOGN] Saved augmented X_train: {X_aug_t.shape}")
    print(f"[SMOGN] Saved augmented y_train: {y_aug_t.shape}")

    # Re-run GMM labelling on augmented train + original val
    y_val = torch.load(args.y_val_path, weights_only=True)
    z_train_aug, _ = fit_label_gmm(y_aug_t, y_val)
    torch.save(z_train_aug, args.z_train_smogn_path)
    print(f"[SMOGN] Saved augmented z_train: {z_train_aug.shape}")

    # Re-predict z_test using augmented train GMM labels
    X_val = torch.load(args.X_val_path, weights_only=True)
    X_test = torch.load(args.X_test_path, weights_only=True)
    z_val = torch.load(args.z_val_path, weights_only=True)
    y_test = torch.load(args.y_test_path, weights_only=True)
    z_test_aug, _, _ = train_lstm_and_predict_test_components(
        X_aug_t, X_val, X_test, z_train_aug, z_val, y_test)
    # Save SMOGN-retrained z_test to its dedicated path
    torch.save(z_test_aug, args.z_test_smogn_path)

    print("[SMOGN] Augmentation and GMM re-labelling complete.")
