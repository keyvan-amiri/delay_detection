# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 12:41:32 2026
@author: kamirel
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from dataclasses import dataclass
#from src.utils.auxiliary_quantile_model import extract_features

def get_calibration_dataframes(result_dir=None, dataset=None, model=None, seed=None):
    calib_name = f"{dataset}_{model}_quantile_wos_seed{seed}_quantile_train_val.csv"
    test_name  = f"{dataset}_{model}_quantile_wos_seed{seed}_inference.csv"
    cal_df = pd.read_csv(os.path.join(result_dir, calib_name))
    test_df  = pd.read_csv(os.path.join(result_dir, test_name))
    return cal_df, test_df

def coverage_report_and_plots(
    df: pd.DataFrame,
    *,
    nominal: float = 0.80,
    cov_col: str = "PI_Coverage_10_90",
    prefix_col: str = "Prefix_length",
    gt_col: str = "GroundTruth",
    b1: int = 10,              # bins for Prefix_length (integer)
    b2: int = 10,              # bins for GroundTruth (continuous)
    path1: str = "coverage_by_prefix.pdf",
    path2: str = "coverage_by_gt.pdf",
    title_prefix: str = "",
    pdf_plot = False,
) -> dict:
    """
    Computes overall coverage (mean of cov_col) and coverage in:
      - b1 bins of Prefix_length (equal-frequency via qcut when possible)
      - b2 bins of GroundTruth (quantile bins)

    Saves two PDFs:
      path1: coverage vs prefix bins
      path2: coverage vs GT bins

    Returns dict with overall + per-bin tables.
    """
    d = df.dropna(subset=[cov_col, prefix_col, gt_col]).copy()
    if len(d) == 0:
        raise ValueError("Empty dataframe after dropping NaNs for coverage computation.")

    d[cov_col] = d[cov_col].astype(float)
    overall = float(d[cov_col].mean())
    n = len(d)

    # --- Prefix_length bins (prefer equal-frequency; fallback to equal-width)
    pref = d[prefix_col].astype(float)
    try:
        pref_bins = pd.qcut(pref, q=b1, duplicates="drop")
    except Exception:
        pref_bins = pd.cut(pref, bins=b1)

    pref_tbl = (
        d.assign(_bin=pref_bins)
         .groupby("_bin", observed=True)[cov_col]
         .agg(["mean", "count"])
         .reset_index()
         .rename(columns={"mean": "coverage", "count": "n"})
    )

    # --- GroundTruth bins (quantile bins)
    gt = d[gt_col].astype(float)
    try:
        gt_bins = pd.qcut(gt, q=b2, duplicates="drop")
    except Exception:
        gt_bins = pd.cut(gt, bins=b2)

    gt_tbl = (
        d.assign(_bin=gt_bins)
         .groupby("_bin", observed=True)[cov_col]
         .agg(["mean", "count"])
         .reset_index()
         .rename(columns={"mean": "coverage", "count": "n"})
    )
    if pdf_plot:
        # --- Plot 1: coverage by prefix bins
        os.makedirs(os.path.dirname(path1) or ".", exist_ok=True)
        plt.figure(figsize=(8, 4.5))
        x = np.arange(len(pref_tbl))
        plt.plot(x, pref_tbl["coverage"].to_numpy(), marker="o")
        plt.axhline(nominal, linestyle="--", linewidth=1, label=f"Nominal={nominal:.2f}")
        plt.xticks(x, pref_tbl["_bin"].astype(str).to_numpy(), rotation=45, ha="right")
        plt.ylim(0.0, 1.0)
        plt.ylabel("Empirical coverage")
        plt.xlabel(f"{prefix_col} bins (qcut)")
        plt.title(f"{title_prefix} Coverage by {prefix_col} | overall={overall:.3f} (n={n})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path1, format="pdf")
        plt.close()

        # --- Plot 2: coverage by GT bins
        os.makedirs(os.path.dirname(path2) or ".", exist_ok=True)
        plt.figure(figsize=(8, 4.5))
        x = np.arange(len(gt_tbl))
        plt.plot(x, gt_tbl["coverage"].to_numpy(), marker="o")
        plt.axhline(nominal, linestyle="--", linewidth=1, label=f"Nominal={nominal:.2f}")
        plt.xticks(x, gt_tbl["_bin"].astype(str).to_numpy(), rotation=45, ha="right")
        plt.ylim(0.0, 1.0)
        plt.ylabel("Empirical coverage")
        plt.xlabel(f"{gt_col} quantile bins (qcut)")
        plt.title(f"{title_prefix} Coverage by {gt_col} | overall={overall:.3f} (n={n})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path2, format="pdf")
        plt.close()

    return {
        "overall_coverage": overall,
        "n": n,
        "prefix_table": pref_tbl,
        "gt_table": gt_tbl,
    }



def _cdf_at_y_from_quantiles(y: np.ndarray, qvals: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """
    Approximate CDF value u = F_hat(y) using piecewise-linear interpolation
    between predicted quantiles (probs -> qvals).

    y: (n,)
    qvals: (n, K) quantile values
    probs: (K,) quantile levels in ascending order

    Returns u in [0,1], shape (n,)
    """
    n, K = qvals.shape
    u = np.empty(n, dtype=float)

    # Ensure probs is ascending
    probs = np.asarray(probs, dtype=float)
    assert np.all(np.diff(probs) > 0), "probs must be strictly increasing."

    for i in range(n):
        yi = y[i]
        qi = qvals[i]

        # Handle NaNs row-wise
        if np.any(np.isnan(qi)) or np.isnan(yi):
            u[i] = np.nan
            continue

        if yi <= qi[0]:
            u[i] = 0.0
        elif yi >= qi[-1]:
            u[i] = 1.0
        else:
            # find j where qi[j] <= yi <= qi[j+1]
            j = np.searchsorted(qi, yi, side="right") - 1
            j = int(np.clip(j, 0, K - 2))
            # linear interpolate within (qi[j], qi[j+1]) to get prob
            denom = (qi[j+1] - qi[j])
            if denom <= 0:
                # degenerate; fall back to midpoint prob
                u[i] = float(0.5 * (probs[j] + probs[j+1]))
            else:
                t = (yi - qi[j]) / denom
                u[i] = float(probs[j] + t * (probs[j+1] - probs[j]))
    return u


def fit_isotonic_recalibrator(
    cal_df: pd.DataFrame,
    *,
    quantile_cols=("Q0_1", "Q0_5", "Q0_6", "Q0_9", "Q0_95", "Q0_99"),
    quantile_probs=(0.1, 0.5, 0.6, 0.9, 0.95, 0.99),
    gt_col="GroundTruth",
):
    """
    Fits isotonic regression R mapping u = F_hat(y) -> empirical_CDF(u),
    following Kuleshov et al. (2018) idea: learn monotone calibration map. :contentReference[oaicite:3]{index=3}
    """
    d = cal_df.dropna(subset=[gt_col, *quantile_cols]).copy()
    y = d[gt_col].to_numpy(dtype=float)
    qvals = d[list(quantile_cols)].to_numpy(dtype=float)

    u = _cdf_at_y_from_quantiles(y, qvals, np.array(quantile_probs, dtype=float))
    m = ~np.isnan(u)
    u = u[m]
    if len(u) < 50:
        raise ValueError(f"Too few calibration points for isotonic calibration: {len(u)}")

    # empirical CDF target: rank(u)/n (use average ranks)
    order = np.argsort(u)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(u) + 1, dtype=float)
    ecdf = ranks / float(len(u))

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(u, ecdf)
    return iso


def isotonic_inverse(iso: IsotonicRegression, p: float) -> float:
    """
    Approximate inverse of fitted isotonic: find u such that iso(u)=p.
    Uses interpolation over iso.y_thresholds_ vs iso.X_thresholds_.
    """
    y_thr = np.asarray(iso.y_thresholds_, dtype=float)
    x_thr = np.asarray(iso.X_thresholds_, dtype=float)

    # Ensure monotone in y_thr
    # np.interp assumes x increasing; y_thr should be non-decreasing for isotonic.
    p = float(np.clip(p, 0.0, 1.0))
    return float(np.interp(p, y_thr, x_thr))


def quantile_at_p_from_row(qvals_row: np.ndarray, probs: np.ndarray, p: float) -> float:
    """
    Given quantile values at probs, return quantile at level p via interpolation.
    """
    probs = np.asarray(probs, dtype=float)
    qvals_row = np.asarray(qvals_row, dtype=float)
    p = float(np.clip(p, probs[0], probs[-1]))
    return float(np.interp(p, probs, qvals_row))


def apply_isotonic_recalibration_to_interval(
    df: pd.DataFrame,
    iso: IsotonicRegression,
    *,
    quantile_cols=("Q0_1", "Q0_5", "Q0_6", "Q0_9", "Q0_95", "Q0_99"),
    quantile_probs=(0.1, 0.5, 0.6, 0.9, 0.95, 0.99),
    gt_col="GroundTruth",
    out_prefix="iso",
):
    """
    Produces recalibrated PI10/PI90 by adjusting quantile levels using R^{-1}.
    Adds columns:
      PI10_{out_prefix}, PI90_{out_prefix}, PI_Width_10_90_{out_prefix}, PI_Coverage_10_90_{out_prefix}
    """
    probs = np.array(quantile_probs, dtype=float)
    qvals = df[list(quantile_cols)].to_numpy(dtype=float)

    p10_adj = isotonic_inverse(iso, 0.10)
    p90_adj = isotonic_inverse(iso, 0.90)

    pi10 = np.empty(len(df), dtype=float)
    pi90 = np.empty(len(df), dtype=float)

    for i in range(len(df)):
        pi10[i] = quantile_at_p_from_row(qvals[i], probs, p10_adj)
        pi90[i] = quantile_at_p_from_row(qvals[i], probs, p90_adj)

    out = df.copy()
    out[f"PI10_{out_prefix}"] = pi10
    out[f"PI90_{out_prefix}"] = pi90
    out[f"PI_Width_10_90_{out_prefix}"] = out[f"PI90_{out_prefix}"] - out[f"PI10_{out_prefix}"]

    if gt_col in out.columns:
        gt = out[gt_col].to_numpy(dtype=float)
        cov = ((gt >= out[f"PI10_{out_prefix}"].to_numpy(dtype=float)) &
               (gt <= out[f"PI90_{out_prefix}"].to_numpy(dtype=float))).astype(float)
        out[f"PI_Coverage_10_90_{out_prefix}"] = cov

    return out

def fit_cqr(
    cal_df: pd.DataFrame,
    *,
    lower_col: str = "PI10",          # or "Q0_1" if that's your lower quantile
    upper_col: str = "PI90",          # or "Q0_9" if that's your upper quantile
    gt_col: str = "GroundTruth",
    alpha: float = 0.20,              # 80% interval => alpha=0.2
) -> float:
    """
    Conformalized Quantile Regression (CQR) for two-sided intervals.
    Computes the conformal slack qhat on the calibration set:
        s_i = max(lower_i - y_i, y_i - upper_i, 0)
        qhat = (1-alpha)-quantile of {s_i} with conformal finite-sample correction.

    Returns:
      qhat (float): nonnegative amount to widen intervals on test:
        [lower - qhat, upper + qhat]
    """
    d = cal_df.dropna(subset=[lower_col, upper_col, gt_col]).copy()
    y = d[gt_col].to_numpy(dtype=float)
    lo = d[lower_col].to_numpy(dtype=float)
    hi = d[upper_col].to_numpy(dtype=float)

    # nonconformity scores for two-sided intervals
    s = np.maximum.reduce([lo - y, y - hi, np.zeros_like(y)])

    n = len(s)
    if n == 0:
        raise ValueError("Calibration dataframe has 0 usable rows for CQR.")
    # finite-sample corrected quantile index (Romano et al. style)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    qhat = float(np.sort(s)[k - 1])
    return qhat


def apply_cqr_to_interval(
    df: pd.DataFrame,
    qhat: float,
    *,
    lower_col: str = "PI10",
    upper_col: str = "PI90",
    gt_col: str = "GroundTruth",
    out_prefix: str = "cqr",
) -> pd.DataFrame:
    """
    Apply CQR widening to any dataframe with lower/upper columns.
    Adds:
      PI10_{out_prefix}, PI90_{out_prefix},
      PI_Width_10_90_{out_prefix}, PI_Coverage_10_90_{out_prefix} (if gt exists)
    """
    out = df.copy()
    lo = out[lower_col].to_numpy(dtype=float)
    hi = out[upper_col].to_numpy(dtype=float)

    out[f"PI10_{out_prefix}"] = lo - qhat
    out[f"PI90_{out_prefix}"] = hi + qhat
    out[f"PI_Width_10_90_{out_prefix}"] = out[f"PI90_{out_prefix}"] - out[f"PI10_{out_prefix}"]

    if gt_col in out.columns:
        y = out[gt_col].to_numpy(dtype=float)
        cov = ((y >= out[f"PI10_{out_prefix}"].to_numpy(dtype=float)) &
               (y <= out[f"PI90_{out_prefix}"].to_numpy(dtype=float))).astype(float)
        out[f"PI_Coverage_10_90_{out_prefix}"] = cov

    return out


def fit_local_cqr_by_width(
    cal_df: pd.DataFrame,
    *,
    lower_col: str = "PI10",
    upper_col: str = "PI90",
    gt_col: str = "GroundTruth",
    width_col: str = "PI_Width_10_90",   # kept for backward compat; used only as fallback
    alpha: float = 0.20,
    n_bins: int = 5,                     # interpreted as max #groups (G)
    binning: str = "quantile",           # kept for backward compat; ignored when Prefix_length exists
    min_bin_size: int = 50,
) -> dict:
    """
    Locally-adaptive CQR with *process-aware* grouping by Prefix_length.
    - If Prefix_length column exists, bins/groups are formed over prefix lengths
      using a mass-balanced partition (your paper's idea).
    - Otherwise, falls back to the original behavior: binning over width_col.

    Returns a dict with:
      - mode: 'prefix' or 'width'
      - upper_bounds: sorted array of group upper bounds (len=G')
      - qhat_per_group: array (len=G') aligned with upper_bounds
      - qhat_global: scalar fallback
      - meta: sizes, group ranges, etc.
    """
    # --- pick binning feature (prefer Prefix_length to keep pipeline unchanged)
    prefer_prefix = "Prefix_length" in cal_df.columns
    bin_col = "Prefix_length" if prefer_prefix else width_col
    mode = "prefix" if prefer_prefix else "width"

    d = cal_df.dropna(subset=[lower_col, upper_col, gt_col, bin_col]).copy()
    if len(d) == 0:
        raise ValueError("Calibration dataframe has 0 usable rows for local CQR.")

    y = d[gt_col].to_numpy(dtype=float)
    lo = d[lower_col].to_numpy(dtype=float)
    hi = d[upper_col].to_numpy(dtype=float)
    x = d[bin_col].to_numpy(dtype=float)

    # nonconformity scores for two-sided intervals
    s = np.maximum.reduce([lo - y, y - hi, np.zeros_like(y)])
    n = len(s)

    def conformal_qhat(scores: np.ndarray) -> float:
        m = len(scores)
        if m == 0:
            return np.nan
        k = int(np.ceil((m + 1) * (1 - alpha)))
        k = min(max(k, 1), m)
        return float(np.sort(scores)[k - 1])

    qhat_global = conformal_qhat(s)

    # If we don't have Prefix_length, fall back to your original width-binning logic
    if mode == "width":
        if n_bins <= 1:
            return {
                "mode": "width",
                "bin_edges": np.array([-np.inf, np.inf], dtype=float),
                "qhat_per_bin": np.array([qhat_global], dtype=float),
                "qhat_global": float(qhat_global),
                "meta": {"n": n, "bin_sizes": [n], "n_bins": 1},
            }

        w = x
        if binning == "quantile":
            qs = np.linspace(0, 1, n_bins + 1)
            edges = np.quantile(w, qs)
            edges = np.unique(edges)
            if len(edges) < 2:
                edges = np.array([np.min(w), np.max(w)], dtype=float)
            edges[0] = -np.inf
            edges[-1] = np.inf
        elif binning == "uniform":
            edges = np.linspace(np.min(w), np.max(w), n_bins + 1)
            edges[0] = -np.inf
            edges[-1] = np.inf
        else:
            raise ValueError("binning must be 'quantile' or 'uniform'")

        bin_idx = np.digitize(w, edges[1:-1], right=True)
        B = len(edges) - 1
        qhat_bins = np.full(B, np.nan, dtype=float)
        bin_sizes = []

        for b in range(B):
            mask = (bin_idx == b)
            sb = s[mask]
            bin_sizes.append(int(mask.sum()))
            qhat_bins[b] = conformal_qhat(sb) if len(sb) >= min_bin_size else qhat_global

        return {
            "mode": "width",
            "bin_edges": edges.astype(float),
            "qhat_per_bin": qhat_bins.astype(float),
            "qhat_global": float(qhat_global),
            "meta": {"n": n, "bin_sizes": bin_sizes, "n_bins": B},
        }

    # --- process-aware partitioning over Prefix_length (mass-balanced, contiguous lengths)
    # treat prefix lengths as discrete; we group contiguous sorted unique lengths
    pref = x
    # if they're floats but represent ints, this still works; grouping is by numeric order
    order = np.argsort(pref)
    pref_sorted = pref[order]
    s_sorted = s[order]

    # counts per unique prefix length (stable, contiguous)
    uniq, start_idx, counts = np.unique(pref_sorted, return_index=True, return_counts=True)
    total = int(counts.sum())
    G = int(max(1, n_bins))
    target = total / float(G)

    # Step 1: build groups by accumulating counts until reaching ~ total/G
    groups = []  # list of (start_u_idx, end_u_idx) inclusive indices into uniq
    cur_start = 0
    cur_mass = 0.0
    for u_i, c in enumerate(counts):
        if u_i == cur_start:
            cur_mass = 0.0
        cur_mass += float(c)

        # if we've hit target and still can create more groups, cut here
        remaining_uniq = (len(uniq) - 1) - u_i
        remaining_groups_possible = (G - 1) - len(groups)
        must_leave_at_least_one = remaining_uniq >= remaining_groups_possible

        if (cur_mass >= target) and must_leave_at_least_one and (len(groups) < G - 1):
            groups.append((cur_start, u_i))
            cur_start = u_i + 1

    # last group takes the rest
    if cur_start <= len(uniq) - 1:
        groups.append((cur_start, len(uniq) - 1))

    # Step 2: merge tiny groups (< min_bin_size) into neighbors (to avoid sparse calibration)
    # compute mass per group
    def group_mass(g):
        a, b = g
        return int(counts[a : b + 1].sum())

    # iterative merging
    merged = True
    while merged and len(groups) > 1:
        merged = False
        masses = [group_mass(g) for g in groups]
        small_idx = [i for i, m in enumerate(masses) if m < min_bin_size]
        if not small_idx:
            break

        i = small_idx[0]
        # merge with neighbor that yields larger combined mass / keeps contiguity
        if i == 0:
            # merge into next
            a1, b1 = groups[i]
            a2, b2 = groups[i + 1]
            groups[i] = (a1, b2)
            del groups[i + 1]
        elif i == len(groups) - 1:
            # merge into prev
            a0, b0 = groups[i - 1]
            a1, b1 = groups[i]
            groups[i - 1] = (a0, b1)
            del groups[i]
        else:
            # choose merge direction that results in larger combined mass
            m_prev = masses[i - 1] + masses[i]
            m_next = masses[i] + masses[i + 1]
            if m_prev >= m_next:
                a0, b0 = groups[i - 1]
                a1, b1 = groups[i]
                groups[i - 1] = (a0, b1)
                del groups[i]
            else:
                a1, b1 = groups[i]
                a2, b2 = groups[i + 1]
                groups[i] = (a1, b2)
                del groups[i + 1]
        merged = True

    # Map each row to a group by its prefix length using upper bounds
    # We'll store upper bounds (numeric) for fast searchsorted in apply().
    group_ranges = []
    upper_bounds = []
    qhat_per_group = []
    group_sizes = []

    # Precompute for each unique prefix length: the slice of rows in pref_sorted
    # start_idx gives first occurrence; last occurrence is start_idx + count - 1
    # We'll build a quick lookup from uniq index -> row slice
    row_slices = [(start_idx[i], start_idx[i] + counts[i]) for i in range(len(uniq))]

    for (a_u, b_u) in groups:
        # rows belonging to uniq indices [a_u..b_u]
        row_lo = row_slices[a_u][0]
        row_hi = row_slices[b_u][1]  # exclusive
        sb = s_sorted[row_lo:row_hi]
        m = len(sb)

        qg = conformal_qhat(sb) if m >= min_bin_size else qhat_global
        qhat_per_group.append(float(qg))
        group_sizes.append(int(m))

        lo_k = float(uniq[a_u])
        hi_k = float(uniq[b_u])
        group_ranges.append((lo_k, hi_k))
        upper_bounds.append(hi_k)

    upper_bounds = np.asarray(upper_bounds, dtype=float)
    qhat_per_group = np.asarray(qhat_per_group, dtype=float)

    return {
        "mode": "prefix",
        "bin_col": bin_col,  # 'Prefix_length'
        "upper_bounds": upper_bounds,
        "qhat_per_group": qhat_per_group,
        "qhat_global": float(qhat_global),
        "meta": {
            "n": int(n),
            "requested_max_groups": int(G),
            "actual_groups": int(len(groups)),
            "group_ranges": group_ranges,   # [(min_prefix, max_prefix), ...]
            "group_sizes": group_sizes,
            "min_bin_size": int(min_bin_size),
        },
    }


def apply_local_cqr_to_interval(
    df: pd.DataFrame,
    local_cqr: dict,
    *,
    lower_col: str = "PI10",
    upper_col: str = "PI90",
    gt_col: str = "GroundTruth",
    width_col: str = "PI_Width_10_90",   # kept for backward compat; used only as fallback
    out_prefix: str = "cqr_local",
) -> pd.DataFrame:
    """
    Apply locally-adaptive CQR widening.
    - If local_cqr['mode'] == 'prefix' (default when Prefix_length exists at fit time),
      uses Prefix_length groups (mass-balanced, contiguous ranges).
    - Otherwise uses width bins (original behavior).

    Adds:
      PI10_{out_prefix}, PI90_{out_prefix},
      PI_Width_10_90_{out_prefix}, PI_Coverage_10_90_{out_prefix} (if gt exists)
    """
    out = df.copy()

    mode = local_cqr.get("mode", "width")

    if mode == "prefix":
        bin_col = local_cqr.get("bin_col", "Prefix_length")
        if bin_col not in out.columns:
            # graceful fallback if test doesn't have Prefix_length for some reason
            mode = "width"
        else:
            out = out.dropna(subset=[lower_col, upper_col, bin_col]).copy()
            pref = out[bin_col].to_numpy(dtype=float)

            upper_bounds = np.asarray(local_cqr["upper_bounds"], dtype=float)
            qhat_groups = np.asarray(local_cqr["qhat_per_group"], dtype=float)

            # group index = first upper_bound >= pref (searchsorted with side='left')
            # clip to valid range
            idx = np.searchsorted(upper_bounds, pref, side="left")
            idx = np.clip(idx, 0, len(qhat_groups) - 1)
            qhat_row = qhat_groups[idx]

            lo = out[lower_col].to_numpy(dtype=float)
            hi = out[upper_col].to_numpy(dtype=float)

            out[f"PI10_{out_prefix}"] = lo - qhat_row
            out[f"PI90_{out_prefix}"] = hi + qhat_row
            out[f"PI_Width_10_90_{out_prefix}"] = out[f"PI90_{out_prefix}"] - out[f"PI10_{out_prefix}"]

            if gt_col in out.columns:
                y = out[gt_col].to_numpy(dtype=float)
                cov = ((y >= out[f"PI10_{out_prefix}"].to_numpy(dtype=float)) &
                       (y <= out[f"PI90_{out_prefix}"].to_numpy(dtype=float))).astype(float)
                out[f"PI_Coverage_10_90_{out_prefix}"] = cov

            return out

    # --- width mode (original behavior)
    out = out.dropna(subset=[lower_col, upper_col, width_col]).copy()

    edges = np.asarray(local_cqr["bin_edges"], dtype=float)
    qhat_bins = np.asarray(local_cqr["qhat_per_bin"], dtype=float)

    w = out[width_col].to_numpy(dtype=float)
    b = np.digitize(w, edges[1:-1], right=True)  # 0..B-1
    qhat_row = qhat_bins[b]

    lo = out[lower_col].to_numpy(dtype=float)
    hi = out[upper_col].to_numpy(dtype=float)

    out[f"PI10_{out_prefix}"] = lo - qhat_row
    out[f"PI90_{out_prefix}"] = hi + qhat_row
    out[f"PI_Width_10_90_{out_prefix}"] = out[f"PI90_{out_prefix}"] - out[f"PI10_{out_prefix}"]

    if gt_col in out.columns:
        y = out[gt_col].to_numpy(dtype=float)
        cov = ((y >= out[f"PI10_{out_prefix}"].to_numpy(dtype=float)) &
               (y <= out[f"PI90_{out_prefix}"].to_numpy(dtype=float))).astype(float)
        out[f"PI_Coverage_10_90_{out_prefix}"] = cov

    return out

@dataclass
class GlobalScaledCQR:
    gamma: float
    hw_floor: float
    alpha: float


def fit_global_scaled_cqr(
    cal_df: pd.DataFrame,
    *,
    lower_col: str = "PI10",
    upper_col: str = "PI90",
    gt_col: str = "GroundTruth",
    alpha: float = 0.20,
    hw_floor_q: float = 0.01,     # robust floor quantile for half-width
    clip_r_q: float = 0.999,      # optional winsorization of extreme ratios
) -> GlobalScaledCQR:
    """
    Global multiplicative conformal scaling:
        center ± gamma * half_width

    gamma is the (1-alpha) conformal quantile of:
        r_i = |y_i - center_i| / max(half_width_i, hw_floor)

    Robustness:
      - hw_floor is set to quantile(hw, hw_floor_q) to prevent ratio explosion when hw≈0
      - optional clipping of r_i at quantile(r, clip_r_q) to reduce sensitivity to rare pathologies
    """
    d = cal_df.dropna(subset=[lower_col, upper_col, gt_col]).copy()
    if len(d) == 0:
        raise ValueError("Calibration dataframe has 0 usable rows for global scaled CQR.")

    y = d[gt_col].to_numpy(dtype=float)
    lo = d[lower_col].to_numpy(dtype=float)
    hi = d[upper_col].to_numpy(dtype=float)

    center = 0.5 * (lo + hi)
    half_width = 0.5 * (hi - lo)
    half_width = np.maximum(half_width, 0.0)

    # robust floor
    hw_floor = float(np.quantile(half_width, hw_floor_q))
    if (not np.isfinite(hw_floor)) or (hw_floor <= 0):
        hw_floor = 1e-6

    denom = np.maximum(half_width, hw_floor)
    r = np.abs(y - center) / denom

    # optional clipping (prevents a couple of crazy points dominating gamma)
    if clip_r_q is not None:
        r_clip = float(np.quantile(r, clip_r_q))
        if np.isfinite(r_clip) and r_clip > 0:
            r = np.minimum(r, r_clip)

    n = len(r)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    gamma = float(np.sort(r)[k - 1])

    return GlobalScaledCQR(gamma=gamma, hw_floor=hw_floor, alpha=alpha)


def apply_global_scaled_cqr_to_interval(
    df: pd.DataFrame,
    model: GlobalScaledCQR,
    *,
    lower_col: str = "PI10",
    upper_col: str = "PI90",
    gt_col: str = "GroundTruth",
    out_prefix: str = "scqr",
) -> pd.DataFrame:
    """
    Apply global scaled CQR:
        new interval = center ± gamma * half_width
    Adds:
      PI10_{out_prefix}, PI90_{out_prefix},
      PI_Width_10_90_{out_prefix}, PI_Coverage_10_90_{out_prefix} (if gt exists)
    """
    out = df.copy()
    lo = out[lower_col].to_numpy(dtype=float)
    hi = out[upper_col].to_numpy(dtype=float)

    center = 0.5 * (lo + hi)
    half_width = 0.5 * (hi - lo)
    half_width = np.maximum(half_width, 0.0)

    new_lo = center - model.gamma * half_width
    new_hi = center + model.gamma * half_width

    out[f"PI10_{out_prefix}"] = new_lo
    out[f"PI90_{out_prefix}"] = new_hi
    out[f"PI_Width_10_90_{out_prefix}"] = out[f"PI90_{out_prefix}"] - out[f"PI10_{out_prefix}"]

    if gt_col in out.columns:
        y = out[gt_col].to_numpy(dtype=float)
        cov = ((y >= new_lo) & (y <= new_hi)).astype(float)
        out[f"PI_Coverage_10_90_{out_prefix}"] = cov

    return out

def fit_multi_cqr_by_width(
    cal_df: pd.DataFrame,
    *,
    lower_col: str = "PI10",
    upper_col: str = "PI90",
    gt_col: str = "GroundTruth",
    width_col: str = "PI_Width_10_90",
    alpha: float = 0.20,
    n_bins: int = 5,
    binning: str = "quantile",
    min_bin_size: int = 50,
) -> dict:
    """
    Robust multiplicative conformal scaling:
        center ± gamma * half_width

    gamma is the (1-alpha) conformal quantile of:
        r_i = |y_i - center_i| / max(half_width_i, hw_floor)

    Key robustness: hw_floor prevents exploding ratios when predicted half-width ~ 0.
    hw_floor is chosen from calibration half-width distribution (global).
    """
    d = cal_df.dropna(subset=[lower_col, upper_col, gt_col, width_col]).copy()
    if len(d) == 0:
        raise ValueError("Calibration dataframe has 0 usable rows for scaled conformal.")

    y = d[gt_col].to_numpy(dtype=float)
    lo = d[lower_col].to_numpy(dtype=float)
    hi = d[upper_col].to_numpy(dtype=float)
    x = d[width_col].to_numpy(dtype=float)  # binning feature (width or Prefix_length)

    center = 0.5 * (lo + hi)
    half_width = 0.5 * (hi - lo)
    half_width = np.maximum(half_width, 0.0)

    # --- Robust floor for half_width used in normalization
    # Pick a small quantile so we only "fix" the near-zero widths.
    # Also ensure it's positive by falling back to a small constant if needed.
    hw_floor = float(np.quantile(half_width, 0.01))
    if not np.isfinite(hw_floor) or hw_floor <= 0:
        hw_floor = 1e-6  # safe fallback in case intervals are degenerate

    denom = np.maximum(half_width, hw_floor)
    r = np.abs(y - center) / denom

    # Optional: clip extreme ratios to reduce sensitivity to a few pathologies
    # (keeps conformal-ish behavior, but stabilizes in practice)
    r_clip = float(np.quantile(r, 0.999))
    if np.isfinite(r_clip) and r_clip > 0:
        r = np.minimum(r, r_clip)

    def conformal_gamma(scores: np.ndarray) -> float:
        m = len(scores)
        k = int(np.ceil((m + 1) * (1 - alpha)))
        k = min(max(k, 1), m)
        return float(np.sort(scores)[k - 1])

    gamma_global = conformal_gamma(r)

    if n_bins <= 1:
        return {
            "mode": "scaled_robust",
            "binning_on": width_col,
            "bin_edges": np.array([-np.inf, np.inf], dtype=float),
            "gamma_per_bin": np.array([gamma_global], dtype=float),
            "gamma_global": float(gamma_global),
            "hw_floor": float(hw_floor),
            "r_clip": float(r_clip),
            "meta": {"n": int(len(r)), "bin_sizes": [int(len(r))], "n_bins": 1},
        }

    # ----- build bin edges over x
    if binning == "quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(x, qs)
        edges = np.unique(edges)
        if len(edges) < 2:
            edges = np.array([np.min(x), np.max(x)], dtype=float)
        edges[0] = -np.inf
        edges[-1] = np.inf
    elif binning == "uniform":
        edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
        edges[0] = -np.inf
        edges[-1] = np.inf
    else:
        raise ValueError("binning must be 'quantile' or 'uniform'")

    bin_idx = np.digitize(x, edges[1:-1], right=True)
    B = len(edges) - 1

    gamma_bins = np.full(B, np.nan, dtype=float)
    bin_sizes = []

    for b in range(B):
        mask = (bin_idx == b)
        rb = r[mask]
        bin_sizes.append(int(mask.sum()))
        gamma_bins[b] = conformal_gamma(rb) if len(rb) >= min_bin_size else gamma_global

    return {
        "mode": "scaled_robust",
        "binning_on": width_col,
        "bin_edges": edges.astype(float),
        "gamma_per_bin": gamma_bins.astype(float),
        "gamma_global": float(gamma_global),
        "hw_floor": float(hw_floor),
        "r_clip": float(r_clip),
        "meta": {"n": int(len(r)), "bin_sizes": bin_sizes, "n_bins": int(B)},
    }


def apply_multi_cqr_to_interval(
    df: pd.DataFrame,
    local_cqr: dict,
    *,
    lower_col: str = "PI10",
    upper_col: str = "PI90",
    gt_col: str = "GroundTruth",
    width_col: str = "PI_Width_10_90",
    out_prefix: str = "cqr_local",
) -> pd.DataFrame:
    """
    Apply robust multiplicative scaling:
        center ± gamma(bin) * half_width

    Uses the same width_col that was used for binning during fit
    (Prefix_length or PI_Width_10_90), and uses hw_floor stored in local_cqr
    to avoid negative/degenerate widths (though the ratio issue is fixed in fit).
    """
    out = df.copy()
    out = out.dropna(subset=[lower_col, upper_col, width_col]).copy()

    edges = np.asarray(local_cqr["bin_edges"], dtype=float)
    gamma_bins = np.asarray(local_cqr["gamma_per_bin"], dtype=float)

    x = out[width_col].to_numpy(dtype=float)
    b = np.digitize(x, edges[1:-1], right=True)
    gamma_row = gamma_bins[b]

    lo = out[lower_col].to_numpy(dtype=float)
    hi = out[upper_col].to_numpy(dtype=float)

    center = 0.5 * (lo + hi)
    half_width = 0.5 * (hi - lo)
    half_width = np.maximum(half_width, 0.0)

    new_lo = center - gamma_row * half_width
    new_hi = center + gamma_row * half_width

    out[f"PI10_{out_prefix}"] = new_lo
    out[f"PI90_{out_prefix}"] = new_hi
    out[f"PI_Width_10_90_{out_prefix}"] = out[f"PI90_{out_prefix}"] - out[f"PI10_{out_prefix}"]

    if gt_col in out.columns:
        y = out[gt_col].to_numpy(dtype=float)
        cov = ((y >= new_lo) & (y <= new_hi)).astype(float)
        out[f"PI_Coverage_10_90_{out_prefix}"] = cov

    return out

def main():
    # ---- settings ----
    model_name = "DALSTM"
    datsets = ["P2P", "BPIC15_1", "BPIC_2017_W", "Sepsis", "BPIC20ID", "BPIC20DD", "BPIC20PTC"]
    seeds = [409, 1824, 3657, 4012, 4506]
    # coverage plot settings
    nominal = 0.80
    b1 = 10  # prefix bins
    b2 = 10  # gt bins
    quantile_cols = ("Q0_1", "Q0_5", "Q0_6", "Q0_9", "Q0_95", "Q0_99")
    quantile_probs = (0.1, 0.5, 0.6, 0.9, 0.95, 0.99)    
    parser = argparse.ArgumentParser(
        description='Calibration for Quantile Regression')
    parser.add_argument('--method', type=str, default='cqr',
                        choices=['cqr', 'cov_prefix', 'cov_width', 
                                 'scqr', 'mul_prefix', 'mul_width',
                                 'isotonic'],
                        help='Calibration Approach to use')  
    parser.add_argument('--pdf_plots', action='store_true', default=False)
    args = parser.parse_args()

    # paths
    root_path = os.getcwd()
    result_path = os.path.join(root_path, "results", model_name)

    # ---- run ----
    for dataset in datsets:
        result_dir = os.path.join(result_path, dataset)
        for seed in seeds:
            cal_df, test_df = get_calibration_dataframes(
                result_dir=result_dir, dataset=dataset, model=model_name,
                seed=seed)
            # --- 1) coverage BEFORE (uses existing PI_Coverage_10_90)
            before_prefix_pdf = os.path.join(
                result_dir, f"{dataset}_{model_name}_quantile_wos_seed{seed}_coverage_by_prefix_before.pdf"
            )
            before_gt_pdf = os.path.join(
                result_dir, f"{dataset}_{model_name}_quantile_wos_seed{seed}_coverage_by_gt_before.pdf"
            )
            rep_before = coverage_report_and_plots(
                test_df,
                nominal=nominal,
                cov_col="PI_Coverage_10_90",
                b1=b1, b2=b2,
                path1=before_prefix_pdf,
                path2=before_gt_pdf,
                title_prefix=f"{dataset}/{model_name} seed={seed} BEFORE",
                )
            print(dataset, f"seed={seed}", "BEFORE overall_cov", rep_before["overall_coverage"])
            mean_width_before = test_df["PI_Width_10_90"].mean()
            print(dataset, f"seed={seed}", "Before  mean_width", mean_width_before)
            if args.method == 'isotonic':
                # --- 2) fit isotonic recalibrator on calibration dataframe
                iso = fit_isotonic_recalibrator(
                    cal_df,
                    quantile_cols=quantile_cols,
                    quantile_probs=quantile_probs,
                    gt_col="GroundTruth",
                    )
                # --- 3) apply isotonic recalibration to test intervals (PI10/PI90)
                test_cal = apply_isotonic_recalibration_to_interval(
                    test_df,
                    iso,
                    quantile_cols=quantile_cols,
                    quantile_probs=quantile_probs,
                    gt_col="GroundTruth",
                    out_prefix="iso",
                    )
                prefix_str = "iso"
            elif args.method == 'cqr':
                # --- 2) fit CQR on calibration dataframe (returns scalar qhat)
                qhat = fit_cqr(
                    cal_df,
                    lower_col="PI10",   # or "Q0_1"
                    upper_col="PI90",   # or "Q0_9"
                    gt_col="GroundTruth",
                    alpha=0.20          # 80% interval
                    )

                # --- 3) apply CQR to test intervals
                test_cal = apply_cqr_to_interval(
                    test_df,
                    qhat,
                    lower_col="PI10",   # or "Q0_1"
                    upper_col="PI90",   # or "Q0_9"
                    gt_col="GroundTruth",
                    out_prefix="cqr"
                    )
                prefix_str = "cqr"
            elif args.method == 'cov_width':
                # UNCERTAINTY-WIDTH MODE
                qhat_width = fit_local_cqr_by_width(
                    cal_df,
                    lower_col="PI10",
                    upper_col="PI90",
                    gt_col="GroundTruth",
                    width_col="PI_Width_10_90",  # <- bin by predicted uncertainty
                    alpha=0.20,
                    n_bins=5,
                    binning="quantile",          # or "uniform"
                    min_bin_size=50
                    )
                test_cal = apply_local_cqr_to_interval(
                    test_df,
                    qhat_width,
                    lower_col="PI10",
                    upper_col="PI90",
                    gt_col="GroundTruth",
                    width_col="PI_Width_10_90",  # <- must match fit
                    out_prefix="cov_width"
                    )
                prefix_str = "cov_width"                
            elif args.method == 'cov_prefix':
                # PREFIX-LENGTH MODE
                qhat_prefix = fit_local_cqr_by_width(
                    cal_df,
                    lower_col="PI10",
                    upper_col="PI90",
                    gt_col="GroundTruth",
                    width_col="Prefix_length",   # <- key: use prefix length as binning variable
                    alpha=0.20,
                    n_bins=10,                   # max number of prefix groups (G)
                    min_bin_size=50
                    )
                test_cal = apply_local_cqr_to_interval(
                    test_df,
                    qhat_prefix,
                    lower_col="PI10",
                    upper_col="PI90",
                    gt_col="GroundTruth",
                    width_col="Prefix_length",   # <- must match fit
                    out_prefix="cov_prefix"
                    )                
                prefix_str = "cov_prefix"
            
            elif args.method == 'scqr':
                scqr_model = fit_global_scaled_cqr(
                    cal_df,
                    lower_col="PI10",
                    upper_col="PI90",
                    gt_col="GroundTruth",
                    alpha=0.20,
                    )

                test_cal = apply_global_scaled_cqr_to_interval(
                    test_df,
                    scqr_model,
                    lower_col="PI10",
                    upper_col="PI90",
                    gt_col="GroundTruth",
                    out_prefix="cov_scqr"
                    )
                prefix_str = "cov_scqr"
                
            elif args.method == 'mul_prefix':
                # PREFIX-LENGTH MODE
                qhat_prefix = fit_multi_cqr_by_width(
                    cal_df,
                    lower_col="PI10",
                    upper_col="PI90",
                    gt_col="GroundTruth",
                    width_col="Prefix_length",   # <- key: use prefix length as binning variable
                    alpha=0.20,
                    n_bins=10,                   # max number of prefix groups (G)
                    min_bin_size=50
                    )
                test_cal = apply_multi_cqr_to_interval(
                    test_df,
                    qhat_prefix,
                    lower_col="PI10",
                    upper_col="PI90",
                    gt_col="GroundTruth",
                    width_col="Prefix_length",   # <- must match fit
                    out_prefix="mul_prefix"
                    )                
                prefix_str = "mul_prefix"

            elif args.method == 'mul_width':
                # UNCERTAINTY-WIDTH MODE
                qhat_width = fit_multi_cqr_by_width(
                    cal_df,
                    lower_col="PI10",
                    upper_col="PI90",
                    gt_col="GroundTruth",
                    width_col="PI_Width_10_90",  # <- bin by predicted uncertainty
                    alpha=0.20,
                    n_bins=5,
                    binning="quantile",          # or "uniform"
                    min_bin_size=50
                    )
                test_cal = apply_multi_cqr_to_interval(
                    test_df,
                    qhat_width,
                    lower_col="PI10",
                    upper_col="PI90",
                    gt_col="GroundTruth",
                    width_col="PI_Width_10_90",  # <- must match fit
                    out_prefix="mul_width"
                    )
                prefix_str = "mul_width"   
            # --- 4) coverage AFTER (uses PI_Coverage_10_90_iso)
            after_prefix_pdf = os.path.join(
                result_dir, f"{dataset}_{model_name}_quantile_wos_seed{seed}_coverage_by_prefix_after_iso.pdf"
            )
            after_gt_pdf = os.path.join(
                result_dir, f"{dataset}_{model_name}_quantile_wos_seed{seed}_coverage_by_gt_after_iso.pdf"
            )
            coverage_column = "PI_Coverage_10_90_" + prefix_str
            rep_after = coverage_report_and_plots(
                test_cal,
                nominal=nominal,
                cov_col=coverage_column,
                b1=b1, b2=b2,
                path1=after_prefix_pdf,
                path2=after_gt_pdf,
                title_prefix=f"{dataset}/{model_name} seed={seed} AFTER isotonic",
            )
            print(dataset, f"seed={seed}", "AFTER  overall_cov", rep_after["overall_coverage"])
            width_column = "PI_Width_10_90_" + prefix_str
            mean_width_after = test_cal[width_column].mean()
            print(dataset, f"seed={seed}", "After  mean_width", mean_width_after)
            


            # Optional: save recalibrated test df next to plots
            #out_csv = os.path.join(result_dir, f"{dataset}_{model_name}_quantile_wos_seed{seed}_calibrated.csv")
            #test_iso.to_csv(out_csv, index=False)
            #print("[OK] wrote", out_csv)




if __name__ == "__main__":
    main()