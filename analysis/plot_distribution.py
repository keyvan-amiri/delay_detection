# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 11:08:35 2026
@author: Keyvan Amiri Elyasi
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def _c_index(y_true, y_pred, eps=1e-12):
    """
    Concordance index for regression-style ranking.
    Compares all pairs (i,j) where y_true differs.
    Ties in predictions get 0.5 credit.

    Returns:
      c_index in [0,1] or np.nan if no comparable pairs.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Remove NaNs
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]
    n = len(y_true)
    if n < 2:
        return np.nan

    # Pairwise differences
    dt = y_true[:, None] - y_true[None, :]
    dp = y_pred[:, None] - y_pred[None, :]

    # Comparable pairs: dt != 0, take upper triangle only to avoid double counting
    iu = np.triu_indices(n, k=1)
    dt_u = dt[iu]
    dp_u = dp[iu]

    comp = np.abs(dt_u) > eps
    if not np.any(comp):
        return np.nan

    dt_u = dt_u[comp]
    dp_u = dp_u[comp]

    # Concordant if signs match, discordant if signs differ, tie pred -> 0.5
    same_sign = (dt_u * dp_u) > 0
    tie_pred  = np.abs(dp_u) <= eps

    c = (same_sign.sum() + 0.5 * tie_pred.sum()) / len(dt_u)
    return float(c)


def cumulative_cindex_and_auc(
    df: pd.DataFrame,
    prefix_col="Prefix_length",
    y_col="GroundTruth",
    yhat_col="Prediction",
    max_prefix=None,
    min_prefix=1,
    weight_scheme="inv",   # "inv", "exp", or callable w(t)->weight
    exp_alpha=0.05,
):
    """
    Computes cumulative C-index curve:
      C(t) computed on all rows with prefix <= t.

    Then computes:
      - AUC (uniform over t)
      - weighted AUC (emphasize early prefixes)

    Returns:
      curve_df: columns = ["t", "c_index", "n_rows"]
      metrics: dict with auc, wauc, and metadata
    """
    needed = {prefix_col, y_col, yhat_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    # Clean and sort by prefix
    d = df[[prefix_col, y_col, yhat_col]].copy()
    d = d[np.isfinite(d[y_col]) & np.isfinite(d[yhat_col]) & np.isfinite(d[prefix_col])]
    d[prefix_col] = d[prefix_col].astype(int)
    d = d.sort_values(prefix_col)

    if len(d) == 0:
        raise ValueError("No valid rows after cleaning NaNs/Infs.")

    # Determine prefix range
    t_min = max(min_prefix, int(d[prefix_col].min()))
    t_max = int(d[prefix_col].max()) if max_prefix is None else min(int(max_prefix), int(d[prefix_col].max()))
    if t_max < t_min:
        raise ValueError(f"Invalid prefix range: [{t_min}, {t_max}]")

    # Precompute index sets for cumulative slices efficiently
    # We'll walk t and expand the slice end.
    ts = np.arange(t_min, t_max + 1, dtype=int)
    prefix_vals = d[prefix_col].to_numpy()
    y_true_all = d[y_col].to_numpy()
    y_pred_all = d[yhat_col].to_numpy()

    curve = []
    end = 0
    n_total = len(d)

    for t in ts:
        # advance 'end' to include all rows with prefix <= t
        while end < n_total and prefix_vals[end] <= t:
            end += 1
        # cumulative subset is [0:end)
        y_true = y_true_all[:end]
        y_pred = y_pred_all[:end]
        c = _c_index(y_true, y_pred)
        curve.append((t, c, end))

    curve_df = pd.DataFrame(curve, columns=["t", "c_index", "n_rows"])

    # AUC (simple mean over available t, ignoring nan)
    cvals = curve_df["c_index"].to_numpy()
    valid = np.isfinite(cvals)
    if not np.any(valid):
        auc = np.nan
        wauc = np.nan
    else:
        auc = float(np.mean(cvals[valid]))

        # weights
        if callable(weight_scheme):
            w = np.array([float(weight_scheme(int(t))) for t in curve_df["t"].to_numpy()], dtype=float)
        elif weight_scheme == "inv":
            w = 1.0 / np.maximum(curve_df["t"].to_numpy().astype(float), 1.0)
        elif weight_scheme == "exp":
            w = np.exp(-exp_alpha * curve_df["t"].to_numpy().astype(float))
        else:
            raise ValueError("weight_scheme must be 'inv', 'exp', or callable")

        w = w[valid]
        c_use = cvals[valid]
        wauc = float(np.sum(w * c_use) / np.sum(w))

    metrics = {
        "t_min": int(t_min),
        "t_max": int(t_max),
        "auc": auc,
        "wauc": wauc,
        "weight_scheme": ("callable" if callable(weight_scheme) else weight_scheme),
        "exp_alpha": exp_alpha if (weight_scheme == "exp") else None,
    }
    return curve_df, metrics

def plot_cindex_curves_and_save(
    dataset,
    model,
    dfs,
    labels,
    result_folder,
    seed_tag="seed409",
    weight_scheme="inv",
    exp_alpha=0.05,
    max_prefix=None,
):
    """
    For each df in dfs, compute cumulative C-index curve and AUC/WAUC.
    Saves:
      - PDF with curves
      - CSV summary with auc/wauc per model
    """
    os.makedirs(result_folder, exist_ok=True)

    all_curves = []
    rows = []

    for df, lab in zip(dfs, labels):
        curve_df, metrics = cumulative_cindex_and_auc(
            df,
            prefix_col="Prefix_length",
            y_col="GroundTruth",
            yhat_col="Prediction",
            max_prefix=max_prefix,
            weight_scheme=weight_scheme,
            exp_alpha=exp_alpha,
        )
        all_curves.append((lab, curve_df))
        rows.append({
            "model_label": lab,
            "auc": metrics["auc"],
            "wauc": metrics["wauc"],
            "t_min": metrics["t_min"],
            "t_max": metrics["t_max"],
            "weight_scheme": metrics["weight_scheme"],
            "exp_alpha": metrics["exp_alpha"],
        })

    summary_df = pd.DataFrame(rows).sort_values("model_label")

    # ---- Plot ----
    plt.figure(figsize=(9, 5))
    for lab, cdf in all_curves:
        plt.plot(cdf["t"].to_numpy(), cdf["c_index"].to_numpy(), label=lab)

    plt.xlabel("Cumulative prefix horizon t (uses prefixes ≤ t)")
    plt.ylabel("C-index")
    plt.title(f"Cumulative C-index curves ({dataset} / {model})")
    plt.ylim(0.0, 1.0)
    plt.legend()

    out_pdf = os.path.join(
        result_folder,
        f"{dataset}_{model}_cindex_cumulative_{seed_tag}.pdf"
    )
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()
    print(f"[OK] Saved C-index curve PDF: {out_pdf}")

    out_csv = os.path.join(
        result_folder,
        f"{dataset}_{model}_cindex_summary_{seed_tag}.csv"
    )
    summary_df.to_csv(out_csv, index=False)
    print(f"[OK] Saved C-index summary CSV: {out_csv}")

    return summary_df, all_curves

def residual_plot(dataset, model, df_lst, loaded_labels, result_folder, seed_tag="seed409"):
    """
    Residuals vs GroundTruth plot for tail-focused error structure.
    Same signature as your KDE function.
    Residual = Prediction - GroundTruth
    """
    gt = df_lst[0]["GroundTruth"].to_numpy()

    plt.figure(figsize=(9, 5))
    for df, lab in zip(df_lst, loaded_labels):
        preds = df["Prediction"].to_numpy()
        mask = ~np.isnan(gt) & ~np.isnan(preds)
        x = gt[mask]
        r = preds[mask] - x
        # Use hexbin for large datasets, scatter otherwise
        if len(x) > 5000:
            plt.hexbin(x, r, gridsize=50, mincnt=1, alpha=0.6)
        else:
            plt.scatter(x, r, s=10, alpha=0.5, label=f"{lab}")
        # Add running median trend line (robust bias indicator)
        order = np.argsort(x)
        xs = x[order]
        rs = r[order]
        bins = np.array_split(np.arange(len(xs)), 40)
        xm = [xs[b].mean() for b in bins if len(b) > 0]
        rm = [np.median(rs[b]) for b in bins if len(b) > 0]
        plt.plot(xm, rm, linewidth=2, label=f"{lab} median")
    # zero-error line
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("GroundTruth")
    plt.ylabel("Residual (Prediction − GroundTruth)")
    plt.title(f"Residuals vs GroundTruth ({dataset} / {model})")
    plt.legend()
    out_pdf = os.path.join(
        result_folder,
        f"{dataset}_{model}_residuals_vs_gt_{seed_tag}.pdf"
    )
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()
    print(f"[OK] Saved residual plot PDF: {out_pdf}")

def distribution_QQ_plot(dataset, model, df_lst, loaded_labels, result_folder, seed_tag="seed409"):
    """
    Quantile–Quantile plot: Predictions vs GroundTruth.
    Same signature as your KDE function.
    """
    gt = df_lst[0]["GroundTruth"].to_numpy()
    gt = gt[~np.isnan(gt)]
    # quantile grid
    q = np.linspace(0.001, 0.999, 400)
    gt_q = np.quantile(gt, q)
    plt.figure(figsize=(6, 6))
    # diagonal reference
    gmin, gmax = gt_q.min(), gt_q.max()
    plt.plot([gmin, gmax], [gmin, gmax], linestyle="--", linewidth=1, label="Ideal")
    for df, lab in zip(df_lst, loaded_labels):
        preds = df["Prediction"].to_numpy()
        preds = preds[~np.isnan(preds)]
        pred_q = np.quantile(preds, q)
        plt.plot(gt_q, pred_q, linewidth=2, label=f"{lab}")
    plt.xlabel("GroundTruth quantiles")
    plt.ylabel("Prediction quantiles")
    plt.title(f"QQ Plot ({dataset} / {model})")
    plt.legend()
    out_pdf = os.path.join(
        result_folder,
        f"{dataset}_{model}_qq_{seed_tag}.pdf"
    )
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()
    print(f"[OK] Saved QQ plot PDF: {out_pdf}")
       
def distribution_plot_cdf(dataset, model, df_lst, loaded_labels, result_folder, seed_tag="seed409"):
    gt = df_lst[0]["GroundTruth"].to_numpy()

    all_vals = [gt] + [df["Prediction"].to_numpy() for df in df_lst]
    all_vals = np.concatenate(all_vals)
    xmin, xmax = np.nanmin(all_vals), np.nanmax(all_vals)

    plt.figure(figsize=(9, 5))
    # ---------- ECDF helper ----------
    def ecdf(arr):
        arr = np.sort(arr[~np.isnan(arr)])
        y = np.arange(1, len(arr) + 1) / len(arr)
        return arr, y
    # ---------- Ground Truth ----------
    x_gt, y_gt = ecdf(gt)
    plt.step(x_gt, y_gt, where="post", label="GroundTruth")
    # ---------- Predictions ----------
    for df, lab in zip(df_lst, loaded_labels):
        preds = df["Prediction"].to_numpy()
        x_p, y_p = ecdf(preds)
        plt.step(x_p, y_p, where="post", label=f"Pred: {lab}")
    plt.xlim(xmin, xmax)
    plt.ylim(0, 1)
    plt.xlabel("Value")
    plt.ylabel("Cumulative Probability")
    plt.title(f"GroundTruth vs Predictions CDF ({dataset} / {model})")
    plt.legend()
    out_pdf = os.path.join(
        result_folder,
        f"{dataset}_{model}_gt_vs_preds_cdf_{seed_tag}.pdf"
    )
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()
    print(f"[OK] Saved CDF PDF: {out_pdf}")
    
def distribution_calibration_plot(dataset, model, df_lst, loaded_labels, result_folder, seed_tag="seed409"):
    """
    Calibration in quantile bins (reliability for regression marginals).
    Same signature as your KDE function.
    Bins are defined by GroundTruth quantiles on the (available) test set.
    For each bin: plot mean(GT) vs mean(Pred). Ideal behavior follows y=x.
    """
    # --- settings (edit if you want) ---
    n_bins = 10  # deciles
    use_median = False  # set True for median calibration (more robust to outliers)
    gt_all = df_lst[0]["GroundTruth"].to_numpy()
    # We'll compute bin edges on the available (non-NaN) GT values
    gt_valid = gt_all[~np.isnan(gt_all)]
    if len(gt_valid) < n_bins * 5:
        print(f"[WARN] Very few GT samples ({len(gt_valid)}) for {n_bins} bins; plot may be noisy.")
    # Quantile bin edges (ensure strictly increasing)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(gt_valid, qs)
    # Make edges strictly increasing to avoid issues when many identical values exist
    edges = np.unique(edges)
    if len(edges) < 3:
        raise ValueError("Not enough unique GroundTruth values to form quantile bins.")
    # Bin indices using the edges; last edge inclusive
    # We'll use np.digitize with right=True so that x==edge goes to the lower bin.
    def assign_bins(x, bin_edges):
        # bins are 0..(K-2) where K=len(bin_edges)
        b = np.digitize(x, bin_edges[1:-1], right=True)
        return b
    plt.figure(figsize=(6.5, 6.5))
    # Ideal calibration line range (based on GT)
    xmin, xmax = np.nanmin(gt_all), np.nanmax(gt_all)
    plt.plot([xmin, xmax], [xmin, xmax], linestyle="--", linewidth=1, label="Ideal")
    for df, lab in zip(df_lst, loaded_labels):
        preds_all = df["Prediction"].to_numpy()
        mask = ~np.isnan(gt_all) & ~np.isnan(preds_all)
        gt = gt_all[mask]
        preds = preds_all[mask]
        bins = assign_bins(gt, edges)
        K = len(edges) - 1  # number of bins
        x_centers = []
        y_centers = []
        x_err = []
        y_err = []
        for b in range(K):
            idx = np.where(bins == b)[0]
            if len(idx) == 0:
                continue
            gt_b = gt[idx]
            pr_b = preds[idx]
            if use_median:
                x0 = np.median(gt_b)
                y0 = np.median(pr_b)
            else:
                x0 = np.mean(gt_b)
                y0 = np.mean(pr_b)
            x_centers.append(x0)
            y_centers.append(y0)
            # Error bars: within-bin spread (1 std). Comment out if you want cleaner plot.
            x_err.append(np.std(gt_b))
            y_err.append(np.std(pr_b))
        x_centers = np.array(x_centers)
        y_centers = np.array(y_centers)
        x_err = np.array(x_err)
        y_err = np.array(y_err)
        plt.errorbar(
            x_centers, y_centers,
            xerr=x_err, yerr=y_err,
            fmt="o-", capsize=3, linewidth=2,
            label=f"{lab}"
        )
    plt.xlabel("GroundTruth (binned by quantiles; mean per bin)")
    plt.ylabel("Prediction (mean per bin)")
    plt.title(f"Quantile-bin Calibration ({dataset} / {model})")
    plt.legend()
    out_pdf = os.path.join(
        result_folder,
        f"{dataset}_{model}_quantile_bin_calibration_{seed_tag}.pdf"
    )
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()
    print(f"[OK] Saved quantile-bin calibration PDF: {out_pdf}")
    
def distribution_plot(dataset, model, df_lst, loaded_labels, result_folder,
                      seed_tag="seed409", n_bins = 20):
    """
    Histogram with shared bins + transparency.
    """
    gt = df_lst[0]["GroundTruth"].to_numpy()
    # collect all values to build shared bins
    all_vals = [gt] + [df["Prediction"].to_numpy() for df in df_lst]
    all_vals = np.concatenate(all_vals)
    all_vals = all_vals[~np.isnan(all_vals)]
    xmin, xmax = np.min(all_vals), np.max(all_vals)
    # choose bin count 
    n_bins = n_bins
    bins = np.linspace(xmin, xmax, n_bins + 1)
    plt.figure(figsize=(9, 5))
    # --- Ground Truth ---
    gt_clean = gt[~np.isnan(gt)]
    plt.hist(
        gt_clean,
        bins=bins,
        density=True,
        alpha=0.35,
        label="GroundTruth"
    )
    # --- Predictions ---
    for df, lab in zip(df_lst, loaded_labels):
        preds = df["Prediction"].to_numpy()
        preds = preds[~np.isnan(preds)]
        plt.hist(
            preds,
            bins=bins,
            density=True,
            alpha=0.35,
            label=f"Pred: {lab}"
        )
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title(f"GroundTruth vs Predictions Histogram ({dataset} / {model})")
    plt.legend()
    out_pdf = os.path.join(
        result_folder,
        f"{dataset}_{model}_gt_vs_preds_hist_{seed_tag}.pdf"
    )
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()
    print(f"[OK] Saved histogram PDF: {out_pdf}")

 
def distribution_kde_plot3(
        dataset, model, df_lst, loaded_labels, result_folder,
        seed_tag="seed409", bw=0.6, tail_focus=True, focus_ratio=0.10,
        tail_computation=False):
    gt = df_lst[0]["GroundTruth"].to_numpy()
    gt_clean = gt[~np.isnan(gt)]
    all_vals = [gt] + [df["Prediction"].to_numpy() for df in df_lst]
    all_vals = np.concatenate(all_vals)
    xmin, xmax = np.nanmin(all_vals), np.nanmax(all_vals)
    x = np.linspace(xmin, xmax, 1000)
    plt.figure(figsize=(9, 5))
    # KDE curves
    kde_gt = gaussian_kde(gt_clean, bw_method=bw)
    plt.plot(x, kde_gt(x), label="GroundTruth")
    # Collect tail-mass stats for annotation (computed if tail_focus)
    tail_stats = []
    for df, lab in zip(df_lst, loaded_labels):
        preds = df["Prediction"].to_numpy()
        preds_clean = preds[~np.isnan(preds)]
        kde_p = gaussian_kde(preds_clean, bw_method=bw)
        plt.plot(x, kde_p(x), label=f"Pred: {lab}")
        tail_stats.append((lab, preds_clean))
    plt.xlabel("Remaining Time (Days)", fontsize=16, fontweight='bold')
    plt.ylabel("Density", fontsize=16, fontweight='bold')
    if tail_focus:
        # Tail defined from GT quantile
        q_tail = np.nanquantile(gt_clean, 1 - focus_ratio)
        # Visual tail markers
        plt.axvline(q_tail, linestyle="--", linewidth=1)
        plt.axvspan(q_tail, xmax, alpha=0.08)
        # Empirical tail mass for GT and each method
        if tail_computation:
            gt_tail_mass = np.mean(gt_clean >= q_tail)
            lines = [f"Tail region: y ≥ q{int((1-focus_ratio)*100)}(GT) = {q_tail:.2f}",
                     f"GT tail mass: {gt_tail_mass*100:.1f}%"]
            for lab, preds_clean in tail_stats:
                pred_tail_mass = np.mean(preds_clean >= q_tail)
                lines.append(f"{lab} tail mass: {pred_tail_mass*100:.1f}%")
            # Put annotation box in axes coords (stable placement)
            plt.text(
                0.98, 0.80,
                "\n".join(lines),
                transform=plt.gca().transAxes,
                ha="right", va="top",
                fontsize=14,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="lightgray")
                )
        plt.text(q_tail + 1, plt.ylim()[1] * 0.70, f"Top {int(focus_ratio*100)}% tail", fontsize=16,fontweight='bold')        
    #plt.title(f"GroundTruth vs Predictions distribution ({dataset} / {model})")
    plt.legend(prop={'size': 16, 'weight': 'bold'})
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    out_pdf = os.path.join(result_folder, f"{dataset}_{model}_gt_vs_preds_kde_{seed_tag}.pdf")
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()
    print(f"[OK] Saved KDE PDF: {out_pdf}")
    
def distribution_kde_plot2(
        dataset, model, df_lst, loaded_labels, result_folder,
        seed_tag="seed409", bw=0.4, tail_focus=True, focus_ratio=0.10,
        tail_computation=False):
    gt = df_lst[0]["GroundTruth"].to_numpy()
    gt_clean = gt[~np.isnan(gt)]

    # compute GT mode
    gt_mean = int(np.round(np.mean(gt_clean), 0))

    all_vals = [gt] + [df["Prediction"].to_numpy() for df in df_lst]
    all_vals = np.concatenate(all_vals)
    xmin, xmax = np.nanmin(all_vals), np.nanmax(all_vals)
    x = np.linspace(xmin, xmax, 1000)
    plt.figure(figsize=(9, 5))

    # KDE curves — CHANGED LABEL to include mean
    kde_gt = gaussian_kde(gt_clean, bw_method=bw)
    plt.plot(x, kde_gt(x), label=f"GroundTruth (mean={gt_mean})")

    # OPTIONAL but useful: still draw the mean line (legend will describe it)
    plt.axvline(gt_mean, linestyle=":", linewidth=1.5)

    tail_stats = []
    for df, lab in zip(df_lst, loaded_labels):
        preds = df["Prediction"].to_numpy()
        preds_clean = preds[~np.isnan(preds)]
        kde_p = gaussian_kde(preds_clean, bw_method=bw)
        plt.plot(x, kde_p(x), label=f"Prediction ({lab})")
        tail_stats.append((lab, preds_clean))

    plt.xlabel("Remaining Time (Days)", fontsize=16, fontweight='bold')
    plt.ylabel("Density", fontsize=16, fontweight='bold')

    if tail_focus:
        q_tail = np.nanquantile(gt_clean, 1 - focus_ratio)
        plt.axvline(q_tail, linestyle="--", linewidth=1)
        plt.axvspan(q_tail, xmax, alpha=0.08)

        if tail_computation:
            gt_tail_mass = np.mean(gt_clean >= q_tail)
            lines = [f"Tail region: y ≥ q{int((1-focus_ratio)*100)}(GT) = {q_tail:.2f}",
                     f"GT tail mass: {gt_tail_mass*100:.1f}%"]
            for lab, preds_clean in tail_stats:
                pred_tail_mass = np.mean(preds_clean >= q_tail)
                lines.append(f"{lab} tail mass: {pred_tail_mass*100:.1f}%")
            plt.text(
                0.98, 0.80,
                "\n".join(lines),
                transform=plt.gca().transAxes,
                ha="right", va="top",
                fontsize=14,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="lightgray")
            )

        plt.text(q_tail + 1, plt.ylim()[1] * 0.70,
                 f"Top {int(focus_ratio*100)}% tail",
                 fontsize=16, fontweight='bold')

    plt.legend(prop={'size': 16, 'weight': 'bold'})
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')

    out_pdf = os.path.join(result_folder, f"{dataset}_{model}_gt_vs_preds_kde_{seed_tag}.pdf")
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()
    print(f"[OK] Saved KDE PDF: {out_pdf}")
    

def distribution_kde_plot(
        dataset, model, df_lst, loaded_labels, result_folder,
        seed_tag="seed409", bw=0.4, tail_focus=True, focus_ratio=0.10,
        tail_computation=False, center_stat="mean",
        center_grid_n=5000, peak_prominence=0.05, max_peaks=5):
    """
    Plots KDEs for GT and predictions and prints summary stats.
    - center_stat:
        * "mean": uses mean(GT) as the center line/label
        * "mode": uses smoothed (KDE) mode of GT as the center line/label
    - Also detects and prints multiple KDE peaks (modes) for each prediction
      using scipy.signal.find_peaks on the KDE curve.
    - peak_prominence is relative to the curve's max height (e.g., 0.05 = 5% of max).
    """
    def kde_peaks(values, bw_method, xmin, xmax, n_grid, rel_prom, max_peaks_keep):
        """Return sorted peak x-positions and peak heights from KDE(values)."""
        v = values[~np.isnan(values)]
        if v.size < 3:
            return [], []
        kde = gaussian_kde(v, bw_method=bw_method)
        xg = np.linspace(xmin, xmax, n_grid)
        yg = kde(xg)

        # prominence threshold relative to max density
        prom_abs = rel_prom * float(np.max(yg)) if np.max(yg) > 0 else 0.0
        peaks, props = find_peaks(yg, prominence=prom_abs)

        if peaks.size == 0:
            return [], []

        px = xg[peaks]
        py = yg[peaks]

        # keep the most prominent/highest peaks (sort by height descending)
        order = np.argsort(py)[::-1]
        px, py = px[order], py[order]

        if max_peaks_keep is not None:
            px, py = px[:max_peaks_keep], py[:max_peaks_keep]

        # for readability in reports: sort modes by x position
        order_x = np.argsort(px)
        px, py = px[order_x], py[order_x]

        return [float(v) for v in px], [float(v) for v in py]

    # --- Ground truth ---
    gt = df_lst[0]["GroundTruth"].to_numpy()
    gt_clean = gt[~np.isnan(gt)]

    # global x-range for consistent peak finding across curves
    all_vals = [gt] + [df["Prediction"].to_numpy() for df in df_lst]
    all_vals = np.concatenate(all_vals)
    xmin, xmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

    # center statistic for GT
    if center_stat == "mode":
        gt_modes, _ = kde_peaks(
            gt_clean, bw_method=bw, xmin=xmin, xmax=xmax,
            n_grid=center_grid_n, rel_prom=peak_prominence, max_peaks_keep=1
        )
        center_val = gt_modes[0] if gt_modes else float(np.mean(gt_clean))
        center_name = "kde_mode"
    else:
        center_val = float(np.mean(gt_clean))
        center_name = "mean"

    # --- Plot setup ---
    x = np.linspace(xmin, xmax, 1000)
    plt.figure(figsize=(9, 5))

    kde_gt = gaussian_kde(gt_clean, bw_method=bw)
    plt.plot(x, kde_gt(x), label=f"GroundTruth ({center_name}={center_val:.1f})")
    plt.axvline(center_val, linestyle=":", linewidth=1.5)

    # --- Peak reporting header ---
    print("\n" + "=" * 70)
    print(f"KDE peak report | dataset={dataset} | model={model} | bw={bw} | "
          f"prominence={peak_prominence} | max_peaks={max_peaks}")
    print(f"GT {center_name}: {center_val:.1f}")
    print("=" * 70)

    # also report GT multi-peaks if you want (kept consistent with request)
    gt_all_modes, _ = kde_peaks(
        gt_clean, bw_method=bw, xmin=xmin, xmax=xmax,
        n_grid=center_grid_n, rel_prom=peak_prominence, max_peaks_keep=max_peaks
    )
    if gt_all_modes:
        print("GroundTruth KDE peaks (modes): " + ", ".join(f"{m:.1f}" for m in gt_all_modes))
    else:
        print("GroundTruth KDE peaks (modes): none detected (check bw/prominence).")

    # --- Predictions: plot + multi-peak extraction ---
    tail_stats = []
    for df, lab in zip(df_lst, loaded_labels):
        preds = df["Prediction"].to_numpy()
        preds_clean = preds[~np.isnan(preds)]

        kde_p = gaussian_kde(preds_clean, bw_method=bw)
        plt.plot(x, kde_p(x), label=f"Prediction ({lab})")
        tail_stats.append((lab, preds_clean))

        pred_modes, _ = kde_peaks(
            preds_clean, bw_method=bw, xmin=xmin, xmax=xmax,
            n_grid=center_grid_n, rel_prom=peak_prominence, max_peaks_keep=max_peaks
        )
        if pred_modes:
            print(f"{lab} KDE peaks (modes): " + ", ".join(f"{m:.1f}" for m in pred_modes))
        else:
            print(f"{lab} KDE peaks (modes): none detected (check bw/prominence).")

    # --- Labels / tail focus ---
    plt.xlabel("Remaining Time (Days)", fontsize=16, fontweight='bold')
    plt.ylabel("Density", fontsize=16, fontweight='bold')

    if tail_focus:
        q_tail = np.nanquantile(gt_clean, 1 - focus_ratio)
        plt.axvline(q_tail, linestyle="--", linewidth=1)
        plt.axvspan(q_tail, xmax, alpha=0.08)

        if tail_computation:
            gt_tail_mass = np.mean(gt_clean >= q_tail)
            lines = [
                f"Tail region: y ≥ q{int((1-focus_ratio)*100)}(GT) = {q_tail:.2f}",
                f"GT tail mass: {gt_tail_mass*100:.1f}%"
            ]
            for lab, preds_clean in tail_stats:
                pred_tail_mass = np.mean(preds_clean >= q_tail)
                lines.append(f"{lab} tail mass: {pred_tail_mass*100:.1f}%")

            plt.text(
                0.98, 0.80,
                "\n".join(lines),
                transform=plt.gca().transAxes,
                ha="right", va="top",
                fontsize=14,
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor="white", alpha=0.85, edgecolor="lightgray")
            )

        plt.text(q_tail + 1, plt.ylim()[1] * 0.70,
                 f"Top {int(focus_ratio*100)}% tail",
                 fontsize=16, fontweight='bold')

    plt.legend(prop={'size': 16, 'weight': 'bold'})
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')

    out_pdf = os.path.join(
        result_folder,
        f"{dataset}_{model}_gt_vs_preds_kde_{center_name}_{seed_tag}.pdf"
    )
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()
    print(f"[OK] Saved KDE PDF: {out_pdf}")


    
def main():
    dataset = "BPIC20PTC"  # HelpDesk BPIC20PTC P2P
    model = "DALSTM"
    IR_lst = ["Vanilla", 'BMSE']
    smooth_lst = ["wos", "wos"]
    labels = ["MAE", 'BMSE']
    #IR_lst = ['Vanilla', 'CSW', 'EAL', 'BMSE', 'SERA']
    #smooth_lst = ['wos', 'wos', 'wos', 'wos', 'wos']
    #labels = ['Vanilla', 'CSW', 'EAL', 'BMSE', 'SERA']
    filter_length = False
    selected_length = 2
    tail_computation = False

    seed_tag = "seed409"
    root_path = os.getcwd()
    result_folder = os.path.join(root_path, "results", "DALSTM", dataset)

    df_lst = []
    loaded_labels = []

    for IR, smooth, lab in zip(IR_lst, smooth_lst, labels):
        name = f"{dataset}_{model}_{IR}_{smooth}_{seed_tag}_inference.csv"
        df_path = os.path.join(result_folder, name)
        if not os.path.exists(df_path):
            print(f"[WARN] Missing file: {df_path} (skipping)")
            continue

        df = pd.read_csv(df_path)
        needed = {"GroundTruth", "Prediction", "Prefix_length"}
        if not needed.issubset(df.columns):
            raise ValueError(f"{df_path} is missing columns: {needed - set(df.columns)}")
            
        if filter_length:
            df = df[df['Prefix_length']==selected_length]

        df_lst.append(df)
        loaded_labels.append(lab)

    if len(df_lst) == 0:
        raise RuntimeError("No CSVs loaded. Check result_folder / filenames.")

    # 1) distribution plot(s)
    #distribution_plot(dataset, model, df_lst, loaded_labels, result_folder, seed_tag=seed_tag)
    distribution_kde_plot(dataset, model, df_lst, loaded_labels, result_folder, seed_tag=seed_tag, center_stat="mode")
    #residual_plot(dataset, model, df_lst, loaded_labels, result_folder, seed_tag=seed_tag)
    #distribution_QQ_plot(dataset, model, df_lst, loaded_labels, result_folder, seed_tag=seed_tag)
    #distribution_plot_cdf(dataset, model, df_lst, loaded_labels, result_folder, seed_tag=seed_tag)
    #distribution_calibration_plot(dataset, model, df_lst, loaded_labels, result_folder, seed_tag=seed_tag)
    if tail_computation:
        # 2) C-index cumulative curves + AUC summaries
        summary_df, _ = plot_cindex_curves_and_save(
            dataset=dataset,
            model=model,
            dfs=df_lst,
            labels=loaded_labels,
            result_folder=result_folder,
            seed_tag=seed_tag,
            weight_scheme="inv",   # early-weighted by 1/t
            exp_alpha=0.05,
            max_prefix=None,       # or set an int (e.g., 50) to cap horizon
            )
        print("\nC-index summary:")
        print(summary_df)


if __name__ == "__main__":
    main()