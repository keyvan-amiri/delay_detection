# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 13:15:19 2026
@author: Keyvan Amiri Elyasi
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde, boxcox
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def get_plot_addresses(model_folder, dataset, technique):
    plot_folder = os.path.join(model_folder, 'plots')
    dist_name = dataset+'_'+technique+'_distribution.pdf'
    dist_path1 = os.path.join(plot_folder, dist_name)
    dist_name = dataset+'_'+technique+'_residuals_vs_target.pdf'
    dist_path2 = os.path.join(plot_folder, dist_name)   
    return (dist_path1,dist_path2)
    
def plot_groundtruth_distributions(
    df, save_path, dataset_name=None, column="GroundTruth", bins=50):
    """
    Plot distribution, log-transform distribution, and Box-Cox transformed distribution
    in three subplots (with KDE overlays) and save to a single PDF.
    Parameters
    ----------
    df : pandas.DataFrame
    save_path : str
        Path to output PDF file.
    dataset_name : str or None
        Used in titles. If None, titles omit dataset prefix.
    column : str
        Column name containing target values.
    bins : int
        Number of histogram bins.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    values = df[column].dropna().astype(float)
    if len(values) == 0:
        raise ValueError("Column contains no valid values")

    prefix = f"{dataset_name}: " if dataset_name else ""
    # ----- Transformations -----
    # Log: safe handling
    if (values <= 0).any():
        log_values = np.log1p(values)
        log_xlabel = "log1p(Value)"
    else:
        log_values = np.log(values)
        log_xlabel = "log(Value)"
    # Box–Cox requires strictly positive values -> shift if needed
    min_v = values.min()
    shift = 0.0
    if min_v <= 0:
        shift = (-min_v) + 1e-6  # ensure strictly positive
    values_for_bc = values + shift
    bc_values, bc_lambda = boxcox(values_for_bc)  # estimate lambda
    # ----- Helper to draw hist + KDE -----
    def hist_with_kde(ax, data, title, xlabel):
        data = np.asarray(data)
        data = data[np.isfinite(data)]
        if data.size == 0:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
            ax.set_title(title)
            return
        # Histogram as density so KDE overlays correctly
        ax.hist(data, bins=bins, density=True, alpha=0.6)
        # KDE overlay
        if data.size >= 2 and np.std(data) > 0:
            kde = gaussian_kde(data)
            x = np.linspace(data.min(), data.max(), 400)
            y = kde(x)
            ax.plot(x, y, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
    # ----- Plot -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    hist_with_kde(
        axes[0],
        values,
        f"{prefix}Remaining time: GroundTruth Distribution",
        "Value (Days)",
    )
    hist_with_kde(
        axes[1],
        log_values,
        f"{prefix}Remaining time: Log Transformed Distribution",
        log_xlabel,
    )
    bc_title = f"{prefix}Remaining time: Box–Cox Distribution (λ={bc_lambda:.3f}"
    if shift != 0.0:
        bc_title += f", shift={shift:.3g}"
    bc_title += ")"
    hist_with_kde(
        axes[2],
        bc_values,
        bc_title,
        "Box–Cox(Value)",
    )
    plt.tight_layout()
    # Save to single PDF
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


def plot_residuals_vs_target(
    df,
    save_path,
    dataset_name=None,
    target_col="GroundTruth",
    residual_col="Absolute_error",
    use_log_target=True,
    alpha=0.35,
    add_binned_trend=True,
    n_bins=25,
):
    """
    Scatter residuals vs target value (optionally also vs log/ log1p target) to diagnose heteroscedasticity.
    Saves to a single-page PDF.
    Parameters
    ----------
    df : pandas.DataFrame
    save_path : str
        Output PDF path.
    dataset_name : str or None
        Prefix in plot titles.
    target_col : str
        Target column name (e.g., GroundTruth).
    residual_col : str
        Residual/absolute error column name (e.g., Absolute_error).
    use_log_target : bool
        If True, creates 2 subplots: residuals vs target AND residuals vs log(target) (or log1p if needed).
        If False, creates 1 subplot: residuals vs target.
    alpha : float
        Point transparency for scatter.
    add_binned_trend : bool
        If True, overlays a binned mean residual trend to make variance patterns obvious.
    n_bins : int
        Number of bins for the binned trend overlay (used if add_binned_trend is True).
    """
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in dataframe")
    if residual_col not in df.columns:
        raise ValueError(f"Column '{residual_col}' not found in dataframe")
    # clean + align
    sub = df[[target_col, residual_col]].dropna().astype(float)
    if len(sub) == 0:
        raise ValueError("No valid rows after dropping NaNs.")
    y = sub[target_col].to_numpy()
    r = sub[residual_col].to_numpy()
    prefix = f"{dataset_name}: " if dataset_name else ""
    def _binned_mean_line(ax, x, y, bins):
        # quantile bins are robust to skew
        edges = np.quantile(x, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            return  # not enough unique edges
        idx = np.digitize(x, edges[1:-1], right=True)
        xs, ys = [], []
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() < 5:
                continue
            xs.append(np.median(x[m]))
            ys.append(np.mean(y[m]))
        if len(xs) >= 2:
            ax.plot(xs, ys, linewidth=2)
    # decide layout
    if use_log_target:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = np.asarray(axes)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        axes = np.asarray([ax])
    # 1) residuals vs target
    ax0 = axes[0]
    ax0.scatter(y, r, alpha=alpha, s=12)
    ax0.set_title(f"{prefix}Residuals vs Target")
    ax0.set_xlabel("Target (Days)")
    ax0.set_ylabel("Absolute error")
    if add_binned_trend:
        _binned_mean_line(ax0, y, r, n_bins)
    # 2) residuals vs log target (optional)
    if use_log_target:
        # safe log transform of target axis
        if (y <= 0).any():
            x2 = np.log1p(y)
            x2_label = "log1p(Target)"
        else:
            x2 = np.log(y)
            x2_label = "log(Target)"
        ax1 = axes[1]
        ax1.scatter(x2, r, alpha=alpha, s=12)
        ax1.set_title(f"{prefix}Residuals vs {x2_label}")
        ax1.set_xlabel(x2_label)
        ax1.set_ylabel("Absolute error")
        if add_binned_trend:
            _binned_mean_line(ax1, x2, r, n_bins)
    plt.tight_layout()
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


def plot_residual_trend_lowess(
    df,
    save_path,
    dataset_name=None,
    target_col="GroundTruth",
    residual_col="Absolute_error",
    use_log_target=True,
    frac=0.2,
    it=0,
    show_spread=True,
    spread_q=0.9,
    spread_bins=40,
):
    """
    Plot ONLY LOWESS trend lines (no scatter) for residuals vs target.
    Optionally also plot vs log/log1p(target). Saves to one-page PDF.

    Lines:
      - LOWESS of mean(|error|) vs x
      - (optional) LOWESS of a high-quantile(|error|) vs x, computed via binning

    Parameters
    ----------
    df : pandas.DataFrame
    save_path : str
        Output PDF file.
    dataset_name : str or None
        Prefix in title.
    target_col : str
        Target column.
    residual_col : str
        Residual column (absolute error).
    use_log_target : bool
        If True, makes 2 subplots: vs target and vs log/log1p target.
    frac : float
        LOWESS smoothing span (0..1). Larger => smoother. Try 0.15–0.35.
    it : int
        LOWESS robust iterations. 0 = standard; 1-3 can reduce outlier influence.
    show_spread : bool
        If True, also plots a smooth "spread" line for a high quantile (e.g. 0.9).
    spread_q : float
        Quantile to summarize spread (e.g., 0.9 or 0.95).
    spread_bins : int
        Number of x-bins used to estimate the quantile curve before LOWESS smoothing.
    """
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in dataframe")
    if residual_col not in df.columns:
        raise ValueError(f"Column '{residual_col}' not found in dataframe")

    sub = df[[target_col, residual_col]].dropna().astype(float)
    if len(sub) == 0:
        raise ValueError("No valid rows after dropping NaNs.")

    y = sub[target_col].to_numpy()
    r = sub[residual_col].to_numpy()

    # Keep finite only
    m = np.isfinite(y) & np.isfinite(r)
    y, r = y[m], r[m]
    if y.size < 5:
        raise ValueError("Not enough finite data points to fit LOWESS.")

    prefix = f"{dataset_name}: " if dataset_name else ""

    def _lowess_line(x, z):
        # sort by x for stable line
        order = np.argsort(x)
        xs = x[order]
        zs = z[order]
        sm = lowess(zs, xs, frac=frac, it=it, return_sorted=True)
        return sm[:, 0], sm[:, 1]

    def _quantile_curve_smoothed(x, z, q, bins):
        # Quantile per bin (quantile-binning by x), then LOWESS smooth that curve
        edges = np.quantile(x, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        if edges.size < 3:
            return None

        centers = []
        qs = []
        for i in range(edges.size - 1):
            lo, hi = edges[i], edges[i + 1]
            mask = (x >= lo) & (x <= hi) if i == edges.size - 2 else (x >= lo) & (x < hi)
            if mask.sum() < 10:
                continue
            centers.append(np.median(x[mask]))
            qs.append(np.quantile(z[mask], q))

        if len(centers) < 5:
            return None

        centers = np.asarray(centers)
        qs = np.asarray(qs)
        return _lowess_line(centers, qs)

    def _plot_panel(ax, x, x_label, title):
        # mean trend
        xl, mean_line = _lowess_line(x, r)
        ax.plot(xl, mean_line, linewidth=2, label="LOWESS mean(|error|)")

        # spread trend (quantile)
        if show_spread:
            res = _quantile_curve_smoothed(x, r, spread_q, spread_bins)
            if res is not None:
                xq, qline = res
                ax.plot(xq, qline, linewidth=2, label=f"LOWESS q{int(spread_q*100)}(|error|)")

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Absolute error")
        ax.legend()

    # layout
    if use_log_target:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes = np.asarray(axes)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        axes = np.asarray([ax])

    # panel 1: raw target
    _plot_panel(
        axes[0],
        y,
        "Target (Days)",
        f"{prefix}Residual trend vs Target",
    )

    # panel 2: log target (optional)
    if use_log_target:
        if (y <= 0).any():
            x2 = np.log1p(y)
            x2_label = "log1p(Target)"
        else:
            x2 = np.log(y)
            x2_label = "log(Target)"

        _plot_panel(
            axes[1],
            x2,
            x2_label,
            f"{prefix}Residual trend vs {x2_label}",
        )

    plt.tight_layout()
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    
def main():
    model = 'DALSTM'
    datasets = ['P2P', 'Sepsis', 'BPIC20DD', 'BPIC20ID', 'BPIC20PTC',
                'BPIC15_1', 'BPIC13I', 'BPIC_2017_W']
    IR_techs = ['Vanilla', 'CSW', 'EAL', 'BMSE', 'SERA']
    root_path = os.getcwd()
    model_folder = result_folder = os.path.join(root_path, 'results', model)
    for dataset in datasets:        
        result_folder = os.path.join(model_folder, dataset)
        for technique in IR_techs:
            csv_name = dataset+'_'+model+'_'+technique+'_wos_seed4012_inference.csv'
            csv_path = os.path.join(result_folder, csv_name)
            res_df = pd.read_csv(csv_path)
            add1, add2 = get_plot_addresses(model_folder, dataset, technique)
            if technique == 'Vanilla':
                plot_groundtruth_distributions(res_df, add1, dataset_name=dataset)
            #plot_residuals_vs_target(res_df, add2, dataset_name=dataset)
            plot_residual_trend_lowess(res_df, add2, dataset_name=dataset)

            
if __name__ == '__main__':
    main()  

