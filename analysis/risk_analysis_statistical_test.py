# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 12:41:32 2026
@author: kamirel
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import friedmanchisquare
from scipy.stats import studentized_range

import scikit_posthocs as sp


# -----------------------------
# Utility: Kendall's W from Friedman chi-square
# W = chi2 / (n * (k - 1))
# n = number of datasets (blocks), k = number of methods
# -----------------------------
def kendalls_w_from_friedman(chi2: float, n: int, k: int) -> float:
    if n <= 0 or k <= 1:
        return np.nan
    return float(chi2 / (n * (k - 1)))


# -----------------------------
# Compute mean ranks (higher metric is better)
# Ranks per dataset across methods, then averaged.
# -----------------------------
def compute_mean_ranks(pivot: pd.DataFrame) -> pd.Series:
    # pivot: index=Dataset, columns=Method, values=metric (higher is better)
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    return ranks.mean(axis=0).sort_values()  # smaller mean rank = better


# -----------------------------
# Nemenyi Critical Difference
# CD = q_alpha * sqrt(k*(k+1)/(6*n))
# q_alpha = studentized_range.ppf(1-alpha, k, inf)
# -----------------------------
def compute_cd(k: int, n: int, alpha: float = 0.05) -> float:
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf)
    return float(q_alpha * math.sqrt(k * (k + 1) / (6.0 * n)))


# -----------------------------
# Plot a simple Critical Difference diagram
# This is a standard "mean ranks on a line" + CD bar + non-significant groups.
# Grouping heuristic: connect contiguous methods in sorted-by-rank order
# when their rank difference <= CD.
# -----------------------------
def plot_cd_diagram(mean_ranks: pd.Series, cd: float, title: str, out_path: str):
    methods = mean_ranks.index.tolist()
    ranks = mean_ranks.values.astype(float)

    # Sort by mean rank (ascending: best at left)
    order = np.argsort(ranks)
    methods = [methods[i] for i in order]
    ranks = ranks[order]

    plt.figure(figsize=(10, 2.8))
    ax = plt.gca()

    # Axis layout
    ax.set_title(title, fontsize=11)
    ax.set_yticks([])
    ax.set_xlabel("Mean rank (lower is better)")
    ax.set_xlim(min(ranks) - 0.5, max(ranks) + 0.5)

    # Plot method labels at their mean ranks
    y0 = 0.6
    for m, r in zip(methods, ranks):
        ax.plot([r], [y0], marker="o")
        ax.text(r, y0 + 0.08, m, ha="center", va="bottom", fontsize=9, rotation=0)

    # Draw CD bar (top)
    cd_y = 1.15
    left = min(ranks)
    ax.plot([left, left + cd], [cd_y, cd_y], linewidth=2)
    ax.text(left + cd / 2, cd_y + 0.05, f"CD = {cd:.3f}", ha="center", va="bottom", fontsize=9)

    # Draw non-significant groups as horizontal segments (contiguous heuristic)
    # If ranks[j] - ranks[i] <= CD, then i..j can be considered not significantly different in Nemenyi.
    group_y = 0.95
    i = 0
    while i < len(ranks):
        j = i
        while j + 1 < len(ranks) and (ranks[j + 1] - ranks[i]) <= cd:
            j += 1
        if j > i:
            ax.plot([ranks[i], ranks[j]], [group_y, group_y], linewidth=4, solid_capstyle="round")
            group_y -= 0.08
        i += 1

    # Clean up
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# One analysis (one of the 12 settings)
# -----------------------------
def analyze_setting(df: pd.DataFrame, quantile: float, level: float, metric_col: str, alpha: float = 0.05):
    sub = df[(df["Quantile"] == quantile) & (df["Execution_Level"] == level)].copy()

    # Pivot to dataset x method
    pivot = sub.pivot(index="Dataset", columns="Method", values=metric_col).dropna()

    n = pivot.shape[0]      # datasets (blocks)
    k = pivot.shape[1]      # methods

    # Friedman
    stat, p = friedmanchisquare(*[pivot[c].values for c in pivot.columns])
    W = kendalls_w_from_friedman(stat, n, k)

    # Mean ranks
    mean_ranks = compute_mean_ranks(pivot)

    # Nemenyi
    nemenyi = sp.posthoc_nemenyi_friedman(pivot)

    # CD
    cd = compute_cd(k, n, alpha=alpha)

    return {
        "pivot": pivot,
        "friedman_stat": float(stat),
        "friedman_p": float(p),
        "kendalls_w": float(W),
        "mean_ranks": mean_ranks,
        "nemenyi": nemenyi,
        "cd": cd,
        "n_datasets": n,
        "k_methods": k,
    }


def main():
    alpha = 0.05
    root_path = os.getcwd()
    csv_path = os.path.join(root_path, "case_level_risk_metrics_total_duration.csv")

    df = pd.read_csv(csv_path)

    # Sanity
    quantiles = sorted(df["Quantile"].unique())
    levels = sorted(df["Execution_Level"].unique())
    metrics = [("AUC-Average", "AUC"), ("PR-AUC-Average", "PR-AUC")]

    # Output containers
    summary_rows = []
    excel_writer_path = os.path.join(root_path, "nemenyi_matrices.xlsx")
    cd_dir = os.path.join(root_path, "cd_diagrams")
    os.makedirs(cd_dir, exist_ok=True)

    with pd.ExcelWriter(excel_writer_path, engine="openpyxl") as writer:
        for q in [0.9, 0.95]:
            for lvl in [0.25, 0.5, 0.75]:
                for metric_col, metric_name in metrics:
                    res = analyze_setting(df, q, lvl, metric_col, alpha=alpha)

                    # Summary row
                    summary_rows.append({
                        "Quantile": q,
                        "Execution_Level": lvl,
                        "Metric": metric_name,
                        "n_datasets": res["n_datasets"],
                        "k_methods": res["k_methods"],
                        "Friedman_chi2": res["friedman_stat"],
                        "Friedman_p": res["friedman_p"],
                        "Kendalls_W": res["kendalls_w"],
                        "CD(alpha=0.05)": res["cd"],
                    })

                    # Save mean ranks (as sheet)
                    ranks_df = res["mean_ranks"].reset_index()
                    ranks_df.columns = ["Method", "MeanRank"]
                    sheet_base = f"q{q}_l{lvl}_{metric_name}".replace(".", "")
                    ranks_df.to_excel(writer, sheet_name=f"{sheet_base}_ranks", index=False)

                    # Save nemenyi matrix (as sheet)
                    res["nemenyi"].to_excel(writer, sheet_name=f"{sheet_base}_nemenyi")

                    # Plot CD diagram
                    title = f"CD Diagram | q={q} | level={lvl} | {metric_name} (n={res['n_datasets']})"
                    out_png = os.path.join(cd_dir, f"CD_q{q}_lvl{lvl}_{metric_name}.png".replace(".", ""))
                    plot_cd_diagram(res["mean_ranks"], res["cd"], title, out_png)

    # Save Friedman summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("friedman_rank_summary.csv", index=False)

    print("Done.")
    print("Wrote: friedman_rank_summary.csv")
    print(f"Wrote: {excel_writer_path}")
    print(f"Wrote CD diagrams to: {cd_dir}/")


if __name__ == "__main__":
    main()