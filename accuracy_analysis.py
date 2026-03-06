# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 12:41:32 2026
@author: kamirel
"""
import os
import pandas as pd
import numpy as np
import torch
import re
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata, studentized_range
import warnings
warnings.filterwarnings("ignore")


def get_inference_results(result_dir=None, labels_name=None, dataset=None, model=None, seed=None):
    quantile_name  = f"{dataset}_{model}_quantile_wos_seed{seed}_inference.csv"
    vanilla_name  = f"{dataset}_{model}_Vanilla_wos_seed{seed}_inference.csv"
    csw_name  = f"{dataset}_{model}_CSW_wos_seed{seed}_inference.csv"
    eal_name  = f"{dataset}_{model}_EAL_wos_seed{seed}_inference.csv"
    bmse_name  = f"{dataset}_{model}_BMSE_wos_seed{seed}_inference.csv"
    sera_name  = f"{dataset}_{model}_SERA_wos_seed{seed}_inference.csv"
    models_name = [vanilla_name, csw_name, eal_name, bmse_name, sera_name]
    df_lst = []
    for name in models_name:
        model_df = pd.read_csv(os.path.join(result_dir, name))
        df_lst.append(model_df)
    quantile_df  = pd.read_csv(os.path.join(result_dir, quantile_name))    
    return quantile_df, labels_name, df_lst

def get_target_stats(temp_dir, model_name, dataset):
    y_train_path = os.path.join(temp_dir, model_name+'_y_train_'+dataset+'.pt')
    y_val_path = os.path.join(temp_dir, model_name+'_y_val_'+dataset+'.pt')
    y_train = torch.load(y_train_path)
    y_val = torch.load(y_val_path)
    y_train_val = torch.cat([y_train, y_val])
    median_rt = float(y_train_val.median())
    quantile_60, quantile_90 = torch.quantile(y_train_val, torch.tensor([0.6, 0.9]))
    q60 = float(quantile_60)
    q90 = float(quantile_90)
    return median_rt, q60, q90

def compute_nmae(df: pd.DataFrame, train_median: float) -> float:
    """
    Computes normalized MAE:
    nMAE = sum(|y_i - yhat_i|) / sum(|y_i - median_train|)    
    df must contain columns: GroundTruth, Prediction (or Absolute_error).
    """
    numerator = (df["GroundTruth"] - df["Prediction"]).abs().sum()
    denominator = (df["GroundTruth"] - train_median).abs().sum()
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)

def compute_region_nmae(df_region: pd.DataFrame, baseline: float) -> float:
    """
    Region-wise normalized MAE using a global baseline.

    nMAE_region =
        mean(|y - y_hat|)_region /
        mean(|y - median_train|)_test
    """
    if len(df_region) == 0:
        return float("nan")

    mae_region = (df_region["GroundTruth"] - df_region["Prediction"]).abs().mean()

    return float(mae_region / baseline)


def split_by_quantiles(df, q60, q90):
    df_le_q60 = df[df["GroundTruth"] <= q60]
    df_q60_q90 = df[(df["GroundTruth"] > q60) & (df["GroundTruth"] <= q90)]
    df_gt_q90 = df[df["GroundTruth"] > q90]    
    return df_le_q60, df_q60_q90, df_gt_q90

def get_nmae_results(labels_name, df_lst, train_median, q60, q90):
    nmae_lst, many_lst, med_lst, few_lst = [], [], [], []
    for name, df in zip(labels_name, df_lst):
        # compute global baseline from full dataset
        baseline = (df["GroundTruth"] - train_median).abs().mean()
        df_le_q60, df_q60_q90, df_gt_q90 = split_by_quantiles(df, q60, q90)
        nmae_lst.append(
            compute_region_nmae(df, baseline)
        )
        many_lst.append(
            compute_region_nmae(df_le_q60, baseline)
        )
        med_lst.append(
            compute_region_nmae(df_q60_q90, baseline)
        )
        few_lst.append(
            compute_region_nmae(df_gt_q90, baseline)
        )
    return (nmae_lst, many_lst, med_lst, few_lst)

def _mean_std(values, ddof=0):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=ddof))

def aggregate_nmae_over_seeds_and_approaches(
        datasets, labels_name, seeds, result_path, temp_path, model_name, ddof=0):
    """
    Returns a dataframe with one row per (Dataset, Approach),
    aggregating mean/std over seeds for:
    - overall nMAE
    - many/medium/few splits
    """
    # metrics[(dataset, approach)] = dict of lists over seeds
    metrics = {}

    for dataset in datasets:
        result_dir = os.path.join(result_path, dataset)
        temp_dir = os.path.join(temp_path, dataset)

        median_rt, q60, q90 = get_target_stats(temp_dir, model_name, dataset)

        for seed in seeds:
            quantile_df, labels_name, df_lst = get_inference_results(
                result_dir=result_dir,
                labels_name=labels_name,
                dataset=dataset,
                model=model_name,
                seed=seed,
            )

            nmae_lst, many_lst, med_lst, few_lst = get_nmae_results(
                labels_name, df_lst, median_rt, q60, q90
            )

            # Ensure we can index by approach
            if not (len(labels_name) == len(nmae_lst) == len(many_lst) == len(med_lst) == len(few_lst)):
                raise ValueError(
                    f"Length mismatch for dataset={dataset}, seed={seed}: "
                    f"len(labels_name)={len(labels_name)}, len(nmae_lst)={len(nmae_lst)}, "
                    f"len(many_lst)={len(many_lst)}, len(med_lst)={len(med_lst)}, len(few_lst)={len(few_lst)}"
                )

            for i, approach in enumerate(labels_name):
                key = (dataset, approach)
                if key not in metrics:
                    metrics[key] = {
                        "all": [],
                        "many": [],
                        "medium": [],
                        "few": [],
                    }

                metrics[key]["all"].append(float(nmae_lst[i]))
                metrics[key]["many"].append(float(many_lst[i]))
                metrics[key]["medium"].append(float(med_lst[i]))
                metrics[key]["few"].append(float(few_lst[i]))

    # Build final rows
    rows = []
    for (dataset, approach), d in metrics.items():
        nMAE_avg, nMAE_std = _mean_std(d["all"], ddof=ddof)
        nMAE_many_avg, nMAE_many_std = _mean_std(d["many"], ddof=ddof)
        nMAE_medium_avg, nMAE_medium_std = _mean_std(d["medium"], ddof=ddof)
        nMAE_few_avg, nMAE_few_std = _mean_std(d["few"], ddof=ddof)

        rows.append({
            "Dataset": dataset,
            "Approach": approach,
            "nMAE_avg": nMAE_avg,
            "nMAE_std": nMAE_std,
            "nMAE_many_avg": nMAE_many_avg,
            "nMAE_many_std": nMAE_many_std,
            "nMAE_medium_avg": nMAE_medium_avg,
            "nMAE_medium_std": nMAE_medium_std,
            "nMAE_few_avg": nMAE_few_avg,
            "nMAE_few_std": nMAE_few_std,
        })

    return pd.DataFrame(rows).sort_values(["Dataset", "Approach"]).reset_index(drop=True)

def save_nmae_latex_table_fixed_order(
    results_df: pd.DataFrame,
    datasets,
    labels_name,
    out_path: str = "nmae_table.txt",
    caption: str = "Results DALSTM",
    label: str = "tab:dalstm_nmae_results",
    metric_title: str = r"$\text{nMAE}$ $\downarrow$",
    decimals: int = 2,
    bold_best: bool = True,
    bold_ties: bool = True,
) -> str:
    """
    Creates a LaTeX table like your example but using nMAE mean/std columns.
    Enforces:
      - dataset order = `datasets`
      - approach order = `labels_name` (Vanilla first)
    Saves LaTeX to `out_path` and returns the LaTeX string.

    Required columns in results_df:
      Dataset, Approach,
      nMAE_avg, nMAE_std,
      nMAE_many_avg, nMAE_many_std,
      nMAE_medium_avg, nMAE_medium_std,
      nMAE_few_avg, nMAE_few_std
    """
    required = [
        "Dataset","Approach",
        "nMAE_avg","nMAE_std",
        "nMAE_many_avg","nMAE_many_std",
        "nMAE_medium_avg","nMAE_medium_std",
        "nMAE_few_avg","nMAE_few_std",
    ]
    missing = [c for c in required if c not in results_df.columns]
    if missing:
        raise ValueError(f"results_df missing required columns: {missing}")

    # Ensure Vanilla is first even if user passes a different order
    labels_name = list(labels_name)
    if "Vanilla" in labels_name:
        labels_name = ["Vanilla"] + [x for x in labels_name if x != "Vanilla"]

    df = results_df.copy()

    # Keep only requested datasets/approaches (optional; comment out if you want all)
    df = df[df["Dataset"].isin(datasets) & df["Approach"].isin(labels_name)].copy()

    # Apply fixed ordering
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=datasets, ordered=True)
    df["Approach"] = pd.Categorical(df["Approach"], categories=labels_name, ordered=True)
    df = df.sort_values(["Dataset", "Approach"]).reset_index(drop=True)

    # Escape LaTeX special chars
    def esc(s: str) -> str:
        s = str(s)
        s = s.replace("\\", r"\textbackslash{}")
        s = re.sub(r"([&_#%${}])", r"\\\1", s)
        s = s.replace("^", r"\^{}").replace("~", r"\~{}")
        return s

    def fmt(mu, sd, do_bold=False) -> str:
        if pd.isna(mu) or pd.isna(sd):
            cell = "--"
        else:
            cell = f"{float(mu):.{decimals}f} ({float(sd):.{decimals}f})"
        return rf"\textbf{{{cell}}}" if do_bold and cell != "--" else cell

    # Determine best (minimum) per dataset for each split
    cols_avg = ["nMAE_avg", "nMAE_many_avg", "nMAE_medium_avg", "nMAE_few_avg"]
    best = {c: {} for c in cols_avg}  # best[col][dataset] = set(approaches)

    if bold_best:
        for ds in datasets:
            dsub = df[df["Dataset"] == ds]
            if dsub.empty:
                continue
            for c in cols_avg:
                vals = dsub[c].astype(float)
                minv = vals.min()
                if pd.isna(minv):
                    continue
                if bold_ties:
                    winners = set(dsub.loc[vals == minv, "Approach"].astype(str).tolist())
                else:
                    winners = {str(dsub.loc[vals.idxmin(), "Approach"])}
                best[c][ds] = winners

    # Build LaTeX
    lines = []
    lines += [r"\begin{table}[htbp]"]
    lines += [r"    \centering"]
    lines += [f"    \\caption{{{esc(caption)}}}"]
    lines += [r"    \setlength{\tabcolsep}{4pt}"]
    lines += [r"    % \scriptsize"]
    lines += [r"    \renewcommand{\arraystretch}{1.2}"]
    lines += [r"    \scriptsize"]
    lines += [r"    \begin{tabular}{ccrrrr}"]
    lines += [r"        \toprule"]
    lines += [r"        \multirow{2}{*}{Log} & \multirow{2}{*}{IR} & \multicolumn{4}{c}{" + metric_title + r"} \\"]
    lines += [r"        \cline{3-6}"]
    lines += [r"         & & All & Many & Med. & Few \\"]
    lines += [r"        \midrule"]

    for dsi, ds in enumerate(datasets):
        dsub = df[df["Dataset"] == ds]
        if dsub.empty:
            continue
        nrows = len(dsub)

        for ri, row in enumerate(dsub.itertuples(index=False)):
            approach = str(row.Approach)
            left = (
                rf"        \multirow{{{nrows}}}{{*}}{{\rotatebox{{90}}{{{esc(ds)}}}}}"
                if ri == 0 else
                r"        "
            )

            b_all  = bold_best and (approach in best["nMAE_avg"].get(ds, set()))
            b_many = bold_best and (approach in best["nMAE_many_avg"].get(ds, set()))
            b_med  = bold_best and (approach in best["nMAE_medium_avg"].get(ds, set()))
            b_few  = bold_best and (approach in best["nMAE_few_avg"].get(ds, set()))

            cell_all  = fmt(row.nMAE_avg,        row.nMAE_std,        b_all)
            cell_many = fmt(row.nMAE_many_avg,   row.nMAE_many_std,   b_many)
            cell_med  = fmt(row.nMAE_medium_avg, row.nMAE_medium_std, b_med)
            cell_few  = fmt(row.nMAE_few_avg,    row.nMAE_few_std,    b_few)

            lines += [f"{left} & {esc(approach)} & {cell_all} & {cell_many} & {cell_med} & {cell_few} \\\\"]

            # add dashed line after Vanilla row (first row) if more approaches exist
            if ri == 0 and nrows > 1:
                lines += [r"        \noalign{\vskip 1mm}"]
                lines += [r"        \cdashline{2-6}"]
                lines += [r"        \noalign{\vskip 1mm}"]

        if dsi != len(datasets) - 1:
            lines += [""]
            lines += [r"        \midrule"]

    lines += [r"        \bottomrule"]
    lines += [r"    \end{tabular}"]
    lines += [f"    \\label{{{esc(label)}}}"]
    lines += [r"\end{table}"]

    latex_str = "\n".join(lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    return latex_str


def save_nmae_boxplot_pdf(
    results_df: pd.DataFrame,
    split: str = "all",
    out_pdf: str = "nmae_boxplot.pdf",
    approach_order=None,
    show_points: bool = True,
):
    """
    Boxplot comparing IR approaches across datasets.

    split:
        "all" | "many" | "medium" | "few"

    approach_order:
        list of approaches to include AND their order.
        If None → all approaches are used.
    """

    col_map = {
        "all": "nMAE_avg",
        "many": "nMAE_many_avg",
        "medium": "nMAE_medium_avg",
        "few": "nMAE_few_avg",
    }

    if split not in col_map:
        raise ValueError(f"split must be one of {list(col_map.keys())}")

    value_col = col_map[split]

    df = results_df[["Dataset", "Approach", value_col]].dropna().copy()

    # If user specifies approaches → filter dataframe
    if approach_order is not None:
        df = df[df["Approach"].isin(approach_order)].copy()
    else:
        approach_order = sorted(df["Approach"].unique())

    # Ensure order
    df["Approach"] = pd.Categorical(df["Approach"], categories=approach_order, ordered=True)
    df = df.sort_values(["Approach", "Dataset"])

    data = [
        df.loc[df["Approach"] == a, value_col].astype(float).values
        for a in approach_order
    ]

    plt.figure(figsize=(max(6, 1.2 * len(approach_order)), 4.5))

    plt.boxplot(
        data,
        labels=approach_order,
        whis=1.5,
        showfliers=True,
    )

    if show_points:
        for i, a in enumerate(approach_order, start=1):
            ys = df.loc[df["Approach"] == a, value_col].values
            xs = [i + (j - (len(ys) - 1) / 2) * 0.03 for j in range(len(ys))]
            plt.scatter(xs, ys, s=18)

    plt.ylabel(f"nMAE ({split})")
    plt.xlabel("IR approach")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(out_pdf, format="pdf")
    plt.close()

    return out_pdf

def friedman_nemenyi_from_results(
    results_df: pd.DataFrame,
    approaches: list,
    mode: str = "all",   # "all" | "many" | "medium" | "few"
):
    """
    Runs Friedman test (blocked by Dataset) and post-hoc Nemenyi using the
    per-dataset mean nMAE values in results_df (the *_avg columns).

    Returns:
      - friedman: dict(statistic, pvalue, N, k)
      - ranks: pd.Series (average rank per approach; lower rank = better)
      - nemenyi_pvals: pd.DataFrame (pairwise p-values)
      - data_matrix: pd.DataFrame (datasets x approaches values used)
    """
    col_map = {
        "all": "nMAE_avg",
        "many": "nMAE_many_avg",
        "medium": "nMAE_medium_avg",
        "few": "nMAE_few_avg",
    }
    if mode not in col_map:
        raise ValueError(f"mode must be one of {list(col_map.keys())}, got {mode!r}")
    value_col = col_map[mode]

    # Build dataset x approach matrix of means
    df = results_df[["Dataset", "Approach", value_col]].copy()
    df = df[df["Approach"].isin(approaches)]
    mat = df.pivot_table(index="Dataset", columns="Approach", values=value_col, aggfunc="mean")

    # Keep only complete blocks (datasets that have all requested approaches)
    mat = mat.dropna(axis=0, how="any")
    if mat.shape[0] < 2:
        raise ValueError("Need at least 2 datasets with all requested approaches for Friedman/Nemenyi.")
    # Ensure requested order
    mat = mat.reindex(columns=approaches)

    N = mat.shape[0]              # number of datasets (blocks)
    k = mat.shape[1]              # number of approaches (treatments)

    # Friedman test (scipy wants each group as a separate array)
    groups = [mat[c].to_numpy(dtype=float) for c in mat.columns]
    stat, p = friedmanchisquare(*groups)

    # Average ranks per approach (lower nMAE -> better rank 1)
    # rankdata ranks ascending by default; per dataset row
    ranks_per_ds = np.apply_along_axis(lambda row: rankdata(row, method="average"), 1, mat.to_numpy())
    avg_ranks = pd.Series(ranks_per_ds.mean(axis=0), index=mat.columns).sort_values()

    # Nemenyi post-hoc (two-sided) using studentized range with df=inf
    # q_ij = |R_i - R_j| / sqrt(k(k+1)/(6N))
    denom = np.sqrt(k * (k + 1) / (6.0 * N))
    pvals = pd.DataFrame(np.ones((k, k), dtype=float), index=mat.columns, columns=mat.columns)

    for i, ai in enumerate(mat.columns):
        for j, aj in enumerate(mat.columns):
            if j <= i:
                continue
            q = abs(avg_ranks[ai] - avg_ranks[aj]) / denom
            # studentized_range.sf gives P(Q >= q); works for Nemenyi with df=inf
            pij = float(studentized_range.sf(q, k, np.inf))
            pvals.loc[ai, aj] = pij
            pvals.loc[aj, ai] = pij

    return {
        "friedman": {"statistic": float(stat), "pvalue": float(p), "N": int(N), "k": int(k)},
        "ranks": avg_ranks,
        "nemenyi_pvals": pvals,
        "data_matrix": mat,
    }


import pandas as pd

def save_stats_summary_latex_table_compact(
    stats_by_mode: dict,
    out_path: str,
    caption: str = "Friedman test and Nemenyi post-hoc summary (nMAE)",
    label: str = "tab:friedman_nemenyi_nmae",
    alpha: float = 0.05,
    chi2_decimals: int = 2,
    p_decimals: int = 3,
    rank_decimals: int = 2,
    max_pairs_per_line: int = 2,
) -> str:
    """
    Writes a compact LaTeX table (tabularx) with wrapping columns.
    Fixes LaTeX errors by formatting p-values in math mode.
    """

    def fmt_p_math(p: float) -> str:
        thr = 10 ** (-p_decimals)
        if p < thr:
            return rf"$<{thr:.{p_decimals}f}$"
        return rf"${p:.{p_decimals}f}$"

    def fmt_ranks(r: pd.Series) -> str:
        r = r.sort_values()
        return ", ".join([f"{name} ({val:.{rank_decimals}f})" for name, val in r.items()])

    def fmt_sig_pairs(pmat: pd.DataFrame) -> str:
        cols = list(pmat.columns)
        pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pij = float(pmat.iloc[i, j])
                if pij < alpha:
                    pairs.append(f"{cols[i]} vs {cols[j]} ({pij:.{p_decimals}f})")
        if not pairs:
            return "None"
        # manual breaks inside cell (works in tabularx X columns)
        lines = []
        for i in range(0, len(pairs), max_pairs_per_line):
            lines.append("; ".join(pairs[i:i + max_pairs_per_line]))
        return r" \\ ".join(lines)

    rows = []
    for mode, out in stats_by_mode.items():
        fr = out["friedman"]
        rows.append({
            "Mode": mode,
            "chi2": float(fr["statistic"]),
            "p": float(fr["pvalue"]),
            "ranks": fmt_ranks(out["ranks"]),
            "pairs": fmt_sig_pairs(out["nemenyi_pvals"]),
        })
    df = pd.DataFrame(rows)

    lines = []
    lines += [r"\begin{table}[htbp]"]
    lines += [r"  \centering"]
    lines += [f"  \\caption{{{caption}}}"]
    lines += [r"  \setlength{\tabcolsep}{4pt}"]
    lines += [r"  \renewcommand{\arraystretch}{1.15}"]
    lines += [r"  \scriptsize"]
    # Define a wrapped, ragged-right X column
    lines += [r"  \newcolumntype{Y}{>{\raggedright\arraybackslash}X}"]
    lines += [r"  \begin{tabularx}{\linewidth}{lrrYY}"]
    lines += [r"    \toprule"]
    lines += [r"    Mode & $\chi^2$ & $p$ & Avg. ranks (best$\rightarrow$worst) & Nemenyi significant pairs ($p<" + f"{alpha:g}" + r"$) \\"]
    lines += [r"    \midrule"]

    for r in df.itertuples(index=False):
        chi2 = f"{r.chi2:.{chi2_decimals}f}"
        pstr = fmt_p_math(r.p)  # <-- FIX: p-value in math mode
        lines += [rf"    {r.Mode} & {chi2} & {pstr} & {r.ranks} & {r.pairs} \\"]

    lines += [r"    \bottomrule"]
    lines += [r"  \end{tabularx}"]
    lines += [f"  \\label{{{label}}}"]
    lines += [r"\end{table}"]

    latex = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)
    return latex

def main():
    # ---- settings ----
    model_name = "DALSTM"
    datasets = ["P2P", "HelpDesk", "Sepsis",
                "BPIC15_1", "BPIC_2017_W",  "BPIC20ID", "BPIC20DD", "BPIC20PTC"]
    labels_name = ['Vanilla', 'CSW', 'EAL', 'BMSE', 'SERA']
    seeds = [409, 1824, 3657, 4012, 4506]    
    # paths
    root_path = os.getcwd()
    result_path = os.path.join(root_path, "results", model_name)
    temp_path = os.path.join(root_path, "temp", model_name)
    results_df = aggregate_nmae_over_seeds_and_approaches(
        datasets=datasets, labels_name=labels_name, seeds=seeds,
        result_path=result_path, temp_path=temp_path, model_name=model_name)

    print(results_df)
    results_df.to_csv("results_nmae_summary.csv", index=False)
    
    save_nmae_latex_table_fixed_order(
        results_df,
        datasets=datasets,
        labels_name=labels_name,
        out_path="dalstm_nmae_table.txt",
        caption="Results DALSTM",
        label="tab:dalstm_nmae_results",
        )
    
    save_nmae_boxplot_pdf(results_df, split="all", out_pdf="nmae_all_boxplot.pdf",
                      approach_order=['Vanilla', 'CSW', 'EAL', 'BMSE'])
    save_nmae_boxplot_pdf(results_df, split="many", out_pdf="nmae_many_boxplot.pdf",
                      approach_order=['Vanilla', 'CSW', 'EAL', 'BMSE'])
    save_nmae_boxplot_pdf(results_df, split="medium", out_pdf="nmae_medium_boxplot.pdf",
                      approach_order=['Vanilla', 'CSW', 'EAL', 'BMSE'])
    save_nmae_boxplot_pdf(results_df, split="few", out_pdf="nmae_few_boxplot.pdf",
                      approach_order=['Vanilla', 'CSW', 'EAL', 'BMSE'])
    
    out_all = friedman_nemenyi_from_results(
        results_df, approaches=["Vanilla", "CSW", "EAL", "BMSE", "SERA"], mode="all")
    print(out_all["friedman"])
    print(out_all["ranks"])
    print(out_all["nemenyi_pvals"])
    
    out_many = friedman_nemenyi_from_results(
        results_df, approaches=["Vanilla", "CSW", "EAL", "BMSE", "SERA"], mode="many")
    print(out_many["friedman"])
    print(out_many["ranks"])
    print(out_many["nemenyi_pvals"])
    
    out_medium = friedman_nemenyi_from_results(
        results_df, approaches=["Vanilla", "CSW", "EAL", "BMSE", "SERA"], mode="medium")
    print(out_medium["friedman"])
    print(out_medium["ranks"])
    print(out_medium["nemenyi_pvals"])
    
    out_few = friedman_nemenyi_from_results(
        results_df, approaches=["Vanilla", "CSW", "EAL", "BMSE", "SERA"], mode="few")
    print(out_few["friedman"])
    print(out_few["ranks"])
    print(out_few["nemenyi_pvals"])
    
    stats_by_mode = {"all": out_all, "many": out_many, "medium": out_medium, "few": out_few}
    save_stats_summary_latex_table_compact(stats_by_mode, out_path="stats_summary.txt")


if __name__ == "__main__":
    main()