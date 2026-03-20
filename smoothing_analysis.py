# -*- coding: utf-8 -*-
"""
Aggregate smoothing results for imbalanced regression approaches and export a LaTeX table.

What this script does
---------------------
1. Reads baseline ("wos") and smoothed inference files.
2. Computes nMAE for:
   - all
   - many
   - medium
   - few
3. Averages results over seeds for each (Dataset, Approach, Smoothing).
4. Computes delta = smoothed - baseline for each dataset.
5. Runs paired Wilcoxon signed-rank tests across datasets.
6. Exports a LaTeX table.

Expected filename patterns
--------------------------
Baseline:
    {dataset}_{model}_{approach}_wos_seed{seed}_inference.csv

Smoothed:
    {dataset}_{model}_{approach}_{smooth_mode}_seed{seed}_inference.csv

Adjust SMOOTHING_CONFIGS if your smoothing tags differ.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
warnings.filterwarnings("ignore")
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL_NAME = "DALSTM"

DATASETS = [
    "P2P",
    "BPIC_2017_W",
    "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5",
    "HelpDesk", "Sepsis",
    "BPIC20ID", "BPIC20DD", "BPIC20PTC", "BPIC20TPD", "BPIC20RFP",
]

SEEDS = [409, 1824, 3657, 4012, 4506]


SMOOTHING_CONFIGS = {
    "FDS": ["CSW", "EAL", "BMSE", "SERA"],
    "LDS": ["CSW", "EAL"],
    "LDS+FDS": ["CSW", "EAL"],
}

REGIONS = ["all", "many", "medium", "few"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def get_target_stats(temp_dir: str, model_name: str, dataset: str) -> Tuple[float, float, float]:
    y_train_path = os.path.join(temp_dir, f"{model_name}_y_train_{dataset}.pt")
    y_val_path = os.path.join(temp_dir, f"{model_name}_y_val_{dataset}.pt")

    y_train = torch.load(y_train_path)
    y_val = torch.load(y_val_path)

    y_train_val = torch.cat([y_train, y_val])
    median_rt = float(y_train_val.median())
    quantile_60, quantile_90 = torch.quantile(y_train_val, torch.tensor([0.6, 0.9]))

    return median_rt, float(quantile_60), float(quantile_90)


def split_by_quantiles(df: pd.DataFrame, q60: float, q90: float):
    df_many = df[df["GroundTruth"] <= q60]
    df_medium = df[(df["GroundTruth"] > q60) & (df["GroundTruth"] <= q90)]
    df_few = df[df["GroundTruth"] > q90]
    return df_many, df_medium, df_few


def compute_region_nmae(df_region: pd.DataFrame, baseline: float) -> float:
    """
    nMAE_region =
        mean(|y - y_hat|)_region /
        mean(|y - median_train|)_test
    """
    if len(df_region) == 0 or baseline == 0:
        return float("nan")

    mae_region = (df_region["GroundTruth"] - df_region["Prediction"]).abs().mean()
    return float(mae_region / baseline)


def compute_nmae_dict(df: pd.DataFrame, train_median: float, q60: float, q90: float) -> Dict[str, float]:
    """
    Returns:
        {
            "all": ...,
            "many": ...,
            "medium": ...,
            "few": ...
        }
    """
    baseline = (df["GroundTruth"] - train_median).abs().mean()
    df_many, df_medium, df_few = split_by_quantiles(df, q60, q90)

    return {
        "all": compute_region_nmae(df, baseline),
        "many": compute_region_nmae(df_many, baseline),
        "medium": compute_region_nmae(df_medium, baseline),
        "few": compute_region_nmae(df_few, baseline),
    }


def get_file_paths(
    result_dir: str,
    dataset: str,
    model: str,
    approach: str,
    smooth_mode: str,
    seed: int,
) -> Tuple[str, str]:
    baseline_name = f"{dataset}_{model}_{approach}_wos_seed{seed}_inference.csv"
    smooth_name = f"{dataset}_{model}_{approach}_{smooth_mode}_seed{seed}_inference.csv"

    baseline_path = os.path.join(result_dir, baseline_name)
    smooth_path = os.path.join(result_dir, smooth_name)
    return baseline_path, smooth_path


def _wilcoxon_paired(x: np.ndarray, y: np.ndarray) -> float:
    """
    Paired Wilcoxon test.
    Returns NaN if not computable.
    """
    if wilcoxon is None:
        return float("nan")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return float("nan")

    diffs = y - x
    if np.allclose(diffs, 0.0):
        return 1.0

    try:
        return float(wilcoxon(y, x, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def format_delta(val: float) -> str:
    if not np.isfinite(val):
        return "--"
    return f"{val:+.2f}\\%"


def format_p_value_latex(p: float) -> str:
    if not np.isfinite(p):
        return "--"
    if p < 0.001:
        return r"$<0.001$"
    return f"{p:.3f}"


def escape_latex(text: str) -> str:
    return (
        str(text)
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


# -----------------------------------------------------------------------------
# Core aggregation
# -----------------------------------------------------------------------------

def build_dataset_level_results(
    datasets: List[str],
    seeds: List[int],
    result_path: str,
    temp_path: str,
    model_name: str,
    smoothing_configs: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Builds dataset-level results after averaging over seeds.

    Output columns:
        Dataset, Approach, Smoothing,
        base_all, smooth_all, delta_all,
        base_many, smooth_many, delta_many,
        base_medium, smooth_medium, delta_medium,
        base_few, smooth_few, delta_few,
        n_seeds
    """
    rows = []

    for dataset in datasets:
        result_dir = os.path.join(result_path, dataset)
        temp_dir = os.path.join(temp_path, dataset)

        median_rt, q60, q90 = get_target_stats(temp_dir, model_name, dataset)

        for smoothing, approaches in smoothing_configs.items():
            for approach in approaches:
                seed_records = []

                for seed in seeds:
                    baseline_path, smooth_path = get_file_paths(
                        result_dir=result_dir,
                        dataset=dataset,
                        model=model_name,
                        approach=approach,
                        smooth_mode=smoothing,
                        seed=seed,
                    )

                    baseline_df = safe_read_csv(baseline_path)
                    smooth_df = safe_read_csv(smooth_path)

                    if baseline_df is None or smooth_df is None:
                        continue

                    base_metrics = compute_nmae_dict(baseline_df, median_rt, q60, q90)
                    smooth_metrics = compute_nmae_dict(smooth_df, median_rt, q60, q90)

                    record = {}
                    for region in REGIONS:
                        record[f"base_{region}"] = base_metrics[region]
                        record[f"smooth_{region}"] = smooth_metrics[region]
                        #record[f"delta_{region}"] = smooth_metrics[region] - base_metrics[region]
                        baseline = base_metrics[region]
                        smooth = smooth_metrics[region]
                        if baseline != 0:
                            record[f"delta_{region}"] = (smooth - baseline) / baseline * 100
                        else:
                            record[f"delta_{region}"] = np.nan
                    seed_records.append(record)
                if not seed_records:
                    continue

                seed_df = pd.DataFrame(seed_records)

                row = {
                    "Dataset": dataset,
                    "Approach": approach,
                    "Smoothing": smoothing,
                    "n_seeds": len(seed_df),
                }
                for region in REGIONS:
                    row[f"base_{region}"] = seed_df[f"base_{region}"].mean()
                    row[f"smooth_{region}"] = seed_df[f"smooth_{region}"].mean()
                    row[f"delta_{region}"] = seed_df[f"delta_{region}"].mean()

                rows.append(row)

    return pd.DataFrame(rows)


def summarize_across_datasets(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates dataset-level results across datasets.

    For each (Approach, Smoothing), returns:
        mean delta for each region
        paired Wilcoxon p-value across datasets for each region
    """
    summary_rows = []

    if dataset_df.empty:
        return pd.DataFrame()

    grouped = dataset_df.groupby(["Approach", "Smoothing"], sort=False)

    for (approach, smoothing), g in grouped:
        row = {
            "Approach": approach,
            "Smoothing": smoothing,
            "n_datasets": len(g),
        }

        for region in REGIONS:
            row[f"delta_{region}"] = g[f"delta_{region}"].mean()
            row[f"p_{region}"] = _wilcoxon_paired(
                g[f"base_{region}"].values,
                g[f"smooth_{region}"].values,
            )

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Optional row ordering
    smoothing_order = {name: i for i, name in enumerate(SMOOTHING_CONFIGS.keys())}
    approach_order = {"CSW": 0, "EAL": 1, "BMSE": 2, "SERA": 3}

    summary_df["__smoothing_order"] = summary_df["Smoothing"].map(lambda x: smoothing_order.get(x, 999))
    summary_df["__approach_order"] = summary_df["Approach"].map(lambda x: approach_order.get(x, 999))
    summary_df = summary_df.sort_values(["__smoothing_order", "__approach_order"]).drop(
        columns=["__smoothing_order", "__approach_order"]
    )

    return summary_df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# LaTeX export
# -----------------------------------------------------------------------------

def make_latex_table(
    summary_df: pd.DataFrame,
    caption: str = "Average relative change in nMAE (\%) after applying smoothing to imbalanced regression approaches. Negative values indicate improvement.",
    label: str = "tab:smoothing_delta_summary",
) -> str:
    """
    Creates a LaTeX table with columns:
    Approach | Smoothing | All (Δ, p) | Many (Δ, p) | Medium (Δ, p) | Few (Δ, p)
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(r"  \renewcommand{\arraystretch}{1.15}")
    lines.append(r"  \scriptsize")
    lines.append(r"  \newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}")
    lines.append(
        r"  \begin{tabularx}{\linewidth}{@{} l l "
        r"C{0.08\linewidth} C{0.07\linewidth} "
        r"C{0.08\linewidth} C{0.07\linewidth} "
        r"C{0.08\linewidth} C{0.07\linewidth} "
        r"C{0.08\linewidth} C{0.07\linewidth} @{} }"
    )
    lines.append(r"    \toprule")
    lines.append(
        r"    \multirow{2}{*}{IR} & \multirow{2}{*}{Smoothing} "
        r"& \multicolumn{2}{c}{All} & \multicolumn{2}{c}{Many} "
        r"& \multicolumn{2}{c}{Medium} & \multicolumn{2}{c}{Few} \\"
    )
    lines.append(
        r"    \cmidrule(lr){3-4} \cmidrule(lr){5-6} "
        r"\cmidrule(lr){7-8} \cmidrule(lr){9-10}"
    )
    lines.append(
        r"    & & $\Delta$ & $p$ & $\Delta$ & $p$ & $\Delta$ & $p$ & $\Delta$ & $p$ \\"
    )
    lines.append(r"    \midrule")

    for _, row in summary_df.iterrows():
        vals = [
            escape_latex(row["Approach"]),
            escape_latex(row["Smoothing"]),
            format_delta(row["delta_all"]),
            format_p_value_latex(row["p_all"]),
            format_delta(row["delta_many"]),
            format_p_value_latex(row["p_many"]),
            format_delta(row["delta_medium"]),
            format_p_value_latex(row["p_medium"]),
            format_delta(row["delta_few"]),
            format_p_value_latex(row["p_few"]),
        ]
        lines.append(
            "    "
            + " & ".join(vals)
            + r" \\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabularx}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    root_path = os.getcwd()
    result_path = os.path.join(root_path, "results", MODEL_NAME)
    temp_path = os.path.join(root_path, "temp", MODEL_NAME)

    dataset_level_df = build_dataset_level_results(
        datasets=DATASETS,
        seeds=SEEDS,
        result_path=result_path,
        temp_path=temp_path,
        model_name=MODEL_NAME,
        smoothing_configs=SMOOTHING_CONFIGS,
    )

    if dataset_level_df.empty:
        print("No results found. Check paths, dataset names, and smoothing tags.")
        return

    # Save dataset-level csv for inspection
    dataset_csv = os.path.join(root_path, f"{MODEL_NAME}_smoothing_dataset_level.csv")
    dataset_level_df.to_csv(dataset_csv, index=False)
    print(f"Saved dataset-level results to: {dataset_csv}")

    summary_df = summarize_across_datasets(dataset_level_df)

    # Save summary csv
    summary_csv = os.path.join(root_path, f"{MODEL_NAME}_smoothing_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary results to: {summary_csv}")

    latex_table = make_latex_table(
        summary_df=summary_df,
        caption=("Average relative change in nMAE (\%) after applying smoothing to imbalanced regression approaches. Negative values indicate improvement."
            "Statistical significance is assessed with a paired Wilcoxon signed-rank test across datasets."
        ),
        label="tab:smoothing_delta_summary",
    )

    tex_path = os.path.join(root_path, f"{MODEL_NAME}_smoothing_summary_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print(f"Saved LaTeX table to: {tex_path}")
    print("\n" + latex_table)


if __name__ == "__main__":
    main()