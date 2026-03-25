# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:49:33 2026
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from scipy.stats import wilcoxon


# Configuration
GROUND_TRUTH_COL = "GroundTruth"
PREDICTION_COL = "Prediction"
CASE_ID_COL = "Case_id"
PREFIX_COL = "Prefix_length"
SCORE_COL = "Prediction" # "Prediction"   "Probability"
# Map folder/file keyword -> final label in results dataframe
APPROACH_NAME_MAP = {
    "base": "Classification",
    "cat": "Survival",
}
EARLINESS_THRESHOLDS = [0.2, 0.4, 0.6, 0.8]
PLOT_METRICS = ["recall", "precision", "f1", "prauc"]
PREFIX_RATIO_COL = "prefix_ratio"
CASE_LENGTH_COL = "case_length"

# Validation
def validate_input_dataframe(
    df: pd.DataFrame,
    required_cols=None
) -> None:
    """
    Validate that the dataframe contains the required columns.
    """
    if required_cols is None:
        required_cols = [
            GROUND_TRUTH_COL,
            PREDICTION_COL,
            CASE_ID_COL,
            PREFIX_COL,
        ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

# Basic confusion counts
def get_confusion_counts(
    df: pd.DataFrame,
    ground_truth_col: str = GROUND_TRUTH_COL,
    prediction_col: str = PREDICTION_COL,
) -> dict:
    """
    Compute TP, TN, FP, FN for binary classification with:
    positive = 1, negative = 0.
    """
    y_true = df[ground_truth_col].astype(int)
    y_pred = df[prediction_col].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safely divide two numbers.
    Returns np.nan if denominator is zero.
    """
    if denominator == 0:
        return np.nan
    return numerator / denominator

# Metric functions
def metric_positive_class_percentage(
    df: pd.DataFrame,
    ground_truth_col: str = GROUND_TRUTH_COL,
    **kwargs
) -> float:
    """
    Percentage of positive class in the ground truth.
    """
    y_true = df[ground_truth_col].astype(int)
    return float((y_true == 1).mean() * 100.0)


def metric_accuracy(df: pd.DataFrame, **kwargs) -> float:
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    counts = get_confusion_counts(df)
    numerator = counts["tp"] + counts["tn"]
    denominator = counts["tp"] + counts["tn"] + counts["fp"] + counts["fn"]
    return float(safe_divide(numerator, denominator))


def metric_precision(df: pd.DataFrame, **kwargs) -> float:
    """
    Precision = TP / (TP + FP)
    """
    counts = get_confusion_counts(df)
    numerator = counts["tp"]
    denominator = counts["tp"] + counts["fp"]
    return float(safe_divide(numerator, denominator))


def metric_recall(df: pd.DataFrame, **kwargs) -> float:
    """
    Recall = TP / (TP + FN)
    """
    counts = get_confusion_counts(df)
    numerator = counts["tp"]
    denominator = counts["tp"] + counts["fn"]
    return float(safe_divide(numerator, denominator))

def metric_f1(df: pd.DataFrame, **kwargs) -> float:
    """
    F1-score for imbalanced binary classification.
    Uses binary predictions.
    """
    y_true = df[GROUND_TRUTH_COL].astype(int)
    y_pred = df[PREDICTION_COL].astype(int)

    return float(f1_score(y_true, y_pred, zero_division=0))

def metric_auroc(
    df: pd.DataFrame,
    ground_truth_col: str = GROUND_TRUTH_COL,
    score_col: str = SCORE_COL,
    **kwargs
) -> float:
    """
    AUROC for binary classification.

    Important:
    This should ideally use continuous prediction scores or probabilities,
    not hard 0/1 predictions.
    """
    y_true = df[ground_truth_col].astype(int)
    y_score = df[score_col].astype(float)
    
    # roc_auc_score fails if only one class is present in y_true
    if y_true.nunique() < 2:
        return np.nan

    return float(roc_auc_score(y_true, y_score))

def metric_prauc(
    df: pd.DataFrame,
    ground_truth_col: str = GROUND_TRUTH_COL,
    score_col: str = SCORE_COL,
    **kwargs
) -> float:
    """
    PRAUC for binary classification.

    Uses Average Precision, which is the standard summary metric
    for precision-recall curves in sklearn.
    """
    y_true = df[ground_truth_col].astype(int)
    y_score = df[score_col].astype(float)

    if y_true.nunique() < 2:
        return np.nan

    return float(average_precision_score(y_true, y_score))

def metric_tp(df: pd.DataFrame, **kwargs) -> int:
    return get_confusion_counts(df)["tp"]


def metric_tn(df: pd.DataFrame, **kwargs) -> int:
    return get_confusion_counts(df)["tn"]


def metric_fp(df: pd.DataFrame, **kwargs) -> int:
    return get_confusion_counts(df)["fp"]


def metric_fn(df: pd.DataFrame, **kwargs) -> int:
    return get_confusion_counts(df)["fn"]

def metric_num_positives(
    df: pd.DataFrame,
    ground_truth_col: str = GROUND_TRUTH_COL,
    **kwargs
) -> int:
    """
    Number of positive samples in the dataset.
    """
    y_true = df[ground_truth_col].astype(int)
    return int((y_true == 1).sum())

def get_metric_functions() -> dict:
    return {
        "positive_class_percentage": metric_positive_class_percentage,
        "accuracy": metric_accuracy,
        "precision": metric_precision,
        "recall": metric_recall,
        "f1": metric_f1,
        "auroc": metric_auroc,
        "prauc": metric_prauc,
        "tp": metric_tp,
        "tn": metric_tn,
        "fp": metric_fp,
        "fn": metric_fn,
        "num_positives": metric_num_positives,
    }


# Evaluation wrappers
def evaluate_single_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    approach_name: str,
    seed: int = None,
    metric_functions: dict = None,
) -> dict:
    """
    Evaluate one dataframe and return a result dictionary.
    """
    validate_input_dataframe(df)

    if metric_functions is None:
        metric_functions = get_metric_functions()

    result = {
        "dataset": dataset_name,
        "approach": approach_name,
        "seed": seed,
        "n_rows": len(df),
        "n_cases": df[CASE_ID_COL].nunique(),
        "min_prefix_length": df[PREFIX_COL].min(),
        "max_prefix_length": df[PREFIX_COL].max(),
    }

    for metric_name, metric_fn in metric_functions.items():
        result[metric_name] = metric_fn(df)

    return result


def evaluate_dataframe_list(
    df_list: list,
    dataset_name: str,
    approach_name: str,
    seeds: list = None,
    metric_functions: dict = None,
) -> list:
    """
    Evaluate a list of dataframes (e.g., one per random seed).
    Returns a list of result dictionaries.
    """
    results = []

    if seeds is None:
        seeds = [None] * len(df_list)

    if len(df_list) != len(seeds):
        raise ValueError(
            f"Length mismatch: len(df_list)={len(df_list)} != len(seeds)={len(seeds)}"
        )

    for df, seed in zip(df_list, seeds):
        row = evaluate_single_dataframe(
            df=df,
            dataset_name=dataset_name,
            approach_name=approach_name,
            seed=seed,
            metric_functions=metric_functions,
        )
        results.append(row)

    return results


def aggregate_results_across_seeds(
    results_df: pd.DataFrame,
    group_cols=None,
    metric_cols=None,
) -> pd.DataFrame:
    """
    Aggregate seed-level results into mean/std tables.
    """
    if group_cols is None:
        group_cols = ["dataset", "approach"]

    if metric_cols is None:
        excluded = set(group_cols + ["seed"])
        metric_cols = [c for c in results_df.columns if c not in excluded]

    agg_df = results_df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()

    # Flatten multi-index columns
    flat_cols = []
    for col in agg_df.columns:
        if isinstance(col, tuple):
            if col[1] == "":
                flat_cols.append(col[0])
            else:
                flat_cols.append(f"{col[0]}_{col[1]}")
        else:
            flat_cols.append(col)
    agg_df.columns = flat_cols

    return agg_df

def prepare_confusion_latex_table(
    summary_df: pd.DataFrame,
    approach_order=None,
    dataset_order=None,
    value_suffix: str = "_mean",
) -> pd.DataFrame:
    """
    Prepare a wide dataframe for LaTeX confusion-matrix table.

    Expected input columns in summary_df:
        dataset, approach, tp_mean, tn_mean, fp_mean, fn_mean, num_positives_mean
    """
    if approach_order is None:
        approach_order = ["Classification", "Survival"]

    confusion_metric_cols = [
        f"tp{value_suffix}",
        f"tn{value_suffix}",
        f"fp{value_suffix}",
        f"fn{value_suffix}",
    ]
    positives_col = f"num_positives{value_suffix}"

    required_cols = ["dataset", "approach"] + confusion_metric_cols + [positives_col]

    missing_cols = [c for c in required_cols if c not in summary_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for LaTeX table: {missing_cols}. "
            f"Available columns: {list(summary_df.columns)}"
        )

    df = summary_df[required_cols].copy()

    # Dataset-level positives column (same for both approaches)
    positives_df = (
        df.groupby("dataset", as_index=False)[positives_col]
        .mean()
        .rename(columns={positives_col: "# Positives"})
    )

    # Pivot only the approach-specific confusion metrics
    wide_df = df.pivot(
        index="dataset",
        columns="approach",
        values=confusion_metric_cols
    )

    # Reorder approaches if requested
    existing_approaches = [
        a for a in approach_order
        if a in wide_df.columns.get_level_values(1)
    ]

    wide_df = wide_df.reindex(
        columns=pd.MultiIndex.from_product([confusion_metric_cols, existing_approaches]),
        fill_value=np.nan
    )

    # Flatten columns
    rename_map = {
        "Classification": "Baseline Classifier",
        "Survival": "Probabilistic Model",
        f"tp{value_suffix}": "TP",
        f"tn{value_suffix}": "TN",
        f"fp{value_suffix}": "FP",
        f"fn{value_suffix}": "FN",
    }

    flat_cols = []
    for metric, approach in wide_df.columns:
        flat_cols.append(f"{rename_map[approach]}_{rename_map[metric]}")
    wide_df.columns = flat_cols

    wide_df = wide_df.reset_index()

    # Merge dataset-level positives after pivot
    wide_df = wide_df.merge(positives_df, on="dataset", how="left")
    cols = ["dataset", "# Positives"] + [
        c for c in wide_df.columns if c not in ["dataset", "# Positives"]
    ]
    wide_df = wide_df[cols]

    if dataset_order is not None:
        wide_df["dataset"] = pd.Categorical(
            wide_df["dataset"],
            categories=dataset_order,
            ordered=True
        )
        wide_df = wide_df.sort_values("dataset").reset_index(drop=True)
        wide_df["dataset"] = wide_df["dataset"].astype(str)

    return wide_df

def prepare_performance_latex_table(
    summary_df: pd.DataFrame,
    approach_order=None,
    dataset_order=None,
    value_suffix: str = "_mean",
) -> pd.DataFrame:
    """
    Prepare a wide dataframe for LaTeX performance table.

    Expected input columns in summary_df:
        dataset, approach, recall_mean, precision_mean, f1_mean, prauc_mean

    Returns a dataframe with columns:
        dataset,
        Baseline Classifier_Recall, Baseline Classifier_Precision, ...
        Probabilistic Model_Recall, Probabilistic Model_Precision, ...
    """
    if approach_order is None:
        approach_order = ["Classification", "Survival"]

    metric_cols = [
        f"recall{value_suffix}",
        f"precision{value_suffix}",
        f"f1{value_suffix}",
        f"prauc{value_suffix}",
    ]
    required_cols = ["dataset", "approach"] + metric_cols

    missing_cols = [c for c in required_cols if c not in summary_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for LaTeX table: {missing_cols}. "
            f"Available columns: {list(summary_df.columns)}"
        )

    df = summary_df[required_cols].copy()

    wide_df = df.pivot(
        index="dataset",
        columns="approach",
        values=metric_cols
    )

    existing_approaches = [
        a for a in approach_order
        if a in wide_df.columns.get_level_values(1)
    ]

    wide_df = wide_df.reindex(
        columns=pd.MultiIndex.from_product([metric_cols, existing_approaches]),
        fill_value=np.nan
    )

    rename_map = {
        "Classification": "Baseline Classifier",
        "Survival": "Probabilistic Model",
        f"recall{value_suffix}": "Recall",
        f"precision{value_suffix}": "Precision",
        f"f1{value_suffix}": "F1-Score",
        f"prauc{value_suffix}": "PRAUC",
    }

    flat_cols = []
    for metric, approach in wide_df.columns:
        flat_cols.append(f"{rename_map[approach]}_{rename_map[metric]}")
    wide_df.columns = flat_cols

    wide_df = wide_df.reset_index()

    if dataset_order is not None:
        wide_df["dataset"] = pd.Categorical(
            wide_df["dataset"],
            categories=dataset_order,
            ordered=True
        )
        wide_df = wide_df.sort_values("dataset").reset_index(drop=True)
        wide_df["dataset"] = wide_df["dataset"].astype(str)

    return wide_df


def dataframe_to_latex_confusion_table(
    wide_df: pd.DataFrame,
    caption: str,
    label: str,
    decimals: int = 0,
    round_values: bool = True,
) -> str:
    """
    Convert wide confusion-table dataframe to LaTeX.

    Expected columns:
        dataset,
        Baseline Classifier_TP, Baseline Classifier_TN, ...
        Probabilistic Model_TP, Probabilistic Model_TN, ...
    """
    expected_cols = [
        "dataset",
        "Baseline Classifier_TP",
        "Baseline Classifier_TN",
        "Baseline Classifier_FP",
        "Baseline Classifier_FN",
        "Probabilistic Model_TP",
        "Probabilistic Model_TN",
        "Probabilistic Model_FP",
        "Probabilistic Model_FN",
    ]

    missing_cols = [c for c in expected_cols if c not in wide_df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in wide_df: {missing_cols}")

    def fmt(x):
        if pd.isna(x):
            return ""
        if round_values:
            x = round(x, decimals)
        if decimals == 0:
            return str(int(x))
        return f"{x:.{decimals}f}"

    def bold_if_best(val, other, higher_is_better):
        if pd.isna(val) or pd.isna(other):
            return fmt(val)
        if higher_is_better:
            is_best = val > other
        else:
            is_best = val < other
        formatted = fmt(val)
        return f"\\textbf{{{formatted}}}" if is_best else formatted
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lrrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"& & \multicolumn{4}{c}{Baseline Classifier} & \multicolumn{4}{c}{Probabilistic Model} \\"
    )
    lines.append(
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}"
    )
    lines.append(
        r"Dataset & \# Positives & TP & TN & FP & FN & TP & TN & FP & FN \\"
    )
    lines.append(r"\midrule")
    
    for _, row in wide_df.iterrows():
        # Baseline
        b_tp = row["Baseline Classifier_TP"]
        b_tn = row["Baseline Classifier_TN"]
        b_fp = row["Baseline Classifier_FP"]
        b_fn = row["Baseline Classifier_FN"]
        # Probabilistic
        p_tp = row["Probabilistic Model_TP"]
        p_tn = row["Probabilistic Model_TN"]
        p_fp = row["Probabilistic Model_FP"]
        p_fn = row["Probabilistic Model_FN"]
        lines.append(
            f"{row['dataset']} "
            f"& {fmt(row['# Positives'])} "
            f"& {bold_if_best(b_tp, p_tp, True)} "
            f"& {bold_if_best(b_tn, p_tn, True)} "
            f"& {bold_if_best(b_fp, p_fp, False)} "
            f"& {bold_if_best(b_fn, p_fn, False)} "
            f"& {bold_if_best(p_tp, b_tp, True)} "
            f"& {bold_if_best(p_tn, b_tn, True)} "
            f"& {bold_if_best(p_fp, b_fp, False)} "
            f"& {bold_if_best(p_fn, b_fn, False)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

def dataframe_to_latex_performance_table(
    wide_df: pd.DataFrame,
    caption: str,
    label: str,
    decimals: int = 3,
    round_values: bool = True,
) -> str:
    """
    Convert wide performance-table dataframe to LaTeX.

    Expected columns:
        dataset,
        Baseline Classifier_Recall, Baseline Classifier_Precision,
        Baseline Classifier_F1-Score, Baseline Classifier_PRAUC,
        Probabilistic Model_Recall, Probabilistic Model_Precision,
        Probabilistic Model_F1-Score, Probabilistic Model_PRAUC
    """
    expected_cols = [
        "dataset",
        "Baseline Classifier_Recall",
        "Baseline Classifier_Precision",
        "Baseline Classifier_F1-Score",
        "Baseline Classifier_PRAUC",
        "Probabilistic Model_Recall",
        "Probabilistic Model_Precision",
        "Probabilistic Model_F1-Score",
        "Probabilistic Model_PRAUC",
    ]

    missing_cols = [c for c in expected_cols if c not in wide_df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in wide_df: {missing_cols}")

    def fmt(x):
        if pd.isna(x):
            return ""
        if round_values:
            x = round(x, decimals)
        return f"{x:.{decimals}f}"

    def bold_if_best(val, other):
        if pd.isna(val) or pd.isna(other):
            return fmt(val)
        formatted = fmt(val)
        return f"\\textbf{{{formatted}}}" if val > other else formatted

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"& \multicolumn{4}{c}{Baseline Classifier} & \multicolumn{4}{c}{Probabilistic Model} \\"
    )
    lines.append(
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}"
    )
    lines.append(
        r"Dataset & Recall & Precision & F1-Score & PRAUC & Recall & Precision & F1-Score & PRAUC \\"
    )
    lines.append(r"\midrule")

    for _, row in wide_df.iterrows():
        b_recall = row["Baseline Classifier_Recall"]
        b_precision = row["Baseline Classifier_Precision"]
        b_f1 = row["Baseline Classifier_F1-Score"]
        b_prauc = row["Baseline Classifier_PRAUC"]

        p_recall = row["Probabilistic Model_Recall"]
        p_precision = row["Probabilistic Model_Precision"]
        p_f1 = row["Probabilistic Model_F1-Score"]
        p_prauc = row["Probabilistic Model_PRAUC"]

        lines.append(
            f"{row['dataset']} "
            f"& {bold_if_best(b_recall, p_recall)} "
            f"& {bold_if_best(b_precision, p_precision)} "
            f"& {bold_if_best(b_f1, p_f1)} "
            f"& {bold_if_best(b_prauc, p_prauc)} "
            f"& {bold_if_best(p_recall, b_recall)} "
            f"& {bold_if_best(p_precision, b_precision)} "
            f"& {bold_if_best(p_f1, b_f1)} "
            f"& {bold_if_best(p_prauc, b_prauc)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def export_confusion_table_to_latex(
    summary_df: pd.DataFrame,
    output_path: str = None,
    dataset_order=None,
    caption: str = "Confusion-matrix statistics aggregated across random seeds.",
    label: str = "tab:confusion_metrics",
    decimals: int = 0,
    round_values: bool = True,
) -> tuple:
    """
    Full wrapper:
      1. prepare wide table
      2. generate LaTeX
      3. optionally save to file

    Returns:
        wide_df, latex_str
    """
    wide_df = prepare_confusion_latex_table(
        summary_df=summary_df,
        dataset_order=dataset_order,
        value_suffix="_mean",
    )

    latex_str = dataframe_to_latex_confusion_table(
        wide_df=wide_df,
        caption=caption,
        label=label,
        decimals=decimals,
        round_values=round_values,
    )

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex_str)

    return wide_df, latex_str

def export_performance_table_to_latex(
    summary_df: pd.DataFrame,
    output_path: str = None,
    dataset_order=None,
    caption: str = "Performance metrics aggregated across random seeds.",
    label: str = "tab:performance_metrics",
    decimals: int = 3,
    round_values: bool = True,
) -> tuple:
    """
    Full wrapper:
      1. prepare wide table
      2. generate LaTeX
      3. optionally save to file

    Returns:
        wide_df, latex_str
    """
    wide_df = prepare_performance_latex_table(
        summary_df=summary_df,
        dataset_order=dataset_order,
        value_suffix="_mean",
    )

    latex_str = dataframe_to_latex_performance_table(
        wide_df=wide_df,
        caption=caption,
        label=label,
        decimals=decimals,
        round_values=round_values,
    )

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex_str)

    return wide_df, latex_str


def format_quantile_for_name(quantile: float) -> str:
    """
    Convert quantile to filename-safe string.
    Example:
        0.8  -> q0p8
        0.85 -> q0p85
    """
    return f"q{str(quantile).replace('.', 'p')}"


def build_expected_filename(
    dataset_name: str,
    model_name: str,
    approach_key: str,
    seed: int,
    quantile: float = 0.9,
) -> str:
    """
    Build the expected inference filename.

    Convention:
    - quantile 0.9 -> old naming without quantile suffix
    - other quantiles -> include suffix like q0p8
    """
    q_part = ""
    if quantile != 0.9:
        q_part = f"_{format_quantile_for_name(quantile)}"

    if approach_key == "base":
        return f"{dataset_name}_{model_name}classification{q_part}_seed_{seed}_inference.csv"
    elif approach_key == "cat":
        return f"{dataset_name}_{model_name}classification_CAT{q_part}_seed_{seed}_inference.csv"
    else:
        raise ValueError(f"Unknown approach_key: {approach_key}")


def discover_seed_file(
    result_dir: str,
    dataset_name: str,
    model_name: str,
    approach_key: str,
    seed: int,
    quantile: float = 0.9,
) -> str:
    """
    Return the inference file for one dataset / approach / seed.

    Rules:
    - For quantile == 0.9, prefer the legacy filename without quantile suffix.
    - For other quantiles, prefer the quantile-specific filename.
    """
    expected_file = build_expected_filename(
        dataset_name=dataset_name,
        model_name=model_name,
        approach_key=approach_key,
        seed=seed,
        quantile=quantile,
    )

    file_path = os.path.join(result_dir, expected_file)
    if os.path.exists(file_path):
        return file_path

    # Fallback patterns
    if approach_key == "base":
        if quantile == 0.9:
            patterns = [
                os.path.join(
                    result_dir,
                    f"{dataset_name}_{model_name}classification_seed_{seed}_inference.csv"
                )
            ]
        else:
            q_name = format_quantile_for_name(quantile)
            patterns = [
                os.path.join(
                    result_dir,
                    f"{dataset_name}_{model_name}classification_{q_name}_seed_{seed}_inference.csv"
                )
            ]
    elif approach_key == "cat":
        if quantile == 0.9:
            patterns = [
                os.path.join(
                    result_dir,
                    f"{dataset_name}_{model_name}classification_CAT_seed_{seed}_inference.csv"
                )
            ]
        else:
            q_name = format_quantile_for_name(quantile)
            patterns = [
                os.path.join(
                    result_dir,
                    f"{dataset_name}_{model_name}classification_CAT_{q_name}_seed_{seed}_inference.csv"
                )
            ]
    else:
        raise ValueError(f"Unknown approach_key: {approach_key}")

    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))

    if len(matches) == 0:
        return None

    if len(matches) > 1:
        print(f"[WARNING] Multiple inference matches found for approach='{approach_key}', seed={seed}, quantile={quantile}:")
        for m in matches:
            print(f"  - {m}")
        print(f"[INFO] Using first match: {matches[0]}")

    return matches[0]


def load_prediction_dataframe(file_path: str) -> pd.DataFrame:
    if file_path is None:
        return None
    return pd.read_csv(file_path)


def load_approach_dataframes_for_dataset(
    result_dir: str,
    dataset_name: str,
    model_name: str,
    approach_key: str,
    seeds: list,
    quantile: float = 0.9,
) -> tuple:
    """
    Load all inference dataframes for one dataset and one approach.
    Returns:
        df_list, found_seeds
    """
    df_list = []
    found_seeds = []

    for seed in seeds:
        file_path = discover_seed_file(
            result_dir=result_dir,
            dataset_name=dataset_name,
            model_name=model_name,
            approach_key=approach_key,
            seed=seed,
            quantile=quantile,
        )

        if file_path is None:
            print(
                f"[WARNING] No inference file found for approach='{approach_key}', "
                f"seed={seed}, quantile={quantile}, dir={result_dir}"
            )
            continue

        df = load_prediction_dataframe(file_path)
        df_list.append(df)
        found_seeds.append(seed)

    return df_list, found_seeds

def run_wilcoxon_tests(
    summary_df: pd.DataFrame,
    metrics=None,
    approach_a: str = "Classification",
    approach_b: str = "Survival",
) -> pd.DataFrame:
    """
    Run paired Wilcoxon signed-rank tests across datasets, comparing two approaches
    on aggregated mean performance per dataset.

    Higher is better for all supplied metrics by default.
    """
    if metrics is None:
        metrics = ["recall", "precision", "f1", "prauc"]

    results = []

    for metric in metrics:
        metric_col = f"{metric}_mean"
        if metric_col not in summary_df.columns:
            raise ValueError(f"Column not found in summary_df: {metric_col}")

        pivot_df = summary_df.pivot(
            index="dataset",
            columns="approach",
            values=metric_col
        )

        required_cols = [approach_a, approach_b]
        missing = [c for c in required_cols if c not in pivot_df.columns]
        if missing:
            raise ValueError(
                f"Missing approaches in pivoted dataframe for metric '{metric}': {missing}"
            )

        paired = pivot_df[[approach_a, approach_b]].dropna()

        x = paired[approach_a].to_numpy()
        y = paired[approach_b].to_numpy()
        diff = y - x  # positive means approach_b is better

        n_pairs = len(paired)
        n_positive = int((diff > 0).sum())
        n_negative = int((diff < 0).sum())
        n_zero = int((diff == 0).sum())

        if np.allclose(diff, 0):
            stat = np.nan
            p_value = 1.0
        else:
            stat, p_value = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")

        results.append({
            "metric": metric,
            "approach_a": approach_a,
            "approach_b": approach_b,
            "n_pairs": n_pairs,
            "mean_" + approach_a: float(np.mean(x)),
            "mean_" + approach_b: float(np.mean(y)),
            "mean_diff_" + approach_b + "_minus_" + approach_a: float(np.mean(diff)),
            "wins_" + approach_b: n_positive,
            "wins_" + approach_a: n_negative,
            "ties": n_zero,
            "wilcoxon_stat": stat,
            "p_value": p_value,
            "significant_0.05": bool(p_value < 0.05),
        })

    return pd.DataFrame(results)

def add_case_length_and_prefix_ratio(
    df: pd.DataFrame,
    case_id_col: str = CASE_ID_COL,
    prefix_col: str = PREFIX_COL,
) -> pd.DataFrame:
    """
    Derive case length as the maximum Prefix_length within each Case_id,
    then compute prefix_ratio = Prefix_length / case_length.
    """
    out = df.copy()

    out[prefix_col] = out[prefix_col].astype(float)

    case_lengths = out.groupby(case_id_col)[prefix_col].transform("max")
    out[CASE_LENGTH_COL] = case_lengths

    if (out[CASE_LENGTH_COL] <= 0).any():
        raise ValueError(
            f"Derived '{CASE_LENGTH_COL}' contains non-positive values."
        )

    out[PREFIX_RATIO_COL] = out[prefix_col] / out[CASE_LENGTH_COL]
    return out

def evaluate_earliness_single_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    approach_name: str,
    seed: int = None,
    thresholds: list = None,
    metric_functions: dict = None,
) -> list:
    """
    Evaluate metrics on cumulative subsets:
    Prefix_length / case_length <= threshold
    """
    validate_input_dataframe(df)

    if thresholds is None:
        thresholds = EARLINESS_THRESHOLDS

    if metric_functions is None:
        metric_functions = {
            "recall": metric_recall,
            "precision": metric_precision,
            "f1": metric_f1,
            "prauc": metric_prauc,
        }

    df = add_case_length_and_prefix_ratio(df)

    rows = []
    for threshold in thresholds:
        subset = df[df[PREFIX_RATIO_COL] <= threshold].copy()

        row = {
            "dataset": dataset_name,
            "approach": approach_name,
            "seed": seed,
            "prefix_ratio_threshold": threshold,
            "n_rows": len(subset),
            "n_cases": subset[CASE_ID_COL].nunique() if len(subset) > 0 else 0,
        }

        if len(subset) == 0:
            for metric_name in metric_functions:
                row[metric_name] = np.nan
        else:
            for metric_name, metric_fn in metric_functions.items():
                row[metric_name] = metric_fn(subset)

        rows.append(row)

    return rows

def evaluate_earliness_dataframe_list(
    df_list: list,
    dataset_name: str,
    approach_name: str,
    seeds: list = None,
    thresholds: list = None,
    metric_functions: dict = None,
) -> list:
    """
    Evaluate earliness metrics for a list of dataframes.
    """
    results = []

    if seeds is None:
        seeds = [None] * len(df_list)

    if len(df_list) != len(seeds):
        raise ValueError(
            f"Length mismatch: len(df_list)={len(df_list)} != len(seeds)={len(seeds)}"
        )

    for df, seed in zip(df_list, seeds):
        rows = evaluate_earliness_single_dataframe(
            df=df,
            dataset_name=dataset_name,
            approach_name=approach_name,
            seed=seed,
            thresholds=thresholds,
            metric_functions=metric_functions,
        )
        results.extend(rows)

    return results

def metric_display_name(metric: str) -> str:
    mapping = {
        "recall": "Recall",
        "precision": "Precision",
        "f1": "F1-Score",
        "prauc": "PRAUC",
    }
    return mapping.get(metric, metric)


def plot_earliness_curves_for_dataset(
    summary_df: pd.DataFrame,
    dataset_name: str,
    output_dir: str,
    metrics: list = None,
    q_suffix: str = "",
):
    """
    Create 4 PDF plots for one dataset: recall, precision, f1, prauc.
    """
    if metrics is None:
        metrics = PLOT_METRICS

    dataset_df = summary_df[summary_df["dataset"] == dataset_name].copy()
    if dataset_df.empty:
        return

    for metric in metrics:
        metric_col = f"{metric}_mean"
        if metric_col not in dataset_df.columns:
            continue

        plt.figure(figsize=(6, 4))

        for approach in ["Classification", "Survival"]:
            approach_df = dataset_df[
                dataset_df["approach"] == approach
            ].sort_values("prefix_ratio_threshold")

            plt.plot(
                approach_df["prefix_ratio_threshold"],
                approach_df[metric_col],
                marker="o",
                label=approach,
            )

        plt.xlabel("Prefix length / case length", fontsize=18, fontweight="bold")
        plt.ylabel(metric_display_name(metric), fontsize=18, fontweight="bold")
        plt.xticks(fontsize=16, fontweight="bold")
        plt.yticks(fontsize=16, fontweight="bold")
        
        #plt.title(f"{dataset_name} - {metric_display_name(metric)}")
        plt.xticks(EARLINESS_THRESHOLDS, [str(x) for x in EARLINESS_THRESHOLDS])
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        output_path = os.path.join(
            output_dir,
            f"{dataset_name}{q_suffix}_{metric}_earliness.pdf"
        )
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close()


def plot_earliness_curves_aggregated(
    summary_df: pd.DataFrame,
    output_dir: str,
    metrics: list = None,
    q_suffix: str = "",
):
    """
    Create 4 PDF plots averaged over datasets.
    """
    if metrics is None:
        metrics = PLOT_METRICS

    agg_df = (
        summary_df.groupby(["approach", "prefix_ratio_threshold"], as_index=False)
        .agg({
            "recall_mean": "mean",
            "precision_mean": "mean",
            "f1_mean": "mean",
            "prauc_mean": "mean",
        })
    )

    for metric in metrics:
        metric_col = f"{metric}_mean"
        if metric_col not in agg_df.columns:
            continue

        plt.figure(figsize=(6, 4))

        for approach in ["Classification", "Survival"]:
            approach_df = agg_df[
                agg_df["approach"] == approach
            ].sort_values("prefix_ratio_threshold")

            plt.plot(
                approach_df["prefix_ratio_threshold"],
                approach_df[metric_col],
                marker="o",
                label=approach,
            )

        plt.xlabel("Prefix length / case length", fontsize=18, fontweight="bold")
        plt.ylabel(metric_display_name(metric), fontsize=18, fontweight="bold")
        plt.xticks(fontsize=16, fontweight="bold")
        plt.yticks(fontsize=16, fontweight="bold")
        
        #plt.title(f"Average across datasets - {metric_display_name(metric)}")
        plt.xticks(EARLINESS_THRESHOLDS, [str(x) for x in EARLINESS_THRESHOLDS])
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        output_path = os.path.join(
            output_dir,
            f"all_datasets{q_suffix}_{metric}_earliness.pdf"
        )
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close()
        

# Main pipeline
def main():
    model_name = "DALSTM"
    datasets = [
        "P2P",
        "BPIC_2017_W",
        "BPIC15_1",
        "BPIC15_2",
        "BPIC15_3",
        "BPIC15_4",
        "BPIC15_5",
        "HelpDesk",
        "Sepsis",
        "BPIC20ID",
        "BPIC20DD",
        "BPIC20PTC",
        "BPIC20TPD",
        "BPIC20RFP",
    ]
    seeds = [409, 1824, 3657, 4012, 4506]
    parser = argparse.ArgumentParser(description="Aggregate delay-classification inference results")
    parser.add_argument(
        '--quantile',
        type=float,
        default=0.8,
        help='Quantile threshold for defining delays'
        )
    args = parser.parse_args()

    root_path = os.getcwd()
    all_results = []
    metric_functions = get_metric_functions()
    earliness_results = []
    earliness_metric_functions = {
        "recall": metric_recall,
        "precision": metric_precision,
        "f1": metric_f1,
        "prauc": metric_prauc,
        }

    for dataset in datasets:
        result_dir = os.path.join(root_path, "results", model_name, dataset)
        for approach_key, approach_name in APPROACH_NAME_MAP.items():
            df_list, found_seeds = load_approach_dataframes_for_dataset(
                result_dir=result_dir,
                dataset_name=dataset,
                model_name=model_name,
                approach_key=approach_key,
                seeds=seeds,
                quantile=args.quantile,
            )
            results = evaluate_dataframe_list(
                df_list=df_list,
                dataset_name=dataset,
                approach_name=approach_name,
                seeds=found_seeds,
                metric_functions=metric_functions,
            )
            all_results.extend(results)
            earliness_rows = evaluate_earliness_dataframe_list(
                df_list=df_list,
                dataset_name=dataset,
                approach_name=approach_name,
                seeds=found_seeds,
                thresholds=EARLINESS_THRESHOLDS,
                metric_functions=earliness_metric_functions,
                )
            earliness_results.extend(earliness_rows)
    # Seed-level results
    results_df = pd.DataFrame(all_results)
    # Aggregate across seeds
    metric_cols = list(metric_functions.keys())
    summary_df = aggregate_results_across_seeds(
        results_df=results_df,
        group_cols=["dataset", "approach"],
        metric_cols=metric_cols,
    )
    print("\nAggregated results across seeds:")
    print(summary_df)
    # Save outputs
    output_dir = os.path.join(root_path, "classification_comparison_results")
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"{model_name}_summary_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary results to: {summary_path}")
    earliness_results_df = pd.DataFrame(earliness_results)
    earliness_summary_df = aggregate_results_across_seeds(
        results_df=earliness_results_df,
        group_cols=["dataset", "approach", "prefix_ratio_threshold"],
        metric_cols=["recall", "precision", "f1", "prauc"],
    )
    dataset_order = [
    "P2P",
    "BPIC_2017_W",
    "BPIC15_1",
    "BPIC15_2",
    "BPIC15_3",
    "BPIC15_4",
    "BPIC15_5",
    "HelpDesk",
    "Sepsis",
    "BPIC20ID",
    "BPIC20DD",
    "BPIC20PTC",
    "BPIC20TPD",
    "BPIC20RFP",
    ]
    
    q_suffix = "" if args.quantile == 0.9 else f"_{format_quantile_for_name(args.quantile)}"
    summary_path = os.path.join(output_dir, f"{model_name}{q_suffix}_summary_results.csv")
    earliness_summary_path = os.path.join(
        output_dir,
        f"{model_name}{q_suffix}_earliness_summary.csv"
        )
    earliness_summary_df.to_csv(earliness_summary_path, index=False)
    (f"Saved earliness summary results to: {earliness_summary_path}")
    latex_output_path = os.path.join(output_dir, f"{model_name}{q_suffix}_confusion_table.tex")

    confusion_wide_df, latex_table = export_confusion_table_to_latex(
        summary_df=summary_df,
        output_path=latex_output_path,
        dataset_order=dataset_order,
        caption="Mean confusion-matrix counts aggregated across five random seeds.",
        label="tab:confusion_metrics",
        decimals=0,
        round_values=True,
    )

    print("\nWide confusion table:")
    print(confusion_wide_df.head())

    print("\nLaTeX table:")
    print(latex_table)
    performance_latex_output_path = os.path.join(
        output_dir,
        f"{model_name}_performance_table.tex"
    )

    performance_wide_df, performance_latex_table = export_performance_table_to_latex(
        summary_df=summary_df,
        output_path=performance_latex_output_path,
        dataset_order=dataset_order,
        caption="Mean Recall, Precision, F1-Score, and PRAUC aggregated across five random seeds.",
        label="tab:performance_metrics",
        decimals=2,
        round_values=True,
        )

    print("\nWide performance table:")
    print(performance_wide_df.head())

    print("\nPerformance LaTeX table:")
    print(performance_latex_table)
    # Wilcoxon signed-rank tests across datasets
    wilcoxon_df = run_wilcoxon_tests(
        summary_df=summary_df,
        metrics=["recall", "precision", "f1", "prauc"],
        approach_a="Classification",
        approach_b="Survival",
    )

    wilcoxon_output_path = os.path.join(
        output_dir,
        f"{model_name}{q_suffix}_wilcoxon_tests.csv"
    )
    wilcoxon_df.to_csv(wilcoxon_output_path, index=False)
    print("\nWilcoxon signed-rank test results:")
    print(wilcoxon_df)
    print(f"Saved Wilcoxon results to: {wilcoxon_output_path}")
    
    for dataset in datasets:
        plot_earliness_curves_for_dataset(
            summary_df=earliness_summary_df,
            dataset_name=dataset,
            output_dir=output_dir,
            metrics=PLOT_METRICS,
            q_suffix=q_suffix,
        )
    plot_earliness_curves_aggregated(
        summary_df=earliness_summary_df,
        output_dir=output_dir,
        metrics=PLOT_METRICS,
        q_suffix=q_suffix,
    )
    print(f"Saved earliness plots to: {output_dir}")

if __name__ == '__main__':
    main()