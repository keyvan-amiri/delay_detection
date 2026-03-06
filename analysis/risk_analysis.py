# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 12:41:32 2026
@author: kamirel
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score, average_precision_score


def get_inference_results(result_dir=None, dataset=None, model=None, seed=None):
    labels_name = ['Vanilla', 'CSW', 'EAL', 'BMSE', 'SERA']
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



def normalize_case_id(df: pd.DataFrame, case_col: str = "Case_id") -> pd.DataFrame:
    """
    Canonicalize Case_id across dataframes.
    Works with pandas StringDtype, object, numeric, categorical.
    Output: pandas 'string' dtype with stripped whitespace and no trailing '.0'.
    """
    df = df.copy()

    s = df[case_col]

    if is_numeric_dtype(s):
        # numeric -> Int64 where possible (avoids 1.0), then to string
        s_int = pd.to_numeric(s, errors="coerce").astype("Int64")
        s_str = s_int.astype("string")
    else:
        # anything else -> string directly
        s_str = s.astype("string")

    # cleanup (strip + remove ".0" artifacts)
    s_str = s_str.str.strip().str.replace(r"\.0$", "", regex=True)

    df[case_col] = s_str
    return df

def _safe_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    if np.unique(y_true).size < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def _safe_prauc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    if np.unique(y_true).size < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def extract_case_level_snapshots(df: pd.DataFrame, levels=(0.25, 0.5, 0.75)) -> dict[float, pd.DataFrame]:
    """
    For each case, select the row whose prefix index is closest to lvl * (#prefixes).
    Returns dict: level -> df(one row per case)
    """
    snapshots = {lvl: [] for lvl in levels}

    for case_id, g in df.groupby("Case_id"):
        g = g.sort_values("Prefix_length")
        n = len(g)
        if n == 0:
            continue

        for lvl in levels:
            idx = int(np.ceil(lvl * n)) - 1
            idx = min(max(idx, 0), n - 1)
            snapshots[lvl].append(g.iloc[idx])

    return {lvl: pd.DataFrame(rows) for lvl, rows in snapshots.items()}


def compute_case_total_duration(df: pd.DataFrame, gt_col="GroundTruth") -> pd.Series:
    """
    True total case duration = remaining time at prefix=1 (assumption in your data).
    Returns Series indexed by Case_id.
    """
    g = df.sort_values("Prefix_length").groupby("Case_id").first()
    return g[gt_col].astype(float)


def add_elapsed_time(df: pd.DataFrame, gt_col="GroundTruth") -> pd.DataFrame:
    """
    Adds elapsed time per row:
        elapsed(p) = GT(prefix=1) - GT(p)
    """
    df = df.copy()
    gt1 = compute_case_total_duration(df, gt_col=gt_col)  # per case
    df["_GT1"] = df["Case_id"].map(gt1)
    df["Elapsed"] = df["_GT1"] - df[gt_col].astype(float)
    return df


def compute_case_level_metrics_total_duration_risk(
    df_lst: list[pd.DataFrame],
    labels_name: list[str],
    quantile_df: pd.DataFrame,
    quantile: float,
    levels=(0.25, 0.5, 0.75),
    *,
    gt_col="GroundTruth",
    pred_col="Prediction",
) -> dict:
    """
    Case-level extreme detection at execution levels.

    Label (case-level):
        Z = 1 if TrueTotalDuration > tau_q
        TrueTotalDuration = GT(prefix=1)

    Risk score at prefix snapshot:
        Deterministic: Elapsed + Prediction
        Quantile:      Elapsed + Q_q  (Q0_9 or Q0_95)
    """
    assert quantile in [0.9, 0.95], "Use 0.9 and 0.95 only for this analysis."
    assert len(df_lst) == len(labels_name)

    q_col = {0.9: "Q0_9", 0.95: "Q0_95"}[quantile]

    # True total duration per case (from quantile_df reference)
    true_total = compute_case_total_duration(quantile_df, gt_col=gt_col)  # index Case_id
    tau = float(true_total.quantile(quantile))
    z_case = (true_total > tau).astype(int)  # index Case_id

    # Add elapsed time columns
    quantile_df_e = add_elapsed_time(quantile_df, gt_col=gt_col)
    det_df_e_list = [add_elapsed_time(d, gt_col=gt_col) for d in df_lst]

    # Build snapshots (one row per case per level)
    q_snaps = extract_case_level_snapshots(quantile_df_e, levels=levels)
    det_snaps = [extract_case_level_snapshots(d, levels=levels) for d in det_df_e_list]

    out = {"tau": tau, "positive_rate": float(z_case.mean())}

    for lvl in levels:
        out[lvl] = {}

        # Quantile risk: elapsed + Q_q
        q_lvl = q_snaps[lvl].dropna(subset=["Case_id", "Elapsed", q_col]).copy()
        q_risk = (q_lvl["Elapsed"].astype(float) + q_lvl[q_col].astype(float))
        q_cases = q_lvl["Case_id"].astype(str).values
        y_true = z_case.reindex(q_cases).values

        out[lvl]["Quantile"] = {
            "AUC": _safe_auc(y_true, q_risk.values),
            "PR_AUC": _safe_prauc(y_true, q_risk.values),
            "n_cases": int(len(q_lvl)),
        }

        # Deterministic risk: elapsed + Prediction
        for name, snaps in zip(labels_name, det_snaps):
            d_lvl = snaps[lvl].dropna(subset=["Case_id", "Elapsed", pred_col]).copy()
            d_risk = (d_lvl["Elapsed"].astype(float) + d_lvl[pred_col].astype(float))
            d_cases = d_lvl["Case_id"].astype(str).values
            y_true_d = z_case.reindex(d_cases).values

            out[lvl][name] = {
                "AUC": _safe_auc(y_true_d, d_risk.values),
                "PR_AUC": _safe_prauc(y_true_d, d_risk.values),
                "n_cases": int(len(d_lvl)),
            }

    return out


def aggregate_case_level_results(seed_results: list[dict], levels=(0.25, 0.5, 0.75)) -> dict:
    """
    Aggregate across seeds: mean/std per method per level.
    """
    if not seed_results:
        return {}

    # determine methods from first seed at first level
    methods = list(seed_results[0][levels[0]].keys())

    agg = {"tau_mean": np.mean([r["tau"] for r in seed_results]),
           "tau_std": np.std([r["tau"] for r in seed_results], ddof=1) if len(seed_results) > 1 else 0.0}

    for lvl in levels:
        agg[lvl] = {}
        for m in methods:
            aucs = np.array([r[lvl][m]["AUC"] for r in seed_results], dtype=float)
            pras = np.array([r[lvl][m]["PR_AUC"] for r in seed_results], dtype=float)

            agg[lvl][m] = {
                "AUC_mean": float(np.nanmean(aucs)),
                "AUC_std": float(np.nanstd(aucs, ddof=1)) if np.sum(~np.isnan(aucs)) > 1 else 0.0,
                "PR_AUC_mean": float(np.nanmean(pras)),
                "PR_AUC_std": float(np.nanstd(pras, ddof=1)) if np.sum(~np.isnan(pras)) > 1 else 0.0,
            }

    return agg


def build_case_level_dataframe(all_results: dict, levels=(0.25, 0.5, 0.75)) -> pd.DataFrame:
    """
    all_results[dataset][quantile] = aggregated dict from aggregate_case_level_results()
    Returns long-form dataframe with Execution_Level column.
    """
    rows = []
    for dataset, q_dict in all_results.items():
        for quantile, agg in q_dict.items():
            for lvl in levels:
                for method, vals in agg[lvl].items():
                    rows.append({
                        "Dataset": dataset,
                        "Quantile": float(quantile),
                        "Execution_Level": float(lvl),
                        "Method": method,
                        "AUC-Average": vals["AUC_mean"],
                        "AUC-Std": vals["AUC_std"],
                        "PR-AUC-Average": vals["PR_AUC_mean"],
                        "PR-AUC-Std": vals["PR_AUC_std"],
                    })
    return pd.DataFrame(rows)


def main():
    model_name = "DALSTM"
    datasets = ["P2P", "BPIC15_1", "BPIC_2017_W", "Sepsis", "BPIC20ID", "BPIC20DD", "BPIC20PTC"]
    seeds = [409, 1824, 3657, 4012, 4506]
    quantiles = [0.9, 0.95]
    levels = [0.25, 0.5, 0.75]
    
    root_path = os.getcwd()
    result_path = os.path.join(root_path, "results", model_name)

    all_results = {}

    for dataset in datasets:
        result_dir = os.path.join(result_path, dataset)
        all_results[dataset] = {}

        for q in quantiles:
            seed_results = []

            for seed in seeds:
                quantile_df, labels_name, df_lst = get_inference_results(
                    result_dir=result_dir, dataset=dataset, model=model_name, seed=seed
                )
                quantile_df = normalize_case_id(quantile_df)
                df_lst = [normalize_case_id(d) for d in df_lst]
                #q_ids = set(quantile_df["Case_id"].unique())
                #for name, d in zip(labels_name, df_lst):
                    #d_ids = set(d["Case_id"].unique())
                    #print(name, "missing in det:", len(q_ids - d_ids), "missing in quant:", len(d_ids - q_ids))

                metrics = compute_case_level_metrics_total_duration_risk(
                    df_lst=df_lst,
                    labels_name=labels_name,
                    quantile_df=quantile_df,
                    quantile=q,
                    levels=levels,
                )
                seed_results.append(metrics)

            all_results[dataset][q] = aggregate_case_level_results(seed_results, levels=levels)

    final_df = build_case_level_dataframe(all_results, levels=levels)
    final_df.to_csv("case_level_risk_metrics_total_duration.csv", index=False)
    print(final_df.head())

if __name__ == "__main__":
    main()