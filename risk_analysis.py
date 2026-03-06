# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 12:41:32 2026
@author: kamirel
"""
import os
import pickle
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings("ignore")


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

def compute_tau_from_train_val(
    train_cases,
    val_cases,
    y_train,
    y_val,
    quantile
):
    """
    Compute quantile threshold τ using train + validation ground truth.

    Assumes y_* contains remaining time values per prefix.
    True total duration per case = remaining time at prefix=1.
    """

    train_cases = np.asarray(train_cases)
    val_cases = np.asarray(val_cases)

    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    df_train = pd.DataFrame({
        "Case_id": train_cases,
        "GT": y_train
    })

    df_val = pd.DataFrame({
        "Case_id": val_cases,
        "GT": y_val
    })

    df = pd.concat([df_train, df_val], ignore_index=True)

    # first prefix per case = prefix=1
    df = df.groupby("Case_id").first()

    true_total = df["GT"].astype(float)

    tau = float(true_total.quantile(quantile))

    return tau

def compute_case_level_metrics_total_duration_risk(
    df_lst: list[pd.DataFrame],
    labels_name: list[str],
    quantile_df: pd.DataFrame,
    tau: float,
    quantile: float,
    levels=(0.25, 0.5, 0.75),
    *,
    gt_col="GroundTruth",
    pred_col="Prediction",
    use_pi_width_for_quantile_risk=False,
) -> dict:

    assert len(df_lst) == len(labels_name)

    q_str = str(quantile).replace(".", "_")
    q_col = f"Q{q_str}"
    pi_col = "PI_Width_10_90"
    if use_pi_width_for_quantile_risk:
        quantile_score_col = pi_col
    else:
        quantile_score_col = q_col
    has_quantile = quantile_score_col in quantile_df.columns

    true_total = compute_case_total_duration(quantile_df, gt_col=gt_col)
    z_case = (true_total > tau).astype(int)

    quantile_df_e = add_elapsed_time(quantile_df, gt_col=gt_col)
    det_df_e_list = [add_elapsed_time(d, gt_col=gt_col) for d in df_lst]

    q_snaps = extract_case_level_snapshots(quantile_df_e, levels=levels)
    det_snaps = [extract_case_level_snapshots(d, levels=levels) for d in det_df_e_list]

    out = {"tau": tau, "positive_rate": float(z_case.mean())}

    for lvl in levels:

        out[lvl] = {}

        # -------- quantile model (only if column exists) --------
        if has_quantile:
            q_lvl = q_snaps[lvl].dropna(
                subset=["Case_id", "Elapsed", quantile_score_col]).copy()
            q_risk = (
                q_lvl["Elapsed"].astype(float)
                + q_lvl[quantile_score_col].astype(float)
            )

            q_cases = q_lvl["Case_id"].astype(str).values
            y_true = z_case.reindex(q_cases).values
            out[lvl]["Quantile"] = {
                "AUC": _safe_auc(y_true, q_risk.values),
                "PR_AUC": _safe_prauc(y_true, q_risk.values),
                "n_cases": int(len(q_lvl)),
            }
        # -------- deterministic models --------
        for name, snaps in zip(labels_name, det_snaps):

            d_lvl = snaps[lvl].dropna(
                subset=["Case_id", "Elapsed", pred_col]
            ).copy()

            d_risk = (
                d_lvl["Elapsed"].astype(float)
                + d_lvl[pred_col].astype(float)
            )

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

def load_case_ids(temp_dir, dataset, model_name):    
    train_case_lst_path = os.path.join(temp_dir, model_name+'_train_cases_'+dataset+'.pkl') 
    val_case_lst_path = os.path.join(temp_dir, model_name+'_val_cases_'+dataset+'.pkl') 
    test_case_lst_path = os.path.join(temp_dir, model_name+'_test_cases_'+dataset+'.pkl') 
    with open(train_case_lst_path, "rb") as f:
        train_cases = pickle.load(f)
    with open(val_case_lst_path, "rb") as f:
        val_cases = pickle.load(f)
    with open(test_case_lst_path, "rb") as f:
        test_cases = pickle.load(f)
    return (train_cases, val_cases, test_cases)
    
def get_ground_truth(temp_dir, model_name, dataset):
    y_train_path = os.path.join(temp_dir, model_name+'_y_train_'+dataset+'.pt')
    y_val_path = os.path.join(temp_dir, model_name+'_y_val_'+dataset+'.pt')
    y_test_path = os.path.join(temp_dir, model_name+'_y_test_'+dataset+'.pt')
    y_train = torch.load(y_train_path)
    y_val = torch.load(y_val_path)
    y_test = torch.load(y_test_path)
    return (y_train, y_val, y_test)   
   
def plot_early_detection_pr_auc(
    results_df: pd.DataFrame,
    methods: list[str],
    quantile: float,
    pdf_path: str,
    *,
    execution_levels=(0.25, 0.5, 0.75),
    dataset: str | None = None,
    aggregate_across_datasets: bool = True,
    title: str | None = None,
):
    """
    Plot early detection ability:
        x-axis: Execution_Level (e.g., 0.25, 0.5, 0.75)
        y-axis: PR-AUC (uses column 'PR-AUC-Average')
        lines: Methods
    Saves figure as a PDF to `pdf_path`.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns:
        ['Dataset','Quantile','Execution_Level','Method','PR-AUC-Average'].
        (Optionally 'PR-AUC-Std' if you want to extend later.)
    methods : list[str]
        Methods to plot (e.g., ['Vanilla','CSW','EAL','BMSE','SERA','Quantile']).
    quantile : float
        Which Quantile value to filter on (e.g., 0.9).
    pdf_path : str
        Output path to save the PDF (directories will be created if needed).
    execution_levels : tuple
        Levels to display on x-axis.
    dataset : str | None
        If provided, restrict plot to a single dataset.
    aggregate_across_datasets : bool
        If True (default), averages PR-AUC across datasets for each (Method, Execution_Level).
        If False, and dataset is None, it will plot per-dataset curves is ambiguous; keep True.
    title : str | None
        Custom plot title. If None, a sensible default is used.
    """

    required_cols = {"Dataset", "Quantile", "Execution_Level", "Method", "PR-AUC-Average"}
    missing = required_cols - set(results_df.columns)
    if missing:
        raise ValueError(f"results_df is missing required columns: {sorted(missing)}")

    df = results_df.copy()

    # Robust float matching for quantile/execution level
    df["Quantile"] = df["Quantile"].astype(float)
    df["Execution_Level"] = df["Execution_Level"].astype(float)

    df = df[np.isclose(df["Quantile"].values, float(quantile))]

    if dataset is not None:
        df = df[df["Dataset"] == dataset]

    df = df[df["Method"].isin(methods)]
    df = df[df["Execution_Level"].isin(execution_levels)]

    if df.empty:
        scope = f"dataset={dataset}, " if dataset is not None else ""
        raise ValueError(
            f"No rows to plot after filtering ({scope}quantile={quantile}, methods={methods})."
        )

    # Aggregate across datasets (recommended for concise plot)
    if dataset is None and aggregate_across_datasets:
        plot_df = (
            df.groupby(["Method", "Execution_Level"], as_index=False)["PR-AUC-Average"]
            .mean()
        )
    else:
        # If dataset is specified, we already have the right slice.
        # If user sets aggregate_across_datasets=False without dataset, we still aggregate to avoid clutter.
        plot_df = (
            df.groupby(["Method", "Execution_Level"], as_index=False)["PR-AUC-Average"]
            .mean()
        )

    # Ensure consistent x ordering
    x_levels = list(execution_levels)

    fig, ax = plt.subplots()
    for m in methods:
        m_df = plot_df[plot_df["Method"] == m].copy()
        if m_df.empty:
            continue
        y_by_x = (
            m_df.set_index("Execution_Level")["PR-AUC-Average"]
            .reindex(x_levels)
            .astype(float)
        )
        ax.plot(x_levels, y_by_x.values, marker="o", label=m)

    ax.set_xlabel("Prefix length to case length ratio")
    ax.set_ylabel(f"PR-AUC (quantile={quantile})")
    ax.set_xticks(x_levels)

    if title is None:
        if dataset is None:
            title = f"Early detection ability (PR-AUC) at quantile={quantile}"
        else:
            title = f"{dataset} — Early detection ability (PR-AUC) at quantile={quantile}"
    #ax.set_title(title)

    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    
def main():
    model_name = "DALSTM"
    datasets = ["P2P", "BPIC15_1", "BPIC_2017_W", "Sepsis", "BPIC20ID", "BPIC20DD", "BPIC20PTC"]
    seeds = [409, 1824, 3657, 4012, 4506]
    quantiles = [0.8, 0.9, 0.95]
    levels = [0.25, 0.5, 0.75]
    use_pi_width_for_quantile_risk = False
    
    root_path = os.getcwd()
    result_path = os.path.join(root_path, "results", model_name)    
    temp_path = os.path.join(root_path, "temp", model_name)
     
    all_results = {}

    for dataset in datasets:
        result_dir = os.path.join(result_path, dataset)
        temp_dir = os.path.join(temp_path, dataset)
        (train_cases, val_cases, test_cases) = load_case_ids(
            temp_dir, dataset, model_name)
        (y_train, y_val, y_test) = get_ground_truth(temp_dir, model_name, dataset)
        all_results[dataset] = {}

        for q in quantiles:
            seed_results = []
            tau = compute_tau_from_train_val(
                train_cases, val_cases, y_train, y_val, q)
            for seed in seeds:
                quantile_df, labels_name, df_lst = get_inference_results(
                    result_dir=result_dir, dataset=dataset, model=model_name, seed=seed
                )
                quantile_df = normalize_case_id(quantile_df)
                df_lst = [normalize_case_id(d) for d in df_lst]
                
                metrics = compute_case_level_metrics_total_duration_risk(
                    df_lst=df_lst,
                    labels_name=labels_name,
                    quantile_df=quantile_df,
                    tau=tau,
                    quantile=q,
                    levels=levels,
                    use_pi_width_for_quantile_risk=use_pi_width_for_quantile_risk,)
                seed_results.append(metrics)

            all_results[dataset][q] = aggregate_case_level_results(seed_results, levels=levels)

    final_df = build_case_level_dataframe(all_results, levels=levels)
    final_df.to_csv("case_level_risk_metrics_total_duration.csv", index=False)
    print(final_df.head())
    plot_early_detection_pr_auc(
        results_df=final_df,
        methods=["Vanilla", "CSW", "EAL", "BMSE", "SERA"],
        quantile=0.95,
        pdf_path="early_detection_q0_95.pdf")
    plot_early_detection_pr_auc(
        results_df=final_df,
        methods=["Vanilla", "CSW", "EAL", "BMSE", "SERA"],
        quantile=0.9,
        pdf_path="early_detection_q0_9.pdf")
    plot_early_detection_pr_auc(
        results_df=final_df,
        methods=["Vanilla", "CSW", "EAL", "BMSE", "SERA"],
        quantile=0.8,
        pdf_path="early_detection_q0_8.pdf")


if __name__ == "__main__":
    main()