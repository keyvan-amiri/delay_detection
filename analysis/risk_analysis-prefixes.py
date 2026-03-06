# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 12:41:32 2026
@author: kamirel
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    ROC-AUC is undefined if y_true has only one class. Return np.nan in that case.
    """
    y_true = np.asarray(y_true).astype(int)
    if np.unique(y_true).size < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def _safe_prauc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    PR-AUC (Average Precision) is defined even under imbalance,
    but if y_true has only one class it's not meaningful. Return np.nan.
    """
    y_true = np.asarray(y_true).astype(int)
    if np.unique(y_true).size < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def get_risk_metrics(
    df_lst: list[pd.DataFrame],
    labels_name: list[str],
    quantile_df: pd.DataFrame,
    quantile: float,
    *,
    gt_col: str = "GroundTruth",
    pred_col: str = "Prediction",
) -> dict:
    """
    Deterministic models:
        Risk score = Prediction

    Quantile model:
        Risk score = corresponding upper quantile:
            0.9  -> Q0_9
            0.95 -> Q0_95

    Extreme label:
        Z = 1 if GroundTruth > tau_q
        where tau_q is q-th percentile of GroundTruth
    """

    assert quantile in [0.9, 0.95], "This setting only supports 0.9 and 0.95"

    assert len(df_lst) == len(labels_name)

    # Define correct quantile column
    quantile_col_map = {
        0.9: "Q0_9",
        0.95: "Q0_95",
    }

    q_col = quantile_col_map[quantile]

    # Threshold tau based on GT distribution
    ref = quantile_df.dropna(subset=[gt_col]).copy()
    tau = float(ref[gt_col].quantile(quantile))

    out = {"tau": tau}

    def _compute_metrics(df: pd.DataFrame, risk_col: str) -> dict:
        d = df.dropna(subset=[gt_col, risk_col]).copy()
        y = d[gt_col].to_numpy(dtype=float)
        z = (y > tau).astype(int)
        s = d[risk_col].to_numpy(dtype=float)

        if len(np.unique(z)) < 2:
            return {"AUC": np.nan, "PR_AUC": np.nan}

        return {
            "AUC": roc_auc_score(z, s),
            "PR_AUC": average_precision_score(z, s),
        }

    # Deterministic models → use Prediction
    for name, df in zip(labels_name, df_lst):
        out[name] = _compute_metrics(df, pred_col)

    # Quantile model → use Q0_9 or Q0_95
    out["Quantile"] = _compute_metrics(quantile_df, q_col)

    return out


def aggregate_risk_metrics(seed_results: list[dict]) -> dict:
    """
    Aggregate per-seed outputs from get_risk_metrics().

    Input: list of dicts (one per seed), each like:
      {"tau": ..., "Vanilla": {"AUC":..., "PR_AUC":...}, ..., "Quantile": {...}}

    Output:
      {
        "Vanilla": {"AUC_mean":..., "AUC_std":..., "PR_AUC_mean":..., "PR_AUC_std":..., "seeds":k},
        ...
        "Quantile": {...},
        "tau_mean":..., "tau_std":...
      }
    """
    if not seed_results:
        return {}

    # collect method keys (excluding "tau")
    method_keys = sorted([k for k in seed_results[0].keys() if k != "tau"])

    agg = {}

    # tau
    taus = np.array([r.get("tau", np.nan) for r in seed_results], dtype=float)
    agg["tau_mean"] = float(np.nanmean(taus))
    agg["tau_std"] = float(np.nanstd(taus, ddof=1)) if np.sum(~np.isnan(taus)) > 1 else 0.0

    for m in method_keys:
        aucs = np.array([r.get(m, {}).get("AUC", np.nan) for r in seed_results], dtype=float)
        pras = np.array([r.get(m, {}).get("PR_AUC", np.nan) for r in seed_results], dtype=float)

        k_auc = int(np.sum(~np.isnan(aucs)))
        k_pr = int(np.sum(~np.isnan(pras)))

        agg[m] = {
            "AUC_mean": float(np.nanmean(aucs)),
            "AUC_std": float(np.nanstd(aucs, ddof=1)) if k_auc > 1 else 0.0,
            "PR_AUC_mean": float(np.nanmean(pras)),
            "PR_AUC_std": float(np.nanstd(pras, ddof=1)) if k_pr > 1 else 0.0,
            "seeds_auc": k_auc,
            "seeds_pr": k_pr,
        }

    return agg


def get_dtaframe_result(all_results: dict) -> pd.DataFrame:
    """
    Convert aggregated results into one long dataframe with columns:
      Dataset, Quantile, Method, AUC-Average, AUC-Std, PR-AUC-Average, PR-AUC-Std

    Expected input structure:
      all_results[dataset][quantile] = aggregated_dict
    where aggregated_dict is the output of aggregate_risk_metrics().
    """
    rows = []
    for dataset, q_dict in all_results.items():
        for q, agg in q_dict.items():
            for method, vals in agg.items():
                if method.startswith("tau_"):
                    continue
                rows.append(
                    {
                        "Dataset": dataset,
                        "Quantile": float(q),
                        "Method": method,
                        "AUC-Average": vals.get("AUC_mean", np.nan),
                        "AUC-Std": vals.get("AUC_std", np.nan),
                        "PR-AUC-Average": vals.get("PR_AUC_mean", np.nan),
                        "PR-AUC-Std": vals.get("PR_AUC_std", np.nan),
                    }
                )
    return pd.DataFrame(rows)

def main():
    # ---- settings ----
    model_name = "DALSTM"
    datsets = ["P2P", "BPIC15_1", "BPIC_2017_W", "Sepsis", "BPIC20ID", "BPIC20DD", "BPIC20PTC"]
    seeds = [409, 1824, 3657, 4012, 4506]
    quantiles = [0.9, 0.95]
    
    # paths
    root_path = os.getcwd()
    result_path = os.path.join(root_path, "results", model_name)
    # all_results[dataset][quantile] = aggregated_metrics
    all_results = {}
    for dataset in datsets:
       result_dir = os.path.join(result_path, dataset)
       all_results[dataset] = {}

       for q in quantiles:
           seed_results = []

           for seed in seeds:
               quantile_df, labels_name, df_lst = get_inference_results(
                   result_dir=result_dir, dataset=dataset, model=model_name, seed=seed
               )

               metrics = get_risk_metrics(
                   df_lst=df_lst,
                   labels_name=labels_name,
                   quantile_df=quantile_df,
                   quantile=q,
               )
               seed_results.append(metrics)

           all_results[dataset][q] = aggregate_risk_metrics(seed_results)

    df_result = get_dtaframe_result(all_results)

    # Optional: save
    out_csv = os.path.join(root_path, f"risk_metrics_{model_name}.csv")
    df_result.to_csv(out_csv, index=False)
    print("[OK] wrote", out_csv)
    print(df_result.head())

if __name__ == "__main__":
    main()