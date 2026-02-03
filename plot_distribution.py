# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 11:08:35 2026
@author: Keyvan Amiri Elyasi
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
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


def distribution_plot(dataset, model, df_lst, loaded_labels, result_folder, seed_tag="seed409"):
    gt = df_lst[0]["GroundTruth"].to_numpy()

    all_vals = [gt] + [df["Prediction"].to_numpy() for df in df_lst]
    all_vals = np.concatenate(all_vals)
    xmin, xmax = np.nanmin(all_vals), np.nanmax(all_vals)
    x = np.linspace(xmin, xmax, 1000)

    plt.figure(figsize=(9, 5))

    kde_gt = gaussian_kde(gt[~np.isnan(gt)])
    plt.plot(x, kde_gt(x), label="GroundTruth")

    for df, lab in zip(df_lst, loaded_labels):
        preds = df["Prediction"].to_numpy()
        preds = preds[~np.isnan(preds)]
        kde_p = gaussian_kde(preds)
        plt.plot(x, kde_p(x), label=f"Pred: {lab}")

    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title(f"GroundTruth vs Predictions distribution ({dataset} / {model})")
    plt.legend()

    out_pdf = os.path.join(result_folder, f"{dataset}_{model}_gt_vs_preds_kde_{seed_tag}.pdf")
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf")
    plt.close()
    print(f"[OK] Saved KDE PDF: {out_pdf}")

def main():
    dataset = "BPIC20PTC"  # HelpDesk / BPIC20PTC
    model = "DALSTM"
    IR_lst = ["Vanilla", "EAL"]
    smooth_lst = ["wos", "wos"]
    labels = ["Vanilla", "EAL"]
    #IR_lst = ['Vanilla', 'CSW', 'EAL', 'BMSE', 'SERA']
    #smooth_lst = ['wos', 'wos', 'wos', 'wos', 'wos']
    #labels = ['Vanilla', 'CSW', 'EAL', 'BMSE', 'SERA']
    filter_length = True
    selected_length = 2

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

    # 1) KDE distribution plot
    distribution_plot(dataset, model, df_lst, loaded_labels, result_folder, seed_tag=seed_tag)
    if not filter_length:
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