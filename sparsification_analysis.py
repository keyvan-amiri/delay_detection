# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 12:41:32 2026
"""

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def get_inference_results(result_dir=None, dataset=None, model=None, seed=None, mode='quantile'):
    if mode=='quantile':
        df_name = f"{dataset}_{model}_quantile_wos_seed{seed}_inference.csv"
    elif mode=='survival':
        df_name = f"{dataset}_{model}_survival_wos_seed{seed}_inference.csv"
    df = pd.read_csv(os.path.join(result_dir, df_name))
    return df


def get_target_stats(temp_dir, model_name, dataset):
    y_train_path = os.path.join(temp_dir, model_name + '_y_train_' + dataset + '.pt')
    y_val_path = os.path.join(temp_dir, model_name + '_y_val_' + dataset + '.pt')
    y_train = torch.load(y_train_path)
    y_val = torch.load(y_val_path)
    y_train_val = torch.cat([y_train, y_val])
    median_rt = float(y_train_val.median())
    quantile_60, quantile_90 = torch.quantile(y_train_val, torch.tensor([0.6, 0.9]))
    q60 = float(quantile_60)
    q90 = float(quantile_90)
    return median_rt, q60, q90


def aggregate_quantile_across_seeds(
    result_dir,
    dataset,
    model_name,
    seeds,
    error_col="Absolute_error",
    uncertainty_col="PI_Width_10_90",
    mode='quantile',
):
    """
    Align rows across seeds by instance ID and compute mean error/uncertainty per instance.
    """
    dfs = []

    for seed in seeds:
        df = get_inference_results(
            result_dir=result_dir,
            dataset=dataset,
            model=model_name,
            seed=seed,
            mode=mode,
        ).copy()
        keep_cols = ["Case_id", "Prefix_length", "GroundTruth", error_col, uncertainty_col]
        df = df[keep_cols].copy()
        df = df.rename(columns={
            error_col: f"{error_col}_seed{seed}",
            uncertainty_col: f"{uncertainty_col}_seed{seed}",
        })
        dfs.append(df)
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(
            df,
            on=["Case_id", "Prefix_length", "GroundTruth"],
            how="inner"
        )
    err_cols = [f"{error_col}_seed{seed}" for seed in seeds]
    unc_cols = [f"{uncertainty_col}_seed{seed}" for seed in seeds]
    merged["Mean_Absolute_error"] = merged[err_cols].mean(axis=1)
    merged["Mean_PI_Width_10_90"] = merged[unc_cols].mean(axis=1)
    return merged


def sparsification_analysis_pdf(
    dataset,
    df,
    result_dir,
    train_median,
    error_col="Mean_Absolute_error",
    uncertainty_col="Mean_PI_Width_10_90",
    n_steps=50,
    n_random_runs=20,
):
    """
    Sparsification analysis using normalized error.

    Normalization:
        normalized_error_i = error_i / mean(|y - median_train|)
    so that dataset scale differences are reduced while ranking is unchanged.
    """

    plot_prefix = os.path.join(result_dir, f"{dataset}_normalized_sparsification")

    # fixed dataset-level denominator
    denom = (df["GroundTruth"] - train_median).abs().mean()
    if denom == 0:
        raise ValueError(f"Normalization denominator is zero for dataset {dataset}.")

    err = (df[error_col] / denom).to_numpy(dtype=float)
    unc = df[uncertainty_col].to_numpy(dtype=float)

    # remove invalid rows
    valid_mask = np.isfinite(err) & np.isfinite(unc)
    err = err[valid_mask]
    unc = unc[valid_mask]

    N = len(err)
    if N == 0:
        raise ValueError(f"No valid rows found for dataset {dataset}.")

    fracs = np.linspace(0, 0.9, n_steps)

    def curve_from_order(order):
        vals = []
        for f in fracs:
            k = int(f * N)
            keep = order[k:]
            if len(keep) == 0:
                vals.append(np.nan)
            else:
                vals.append(err[keep].mean())
        return np.array(vals, dtype=float)

    # highest uncertainty / error removed first
    order_unc = np.argsort(-unc)
    order_oracle = np.argsort(-err)

    curve_unc = curve_from_order(order_unc)
    curve_oracle = curve_from_order(order_oracle)

    curves_rand = []
    for _ in range(n_random_runs):
        order_rand = np.random.permutation(N)
        curves_rand.append(curve_from_order(order_rand))
    curve_rand = np.nanmean(curves_rand, axis=0)

    sparsification_error = curve_unc - curve_oracle

    nAUSE = np.trapz(sparsification_error, fracs)
    nAURG = np.trapz(curve_rand - curve_unc, fracs)

    denom_ratio = nAUSE + nAURG
    nAURG_ratio = float(nAURG / denom_ratio) if denom_ratio != 0 else np.nan

    # ---- plots ----
    plt.figure(figsize=(7, 5))
    plt.plot(fracs, curve_unc, label="Uncertainty-based", linewidth=2)
    plt.plot(fracs, curve_oracle, label="Oracle", linestyle="--")
    plt.plot(fracs, curve_rand, label="Random", linestyle=":")
    plt.xlabel("Fraction removed")
    plt.ylabel("Mean normalized error (remaining)")
    plt.title(f"{dataset}: Normalized Sparsification Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_prefix + "_curves.pdf")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(fracs, sparsification_error, linewidth=2)
    plt.fill_between(fracs, 0, sparsification_error, alpha=0.25)
    plt.xlabel("Fraction removed")
    plt.ylabel("Normalized sparsification error")
    plt.title(f"{dataset}: Normalized Sparsification Error\nnAUSE={nAUSE:.4f}  nAURG={nAURG:.4f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_prefix + "_error.pdf")
    plt.close()

    return {
        "nAUSE": float(nAUSE),
        "nAURG": float(nAURG),
        "nAURG_ratio": nAURG_ratio,
        "fractions": fracs,
        "unc_curve": curve_unc,
        "oracle_curve": curve_oracle,
        "random_curve": curve_rand,
        "sparsification_error": sparsification_error,
        "normalization_denominator": float(denom),
        "n_instances": int(N),
    }


def main():
    # ---- settings ----
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
    datasets = [
        "P2P",
        "BPIC15_2",
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
    uncertainty_model = 'survival' # 'quantile' 'survival'
    error_col = "Absolute_error"
    uncertainty_col = "PI80_width" #"PI_Width_10_90" "PI80_width" "PI90_width" "Tail_mass"

    # ---- paths ----
    root_path = os.getcwd()
    result_path = os.path.join(root_path, "results", model_name)
    temp_path = os.path.join(root_path, "temp", model_name)

    summary_rows = []

    for dataset in datasets:
        print(f"Processing {dataset} ...")
        result_dir = os.path.join(result_path, dataset)
        temp_dir = os.path.join(temp_path, dataset)

        train_median, _, _ = get_target_stats(temp_dir, model_name, dataset)

        # aggregate across seeds first
        agg_df = aggregate_quantile_across_seeds(
            result_dir=result_dir,
            dataset=dataset,
            model_name=model_name,
            seeds=seeds,
            error_col=error_col,
            uncertainty_col=uncertainty_col,
            mode = uncertainty_model,
        )

        # sparsification on seed-averaged values
        metrics = sparsification_analysis_pdf(
            dataset=dataset,
            df=agg_df,
            result_dir=result_dir,
            train_median=train_median,
            error_col="Mean_Absolute_error",
            uncertainty_col="Mean_PI_Width_10_90",
            n_steps=50,
            n_random_runs=20,
        )

        summary_rows.append({
            "dataset": dataset,
            "nAUSE": metrics["nAUSE"],
            "nAURG": metrics["nAURG"],
            "nAURG_ratio": metrics["nAURG_ratio"],
        })

    summary_df = pd.DataFrame(summary_rows)

    save_path = os.path.join(result_path, "normalized_sparsification_summary.csv")
    summary_df.to_csv(save_path, index=False)

    print("\nNormalized sparsification summary:")
    print(summary_df)
    print(f"\nSaved summary to: {save_path}")


if __name__ == "__main__":
    main()