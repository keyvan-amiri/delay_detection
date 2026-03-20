# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 12:41:32 2026
"""

import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def get_dataframes(result_dir=None, dataset=None, model=None, seed=None):
    vanilla_name = f"{dataset}_{model}_Vanilla_wos_seed{seed}_inference.csv"
    quantile_name = f"{dataset}_{model}_quantile_wos_seed{seed}_inference.csv"
    survival_name = f"{dataset}_{model}_survival_wos_seed{seed}_inference.csv"
    vanilla_df = pd.read_csv(os.path.join(result_dir, vanilla_name))
    quantile_df = pd.read_csv(os.path.join(result_dir, quantile_name))
    survival_df = pd.read_csv(os.path.join(result_dir, survival_name))
    return vanilla_df, quantile_df, survival_df


def aggregate_vanilla_across_seeds(result_dir, dataset, model_name, seeds):
    dfs = []
    for seed in seeds:
        vanilla_df, _, _ = get_dataframes(
            result_dir=result_dir,
            dataset=dataset,
            model=model_name,
            seed=seed)
        # Keep only needed columns
        vanilla_df = vanilla_df[
            ["Case_id", "Prefix_length", "GroundTruth", "Absolute_error"]
        ].copy()
        vanilla_df = vanilla_df.rename(
            columns={"Absolute_error": f"Absolute_error_seed{seed}"})
        dfs.append(vanilla_df)
    # Merge all seed-specific dfs on the instance identifiers
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(
            df,
            on=["Case_id", "Prefix_length", "GroundTruth"],
            how="inner"
        )
    error_cols = [f"Absolute_error_seed{seed}" for seed in seeds]
    merged_df["Absolute_error_mean"] = merged_df[error_cols].mean(axis=1)
    return merged_df


def aggregate_quantile_across_seeds(result_dir, dataset, model_name, seeds):
    dfs = []
    for seed in seeds:
        _, quantile_df, _ = get_dataframes(
            result_dir=result_dir,
            dataset=dataset,
            model=model_name,
            seed=seed)
        # Keep only needed columns
        quantile_df = quantile_df[
            ["Case_id", "Prefix_length", "GroundTruth", "PI_Width_10_90"]
        ].copy()
        quantile_df = quantile_df.rename(
            columns={"PI_Width_10_90": f"PI_Width_10_90_seed{seed}"}
        )
        dfs.append(quantile_df)
    # Merge all seed-specific dfs on the instance identifiers
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(
            df,
            on=["Case_id", "Prefix_length", "GroundTruth"],
            how="inner"
        )
    pi_cols = [f"PI_Width_10_90_seed{seed}" for seed in seeds]
    merged_df["PI_Width_10_90_mean"] = merged_df[pi_cols].mean(axis=1)
    return merged_df


def aggregate_survival_across_seeds(result_dir, dataset, model_name, seeds):
    dfs = []
    for seed in seeds:
        _, _, survival_df = get_dataframes(
            result_dir=result_dir,
            dataset=dataset,
            model=model_name,
            seed=seed)
        # Keep only needed columns
        survival_df = survival_df[
            ["Case_id", "Prefix_length", "GroundTruth", "PI80_width"]
        ].copy()
        survival_df = survival_df.rename(
            columns={"PI80_width": f"PI80_width_seed{seed}"}
        )
        dfs.append(survival_df)
    # Merge all seed-specific dfs on the instance identifiers
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(
            df,
            on=["Case_id", "Prefix_length", "GroundTruth"],
            how="inner"
        )
    pi_cols = [f"PI80_width_seed{seed}" for seed in seeds]
    merged_df["PI80_width_mean"] = merged_df[pi_cols].mean(axis=1)
    return merged_df


def compute_spearman_results(result_dir, dataset, model_name, seeds):
    vanilla_agg = aggregate_vanilla_across_seeds(
        result_dir=result_dir,
        dataset=dataset,
        model_name=model_name,
        seeds=seeds)
    spearman1 = vanilla_agg["GroundTruth"].corr(
        vanilla_agg["Absolute_error_mean"],
        method="spearman")
    quantile_agg = aggregate_quantile_across_seeds(
        result_dir=result_dir,
        dataset=dataset,
        model_name=model_name,
        seeds=seeds)
    spearman2 = quantile_agg["GroundTruth"].corr(
        quantile_agg["PI_Width_10_90_mean"],
        method="spearman")
    survival_agg = aggregate_survival_across_seeds(
        result_dir=result_dir,
        dataset=dataset,
        model_name=model_name,
        seeds=seeds)
    spearman3 = survival_agg["GroundTruth"].corr(
        survival_agg["PI80_width_mean"],
        method="spearman")
    return spearman1, spearman2, spearman3


def main():
    model_name = "DALSTM"
    datasets = ["P2P", "BPIC_2017_W",
                "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5",
                "HelpDesk", "Sepsis",
                "BPIC20ID", "BPIC20DD", "BPIC20PTC", "BPIC20TPD", "BPIC20RFP",] 
    seeds = [409, 1824, 3657, 4012, 4506]
    root_path = os.getcwd()
    result_path = os.path.join(root_path, "results", model_name)
    results = []
    for dataset in datasets:
        result_dir = os.path.join(result_path, dataset)
        spearman1, spearman2, spearman3 = compute_spearman_results(
            result_dir=result_dir,
            dataset=dataset,
            model_name=model_name,
            seeds=seeds)
        results.append({
            "dataset": dataset,
            "Spearman1": spearman1,
            "Spearman2": spearman2,
            "Spearman3": spearman3})
    results_df = pd.DataFrame(results)
    print(results_df)
    save_path = os.path.join(result_path, "spearman_summary.csv")
    results_df.to_csv(save_path, index=False)
    print(f"\nSaved results to: {save_path}")


if __name__ == "__main__":
    main()