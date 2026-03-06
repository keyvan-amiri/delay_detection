# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 07:52:18 2025
@author: Keyvan Amiri Elyasi
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt


def aggregate_quantile_seed_dfs(df_list):
    if len(df_list) == 0:
        raise ValueError("Empty dataframe list")
    base = df_list[0].copy()
    id_cols = ["Case_id", "Prefix_length", "GroundTruth"]
    quantile_cols = [c for c in base.columns if c.startswith("Q")]
    pi_bound_cols = [c for c in ["PI10", "PI90"] if c in base.columns]
    avg_cols = ["Prediction"] + quantile_cols + pi_bound_cols
    # ---------- alignment check ----------
    for df in df_list[1:]:
        if not base[id_cols].equals(df[id_cols]):
            raise ValueError("DataFrames not aligned")
    # ---------- stack predictions across seeds ----------
    pred_stack = np.stack([df["Prediction"].to_numpy() for df in df_list], axis=0)
    epistemic_std = pred_stack.std(axis=0)
    epistemic_var = pred_stack.var(axis=0)
    # ---------- mean aggregation ----------
    for col in avg_cols:
        base[col] = np.mean([df[col].to_numpy() for df in df_list], axis=0)
    # ---------- derived metrics ----------
    base["Absolute_error"] = np.abs(base["GroundTruth"] - base["Prediction"])
    if "PI10" in base.columns and "PI90" in base.columns:
        base["PI_Width_10_90"] = base["PI90"] - base["PI10"]
        base["PI_Coverage_10_90"] = (
            (base["GroundTruth"] >= base["PI10"]) &
            (base["GroundTruth"] <= base["PI90"])
        ).astype(float)
    # ---------- epistemic ----------
    base["Epistemic_std"] = epistemic_std
    base["Epistemic_var"] = epistemic_var
    # ---------- aleatoric proxy ----------
    if "PI_Width_10_90" in base.columns:
        base["Aleatoric_proxy"] = base["PI_Width_10_90"]
        # combine (variance-style)
        base["Total_uncertainty"] = np.sqrt(
            base["Aleatoric_proxy"]**2 +
            base["Epistemic_std"]**2
        )
    return base


def plot_uncertainty_vs_target_pdf(
        args, 
        df,
        result_dir,
        target_col="GroundTruth",
        uncertainty_col="PI_Width_10_90",
        n_bins=10,
        binning="quantile",   # "quantile" or "width"
        ):    
    df = df.copy()
    # --- choose binning strategy ---
    if binning == "quantile":
        df["target_bin"] = pd.qcut(df[target_col], q=n_bins, duplicates="drop")
    elif binning == "width":
        df["target_bin"] = pd.cut(df[target_col], bins=n_bins)
    else:
        raise ValueError("binning must be 'quantile' or 'width'")
    # --- aggregate stats per bin ---
    stats = (
        df.groupby("target_bin")[uncertainty_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    # --- bin centers ---
    centers = np.array([interval.mid for interval in stats["target_bin"]])
    means = stats["mean"].to_numpy()
    stds = stats["std"].fillna(0).to_numpy()
    lower = means - stds
    upper = means + stds
    # --- plot line + shaded band ---
    plt.figure(figsize=(7, 5))
    plt.plot(centers, means, marker="o", label="Mean uncertainty")
    plt.fill_between(centers, lower, upper, alpha=0.25, label="±1 std")
    plt.xlabel("Target value (bin center)")
    plt.ylabel("Prediction interval width (uncertainty)")
    plt.title(f"{args.dataset}: Uncertainty vs Target Value Range")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if args.log_trans:
        plot_path = os.path.join(result_dir, args.dataset+"_logtrans_uncertainty_plot.pdf")
    elif args.box_cox:
        plot_path = os.path.join(result_dir, args.dataset+"_boxcox_uncertainty_plot.pdf")
    else:
        plot_path = os.path.join(result_dir, args.dataset+"_uncertainty_plot.pdf") 
    plt.savefig(plot_path, format="pdf")
    plt.close()
    return stats

def plot_uncertainty_vs_prefixlen_pdf(
        args,
        df,
        result_dir,
        prefix_col="Prefix_length",
        uncertainty_col="PI_Width_10_90",
        n_bins=5,
        binning="quantile",
        verbose=True):
    df = df.copy()
    if binning == "quantile":
        df["prefix_bin"] = pd.qcut(df[prefix_col], q=n_bins, duplicates="drop")
    elif binning == "width":
        df["prefix_bin"] = pd.cut(df[prefix_col], bins=n_bins)
    else:
        raise ValueError("binning must be 'quantile' or 'width'")
    stats = (
        df.groupby("prefix_bin")[uncertainty_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    if verbose:
        print(f"Requested bins: {n_bins}, actual bins: {len(stats)}")
        print(stats[["prefix_bin", "count"]])
    centers = np.array([interval.mid for interval in stats["prefix_bin"]])
    means = stats["mean"].to_numpy()
    stds = stats["std"].fillna(0).to_numpy()
    plt.figure(figsize=(7, 5))
    plt.plot(centers, means, marker="o", label="Mean uncertainty")
    plt.fill_between(centers, means - stds, means + stds, alpha=0.25, label="±1 std")
    plt.xlabel("Prefix length (bin center)")
    plt.ylabel("Prediction interval width (uncertainty)")
    plt.title(f"{args.dataset}: Uncertainty vs Prefix Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if args.log_trans:
        plot_path = os.path.join(result_dir, args.dataset+"_logtrans_uncertainty_plot_prefix_length.pdf")
    elif args.box_cox:
        plot_path = os.path.join(result_dir, args.dataset+"_boxcox_uncertainty_plot_prefix_length.pdf")
    else:
        plot_path = os.path.join(result_dir, args.dataset+"_uncertainty_plot_prefix_length.pdf")          
    plt.savefig(plot_path, format="pdf")
    plt.close()
    return stats

def plot_coverage_vs_target_pdf(
        args,
        df,
        result_dir,
        target_col="GroundTruth",
        coverage_col="PI_Coverage_10_90",
        n_bins=10,
        binning="quantile",   # "quantile" or "width"
        ):
    df = df.copy()
    # --- choose binning strategy ---
    if binning == "quantile":
        df["target_bin"] = pd.qcut(df[target_col], q=n_bins, duplicates="drop")
    elif binning == "width":
        df["target_bin"] = pd.cut(df[target_col], bins=n_bins)
    else:
        raise ValueError("binning must be 'quantile' or 'width'")
    # --- aggregate coverage stats ---
    stats = (
        df.groupby("target_bin")[coverage_col]
        .agg(["mean", "count"])
        .reset_index()
    )
    centers = np.array([interval.mid for interval in stats["target_bin"]])
    p = stats["mean"].to_numpy()          # empirical coverage
    n = stats["count"].to_numpy()
    # --- binomial 95% CI ---
    se = np.sqrt(p * (1 - p) / np.maximum(n, 1))
    lower = np.clip(p - 1.96 * se, 0, 1)
    upper = np.clip(p + 1.96 * se, 0, 1)
    # --- plot ---
    plt.figure(figsize=(7, 5))
    plt.plot(centers, p, marker="o", label="Empirical coverage")
    plt.fill_between(centers, lower, upper, alpha=0.25, label="95% CI")
    plt.axhline(0.8, linestyle="--", linewidth=1, label="Ideal 80%")
    plt.xlabel("Target value (bin center)")
    plt.ylabel("PI10–90 Coverage")
    plt.title(f"{args.dataset}: Coverage vs Target Value Range")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if args.log_trans:
        plot_path = os.path.join(result_dir, args.dataset+"_logtrans_coverage_plot.pdf")
    elif args.box_cox:
        plot_path = os.path.join(result_dir, args.dataset+"_boxcox_coverage_plot.pdf")
    else:
        plot_path = os.path.join(result_dir, args.dataset+"_coverage_plot.pdf")   
    plt.savefig(plot_path, format="pdf")
    plt.close()
    return stats

def plot_coverage_vs_prefixlen_pdf(
        args,
        df,
        result_dir,
        prefix_col="Prefix_length",
        coverage_col="PI_Coverage_10_90",
        n_bins=10,
        binning="quantile",   # "quantile" or "width"
        ):
    
    df = df.copy()
    # --- choose binning strategy ---
    if binning == "quantile":
        df["prefix_bin"] = pd.qcut(df[prefix_col], q=n_bins, duplicates="drop")
    elif binning == "width":
        df["prefix_bin"] = pd.cut(df[prefix_col], bins=n_bins)
    else:
        raise ValueError("binning must be 'quantile' or 'width'")
    # --- aggregate coverage stats ---
    stats = (
        df.groupby("prefix_bin")[coverage_col]
        .agg(["mean", "count"])
        .reset_index()
    )
    centers = np.array([interval.mid for interval in stats["prefix_bin"]])
    p = stats["mean"].to_numpy()          # empirical coverage
    n = stats["count"].to_numpy()
    # --- binomial 95% CI ---
    se = np.sqrt(p * (1 - p) / np.maximum(n, 1))
    lower = np.clip(p - 1.96 * se, 0, 1)
    upper = np.clip(p + 1.96 * se, 0, 1)
    # --- plot ---
    plt.figure(figsize=(7, 5))
    plt.plot(centers, p, marker="o", label="Empirical coverage")
    plt.fill_between(centers, lower, upper, alpha=0.25, label="95% CI")
    plt.axhline(0.8, linestyle="--", linewidth=1, label="Ideal 80%")
    plt.xlabel("Prefix length (bin center)")
    plt.ylabel("PI10–90 Coverage")
    plt.title(f"{args.dataset}: Coverage vs Prefix Length")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if args.log_trans:
        plot_path = os.path.join(result_dir, args.dataset+"_logtrans_coverage_plot_prefix_length.pdf")
    elif args.box_cox:
        plot_path = os.path.join(result_dir, args.dataset+"_boxcox_coverage_plot_prefix_length.pdf")
    else:
        plot_path = os.path.join(result_dir, args.dataset+"_coverage_plot_prefix_length.pdf")
    plt.savefig(plot_path, format="pdf")
    plt.close()
    return stats



def sparsification_analysis_pdf(
        args,
        df,
        result_dir,
        error_col="Absolute_error",
        uncertainty_col="PI_Width_10_90",
        n_steps=50,
        n_random_runs=20):  
    if args.log_trans:
        plot_path = os.path.join(result_dir, args.dataset+"_logtrans_sparsification_plot.pdf")
    elif args.box_cox:
        plot_path = os.path.join(result_dir, args.dataset+"_boxcox_sparsification_plot.pdf")
    else:
        plot_path = os.path.join(result_dir, args.dataset+"_sparsification_plot.pdf")   
    err = df[error_col].to_numpy()
    unc = df[uncertainty_col].to_numpy()
    N = len(err)
    fracs = np.linspace(0, 0.9, n_steps)  # fraction removed
    # ---------- helper ----------
    def curve_from_order(order):
        vals = []
        for f in fracs:
            k = int(f * N)
            keep = order[k:]
            vals.append(err[keep].mean())
        return np.array(vals)
    # ---------- curves ----------
    order_unc = np.argsort(-unc)       # highest uncertainty removed first
    order_oracle = np.argsort(-err)    # highest error removed first
    curve_unc = curve_from_order(order_unc)
    curve_oracle = curve_from_order(order_oracle)
    # random baseline (average of runs)
    curves_rand = []
    for _ in range(n_random_runs):
        order_rand = np.random.permutation(N)
        curves_rand.append(curve_from_order(order_rand))
    curve_rand = np.mean(curves_rand, axis=0)
    # ---------- metrics ----------
    ause = np.trapz(curve_unc - curve_oracle, fracs)
    aurg = np.trapz(curve_rand - curve_unc, fracs)
    # sparsification error curve
    spar_err = curve_unc - curve_oracle
    # ---------- plotting ----------
    plt.figure(figsize=(7,5))
    plt.plot(fracs, curve_unc, label="Uncertainty-based", linewidth=2)
    plt.plot(fracs, curve_oracle, label="Oracle", linestyle="--")
    plt.plot(fracs, curve_rand, label="Random", linestyle=":")
    plt.xlabel("Fraction removed")
    plt.ylabel("Mean error (remaining)")
    plt.title(f"{args.dataset}: Sparsification Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()    
    plt.savefig(plot_path.replace(".pdf","_curves.pdf"))
    plt.close()
    # ----- sparsification error plot -----
    plt.figure(figsize=(7,5))
    plt.plot(fracs, spar_err, linewidth=2)
    plt.fill_between(fracs, 0, spar_err, alpha=0.25)
    plt.xlabel("Fraction removed")
    plt.ylabel("Sparsification error")
    plt.title(f"{args.dataset}: Sparsification Error Curve\nAUSE={ause:.4f}  AURG={aurg:.4f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path.replace(".pdf","_error.pdf"))
    plt.close()
    return {
        "AUSE": float(ause),
        "AURG": float(aurg),
        "fractions": fracs,
        "unc_curve": curve_unc,
        "oracle_curve": curve_oracle,
        "random_curve": curve_rand,
        "sparsification_error": spar_err,
    }



def evaluate_extreme_detection(df, target_q=0.9,
                               target_col="GroundTruth",
                               uncertainty_col="PI_Width_10_90"):
    thr = df[target_col].quantile(target_q)
    y = (df[target_col] >= thr).astype(int).values
    s = df[uncertainty_col].values
    roc = roc_auc_score(y, s)
    pr  = average_precision_score(y, s)
    def p_at(k):
        k = int(len(y)*k)
        idx = np.argsort(-s)[:k]
        return y[idx].mean()
    return {
        "threshold": thr,
        "ROC_AUC": roc,
        "PR_AUC": pr,
        "P@10%": p_at(0.10),
        "P@20%": p_at(0.20),
    }

def get_seed_results(args, result_dir, seed):
    if args.log_trans:
        df_name = args.dataset+'_'+args.model+'_quantile_wos_logtrans_seed'+str(seed)+'_inference.csv'
    elif args.box_cox:
        df_name = args.dataset+'_'+args.model+'_quantile_wos_boxcox_seed'+str(seed)+'_inference.csv'
    else:
        df_name = args.dataset+'_'+args.model+'_quantile_wos_seed'+str(seed)+'_inference.csv'    
    df = pd.read_csv(os.path.join(result_dir,df_name))
    return df

def main():
    parser = argparse.ArgumentParser(
        description='Imbalanced Regression for Remaining Time Prediction')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str, default='DALSTM',
                        choices=['DALSTM', 'PT'],
                        help='Remaining Time Prediction Baseline Model')
    parser.add_argument('--log_trans', action='store_true', default=False, 
                        help='Whether to use log transformation on target variable') 
    parser.add_argument('--box_cox', action='store_true', default=False, 
                        help='Whether to use Box-Cox transformation on target variable')     
    args = parser.parse_args()
    root_path = os.getcwd()
    result_dir = os.path.join(root_path, 'results', args.model, args.dataset)
    seeds = [4012, 4506, 409, 1824, 3657]
    # aggregate quntile results
    df_lst = []
    for seed in seeds:
        df = get_seed_results(args, result_dir, seed)
        df_lst.append(df)
    agg_df = aggregate_quantile_seed_dfs(df_lst)
    # visualization    
    _ = plot_uncertainty_vs_target_pdf(
        args, agg_df, result_dir, n_bins=10, uncertainty_col="PI_Width_10_90")    
    _ = plot_uncertainty_vs_prefixlen_pdf(
        args, agg_df, result_dir, n_bins=10, uncertainty_col="PI_Width_10_90") 
    inside = agg_df["PI_Coverage_10_90"].sum()
    total = len(agg_df)
    print(f"Coverage: {inside}/{total} = {inside/total:.3f}")    
    plot_coverage_vs_target_pdf(args, agg_df, result_dir)  
    plot_coverage_vs_prefixlen_pdf(args, agg_df, result_dir)  
    res = sparsification_analysis_pdf(args, agg_df, result_dir, 
                                      uncertainty_col="PI_Width_10_90")    
    print(res["AUSE"], res["AURG"])
    delay_performace = evaluate_extreme_detection(agg_df, target_q=0.8,
                                                  uncertainty_col="PI_Width_10_90")
    print(delay_performace)
   



if __name__ == '__main__':
    main() 