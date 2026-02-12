# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 09:44:50 2026

@author: kamirel
"""
import os
import yaml
import argparse
import pickle
import numpy as np
import pandas as pd
from filelock import FileLock

from src.utils.utils import add_shots_quantile
from src.utils.loss_functions import sera_loss    
from src.utils.set_args import handle_experiment


def get_smooth_list(IR: str):
    """Match your experiment naming."""
    if IR in {"Vanilla", "GMM"}:
        return ["wos"]
    elif IR in {"CSW", "EAL"}:
        return ["wos", "LDS", "FDS", "LDS+FDS"]
    elif IR in {"BMSE", "SERA"}:
        return ["wos", "FDS"]
    else:
        return ["wos"]


def compute_metrics_from_df(results_df: pd.DataFrame):
    """Compute MAE and MAE on many/med/few from an inference CSV."""
    required = {"Absolute_error"}
    missing = required - set(results_df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    mae = float(results_df["Absolute_error"].mean())

    df = add_shots_quantile(results_df)

    # If there are no rows in a bucket, mean() => nan; that's fine.
    mae_many = float(df.loc[df["many"] == 1, "Absolute_error"].mean())
    mae_med  = float(df.loc[df["med"] == 1, "Absolute_error"].mean())
    mae_few  = float(df.loc[df["few"] == 1, "Absolute_error"].mean())

    # You cannot recompute SERA from CSV alone unless phi/relevance is saved too.
    sera_val = np.nan

    return {
        "MAE": mae,
        "MAE-Many": mae_many,
        "MAE-Med": mae_med,
        "MAE-Few": mae_few,
        "SERA": sera_val,
    }


def aggregate_seed_metrics(per_seed_metrics):
    """Turn list of per-seed scalar dicts into (mean,std) tuples."""
    keys = ["MAE", "MAE-Many", "MAE-Med", "MAE-Few", "SERA"]
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seed_metrics], dtype=float)
        out[k] = (float(np.nanmean(vals)), float(np.nanstd(vals))) if len(vals) else (np.nan, np.nan)
    return out


def main():
    parser = argparse.ArgumentParser(description="Backfill overall_results.pkl from saved inference CSV files only")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="DALSTM", choices=["DALSTM", "PT"])
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing (IR,smooth) keys too; default only fills missing.")
    parser.add_argument('--cfg', default=None)   
    
    args = parser.parse_args()
    args.root_path = os.getcwd()
    cfg_file = args.cfg if args.cfg is not None else args.dataset + '.yaml' 
    with open(os.path.join(args.root_path, 'cfg', cfg_file) , 'r') as f:
        cfg = yaml.safe_load(f)
    

    # Your seed list
    seeds = [409, 1824, 3657, 4012, 4506][:args.num_seeds]

    IR_techs = ["Vanilla", "CSW", "EAL", "BMSE", "SERA"]

    root_path = os.getcwd()
    result_dir = os.path.join(root_path, "results", args.model, args.dataset)
    data_dir = os.path.join(root_path, "temp", args.model, args.dataset)

    overall_name = f"{args.dataset}_{args.model}_overall_results.pkl"
    overall_path = os.path.join(result_dir, overall_name)
    lock_path = overall_path + ".lock"
    tmp_path = overall_path + ".tmp"

    if not os.path.exists(overall_path):
        raise FileNotFoundError(f"overall_results.pkl not found: {overall_path}")

    # Load existing results once (inside lock)
    with FileLock(lock_path, timeout=600):
        with open(overall_path, "rb") as f:
            overall_results = pickle.load(f)

    updated = 0
    skipped_existing = 0
    missing_csv = 0
    combos_seen = 0

    for IR in IR_techs:
        smooth_lst = get_smooth_list(IR)
        for smooth in smooth_lst:
            combos_seen += 1
            key = (IR, smooth)

            if (not args.overwrite) and (key in overall_results):
                skipped_existing += 1
                continue

            per_seed_metrics = []

            for seed in seeds:
                model_prefix = f"{args.dataset}_{args.model}_{IR}_{smooth}_"
                csv_name = f"{model_prefix}seed{seed}_inference.csv"
                csv_path = os.path.join(result_dir, csv_name)

                if not os.path.exists(csv_path):
                    missing_csv += 1
                    continue

                try:
                    df = pd.read_csv(csv_path)
                    m = compute_metrics_from_df(df)
                    per_seed_metrics.append(m)
                except Exception:
                    missing_csv += 1
                    continue

            perf = aggregate_seed_metrics(per_seed_metrics)

            # best_params cannot be recovered from CSVs -> set None (or keep existing if you prefer)
            existing_best = overall_results.get(key, {}).get("best_params", None)
            best_params = existing_best if (key in overall_results and not args.overwrite) else None

            overall_results[key] = {
                "best_params": best_params,
                "performance": perf
            }
            updated += 1

    # Save updated results (lock + atomic replace)
    with FileLock(lock_path, timeout=600):
        with open(tmp_path, "wb") as f:
            pickle.dump(overall_results, f)
        os.replace(tmp_path, overall_path)

    print("Done.")
    print(f"Combos considered: {combos_seen}")
    print(f"Updated keys: {updated}")
    print(f"Skipped existing keys: {skipped_existing}")
    print(f"Missing/unreadable CSV count: {missing_csv}")
    print(f"Saved: {overall_path}")

    # Print results table
    print(f"\n{'='*80}")
    print(f"Results for {args.dataset} / {args.model}")
    print(f"{'='*80}")
    header = f"{'IR':<10} {'Smooth':<10} {'MAE':>14} {'MAE-Many':>14} {'MAE-Med':>14} {'MAE-Few':>14} {'SERA':>14}"
    print(header)
    print("-" * len(header))
    for key in sorted(overall_results.keys(), key=str):
        if not (isinstance(key, tuple) and len(key) == 2):
            continue
        ir, smooth = key
        perf = overall_results[key].get("performance", {})
        cols = []
        for metric in ["MAE", "MAE-Many", "MAE-Med", "MAE-Few", "SERA"]:
            if metric in perf:
                mean, std = perf[metric]
                if np.isnan(mean):
                    cols.append(f"{'N/A':>14}")
                else:
                    cols.append(f"{mean:>7.3f}±{std:<5.3f}")
            else:
                cols.append(f"{'N/A':>14}")
        print(f"{ir:<10} {smooth:<10} {''.join(cols)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()



    
    
