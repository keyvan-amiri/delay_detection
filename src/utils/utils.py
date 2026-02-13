# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 13:29:04 2025
@author: Keyvan Amiri Elyasi
"""
from filelock import FileLock
import os, time, pickle
import pandas as pd
import numpy as np

def safe_update_results(results_path, key, value, timeout=600):
    lock_path = results_path + ".lock"
    tmp_path = f"{results_path}.{os.getpid()}.{int(time.time()*1000)}.tmp"
    with FileLock(lock_path, timeout=timeout):
        # reload latest from disk (THIS is what prevents lost updates)
        if os.path.exists(results_path):
            with open(results_path, "rb") as f:
                results = pickle.load(f)
        else:
            results = {}
        # update only your key
        results[key] = value
        # atomic write
        with open(tmp_path, "wb") as f:
            pickle.dump(results, f)
        os.replace(tmp_path, results_path)

def add_shots_quantile(
    df_inp,
    trg_col="GroundTruth",
    many_frac=0.6,
    med_frac=0.3,
    tail="high",   # "high", "low", or "both"
):
    """
    Quantile-based grouping for regression targets.

    Groups samples into many / med / few using quantiles.

    Parameters
    ----------
    df_inp : pandas DataFrame
    trg_col : str
        Target column name
    many_frac : float
        Fraction assigned to many-shot region
    med_frac : float
        Fraction assigned to medium-shot region
    tail : str
        Tail definition:
            "high"  -> few = highest targets (default)
            "low"   -> few = lowest targets
            "both"  -> few = both extreme tails (symmetric)

    Returns
    -------
    DataFrame with added columns: many, med, few
    """

    if many_frac <= 0 or med_frac < 0 or many_frac + med_frac >= 1:
        raise ValueError("Require: many_frac > 0, med_frac >= 0, and many_frac + med_frac < 1.")

    if tail not in {"high", "low", "both"}:
        raise ValueError("tail must be one of: 'high', 'low', 'both'")

    df = df_inp.copy()
    y = df[trg_col]

    few_frac = 1.0 - (many_frac + med_frac)

    df[["many", "med", "few"]] = 0

    # ---------- ONE-SIDED: HIGH ----------
    if tail == "high":
        q_many = y.quantile(many_frac)
        q_med  = y.quantile(many_frac + med_frac)

        df.loc[y <= q_many, "many"] = 1
        df.loc[(y > q_many) & (y <= q_med), "med"] = 1
        df.loc[y > q_med, "few"] = 1

    # ---------- ONE-SIDED: LOW ----------
    elif tail == "low":
        q_few = y.quantile(few_frac)
        q_med = y.quantile(few_frac + med_frac)

        df.loc[y <= q_few, "few"] = 1
        df.loc[(y > q_few) & (y <= q_med), "med"] = 1
        df.loc[y > q_med, "many"] = 1

    # ---------- TWO-SIDED ----------
    else:  # both
        q1 = few_frac / 2
        q2 = q1 + med_frac / 2
        q3 = 1 - q2
        q4 = 1 - q1

        b1, b2, b3, b4 = y.quantile([q1, q2, q3, q4]).to_list()

        few_mask  = (y <= b1) | (y >= b4)
        many_mask = (y > b2) & (y < b3)
        med_mask  = ~(few_mask | many_mask)

        df.loc[few_mask, "few"] = 1
        df.loc[many_mask, "many"] = 1
        df.loc[med_mask, "med"] = 1

    return df


def add_shots(df_inp, trg_col="GroundTruth",
              many_threshold=0.6, med_threshold=0.3):
    """
    Divide labels into frequency-based groups around the mode.    
    Parameters:
    df_inp: DataFrame containing the GroundTruth column
    trg_col: target that includes ground truth (target) values
    many_threshold: percentage of data points closest to mode labeled as 'many' (default: 0.6)
    med_threshold: next percentage labeled as 'med' (default: 0.3)
    few_threshold: remaining percentage labeled as 'few' (automatically calculated)    
    Returns:
    DataFrame with added 'many', 'med', 'few' columns
    """
    df = df_inp.copy()       
    # Calculate the mode (most frequent value)
    mode_value = df[trg_col].mode()
    mode_value = mode_value[0]    
    # Calculate absolute distance from mode
    df["distance_from_mode"] = abs(df[trg_col] - mode_value)    
    # Sort by distance from mode (closest first)
    df_sorted = df.sort_values("distance_from_mode").reset_index(drop=True)    
    # Calculate cutoff indices based on percentages
    total_points = len(df_sorted)
    many_cutoff = int(total_points * many_threshold)
    med_cutoff = int(total_points * (many_threshold + med_threshold))    
    # Initialize with zeros
    df_sorted[["many", "med", "few"]] = 0    
    # Assign categories based on distance from mode
    df_sorted.loc[:many_cutoff-1, "many"] = 1
    df_sorted.loc[many_cutoff:med_cutoff-1, "med"] = 1
    df_sorted.loc[med_cutoff:, "few"] = 1    
    # Return to original order and drop temporary column
    df_sorted = df_sorted.sort_index().drop("distance_from_mode", axis=1)    
    return df_sorted

def results_to_dataframe(results_dict):
    rows = []
    for (ir, smooth), data in results_dict.items():
        row = {
            'IR': ir,
            'Smooth': smooth,
            'MAE_mean': data['performance']['MAE'][0],
            'MAE_std': data['performance']['MAE'][1],
            'MAE-Many_mean': data['performance']['MAE-Many'][0],
            'MAE-Many_std': data['performance']['MAE-Many'][1],
            'MAE-Med_mean': data['performance']['MAE-Med'][0],
            'MAE-Med_std': data['performance']['MAE-Med'][1],
            'MAE-Few_mean': data['performance']['MAE-Few'][0],
            'MAE-Few_std': data['performance']['MAE-Few'][1],
            'SERA_mean': data['performance']['SERA'][0],
            'SERA_std': data['performance']['SERA'][1]
        }
        rows.append(row)
    
    return pd.DataFrame(rows)



def weighted_metrics(metric_lst, gmm_freq_lst):
    w = np.asarray(gmm_freq_lst, dtype=float)
    w = w / w.sum()
    keys = metric_lst[0].keys()
    out = {}
    for k in keys:
        means = np.array([d[k][0] for d in metric_lst], dtype=float)
        stds  = np.array([d[k][1] for d in metric_lst], dtype=float)

        out[k] = (np.sum(w * means), np.sum(w * stds))  # weighted mean, weighted std (as you stored it)
    return out


    


