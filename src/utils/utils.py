# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 13:29:04 2025
@author: Keyvan Amiri Elyasi
"""
import pandas as pd
import numpy as np

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


    


