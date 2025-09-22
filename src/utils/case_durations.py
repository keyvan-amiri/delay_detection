# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 13:40:47 2025
@author: Keyvan Amiri Elyasi
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def get_case_duration(args, log, log_ids):
    df = log.copy()
    grouped = df.groupby(log_ids.case)
    max_times = grouped[log_ids.end_time].max()
    if log_ids.start_time in df.columns:
        min_times = grouped[log_ids.start_time].min()
    else:
        min_times = grouped[log_ids.end_time].min()
    durations = (max_times - min_times).dt.total_seconds()/3600/24
    long_cases = durations.nlargest(int(len(durations) * args.delay_thresh)).index.tolist()
    shortest_long_duration = durations.loc[long_cases].min()
    prefix_time = get_prefix_durations(log, log_ids)
    #dur_dict = durations.to_dict()
    return long_cases, prefix_time, shortest_long_duration

def get_prefix_durations(log, log_ids):
    df = log.copy()
    sort_col = log_ids.start_time if log_ids.start_time in df.columns else log_ids.end_time
    result = {}
    for case_id, group in df.groupby(log_ids.case):
        group = group.sort_values(sort_col)
        start_time = group.iloc[0][sort_col]
        for i, (_, row) in enumerate(group.iloc[:-1].iterrows(), start=1):  # skip last row
            passed = (row[sort_col] - start_time).total_seconds()/3600/24
            result[(case_id, i)] = passed
    return result


def expand_case_ids(log, log_ids):
    df = log.copy()
    grouped = df.groupby(log_ids.case, sort=False)
    cid_lst = [cid for cid, g in grouped for _ in range(len(g) - 1)]
    #print('length of collected ids:', len(cid_lst))
    return cid_lst

def analyze_delays(args, df_inp, prefix_time, min_delay, long_cases, 
                   seed=None, exp_id=1, logger=None):
    delay_plot_name = args.model_name+'seed_'+str(seed)+'_exp_'+str(exp_id)+'_delay_plot.pdf'
    delay_plot_path = os.path.join(args.result_path, delay_plot_name) 
    dist_plot_name = args.model_name+'seed_'+str(seed)+'_exp_'+str(exp_id)+'_distribution_plot.pdf'
    dist_plot_path = os.path.join(args.result_path, dist_plot_name) 
    df = df_inp.copy()
    df['passed_time'] = df.apply(
        lambda row: prefix_time.get((row['Case_id'], row['Prefix_length']),
                                    None), axis=1)
    df['predicted_delay'] = ((df['Prediction'] + df['passed_time']) >= min_delay).astype(int)
    df['actual_delay'] = df['Case_id'].isin(long_cases).astype(int)
    _, _, _ = compute_metrics_and_f1_curve(
        df, min_delay, pdf_path=delay_plot_path, logger=logger)        
    plot_groundtruth_vs_prediction_kde(df, pdf_path=dist_plot_path)
    return df


def compute_metrics_and_f1_curve(df_inp, min_delay, 
                                 time_bins=10, pdf_path=None, logger=None):
    # exclude trivial predictions (i.e., predicting delay after SLA vialoation)
    df = df_inp[df_inp['passed_time'] < min_delay]
    print("Number of positive ground-truths:", df['actual_delay'].sum())
    print("Number of positive predictions:", df['predicted_delay'].sum())
    print("Total rows:", len(df))
    
    # Compute overall metrics
    y_true = df['actual_delay']
    y_pred = df['predicted_delay']
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    metrics = {k: float(v) for k, v in metrics.items()}
    print(metrics)
    if logger is not None:
        logger.info(f'Assuming SLA with threshold: {min_delay} days:')
        logger.info(f'Acuuracy: {metrics["accuracy"]}')
        logger.info(f'Precision: {metrics["precision"]}')
        logger.info(f'Recall: {metrics["recall"]}')
        logger.info(f'F1-Score: {metrics["f1_score"]}')
    
    # F1-score over passed_time
    passed_time_sorted = np.sort(df['passed_time'].unique())
    
    # Create thresholds based on unique values or bins
    if len(passed_time_sorted) > time_bins:
        thresholds = np.linspace(passed_time_sorted.min(), passed_time_sorted.max(), time_bins)
    else:
        thresholds = passed_time_sorted
    
    f1_scores = []
    thresholds_list = []
    
    for t in thresholds:
        subset = df[df['passed_time'] <= t]
        if len(subset) == 0:
            continue
        f1 = f1_score(subset['actual_delay'], subset['predicted_delay'], zero_division=0)
        f1_scores.append(f1)
        thresholds_list.append(t)
    
    # Plot improvement and save as PDF
    plt.figure(figsize=(8,5))
    plt.plot(thresholds_list, f1_scores, marker='o')
    plt.xlabel('Time since start of case (days)')
    plt.ylabel('F1-score (Delay detection)')
    #plt.title('F1-score improvement over passed_time')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()
    
    return metrics, thresholds_list, f1_scores

def plot_groundtruth_vs_prediction_kde(df, pdf_path=None):
    plt.figure(figsize=(8,5))    
    # Extract values
    gt_values = df['GroundTruth'].values
    pred_values = df['Prediction'].values    
    # KDE estimation
    gt_kde = gaussian_kde(gt_values)
    pred_kde = gaussian_kde(pred_values)    
    # Create a common range for x-axis
    x_min = min(gt_values.min(), pred_values.min())
    x_max = max(gt_values.max(), pred_values.max())
    x = np.linspace(x_min, x_max, 500)    
    # Plot KDEs
    plt.plot(x, gt_kde(x), color='blue', label='GroundTruth')
    plt.fill_between(x, 0, gt_kde(x), color='blue', alpha=0.4)    
    plt.plot(x, pred_kde(x), color='red', label='Prediction')
    plt.fill_between(x, 0, pred_kde(x), color='red', alpha=0.4)    
    plt.xlabel('Remaining Time (Days)')
    plt.ylabel('Density')
    #plt.title('Distribution of GroundTruth vs Prediction')
    plt.legend()
    plt.tight_layout()    
    # Save as PDF
    plt.savefig(pdf_path)
    plt.close()