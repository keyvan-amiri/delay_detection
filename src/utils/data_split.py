# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 09:28:05 2025
@author: Keyvan Amiri Elyasi
"""
import pickle

def split_cases(df, args, log_ids=None, time_col='start', validation=False,
                train_ratio=0.8, val_ratio=0.2, add_pl=False, drop_set=False):
    if log_ids is not None:
        case_col = log_ids.case
    else:
        case_col = args.case_col
    log = df.copy()
    # Add prefix lengths
    log = log.sort_values([case_col, time_col])
    if add_pl:
        log['Prefix_length'] = (log.groupby(case_col).cumcount() + 1).astype(int)
    # Get min timestamp per case
    case_start_times = log.groupby(case_col)[time_col].min()
    # Sort cases by start time
    sorted_case_ids = case_start_times.sort_values().index.tolist()
    # Split case IDs
    split_index = int(len(sorted_case_ids) * train_ratio)
    if validation:
        train_val_case_ids = sorted_case_ids[:split_index]
        val_index = int(len(train_val_case_ids) * (1-val_ratio))
        train_case_ids = train_val_case_ids[:val_index]
        val_case_ids = train_val_case_ids[val_index:]
    else:        
        train_case_ids = sorted_case_ids[:split_index]
    test_case_ids = sorted_case_ids[split_index:]
    log.loc[:, "set"] = log[case_col].apply(
        lambda x: (
            "Train" if x in train_case_ids
            else "Validation" if validation and x in val_case_ids
            else "Test" if x in test_case_ids
            else None
        )
    )
    train_df = log[log[case_col].isin(train_case_ids)].copy().drop(columns=["set"])
    val_df   = log[log[case_col].isin(val_case_ids)].copy().drop(columns=["set"]) if validation else None
    test_df  = log[log[case_col].isin(test_case_ids)].copy().drop(columns=["set"])
    if drop_set:
        log.drop(columns=["set"])
    with open(args.train_id, 'wb') as f:
        pickle.dump(train_case_ids, f)
    with open(args.test_id, 'wb') as f:
        pickle.dump(test_case_ids, f)  
    if validation:
        with open(args.val_id, 'wb') as f:
            pickle.dump(val_case_ids, f) 
    return sorted_case_ids, train_df, val_df, test_df, log