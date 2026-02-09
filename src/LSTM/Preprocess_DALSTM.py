# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 07:50:00 2025
@author: Keyvan Amiri Elyasi
"""
import os
import numpy as np
import pandas as pd
import torch
import pickle

from src.utils.data_split import split_cases
from src.LSTM.Load_DALSTM import dalstm_load_dataset
from src.LSTM.Load_DALSTM import pad_arrays, normalize_tensors
from src.LSTM.Load_DALSTM import remove_small_values, check_processed_tensors
from src.utils.case_durations import expand_case_ids
from src.utils.GMM import fit_label_gmm
from src.utils.GMM import train_lstm_and_predict_test_components
from src.utils.utils import add_shots_quantile


class DALSTM_preprocessing ():      
    def __init__ (self, log, log_ids, args, overwrite=False, 
                  perform_lifecycle_trick=None, threshold=0.0):
        self.dataset_name = args.dataset
        files_exist = check_processed_tensors(args)
        if overwrite or not files_exist:        
            self.log = log.copy()
            self.log_ids = log_ids
            self.args = args
            self.event_attributes = log_ids.event_cat_features
            # to exclude prefixes with very small remaining times 
            self.threshold = threshold 
            self.perform_lifecycle_trick = perform_lifecycle_trick\
                if perform_lifecycle_trick is not None else True
            # execute preprocessing in two steps:
            self.executute_pipeline()         
        else:
            print(f"For '{self.dataset_name}' DALST preprocessing is already done.")
        
    # Execute some basic preprocessing steps
    def executute_pipeline(self, time_format="%Y-%m-%d %H:%M:%S"):  
        pd_log = self.log
        # Use integers always for case identifiers.
        # We need this to make a split that is equal for every dataset
        #pd_log[self.log_ids.case] = pd.Categorical(pd_log[self.log_ids.case])
        #pd_log[self.log_ids.case] = pd_log[self.log_ids.case].cat.codes 
        if self.log_ids.transition in pd_log.columns:
            if (self.perform_lifecycle_trick and 
                pd_log[self.log_ids.transition].nunique() > 1):
                pd_log[self.log_ids.activity] = (
                    pd_log[self.log_ids.activity].astype(str) + "+" + pd_log[self.log_ids.transition])
        else:
            print("No lifecycle column in the log!")
        # Define relevant attributes
        attributes = self.event_attributes
        if self.log_ids.resource is not None:
            attributes.append(self.log_ids.resource)
        # Handling special cases for input event logs
        if self.dataset_name == "Traffic_Fine":
            attributes.remove('dismissal')     
        if (self.dataset_name == "BPI_2012" or self.dataset_name == "BPI_2012W" 
            or self.dataset_name == "BPI_2013_I"):
            attributes.append(self.log_ids.transition)
        sel_cols = [self.log_ids.case, self.log_ids.activity, self.log_ids.end_time] + attributes
        pd_log = pd_log[sel_cols]
        pd_log[self.log_ids.end_time] = pd.to_datetime(pd_log[self.log_ids.end_time], utc=True)
        pd_log[self.log_ids.end_time] = pd_log[self.log_ids.end_time].dt.strftime(time_format)
        ordered_columns=[self.log_ids.case, self.log_ids.activity, self.log_ids.end_time]
        pd_log = pd_log.reindex(columns=(ordered_columns + list(
            [a for a in pd_log.columns if a not in ordered_columns])))
        # split data
        if self.log_ids.start_time in pd_log.columns:
            sort_col = self.log_ids.start_time
        else:
            sort_col = self.log_ids.end_time       
        _, train_df, val_df, test_df, df = split_cases(
            pd_log, self.args, log_ids=self.log_ids, time_col=sort_col,
            validation=True, train_ratio=self.args.train_ratio, 
            val_ratio=self.args.val_ratio, drop_set=True)
        # call dalstm_load_dataset for the whole dataset
        (_, _, _), values = dalstm_load_dataset(df)
        # call dalstm_load_dataset for training, validation, and test sets
        (X_train,y_train, train_lengths), _ =  dalstm_load_dataset(
            train_df, prev_values=values)
        (X_val, y_val, valid_lengths), _ = dalstm_load_dataset(
            val_df, prev_values=values)
        (X_test, y_test, test_lengths), _ = dalstm_load_dataset(
            test_df, prev_values=values)
        # get list of case_ids in the test set for inference dataframe
        test_cases = expand_case_ids(test_df, self.log_ids)        
        # normalize tensors
        X_train, X_val, X_test = normalize_tensors(X_train, X_val, X_test)
        # convert the results to numpy arrays
        X_train = np.asarray(X_train, dtype='object')
        X_val = np.asarray(X_val, dtype='object')
        X_test = np.asarray(X_test, dtype='object')
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)
        y_test = np.asarray(y_test)
        X_train, X_val, X_test = pad_arrays(X_train, X_val, X_test,
                                            self.dataset_name)
        # Convert target attribute to days
        y_train /= (24*3600) 
        y_val /= (24*3600) 
        y_test /= (24*3600) 
        # remove samples with very small remaining time from training
        X_train, X_val, y_train, y_val = remove_small_values(
            X_train, X_val, y_train, y_val, self.threshold)
        # convert numpy arrays to tensors
        # manage disk space for huge event logs
        if (('BPIC15' in self.dataset_name) or 
            (self.dataset_name== 'Traffic_Fine') or
            (self.dataset_name== 'Hospital')):
            X_train = torch.tensor(X_train).type(torch.bfloat16)
            X_val = torch.tensor(X_val).type(torch.bfloat16)
            X_test = torch.tensor(X_test).type(torch.bfloat16)
        else:
            X_train = torch.tensor(X_train).type(torch.float)
            X_val = torch.tensor(X_val).type(torch.float)
            X_test = torch.tensor(X_test).type(torch.float)
        y_train = torch.tensor(y_train).type(torch.float)
        y_val = torch.tensor(y_val).type(torch.float)
        y_test = torch.tensor(y_test).type(torch.float)
        input_size = X_train.size(2)
        # save training, validation, test tensors
        torch.save(X_train, self.args.X_train_path)                  
        torch.save(X_val, self.args.X_val_path)
        torch.save(X_test, self.args.X_test_path)                      
        torch.save(y_train, self.args.y_train_path)
        torch.save(y_val, self.args.y_val_path)
        torch.save(y_test, self.args.y_test_path)
        print('shape of features:', X_train.shape)
        print('shape of labels:', y_train.shape)
        # save test prefix lengths, and test case ids
        with open(self.args.test_length_path, 'wb') as file:
            pickle.dump(test_lengths, file)
        with open(self.args.test_cases_path, 'wb') as file:
            pickle.dump(test_cases, file)
        # save input_size to be used in the definition of model
        with open(self.args.input_size_path, 'wb') as file:
            pickle.dump(input_size, file) 
        # Apply mixture of Gaussian to remaining time prediction (two-step approach)
        z_train, z_val = fit_label_gmm(y_train, y_val)
        z_test, _, _ = train_lstm_and_predict_test_components(
            X_train, X_val, X_test, z_train, z_val, y_test)
        # get statistics
        num_comp = int(max(z_train.max(), z_val.max()).item()) + 1
        train_freq = torch.bincount(z_train.view(-1), minlength=num_comp).float() / z_train.numel()
        val_freq   = torch.bincount(z_val.view(-1),   minlength=num_comp).float() / z_val.numel()
        test_freq   = torch.bincount(z_test.view(-1),   minlength=num_comp).float() / z_test.numel()
        train_means = torch.stack([y_train[z_train == c].mean() for c in range(num_comp)])
        val_means = torch.stack([y_val[z_val == c].mean() for c in range(num_comp)])
        test_means = torch.stack([y_test[z_test == c].mean() for c in range(num_comp)])
        print("GMM statistics")
        print("Train- frequency:", train_freq.tolist(), "mean:", train_means)
        print("Val- frequency:", val_freq.tolist(), "mean:", val_means)
        print("Test- frequency:", test_freq.tolist(), "mean:", test_means)
        torch.save(z_train, self.args.z_train_path)
        torch.save(z_val, self.args.z_val_path)
        torch.save(z_test, self.args.z_test_path)
        print('Preprocessing is done.')


def _get_base_paths(args):
    """Reconstruct the base (unfiltered) tensor paths for the dataset."""
    pp = args.process_path
    ds = args.dataset
    return {
        'X_train': os.path.join(pp, f"DALSTM_X_train_{ds}.pt"),
        'X_val':   os.path.join(pp, f"DALSTM_X_val_{ds}.pt"),
        'X_test':  os.path.join(pp, f"DALSTM_X_test_{ds}.pt"),
        'y_train': os.path.join(pp, f"DALSTM_y_train_{ds}.pt"),
        'y_val':   os.path.join(pp, f"DALSTM_y_val_{ds}.pt"),
        'y_test':  os.path.join(pp, f"DALSTM_y_test_{ds}.pt"),
        'z_train': os.path.join(pp, f"DALSTM_z_train_{ds}.pt"),
        'z_val':   os.path.join(pp, f"DALSTM_z_val_{ds}.pt"),
        'z_test':  os.path.join(pp, f"DALSTM_z_test_{ds}.pt"),
        'test_length': os.path.join(pp, f"DALSTM_test_length_list_{ds}.pkl"),
        'test_cases':  os.path.join(pp, f"DALSTM_test_cases_{ds}.pkl"),
        'input_size':  os.path.join(pp, f"DALSTM_input_size_{ds}.pkl"),
    }


def filter_and_save_subset(args, overwrite=False):
    """
    Load full preprocessed tensors, filter by the frequency subset
    specified in args.subset (many/med/few), and save to subset-specific
    paths (already set in args by update_paths_for_subset).
    """
    subset = args.subset
    # Check if subset files already exist
    subset_files = [args.X_train_path, args.X_val_path, args.X_test_path,
                    args.y_train_path, args.y_val_path, args.y_test_path,
                    args.test_length_path, args.input_size_path]
    if not overwrite and all(os.path.exists(f) for f in subset_files):
        print(f"Subset '{subset}' files already exist — skipping filtering.")
        return

    # Load full (unfiltered) tensors from base paths
    bp = _get_base_paths(args)
    X_train = torch.load(bp['X_train'], weights_only=True)
    X_val   = torch.load(bp['X_val'],   weights_only=True)
    X_test  = torch.load(bp['X_test'],  weights_only=True)
    y_train = torch.load(bp['y_train'], weights_only=True)
    y_val   = torch.load(bp['y_val'],   weights_only=True)
    y_test  = torch.load(bp['y_test'],  weights_only=True)
    z_train = torch.load(bp['z_train'], weights_only=True)
    z_val   = torch.load(bp['z_val'],   weights_only=True)
    z_test  = torch.load(bp['z_test'],  weights_only=True)

    with open(bp['test_length'], 'rb') as f:
        test_lengths = pickle.load(f)
    with open(bp['test_cases'], 'rb') as f:
        test_cases = pickle.load(f)
    with open(bp['input_size'], 'rb') as f:
        input_size = pickle.load(f)

    # Combine all targets to compute consistent quantile thresholds
    y_all = torch.cat([y_train, y_val, y_test])
    df_all = pd.DataFrame({'GroundTruth': y_all.numpy()})
    df_all = add_shots_quantile(df_all)

    mask = df_all[subset].values == 1
    n_train, n_val = len(y_train), len(y_val)

    train_mask = torch.tensor(mask[:n_train])
    val_mask   = torch.tensor(mask[n_train:n_train + n_val])
    test_mask  = torch.tensor(mask[n_train + n_val:])

    # Filter tensors
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    z_train = z_train[train_mask]
    X_val   = X_val[val_mask]
    y_val   = y_val[val_mask]
    z_val   = z_val[val_mask]
    X_test  = X_test[test_mask]
    y_test  = y_test[test_mask]
    z_test  = z_test[test_mask]

    # Filter test metadata
    test_mask_np = test_mask.numpy()
    test_lengths = [l for l, m in zip(test_lengths, test_mask_np) if m]
    test_cases   = [c for c, m in zip(test_cases, test_mask_np) if m]

    print(f"Subset '{subset}' filtering done: "
          f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    # Save to subset-specific paths
    torch.save(X_train, args.X_train_path)
    torch.save(X_val,   args.X_val_path)
    torch.save(X_test,  args.X_test_path)
    torch.save(y_train, args.y_train_path)
    torch.save(y_val,   args.y_val_path)
    torch.save(y_test,  args.y_test_path)
    torch.save(z_train, args.z_train_path)
    torch.save(z_val,   args.z_val_path)
    torch.save(z_test,  args.z_test_path)

    with open(args.test_length_path, 'wb') as f:
        pickle.dump(test_lengths, f)
    with open(args.test_cases_path, 'wb') as f:
        pickle.dump(test_cases, f)
    with open(args.input_size_path, 'wb') as f:
        pickle.dump(input_size, f)