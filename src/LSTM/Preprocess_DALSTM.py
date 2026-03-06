# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 07:50:00 2025
@author: Keyvan Amiri Elyasi
"""
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
        attributes = list(self.event_attributes)
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
        # get list of case_ids in the train, val, test sets
        train_cases = expand_case_ids(train_df, self.log_ids) 
        val_cases = expand_case_ids(val_df, self.log_ids) 
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
        # save prefix lengths, and case ids
        with open(self.args.train_length_path, 'wb') as file:
            pickle.dump(train_lengths, file)
        with open(self.args.val_length_path, 'wb') as file:
            pickle.dump(valid_lengths, file)
        with open(self.args.test_length_path, 'wb') as file:
            pickle.dump(test_lengths, file)
        with open(self.args.train_cases_path, 'wb') as file:
            pickle.dump(train_cases, file)
        with open(self.args.val_cases_path, 'wb') as file:
            pickle.dump(val_cases, file)
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