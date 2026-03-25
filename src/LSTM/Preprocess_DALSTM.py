# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 07:50:00 2025
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.data_split import split_cases
from src.LSTM.Load_DALSTM import dalstm_load_dataset
from src.LSTM.Load_DALSTM import pad_arrays, normalize_tensors
from src.LSTM.Load_DALSTM import remove_small_values, check_processed_tensors
from src.LSTM.load_dataset import get_train_params
from src.utils.utils import expand_case_ids
from src.LSTM.dataset_class import DALSTM_dataset
from src.LSTM.model_DALSTM import DALSTMModel
from src.utils.loss_functions import weighted_l1_loss
from src.LSTM.Train_DALSTM import train_epoch, validate_epoch
from src.utils.GMM import train_lstm_and_predict_test_components
from src.utils.GMM import fit_joint_behavior_time_gmm, extract_prefix_embeddings


def compute_quantile_bin_edges(y_train: torch.Tensor, num_bins: int):
    """
    Compute quantile-based bin edges from train targets only.
    Returns a 1D torch tensor of shape [num_bins + 1].
    """
    y_np = y_train.detach().cpu().view(-1).numpy().astype(np.float64)
    # quantiles from 0 to 1
    q = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(y_np, q)
    # make strictly nondecreasing and robust to repeated quantiles
    edges = np.asarray(edges, dtype=np.float64)
    edges[0] = min(edges[0], y_np.min())
    edges[-1] = max(edges[-1], y_np.max())
    # if some quantile edges collapse, force tiny monotone increments
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-8
    return torch.tensor(edges, dtype=torch.float32)

def compute_survival_bin_edges(
        y_train: torch.Tensor,
        num_bins: int,
        method: str = "quantile",
        tail_frac: float = 0.2,
        tail_bin_frac: float = 0.4):
    """
    Build survival bin edges from TRAIN targets only.
    method:
        - "quantile"
        - "uniform"
        - "hybrid_tail"

    hybrid_tail:
        allocate more bins to the top tail_frac region.
    """
    y_np = y_train.detach().cpu().view(-1).numpy().astype(np.float64)
    y_np = np.sort(y_np)
    if method == "quantile":
        q = np.linspace(0.0, 1.0, num_bins + 1)
        edges = np.quantile(y_np, q)
    elif method == "uniform":
        edges = np.linspace(y_np.min(), y_np.max(), num_bins + 1)
    elif method == "hybrid_tail":
        tail_frac = float(tail_frac)
        tail_bin_frac = float(tail_bin_frac)
        tail_frac = min(max(tail_frac, 0.05), 0.5)
        tail_bin_frac = min(max(tail_bin_frac, 0.2), 0.8)
        tail_bins = max(2, int(round(num_bins * tail_bin_frac)))
        head_bins = num_bins - tail_bins
        if head_bins < 2:
            head_bins = 2
            tail_bins = num_bins - head_bins
        split_q = 1.0 - tail_frac
        q_head = np.linspace(0.0, split_q, head_bins + 1)
        q_tail = np.linspace(split_q, 1.0, tail_bins + 1)
        head_edges = np.quantile(y_np, q_head)
        tail_edges = np.quantile(y_np, q_tail)
        edges = np.concatenate([head_edges[:-1], tail_edges])
    else:
        raise ValueError(f"Unknown survival binning method: {method}")
    edges = np.asarray(edges, dtype=np.float64)
    edges[0] = min(edges[0], y_np.min())
    edges[-1] = max(edges[-1], y_np.max())
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-8
    return torch.tensor(edges, dtype=torch.float32)

class DALSTM_preprocessing ():      
    def __init__ (self, log, log_ids, args, cfg, overwrite=False, 
                  perform_lifecycle_trick=None, threshold=0.0):
        self.dataset_name = args.dataset
        files_exist = check_processed_tensors(args)
        if overwrite or not files_exist:        
            self.log = log.copy()
            self.log_ids = log_ids
            self.args = args
            self.cfg = cfg
            self.event_attributes = log_ids.event_cat_features
            # to exclude prefixes with very small remaining times 
            self.threshold = threshold 
            self.perform_lifecycle_trick = perform_lifecycle_trick\
                if perform_lifecycle_trick is not None else True
            # execute preprocessing in two steps:
            self.executute_pipeline()         
        else:
            print(f"For '{self.dataset_name}' DALST preprocessing is already done.")
            
    # function for feature-based GMM
    def _train_backbone_for_embeddings(self, X_train, y_train, X_val, y_val, input_size):
        """
        Train one vanilla DALSTM once and return an encoder version
        (same weights, last linear layer removed) for embedding extraction.
        """
        device = f'cuda:{os.environ.get("CUDA_VISIBLE_DEVICES", "0")}' if torch.cuda.is_available() else 'cpu'
        n_layers = self.cfg['DALSTM']['n_layers'] or 2
        hidden_size = self.cfg['DALSTM']['hidden_size'] or 150
        dropout = self.cfg['DALSTM']['dropout']
        if dropout is None:
            dropout = True
        dropout_prob = self.cfg['DALSTM']['dropout_prob'] or 0.1
        batch_size = self.cfg['DALSTM']['batch_size']
        max_epochs, early_stop, patience, min_delta = get_train_params(self.cfg)
        # keep preprocessing practical
        max_epochs = min(max_epochs, 50)
        train_dataset = DALSTM_dataset(X_train, y_train, args=None)
        val_dataset = DALSTM_dataset(X_val, y_val, args=None)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        model = DALSTMModel(
            input_size=input_size, hidden_size=hidden_size, n_layers=n_layers,
            dropout=dropout, p_fix=dropout_prob, exclude_last_layer=False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = weighted_l1_loss
        best_val = float("inf")
        best_state = None
        current_patience = 0
        for epoch in range(max_epochs):
            _ = train_epoch(
                model=model, train_loader=train_loader, criterion=criterion,
                optimizer=optimizer, epoch=epoch, bmse=False, fds_model=False,
                heteroscedastic=False, quantile_regression=False,
                device=device)
            val_loss = validate_epoch(
                model=model, val_loader=val_loader, criterion=criterion,
                epoch=epoch, hpo_mode=True, bmse=False, fds_model=False,
                heteroscedastic=False, quantile_regression=False,
                device=device)
            if val_loss < best_val - min_delta:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                current_patience = 0
            else:
                current_patience += 1
                if early_stop and current_patience >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        # Build encoder-only model and copy shared weights
        encoder = DALSTMModel(
            input_size=input_size, hidden_size=hidden_size, n_layers=n_layers,
            dropout=dropout, p_fix=dropout_prob, exclude_last_layer=True).to(device)
        encoder_state = encoder.state_dict()
        trained_state = model.state_dict()
        filtered = {
            k: v for k, v in trained_state.items()
            if k in encoder_state and not k.startswith("linear1")}
        encoder_state.update(filtered)
        encoder.load_state_dict(encoder_state, strict=False)
        encoder.eval()
        return encoder  
      
    # Execute some basic preprocessing steps
    def executute_pipeline(self, time_format="%Y-%m-%d %H:%M:%S"):  
        pd_log = self.log
        # Use integers always for case identifiers.
        # We need this to make a split that is equal for every dataset
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
        pd_log[self.log_ids.end_time] = pd.to_datetime(
            pd_log[self.log_ids.end_time], utc=True)
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
        X_train, X_val, X_test = pad_arrays(
            X_train, X_val, X_test, self.dataset_name)
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
        # Save survival bin edges computed from TRAIN targets only
        surv_bin_edges = compute_quantile_bin_edges(
            y_train, num_bins=self.args.surv_num_bins)
        torch.save(surv_bin_edges, self.args.surv_bin_edges_path)
        # Apply mixture of Gaussian to remaining time prediction (two-step approach)
        # Behavior + time clustering:
        # 1) train one global DALSTM backbone
        # 2) extract prefix embeddings
        # 3) fit GMM on [embedding, scaled y]
        # 4) train router on X -> z as before
        encoder = self._train_backbone_for_embeddings(
            X_train, y_train, X_val, y_val, input_size)
        device = f'cuda:{os.environ.get("CUDA_VISIBLE_DEVICES", "0")}' if torch.cuda.is_available() else 'cpu'
        emb_batch_size = self.cfg['DALSTM']['test_batch_size'] or 512
        H_train = extract_prefix_embeddings(
            encoder, X_train, batch_size=emb_batch_size, device=device)
        H_val = extract_prefix_embeddings(
            encoder, X_val,   batch_size=emb_batch_size, device=device)
        # y_weight controls how much the target contributes relative to behavior
        # pca_dim controls how much embedding compression is used before GMM
        # lower y_weight = more behavior-driven clusters
        # higher y_weight = more time-driven clusters
        z_train, z_val, joint_meta = fit_joint_behavior_time_gmm(
            H_train=H_train, H_val=H_val, y_train=y_train, y_val=y_val,
            y_weight=1.0, pca_dim=16, min_fraction_per_component=0.10, 
            max_components=10, covariance_type="diag", n_init=10, 
            reg_covar=1e-6, random_state=0,)
        z_test, p_test, _, _ = train_lstm_and_predict_test_components(
            X_train, X_val, X_test, z_train, z_val, y_test)
        # get statistics
        num_comp = int(max(z_train.max(), z_val.max()).item()) + 1
        train_freq = torch.bincount(
            z_train.view(-1), minlength=num_comp).float() / z_train.numel()
        val_freq   = torch.bincount(
            z_val.view(-1),   minlength=num_comp).float() / z_val.numel()
        test_freq   = torch.bincount(
            z_test.view(-1),   minlength=num_comp).float() / z_test.numel()
        train_means = torch.stack(
            [y_train[z_train == c].mean() for c in range(num_comp)])
        val_means = torch.stack(
            [y_val[z_val == c].mean() for c in range(num_comp)])
        test_means = torch.stack(
            [y_test[z_test == c].mean() for c in range(num_comp)])
        print("GMM statistics")
        print("Train- frequency:", train_freq.tolist(), "mean:", train_means)
        print("Val- frequency:", val_freq.tolist(), "mean:", val_means)
        print("Test- frequency:", test_freq.tolist(), "mean:", test_means)
        torch.save(z_train, self.args.z_train_path)
        torch.save(z_val, self.args.z_val_path)
        torch.save(z_test, self.args.z_test_path)
        torch.save(p_test, self.args.p_test_path)
        print('Preprocessing is done.')