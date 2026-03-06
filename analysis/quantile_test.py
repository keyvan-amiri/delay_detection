# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 07:52:18 2025
@author: Keyvan Amiri Elyasi
"""
import os
import argparse
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
from torch.utils.data import DataLoader

from src.utils.set_args import define_experiments, add_arguments, get_auxiliary_logger
from src.utils.import_log import get_event_log
from src.LSTM.load_dataset import load_data, load_quantile_lenght_and_ids
from src.LSTM.dataset_class import DALSTM_dataset
from src.utils.utils import add_shots_quantile
from src.utils.auxiliary_quantile_model import get_training_dataframe
from src.utils.auxiliary_quantile_model import get_test_dataframe
from src.utils.auxiliary_quantile_model import extract_features
from src.utils.auxiliary_quantile_model import build_prefix_feature_table
from src.utils.auxiliary_quantile_model import merge_prefix_features
from src.utils.auxiliary_quantile_model import fit_tabular_preprocessor
from src.utils.auxiliary_quantile_model import transform_tabular_preprocessor
from src.utils.auxiliary_quantile_model import catboost_log_residual_with_conditional_shrinkage
#from src.utils.auxiliary_quantile_model import add_many_med_few



def main():
    device = f'cuda:{os.environ.get("CUDA_VISIBLE_DEVICES", "0")}' if torch.cuda.is_available() else 'cpu'
    quantiles=(0.1, 0.5, 0.6, 0.9, 0.95, 0.99)
    parser = argparse.ArgumentParser(
        description='Tabular Predictions')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str, default='DALSTM')
    parser.add_argument('--IR', type=str, default='quantile')
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--iclude_train', action='store_true', default=False) 
    # TODO: use this argument for control
    parser.add_argument('--iclude_features', action='store_true', default=False)    
    parser.add_argument('--log_trans', action='store_true', default=False)     
    args = parser.parse_args()
    args.model_name = args.dataset+'_'+args.model+'_'+args.IR+'_wos_'
    args.box_cox = False
    args.heteroscedastic = False
    args.root_path = os.getcwd()
    cfg_file = args.cfg if args.cfg is not None else args.dataset + '.yaml'       
    with open(os.path.join(args.root_path, 'cfg', cfg_file) , 'r') as f:
        cfg = yaml.safe_load(f)
    args, _, _ = define_experiments(args)
    args = add_arguments(args, cfg)
    log, log_ids = get_event_log(args, cfg)
    prefix_feat = build_prefix_feature_table(
        log, log_ids,
        window=args.window if hasattr(args, "window") else 5,)
    exclude = {"Case_id", "Prefix_length"}
    feature_cols = [c for c in prefix_feat.columns if c not in exclude]
    prep = fit_tabular_preprocessor(prefix_feat, feature_cols, top_k=20)
    prefix_feat = transform_tabular_preprocessor(prefix_feat, prep)
    logger = get_auxiliary_logger(args.result_path)
    logger.info("Now Start training Tabular models")   
    # load data
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _, _ = load_data(args)
    (train_lengths, train_cases, val_lengths, val_cases, 
     test_lengths, test_cases) = load_quantile_lenght_and_ids(args)
    test_batch_size = cfg['DALSTM']['test_batch_size'] 
    if args.iclude_train:
        train_val_dataset = DALSTM_dataset(
            X=torch.cat([X_train, X_val], dim=0),
            y=torch.cat([y_train, y_val], dim=0))     
        quantile_loader = DataLoader(
            train_val_dataset, batch_size=test_batch_size, shuffle=False)
        quantile_lengths = train_lengths+val_lengths
        quantile_cases = train_cases+val_cases
    else:
        val_dataset = DALSTM_dataset(X_val, y_val)
        quantile_loader = DataLoader(
            val_dataset, batch_size=test_batch_size, shuffle=False)
        quantile_lengths = val_lengths
        quantile_cases = val_cases          
    cat_mae_list, cat_mae_many_list, cat_mae_med_list, cat_mae_few_list = [], [], [], []    
    for seed in args.seeds:
        # get train dataframe
        train_df = get_training_dataframe(
            args, cfg, quantile_loader=quantile_loader,
            quantile_lengths=quantile_lengths, quantile_cases=quantile_cases, 
            quantiles=quantiles, seed=seed, device=device)
        test_df = get_test_dataframe(args, seed=seed)
        train_df = merge_prefix_features(train_df, prefix_feat)
        test_df  = merge_prefix_features(test_df, prefix_feat)
        lag_cols = [c for c in train_df.columns if c.startswith("act_lag_")]
        if lag_cols:
            missing_rate = train_df[lag_cols].isna().mean().mean()
            logger.info(f"[seed={seed}] Missing lag feature rate: {missing_rate:.3f}")
        train_f = extract_features(
            train_df, train_flag=True, add_history=True, roll_k=3,
            base_point_col="Q0_5")
        test_f  = extract_features(
            test_df,  train_flag=False, add_history=True, roll_k=3,
            base_point_col="Q0_5")
        # residual correction in log space with conditional shrinkage
        results_df, corr = catboost_log_residual_with_conditional_shrinkage(
            train_f, test_f, base_point_col="Q0_5", w_col="W10_90", use_log=False,
            k_folds=5,
            shuffle_cases=True,
            alpha_bins=5,
            min_bin=100,
            alpha_cap=0.7)       
        cat_base_name = args.model_name+'CAT_'
        if args.log_trans:
            cat_name = cat_base_name+'logtrans_seed'+str(seed)+'_inference.csv'
        else:
            cat_name = cat_base_name+'seed'+str(seed)+'_inference.csv'
        cat_path = os.path.join(args.result_path, cat_name)
        results_df.to_csv(cat_path, index=False)  
        # add quantile shots
        df_all = add_shots_quantile(results_df)  
        df_many = df_all[df_all["many"] == 1]
        df_med  = df_all[df_all["med"] == 1]
        df_few  = df_all[df_all["few"] == 1]
        cat_mae_list.append(df_all["Absolute_error"].mean())
        cat_mae_many_list.append(df_many["Absolute_error"].mean())
        cat_mae_med_list.append(df_med["Absolute_error"].mean())
        cat_mae_few_list.append(df_few["Absolute_error"].mean())
    # ---- aggregate MAE
    MAE_mean, MAE_std = np.mean(cat_mae_list), np.std(cat_mae_list)
    MAE_many_mean, MAE_many_std = np.mean(cat_mae_many_list), np.std(cat_mae_many_list)
    MAE_med_mean,  MAE_med_std  = np.mean(cat_mae_med_list),  np.std(cat_mae_med_list)
    MAE_few_mean,  MAE_few_std  = np.mean(cat_mae_few_list),  np.std(cat_mae_few_list)
    print(f"MAE (CAT): Average: {MAE_mean:.4f} std: {MAE_std:.4f}")
    logger.info(f"MAE (CAT): Average: {MAE_mean:.4f} std: {MAE_std:.4f}")
    print(f"MAE_many (CAT): Average: {MAE_many_mean:.4f} std: {MAE_many_std:.4f}")
    logger.info(f"MAE_many (CAT): Average: {MAE_many_mean:.4f} std: {MAE_many_std:.4f}")
    print(f"MAE_med (CAT): Average: {MAE_med_mean:.4f} std: {MAE_med_std:.4f}")
    logger.info(f"MAE_med (CAT): Average: {MAE_med_mean:.4f} std: {MAE_med_std:.4f}")
    print(f"MAE_few (CAT): Average: {MAE_few_mean:.4f} std: {MAE_few_std:.4f}")
    logger.info(f"MAE_few (CAT): Average: {MAE_few_mean:.4f} std: {MAE_few_std:.4f}")       
   

if __name__ == '__main__':
    main() 