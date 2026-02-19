# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 07:52:18 2025
@author: Keyvan Amiri Elyasi
"""
import os
import argparse
import pandas as pd
import numpy as np

from src.utils.utils import add_shots_quantile

def split_by_uncertainty(
        df,
        unc_col="PI_Width_10_90",
        case_col="Case_id",
        prefix_col="Prefix_length",
        frac=0.20):
    if unc_col not in df.columns:
        raise ValueError(f"{unc_col} not found in dataframe")
    n = len(df)
    k = int(np.ceil(n * frac))
    # indices of rows sorted by uncertainty (descending)
    order = df[unc_col].to_numpy().argsort()[::-1]
    top_idx = order[:k]
    rest_idx = order[k:]
    top_list = list(zip(df.iloc[top_idx][case_col],
                        df.iloc[top_idx][prefix_col]))
    rest_list = list(zip(df.iloc[rest_idx][case_col],
                         df.iloc[rest_idx][prefix_col]))
    return top_list, rest_list

def subset_by_case_prefix(
        df,
        pairs,
        case_col="Case_id",
        prefix_col="Prefix_length"):
    if len(pairs) == 0:
        return df.iloc[0:0].copy()
    key_df = pd.DataFrame(pairs, columns=[case_col, prefix_col])
    out = df.merge(key_df, on=[case_col, prefix_col], how="inner")
    return out



def main():
    parser = argparse.ArgumentParser(
        description='Imbalanced Regression for Remaining Time Prediction')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--IR', type=str, default='BMSE',
                        choices=['CSW', 'EAL', 'BMSE', 'SERA'],
                        help='Imbalanced Regression Approach to use') 
    parser.add_argument("--unc_frac", type=float, default=0.40, 
                        help="Fraction of highest-uncertainty rows to select")
    args = parser.parse_args()
    args.model = 'DALSTM'
    root_path = os.getcwd()
    result_dir = os.path.join(root_path, 'results', args.model, args.dataset)
    seeds = [4012, 4506, 409, 1824, 3657]
    mae_list, mae_many_list, mae_med_list, mae_few_list = [], [], [], []
    for seed in seeds:
        quantile_name = args.dataset+'_'+args.model+'_quantile_wos_seed'+str(seed)+'_inference.csv'
        imb_name = args.dataset+'_'+args.model+'_'+args.IR+'_wos_seed'+str(seed)+'_inference.csv'
        vanilla_name = args.dataset+'_'+args.model+'_Vanilla_wos_seed'+str(seed)+'_inference.csv'       
        quantile_df = pd.read_csv(os.path.join(result_dir, quantile_name))
        imb_df = pd.read_csv(os.path.join(result_dir, imb_name))
        vanilla_df = pd.read_csv(os.path.join(result_dir, vanilla_name))
        top_list, rest_list = split_by_uncertainty(quantile_df, frac=args.unc_frac)
        df_top = subset_by_case_prefix(imb_df, top_list)
        df_rest = subset_by_case_prefix(vanilla_df, rest_list)
        df_all = pd.concat([df_top, df_rest], axis=0)
        df = add_shots_quantile(df_all)
        df_many = df[df["many"] == 1]
        df_med  = df[df["med"] == 1]
        df_few  = df[df["few"] == 1]
        mae_list.append(df["Absolute_error"].mean())
        mae_many_list.append(df_many["Absolute_error"].mean())
        mae_med_list.append(df_med["Absolute_error"].mean())
        mae_few_list.append(df_few["Absolute_error"].mean())
        
    MAE_mean, MAE_std = np.mean(mae_list), np.std(mae_list)
    MAE_many_mean, MAE_many_std = np.mean(mae_many_list), np.std(mae_many_list)
    MAE_med_mean,  MAE_med_std  = np.mean(mae_med_list),  np.std(mae_med_list)
    MAE_few_mean,  MAE_few_std  = np.mean(mae_few_list),  np.std(mae_few_list)
    print(f"MAE:      {MAE_mean:.4f} ± {MAE_std:.4f}")
    print(f"MAE_many: {MAE_many_mean:.4f} ± {MAE_many_std:.4f}")
    print(f"MAE_med:  {MAE_med_mean:.4f} ± {MAE_med_std:.4f}")
    print(f"MAE_few:  {MAE_few_mean:.4f} ± {MAE_few_std:.4f}")

        

if __name__ == '__main__':
    main() 