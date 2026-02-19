# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 07:52:18 2025
@author: Keyvan Amiri Elyasi
"""
import os
import pickle
import argparse
import pandas as pd

from src.utils.utils import results_to_dataframe

def main():
    parser = argparse.ArgumentParser(
        description='Imbalanced Regression for Remaining Time Prediction')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str, default='DALSTM',
                        choices=['DALSTM', 'PT'],
                        help='Remaining Time Prediction Baseline Model')
    args = parser.parse_args()
    root_path = os.getcwd()
    result_dir = os.path.join(root_path, 'results', args.model, args.dataset)
    result_name = args.dataset+'_'+args.model+'_overall_results.pkl'
    with open(os.path.join(result_dir,result_name), 'rb') as f:
        overall_results  =  pickle.load(f)
    df = results_to_dataframe(overall_results)
    # sort dataframe
    custom_order = ['wos', 'LDS', 'FDS', 'LDS+FDS']
    df['Smooth'] = pd.Categorical(df['Smooth'], categories=custom_order, ordered=True)
    df_sorted = df.sort_values('Smooth').reset_index(drop=True)
    df_sorted = df_sorted.drop(columns=["SERA_mean", "SERA_std"])
    #print(df_sorted.head(25))
    print(df_sorted)

if __name__ == '__main__':
    main() 