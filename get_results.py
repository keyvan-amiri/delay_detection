# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 17:14:54 2025
@author: Keyvan Amiri Elyasi
"""

import os
import argparse
import pickle
import yaml
import pandas as pd

from src.utils.utils import add_shots
from src.LSTM.load_dataset import load_DALSTM_data
from src.utils.set_args import define_experiments
from src.utils.set_args import add_arguments
from src.utils.import_log import get_event_log
from src.LSTM.Preprocess_DALSTM import DALSTM_preprocessing

def get_string(IR):
    if IR in {'Vanilla'}:    
        smooth_lst = ['wos']
    elif IR in {'CSW', 'EAL'}:
        smooth_lst = ['wos', 'LDS', 'FDS', 'LDS+FDS']
    elif IR in {'BMSE', 'SERA'}:
        smooth_lst = ['wos', 'FDS']
    return smooth_lst

def update_results(results_df):
    MAE = results_df["Absolute_error"].mean()
    df = add_shots(results_df)
    df_many = df[df["many"] == 1]
    df_med  = df[df["med"] == 1]
    df_few  = df[df["few"] == 1]
    MAE_many = df_many["Absolute_error"].mean()
    MAE_med = df_med["Absolute_error"].mean()
    MAE_few = df_few["Absolute_error"].mean()


def main():
    parser = argparse.ArgumentParser(description='Add existing results')
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--overwrite', action='store_true', default=False, 
                        help='Repeat preprocessing for an existing dataset')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str, default='DALSTM',
                        choices=['DALSTM', 'PT'],
                        help='Remaining Time Prediction Baseline Model')
    parser.add_argument('--num_seeds', type=int, default=5)
    args = parser.parse_args()
    seeds = [409, 1824, 3657, 4012, 4506]
    IR_techs = ['Vanilla', 'CSW', 'EAL', 'BMSE', 'SERA']
    root_path = os.getcwd()
    args.root_path = os.getcwd()
    args.extreme_type = 'both'
    args.asym = False
    args.reweight = 'none'
    args.LDS = False
    args.FDS = False
    args.lds_kernel = 'gaussian'
    result_dir = os.path.join(root_path, 'results', args.model, args.dataset)
    result_name = args.dataset+'_'+args.model+'_overall_results.pkl'
    with open(os.path.join(result_dir,result_name), 'rb') as f:
        overall_results  =  pickle.load(f)
    cfg_file = args.cfg if args.cfg is not None else args.dataset + '.yaml'  
    with open(os.path.join(root_path, 'cfg', cfg_file) , 'r') as f:
        cfg = yaml.safe_load(f)        
    for IR in IR_techs:
        args.IR = IR
        args, exp_ids, smooth_str = define_experiments(args)
        args = add_arguments(args, cfg)
        log, log_ids = get_event_log(args, cfg)
        # training and inference pipeline
        if args.model == 'DALSTM':
            DALSTM_preprocessing (log, log_ids, args, overwrite=args.overwrite) 
        if args.model == 'DALSTM':
            (_, _, _, _, _, relevance_test) = load_DALSTM_data(args, cfg)
        smooth_lst = get_string(IR)
        for exp_str in smooth_lst:
            for seed in seeds:
                model_name = args.dataset+'_'+args.model+'_'+IR+'_'+exp_str+'_'
                res_name = model_name+'seed'+str(seed)+'_inference.csv'
                res_path = os.path.join(
                    root_path, 'results', args.model, args.dataset, res_name)
                try:
                    res_df = pd.read_csv(res_path)
                except:
                    print(res_path)

                
        
    
    
    
    
    

if __name__ == '__main__':
    main() 
    

