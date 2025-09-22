# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 12:25:40 2025
@author: Keyvan Amiri Elyasi
"""
import os
import argparse
import yaml
# TODO: deactivete filter warnings and solve them as much as possible
import warnings
warnings.filterwarnings("ignore")

from src.utils.set_args import add_arguments, get_logger
from src.utils.case_durations import get_case_duration, analyze_delays
from src.utils.import_log import get_event_log
from src.LSTM.Preprocess_DALSTM import DALSTM_preprocessing
from src.LSTM.Pipeline_DALSTM import DALSTM_train_evaluate

def main():
    parser = argparse.ArgumentParser(
        description='Imbalanced Regression for Remaining Time Prediction')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--overwrite', action='store_true', default=False, 
                        help='Repeat preprocessing for an existing dataset')
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--loss', type=str, default='mae',
                        choices=['mae', 'mse', 'focal_mae', 'focal_mse', 'huber'],
                        help='training loss type')   
    parser.add_argument('--reweight', type=str, default='none', 
                        choices=['none', 'sqrt_inv', 'inverse'],
                        help='cost-sensitive reweighting scheme')      
    parser.add_argument('--LDS', action='store_true', default=False, 
                        help='Use Label Distribution Smoothing')    
    parser.add_argument('--FDS', action='store_true', default=False, 
                        help='Use Feature Distribution Smoothing')
    parser.add_argument('--bmse', action='store_true', default=False, 
                        help='Whether to use balanced mse')    
    parser.add_argument('--sera', action='store_true', default=False, 
                        help='Whether to use SERA as objective function')
    parser.add_argument('--extreme_type', type=str, default='both', 
                        choices=['high', 'low', 'both'],
                        help='importance of extremes for SERA computation') 
    parser.add_argument('--asym', action='store_true', default=False, 
                        help='Whether to adjust box-plots for SERA')     
    parser.add_argument('--heteroscedastic', action='store_true', default=False, 
                        help='Whether to use heteroscedastic regression')    
    args = parser.parse_args()
    args.root_path = os.getcwd()
    cfg_file = args.cfg if args.cfg is not None else args.dataset + '.yaml'       
    with open(os.path.join(args.root_path, 'cfg', cfg_file) , 'r') as f:
        cfg = yaml.safe_load(f)
    args = add_arguments(args, cfg)
    logger = get_logger(args)
    log, log_ids = get_event_log(args, cfg)
    # get information for delay detection analysis
    long_cases, prefix_time, min_delay = get_case_duration(args, log, log_ids)
    # training and inference pipeline
    if args.model == 'DALSTM':
        DALSTM_preprocessing (log, log_ids, args, overwrite=args.overwrite)       
        dalstm_pipeline = DALSTM_train_evaluate(args, cfg)
        for seed in args.seeds:
            dalstm_pipeline.train(seed, logger=logger)
            predictions = dalstm_pipeline.inference(seed, logger=logger)   
            res_df = analyze_delays(
                args, predictions, prefix_time, min_delay, long_cases,
                seed=seed, logger=logger)
            #print(res_df.head())

if __name__ == '__main__':
    main()  