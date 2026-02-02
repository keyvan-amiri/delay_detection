# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 12:25:40 2025
@author: Keyvan Amiri Elyasi
"""
from filelock import FileLock
import os
import argparse
import yaml
import numpy as np
import pickle
# TODO: deactivete filter warnings and solve them as much as possible
import warnings
warnings.filterwarnings("ignore")

from src.utils.set_args import define_experiments, handle_experiment
from src.utils.set_args import add_arguments, get_logger, get_num_component
from src.utils.import_log import get_event_log
from src.utils.pipeline import conduct_HPO, train_evaluate_best_model
from src.utils.utils import weighted_metrics
#from src.utils.case_durations import get_case_duration, analyze_delays
from src.LSTM.Preprocess_DALSTM import DALSTM_preprocessing


def main():
    parser = argparse.ArgumentParser(
        description='Imbalanced Regression for Remaining Time Prediction')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str, default='DALSTM',
                        choices=['DALSTM', 'PT'],
                        help='Remaining Time Prediction Baseline Model')
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--overwrite', action='store_true', default=False, 
                        help='Repeat preprocessing for an existing dataset')
    parser.add_argument('--num_seeds', type=int, default=5)
    # TODO: add SMOTE-based approaches if necessary
    parser.add_argument('--IR', type=str, default='Vanilla',
                        choices=['Vanilla', 'CSW', 'EAL', 'BMSE', 'SERA', 'GMM'],
                        help='Imbalanced Regression Approach to use')   
    parser.add_argument('--heteroscedastic', action='store_true', default=False, 
                        help='Whether to use heteroscedastic regression')    
    args = parser.parse_args()
    args.root_path = os.getcwd()
    cfg_file = args.cfg if args.cfg is not None else args.dataset + '.yaml'       
    with open(os.path.join(args.root_path, 'cfg', cfg_file) , 'r') as f:
        cfg = yaml.safe_load(f)
    args, exp_ids, smooth_str = define_experiments(args)
    args = add_arguments(args, cfg)
    log, log_ids = get_event_log(args, cfg)
    # training and inference pipeline
    if args.model == 'DALSTM':
        DALSTM_preprocessing (log, log_ids, args, overwrite=args.overwrite)    
    ovarall_result_name = args.dataset+'_'+args.model+'_overall_results.pkl'
    ovarall_result_path = os.path.join(args.result_path, ovarall_result_name)
    lock_path = ovarall_result_path + ".lock"
    tmp_path  = ovarall_result_path + ".tmp"
    with FileLock(lock_path, timeout=600):  # 10 min timeout
        # reload latest results inside the lock (critical)
        if os.path.exists(ovarall_result_path):
            with open(ovarall_result_path, "rb") as f:
                results = pickle.load(f)
        else:
            results = {}        
    gmm_freq_lst, distinct_labels = get_num_component(args)
    print(gmm_freq_lst, distinct_labels)
    for exp_id, smooth in zip(exp_ids, smooth_str):
        args = handle_experiment(args, smooth)
        logger = get_logger(args)        
        best_par_lst, metric_lst = [], []
        for gmm_id in gmm_freq_lst:
            gmm_dict = {}
            # HPO with first seed
            seed = args.seeds[0]
            best_parameters = conduct_HPO(
                args, cfg, seed=seed, logger=logger, gmm_label=gmm_id)
            mae_lst, many_lst, med_lst, few_lst, sera_lst = [], [], [], [], []
            for seed in args.seeds: 
                (MAE, MAE_many, MAE_med, MAE_few, SERA
                 ) = train_evaluate_best_model(
                     args, cfg, best_parameters, seed=seed, logger=logger)
                mae_lst.append(MAE)
                many_lst.append(MAE_many)
                med_lst.append(MAE_med)
                few_lst.append(MAE_few)
                sera_lst.append(SERA)               
            gmm_dict['MAE'] = (np.mean(mae_lst), np.std(mae_lst))
            gmm_dict['MAE-Many'] = (np.mean(many_lst), np.std(many_lst))
            gmm_dict['MAE-Med'] = (np.mean(med_lst), np.std(med_lst))
            gmm_dict['MAE-Few'] = (np.mean(few_lst), np.std(few_lst))
            gmm_dict['SERA'] = (np.mean(sera_lst), np.std(sera_lst))
            best_par_lst.append(best_parameters)
            metric_lst.append(gmm_dict)
        exp_dict = weighted_metrics(metric_lst, gmm_freq_lst)    
        results[(args.IR, smooth)] = {'best_params': best_par_lst,
                                      'performance': exp_dict}
    tmp_path = ovarall_result_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(results, f)
    os.replace(tmp_path, ovarall_result_path)  # atomic rename
    # TODO: maybe in future delay analysis
    #long_cases, prefix_time, min_delay = get_case_duration(args, log, log_ids)
    #res_df = analyze_delays(args, predictions, prefix_time, min_delay, long_cases, seed=seed, logger=logger)
    
if __name__ == '__main__':
    main()  