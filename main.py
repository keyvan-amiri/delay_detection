# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 12:25:40 2025
"""
import os
import argparse, yaml
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from src.utils.set_args import define_experiments, handle_experiment
from src.utils.set_args import add_arguments, get_logger, get_num_component
from src.utils.import_log import get_event_log
from src.utils.pipeline import conduct_HPO, train_evaluate_best_model
from src.utils.pipeline import train_evaluate_soft_gmm
from src.utils.utils import safe_update_results
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
    parser.add_argument('--IR', type=str, default='Vanilla',
                    choices=['Vanilla', 'CSW', 'EAL', 'BMSE', 'SERA',
                             'quantile', 'GMM', 'survival'],
                    help='Imbalanced Regression Approach to use')
    parser.add_argument('--surv_num_bins', type=int, default=10,
                    help='Number of quantile bins for discrete-time survival')
    parser.add_argument('--surv_binning', type=str, default='quantile',
                    choices=['quantile', 'uniform', 'hybrid_tail'])
    parser.add_argument('--surv_tail_frac', type=float, default=0.2,
                    help='Top fraction of targets treated as tail in hybrid survival binning')
    parser.add_argument('--surv_tail_bin_frac', type=float, default=0.4,
                    help='Fraction of bins allocated to the tail in hybrid survival binning')
    parser.add_argument('--surv_pred_type', type=str, default='mean',
                    choices=['mean', 'median'],
                    help='How to convert survival distribution to scalar prediction')
    parser.add_argument('--hpo_metric', type=str, default='tail_blend',
                    choices=['val_loss', 'tail_blend'],
                    help='Metric used for HPO / model selection')
    parser.add_argument('--hpo_alpha', type=float, default=0.5,
                    help='Blend weight: alpha*MAE_all + (1-alpha)*MAE_tail')
    parser.add_argument('--hpo_tail_q', type=float, default=0.9,
                    help='Quantile threshold for defining tail on validation targets')
    parser.add_argument('--surv_width_grid_size', type=int, default=41,
                    help='Number of candidate thresholds for PI80_width search on validation.')
    parser.add_argument('--log_trans', action='store_true', default=False, 
                        help='Whether to use log transformation on target variable') 
    parser.add_argument('--box_cox', action='store_true', default=False, 
                        help='Whether to use Box-Cox transformation on target variable')     
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
    # preprocessing
    if args.model == 'DALSTM':
        DALSTM_preprocessing(log, log_ids, args, cfg, overwrite=args.overwrite)    
    overall_result_name = args.dataset+'_'+args.model+'_overall_results.pkl'
    overall_result_path = os.path.join(args.result_path, overall_result_name) 
    gmm_freq_lst, distinct_labels = get_num_component(args)
    print(gmm_freq_lst, distinct_labels)
    # Experiment loop
    for exp_id, smooth in zip(exp_ids, smooth_str):
        args = handle_experiment(args, smooth)
        logger = get_logger(args)
        # GMM path: HPO per expert, final evaluation via soft routing
        if args.IR == 'GMM':
            best_par_lst = []
            # HPO once per GMM component
            seed = args.seeds[0]
            for gmm_label in distinct_labels:
                best_parameters = conduct_HPO(
                    args, cfg, seed=seed, logger=logger, gmm_label=gmm_label)
                best_par_lst.append(best_parameters)
            # Final evaluation by seed
            mae_lst, many_lst, med_lst, few_lst, sera_lst = [], [], [], [], []
            for seed in args.seeds:
                (MAE, MAE_many, MAE_med, MAE_few, SERA, _
                 ) = train_evaluate_soft_gmm(
                     args=args, cfg=cfg, best_param_list=best_par_lst,
                    distinct_labels=distinct_labels, seed=seed, logger=logger)
                mae_lst.append(MAE)
                many_lst.append(MAE_many)
                med_lst.append(MAE_med)
                few_lst.append(MAE_few)
                sera_lst.append(SERA)
            exp_dict = {
                'MAE': (np.mean(mae_lst), np.std(mae_lst)),
                'MAE-Many': (np.mean(many_lst), np.std(many_lst)),
                'MAE-Med': (np.mean(med_lst), np.std(med_lst)),
                'MAE-Few': (np.mean(few_lst), np.std(few_lst)),
                'SERA': (np.mean(sera_lst), np.std(sera_lst)),}
        # Non-GMM path
        else:
            best_par_lst = []
            seed = args.seeds[0]
            best_parameters = conduct_HPO(args, cfg, seed=seed, logger=logger)
            best_par_lst.append(best_parameters)
            mae_lst, many_lst, med_lst, few_lst, sera_lst = [], [], [], [], []
            for seed in args.seeds:
                MAE, MAE_many, MAE_med, MAE_few, SERA = train_evaluate_best_model(
                    args, cfg, best_parameters, seed=seed, logger=logger)
                mae_lst.append(MAE)
                many_lst.append(MAE_many)
                med_lst.append(MAE_med)
                few_lst.append(MAE_few)
                sera_lst.append(SERA)
            exp_dict = {
                'MAE': (np.mean(mae_lst), np.std(mae_lst)),
                'MAE-Many': (np.mean(many_lst), np.std(many_lst)),
                'MAE-Med': (np.mean(med_lst), np.std(med_lst)),
                'MAE-Few': (np.mean(few_lst), np.std(few_lst)),
                'SERA': (np.mean(sera_lst), np.std(sera_lst)),}
        # Save results
        if args.log_trans:
            IR_str = args.IR + '_log'
        elif args.box_cox:
            IR_str = args.IR + '_cox'
        else:
            IR_str = args.IR
        key = (IR_str, smooth)
        value = {'best_params': best_par_lst, 'performance': exp_dict}
        safe_update_results(overall_result_path, key, value)
        if logger is not None:
            logger.info(f"Saved results for {key} safely.")
    print("All experiments finished.")

if __name__ == '__main__':
    main()