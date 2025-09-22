# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 13:29:37 2025
@author: Keyvan Amiri Elyasi
"""
import os
import logging
import random

def add_arguments(args, cfg):
    # paths to processed data and results
    args.process_path = os.path.join(args.root_path, 'temp', args.dataset)
    args.result_path = os.path.join(args.root_path, 'results', args.dataset)
    path_lst = [args.process_path, args.result_path]
    for dir_path in path_lst:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    # data split arguments
    args.train_ratio = cfg['data']['train_ratio']
    args.val_ratio = cfg['data']['val_ratio']
    train_id_name = args.dataset + '_train_ids.pkl'
    val_id_name = args.dataset + '_val_ids.pkl'
    test_id_name = args.dataset + '_test_ids.pkl'    
    args.train_id = os.path.join(args.process_path, train_id_name)
    args.val_id = os.path.join(args.process_path, val_id_name)
    args.test_id = os.path.join(args.process_path, test_id_name)
    # Label and Feature smoothing arguments
    args.lds_kernel = cfg['imbalanced']['lds_kernel']
    args.lds_ks = cfg['imbalanced']['lds_ks']
    args.lds_sigma = cfg['imbalanced']['lds_sigma']
    args.fds_kernel = cfg['imbalanced']['fds_kernel']
    args.fds_ks = cfg['imbalanced']['fds_ks']
    args.fds_sigma = cfg['imbalanced']['fds_sigma']
    args.fds_start_update = cfg['imbalanced']['fds_start_update']
    args.fds_start_smooth = cfg['imbalanced']['fds_start_smooth']
    # model and checkpoint arguments
    if args.bmse:
        loss_string = 'bmse'
    elif args.sera:
        loss_string = 'sera'
    else:
        loss_string = args.loss
    if args.heteroscedastic:
        loss_string = loss_string + '_heteroscedastic'
    if args.LDS and args.FDS:
        smoothing_str = "LDS+FDS"
    elif args.LDS:
        smoothing_str = "LDS"
    elif args.FDS:
        smoothing_str = "FDS"
    else:
        smoothing_str = "NoSmoothing"       
    args.model_name = args.dataset+'_'+args.model+'_'+loss_string+'_'+smoothing_str+'_'+args.reweight+'_'     
    # set seeds
    args.seeds = generate_seeds(args.num_seeds)
    # delay threshold (e.g., 10% of cases with longest durations)
    args.delay_thresh = cfg['data']['delay_thresh']
    # DALSTM arguments
    if args.model == 'DALSTM':
        args = add_DALSTM_arguments(args)
    return args      


def add_DALSTM_arguments(args):
    args.X_train_path = os.path.join(
        args.process_path, "DALSTM_X_train_"+args.dataset+".pt")
    args.X_val_path = os.path.join(
        args.process_path, "DALSTM_X_val_"+args.dataset+".pt")
    args.X_test_path = os.path.join(
        args.process_path, "DALSTM_X_test_"+args.dataset+".pt")
    args.y_train_path = os.path.join(
        args.process_path, "DALSTM_y_train_"+args.dataset+".pt")
    args.y_val_path = os.path.join(
        args.process_path, "DALSTM_y_val_"+args.dataset+".pt")
    args.y_test_path = os.path.join(
        args.process_path, "DALSTM_y_test_"+args.dataset+".pt") 
    args.test_length_path = os.path.join(
        args.process_path, "DALSTM_test_length_list_"+args.dataset+".pkl")  
    args.test_cases_path = os.path.join(
        args.process_path, "DALSTM_test_cases_"+args.dataset+".pkl")
    args.input_size_path = os.path.join(
        args.process_path, "DALSTM_input_size_"+args.dataset+".pkl")
    args.max_len_path = os.path.join(
        args.process_path, "DALSTM_max_len_"+args.dataset+".pkl")  
    return args
    
def generate_seeds(num_seeds=5, base_seed=42, max_seed=10000):
    rng = random.Random(base_seed)  # deterministic generator
    return [rng.randint(0, max_seed) for _ in range(num_seeds)]

def get_logger(args):    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_par = logging.getLogger('Train_Evaluation_Logger') 
    logger_par.setLevel(logging.INFO) 
    logger_par_name = args.model_name + 'report.log'
    logger_par_path = os.path.join(args.result_path, logger_par_name)            
    file_handler_par = logging.FileHandler(logger_par_path)
    file_handler_par.setLevel(logging.INFO)
    file_handler_par.setFormatter(formatter)
    logger_par.addHandler(file_handler_par)
    return logger_par