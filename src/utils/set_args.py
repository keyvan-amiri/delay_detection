# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 13:29:37 2025
@author: Keyvan Amiri Elyasi
"""
import os
import logging
import random
import torch


def define_experiments(args):
    args.bmse = False
    args.sera = False
    if args.IR in {'Vanilla', 'quantile', 'GMM'}:
        exp_ids = [1]
        smooth_str = ['wos']
    elif args.IR in {'CSW', 'EAL'}:
        exp_ids = [1, 2, 3, 4]
        smooth_str = ['wos', 'LDS', 'FDS', 'LDS+FDS']
    elif args.IR in {'BMSE', 'SERA'}:
        exp_ids = [1, 3]
        smooth_str = ['wos', 'FDS']
        if args.IR == 'BMSE':
            args.bmse = True
        else:
            args.sera = True
    else:
        raise NotImplementedError(f'Imbalanced regression with {args.IR} is not implemented.')
    return args, exp_ids, smooth_str

def add_arguments(args, cfg):
    # handle important paths
    args = handle_paths(args)
    # set seeds
    args.seeds = generate_seeds(args.num_seeds)
    # data split arguments
    args.train_ratio = cfg['data']['train_ratio']
    args.val_ratio = cfg['data']['val_ratio']
    # delay threshold (e.g., 10% of cases with longest durations)
    args.delay_thresh = cfg['data']['delay_thresh']
    if args.log_trans and args.box_cox:
        # Only one transformation at the time
        args.box_cox = False   
    return args      


def handle_paths(args):
    # paths to processed data and results
    args.process_path = os.path.join(args.root_path, 'temp', args.model, args.dataset)
    args.result_path = os.path.join(args.root_path, 'results', args.model, args.dataset)
    path_lst = [args.process_path, args.result_path]
    for dir_path in path_lst:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    # save train, val, test case ids for further analysis
    train_id_name = args.dataset + '_train_ids.pkl'
    val_id_name = args.dataset + '_val_ids.pkl'
    test_id_name = args.dataset + '_test_ids.pkl'    
    args.train_id = os.path.join(args.process_path, train_id_name)
    args.val_id = os.path.join(args.process_path, val_id_name)
    args.test_id = os.path.join(args.process_path, test_id_name)
    # DALSTM arguments
    if args.model == 'DALSTM':
        args = add_DALSTM_paths(args)    
    return args

def add_DALSTM_paths(args):
    # save addresses for train, validation, test tensors
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
    # GMM labels for two-step approach
    args.z_train_path = os.path.join(
        args.process_path, "DALSTM_z_train_"+args.dataset+".pt")
    args.z_val_path = os.path.join(
        args.process_path, "DALSTM_z_val_"+args.dataset+".pt")
    args.z_test_path = os.path.join(
        args.process_path, "DALSTM_z_test_"+args.dataset+".pt")
    # save length of the prefixes in the test set
    args.test_length_path = os.path.join(
        args.process_path, "DALSTM_test_length_list_"+args.dataset+".pkl")  
    args.test_cases_path = os.path.join(
        args.process_path, "DALSTM_test_cases_"+args.dataset+".pkl")
    args.input_size_path = os.path.join(
        args.process_path, "DALSTM_input_size_"+args.dataset+".pkl")
    return args

def handle_experiment(args, exp_str):
    args.model_name = args.dataset+'_'+args.model+'_'+args.IR+'_'+exp_str+'_'
    # args.LDS = True: Use Label Distribution Smoothing
    # args.LDS = True: Use Feature Distribution Smoothing
    if exp_str == 'wos':
        args.LDS = False
        args.FDS = False
    elif exp_str == 'LDS':
        args.LDS = True
        args.FDS = False
    elif exp_str == 'FDS':
        args.LDS = False
        args.FDS = True
    else:
        args.LDS = True
        args.FDS = True
    return args
   
def generate_seeds(num_seeds=5, base_seed=42, max_seed=10000):
    rng = random.Random(base_seed)  # deterministic generator
    return [rng.randint(0, max_seed) for _ in range(num_seeds)]

def get_logger(args):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_par = logging.getLogger('Train_Evaluation_Logger')
    logger_par.setLevel(logging.INFO)
    # Clear previous handlers
    if logger_par.hasHandlers():
        logger_par.handlers.clear()
    if args.log_trans:
        logger_par_name = args.model_name + '_logtrans_report.log'
    elif args.box_cox:
        logger_par_name =  args.model_name + '_box_cox_report.log'
    else:
        logger_par_name = args.model_name + 'report.log'
    logger_par_path = os.path.join(args.result_path, logger_par_name)
    file_handler_par = logging.FileHandler(logger_par_path)
    file_handler_par.setLevel(logging.INFO)
    file_handler_par.setFormatter(formatter)
    logger_par.addHandler(file_handler_par)
    return logger_par

def get_num_component(args):   
    if args.IR == 'GMM':
        z_test = torch.load(args.z_test_path, weights_only=True)
        distinct_labels_tensor, counts = torch.unique(z_test, return_counts=True)
        distinct_labels = distinct_labels_tensor.tolist()
        gmm_freq_lst = (counts.float() / z_test.numel()).tolist()
    else:
        distinct_labels = None
        gmm_freq_lst = [1.0]
    return gmm_freq_lst, distinct_labels

def add_result_paths(args, val_mode, gmm_label, seed):
    if val_mode:    
        if args.IR == 'GMM':
            if args.log_trans:
                res_name = args.model_name+'logtrans_gmm_'+str(gmm_label)+'_seed'+str(seed)+'_inference_validation.csv'
            elif args.box_cox:                
                res_name = args.model_name+'boxcox_gmm_'+str(gmm_label)+'_seed'+str(seed)+'_inference_validation.csv'
            else:
                res_name = args.model_name+'gmm_'+str(gmm_label)+'_seed'+str(seed)+'_inference_validation.csv'
        else:
            if args.log_trans:
                res_name = args.model_name+'logtrans_seed'+str(seed)+'_inference_validation.csv'
            elif args.box_cox: 
                res_name = args.model_name+'boxcox_seed'+str(seed)+'_inference_validation.csv'
            else:                
                res_name = args.model_name+'seed'+str(seed)+'_inference_validation.csv'
        res_path = os.path.join(args.process_path, res_name)
    else:
        if args.IR == 'GMM':
            if args.log_trans:
                res_name = args.model_name+'logtrans_gmm_'+str(gmm_label)+'_seed'+str(seed)+'_inference.csv'
            elif args.box_cox:
                res_name = args.model_name+'boxcox_gmm_'+str(gmm_label)+'_seed'+str(seed)+'_inference.csv'
            else:
                res_name = args.model_name+'gmm_'+str(gmm_label)+'_seed'+str(seed)+'_inference.csv'
        else:
            if args.log_trans:
                res_name = args.model_name+'logtrans_seed'+str(seed)+'_inference.csv'
            elif args.box_cox:
                res_name = args.model_name+'boxcox_seed'+str(seed)+'_inference.csv'
            else:
                res_name = args.model_name+'seed'+str(seed)+'_inference.csv'        
        res_path = os.path.join(args.result_path, res_name)
    return res_path