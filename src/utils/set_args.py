# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 13:29:37 2025
@author: Keyvan Amiri Elyasi
"""
import os
import logging
import random


def define_experiments(args):
    args.bmse = False
    args.sera = False
    if args.IR in {'Vanilla'}:
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
    # DALSTM/PT arguments
    if args.model == 'DALSTM':
        args = add_DALSTM_paths(args)
    elif args.model == 'PT':
        args = add_PT_paths(args)
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
    # save length of the prefixes in the test set
    args.test_length_path = os.path.join(
        args.process_path, "DALSTM_test_length_list_"+args.dataset+".pkl")  
    args.test_cases_path = os.path.join(
        args.process_path, "DALSTM_test_cases_"+args.dataset+".pkl")
    args.input_size_path = os.path.join(
        args.process_path, "DALSTM_input_size_"+args.dataset+".pkl")
    # TODO: remove this part
    #args.max_len_path = os.path.join(
        #args.process_path, "DALSTM_max_len_"+args.dataset+".pkl")  
    return args

def add_PT_paths(args):
    # tensors for PT: token seqs, time features, targets
    args.Xtok_train_path = os.path.join(
        args.process_path, "PT_Xtok_train_"+args.dataset+".pt")
    args.Xtok_val_path = os.path.join(
        args.process_path, "PT_Xtok_val_"+args.dataset+".pt")
    args.Xtok_test_path = os.path.join(
        args.process_path, "PT_Xtok_test_"+args.dataset+".pt")
    args.Xtime_train_path = os.path.join(
        args.process_path, "PT_Xtime_train_"+args.dataset+".pt")
    args.Xtime_val_path = os.path.join(
        args.process_path, "PT_Xtime_val_"+args.dataset+".pt")
    args.Xtime_test_path = os.path.join(
        args.process_path, "PT_Xtime_test_"+args.dataset+".pt")
    args.y_train_path = os.path.join(
        args.process_path, "PT_y_train_"+args.dataset+".pt")
    args.y_val_path = os.path.join(
        args.process_path, "PT_y_val_"+args.dataset+".pt")
    args.y_test_path = os.path.join(
        args.process_path, "PT_y_test_"+args.dataset+".pt")
    args.test_length_path = os.path.join(
        args.process_path, "PT_test_length_list_"+args.dataset+".pkl")
    args.test_cases_path = os.path.join(
        args.process_path, "PT_test_cases_"+args.dataset+".pkl")
    args.vocab_size_path = os.path.join(
        args.process_path, "PT_vocab_size_"+args.dataset+".pkl")
    args.max_len_path = os.path.join(
        args.process_path, "PT_max_len_"+args.dataset+".pkl")
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
    logger_par_name = args.model_name + 'report.log'
    logger_par_path = os.path.join(args.result_path, logger_par_name)
    file_handler_par = logging.FileHandler(logger_par_path)
    file_handler_par.setLevel(logging.INFO)
    file_handler_par.setFormatter(formatter)
    logger_par.addHandler(file_handler_par)
    return logger_par