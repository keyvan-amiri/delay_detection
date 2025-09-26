# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 14:48:03 2025
@author: Keyvan Amiri Elyasi
"""
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

def get_hpo_params(args):
    # learning rate search space
    lr_sp = {"name": "lr", "type": "range", "bounds": [1e-5, 1e-2], 
             "value_type": "float", "log_scale": True}    
    if args.IR == 'Vanilla':
        weight_sp = {"name": "reweight", "type": "fixed", "value": "none",
                     'value_type': 'str'}
        # Use MAE for Vanilla across models (DALSTM, PT)
        loss_sp = {"name": "loss_func", "type": "fixed", "value": "mae",
                   'value_type': 'str'}
    elif args.IR == 'CSW':
        weight_sp = {"name": "reweight", "type": "choice",
                     "values": ["sqrt_inv", "inverse"], 'value_type': 'str'}
        loss_sp = {"name": "loss_func", "type": "choice",
                   "values": ["mae", "mse", "huber"], 'value_type': 'str'}
    elif args.IR == 'EAL':
        weight_sp = {"name": "reweight", "type": "choice",
                     "values": ["inverse"], 'value_type': 'str'}
        loss_sp = {"name": "loss_func", "type": "choice",
                   "values": ["focal_mae", "focal_mse"], 'value_type': 'str'}
    elif args.IR == 'BMSE':
        weight_sp = {"name": "reweight", "type": "fixed", "value": "none",
                     'value_type': 'str'}
        loss_sp = {"name": "loss_func", "type": "fixed", "value": "bmse",
                   'value_type': 'str'}
    elif args.IR == 'SERA':
        weight_sp = {"name": "reweight", "type": "fixed", "value": "none",
                     'value_type': 'str'}
        loss_sp = {"name": "loss_func", "type": "fixed", "value": "sera",
                   'value_type': 'str'}  
    if args.IR == 'SERA':
        ext_sp = {"name": "extreme_type", "type": "choice",
                  "values": ["both", "high"], 'value_type': 'str'}
        asym_sp = {"name": "asym", "type": "choice", "values": [False, True],
                   'value_type': 'bool'}
    else:
        ext_sp = {"name": "extreme_type", "type": "fixed", "value": "both",
                  'value_type': 'str'}
        asym_sp = {"name": "asym", "type": "fixed", "value": False,
                   'value_type': 'bool'}
    if args.LDS:
        lds_ks_sp = {"name": "lds_ks", "type": "choice", "values": [5, 9, 15],
                     'value_type': 'int', "is_ordered": True}
        lds_sigma_sp = {"name": "lds_sigma", "type": "choice",
                        "values": [1, 2, 3], 'value_type': 'int',
                        "is_ordered": True}
        lds_kernel_sp = {"name": "lds_kernel", "type": "fixed",
                         "value": "gaussian", 'value_type': 'str'}
    else:
        lds_ks_sp = {"name": "lds_ks", "type": "fixed", "value": 5}
        lds_sigma_sp = {"name": "lds_sigma", "type": "fixed", "value": 2}
        lds_kernel_sp = {"name": "lds_kernel", "type": "fixed",
                         "value": "gaussian", 'value_type': 'str'}
    if args.FDS:
        fds_ks_sp = {"name": "fds_ks", "type": "choice", "values": [5, 9, 15],
                     'value_type': 'int', "is_ordered": True}
        fds_sigma_sp = {"name": "fds_sigma", "type": "choice",
                        "values": [1, 2, 3], 'value_type': 'int', 
                        "is_ordered": True}
        fds_kernel_sp = {"name": "fds_kernel", "type": "fixed",
                         "value": "gaussian", 'value_type': 'str'}
    else:
        fds_ks_sp = {"name": "fds_ks", "type": "fixed",
                     "value": 5, 'value_type': 'int'}
        fds_sigma_sp = {"name": "fds_sigma", "type": "fixed",
                        "value": 2, 'value_type': 'int'}
        fds_kernel_sp = {"name": "fds_kernel", "type": "fixed",
                         "value": "gaussian", 'value_type': 'str'}
    params = [lr_sp, weight_sp, loss_sp, ext_sp, asym_sp, 
              lds_ks_sp, lds_sigma_sp, lds_kernel_sp,
              fds_ks_sp, fds_sigma_sp, fds_kernel_sp]  
    if args.IR == 'Vanilla':
        num_trials = 10
    else:
        num_trials = 45
    return params, num_trials

def get_hpo_client(args):
    # define searh space base on arguments
    params, trials = get_hpo_params(args)
    # Define HPO strategy: first Sobol trials: num_sobol , then Bayesian Optimization
    # Initialize AX client with this strategy
    ax_client = AxClient()
   
    # Define the search space
    ax_client.create_experiment(
        name="HPO_DIR",
        parameters=params,
        objectives={
            "valid_loss": ObjectiveProperties(minimize=True)
        },
        parameter_constraints=[],
        outcome_constraints=[],
    ) 
    return ax_client, trials