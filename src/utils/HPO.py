# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 14:48:03 2025
@author: Keyvan Amiri Elyasi
"""
import numpy as np
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties    

def get_hpo_params(args):
    # Define learning rate
    lr_sp = {"name": "lr", "type": "range", "bounds": [1e-5, 1e-2],
             "value_type": "float", "log_scale": True}
    # Define loss function
    loss_sp = get_loss_param(args)
    # Define weighting parameters
    weight_sp, bins_sp = get_weighting_params(args)
    # Define parameters for Focal-R losss
    beta_sp, gamma_sp = get_focal_r_params(args)
    # Define parameters for SERA loss
    ext_sp, asym_sp = get_sera_params(args)
    # Define parameters for Label Distribution Smoothing 
    lds_ks_sp, lds_sigma_sp, lds_kernel_sp = get_lds_params(args)
    # Define parameters for Feature Distribution Smoothing 
    fds_ks_sp, fds_sigma_sp, fds_kernel_sp = get_fds_params(args)
    # collect all parameters and define number of trials
    params = [lr_sp, loss_sp, weight_sp, bins_sp, beta_sp, gamma_sp,
              ext_sp, asym_sp,
              lds_ks_sp, lds_sigma_sp, lds_kernel_sp,
              fds_ks_sp, fds_sigma_sp, fds_kernel_sp]
    if args.IR in {'Vanilla', 'GMM'}:
        num_trials = 10
    elif args.IR in {'quantile'}:
        num_trials = 15
    else:
        num_trials = 45
    return params, num_trials

def get_hpo_client(args):
    # define searh space base on arguments
    params, trials = get_hpo_params(args)
    # Define HPO strategy: first Sobol trials: num_sobol, then Bayesian Optimization
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

def get_loss_param(args):
    """
    loss function map:
        {"mae": 0, "mse": 1, "huber": 2
         "focal_mae": 3, "focal_mse": 4,
         "bmse": 5, "sera": 6, "quantile": 7}
    """
    # LOSSFUNC = 
    if args.IR in {'Vanilla', 'GMM'}:
        loss_sp = {"name":"loss_func_id", 
                   "type":"fixed",
                   "value":0,
                   "value_type":"int"}
    elif args.IR == 'CSW':
        loss_sp = {"name":"loss_func_id",
                   "type":"choice",
                   "values":[0,1,2],
                   "value_type":"int"}
    elif args.IR == 'EAL':
        loss_sp = {"name":"loss_func_id",
                   "type":"choice",
                   "values":[3,4],
                   "value_type":"int"}
    elif args.IR == 'BMSE':
        loss_sp = {"name":"loss_func_id", 
                   "type":"fixed",
                   "value":5,
                   "value_type":"int"}
    elif args.IR == 'SERA':
        loss_sp = {"name":"loss_func_id", 
                   "type":"fixed",
                   "value":6,
                   "value_type":"int"}
    elif args.IR == 'quantile':
        loss_sp = {"name":"loss_func_id",
                   "type":"fixed",
                   "value":7,
                   "value_type":"int"}
    return loss_sp

def get_weighting_params(args):
    # REWEIGHT = {"none": 0, "inverse": 1, "sqrt_inv": 2}
    if args.IR == 'CSW':
        weight_sp = {"name":"reweight_id",
                     "type":"choice",
                     "values":[2,1],
                     "value_type":"int"}
        bins_sp = {"name": "n_bins",
                     "type": "choice",
                     "values": [10, 20, 50],
                     "value_type": "int",
                     "is_ordered": True}
    else:
        weight_sp = {"name":"reweight_id", 
                     "type":"fixed",
                     "value":0,
                     "value_type":"int"}
        bins_sp = {"name": "n_bins",
                     "type": "fixed",
                     "value":20,
                     "value_type": "int"}
    return weight_sp, bins_sp

def get_focal_r_params(args):
    if args.IR == 'EAL':
        # tune beta/gamma only for EAL
        beta_sp = {"name": "focal_beta",
                   "type": "range",
                   "bounds": [1e-2, 1.0],
                   "value_type": "float",
                   "log_scale": True}
        gamma_sp = {"name": "focal_gamma",
                    "type": "range",
                    "bounds": [0.5, 5.0],
                    "value_type": "float"}
    else:
        beta_sp  = {"name": "focal_beta",
                    "type": "fixed",
                    "value": 0.2,
                    "value_type": "float"}
        gamma_sp = {"name": "focal_gamma",
                    "type": "fixed",
                    "value": 1.0,
                    "value_type": "float"}
    return beta_sp, gamma_sp

def get_lds_params(args): 
    #KERNEL   = {"gaussian": 0} 
    if args.LDS:
        lds_ks_sp = {"name": "lds_ks",
                     "type": "choice",
                     "values": [5, 9, 15],
                     "value_type": "int",
                     "is_ordered": True}
        lds_sigma_sp = {"name": "lds_sigma",
                        "type": "choice",
                        "values": [1, 2, 3],
                        "value_type": "int",
                        "is_ordered": True}
    else:
        lds_ks_sp = {"name": "lds_ks",
                     "type": "fixed",
                     "value": 5,
                     "value_type": "int"}
        lds_sigma_sp = {"name": "lds_sigma",
                        "type": "fixed",
                        "value": 2,
                        "value_type": "int"}
    lds_kernel_sp = {"name":"lds_kernel_id",
                     "type":"fixed",
                     "value":0,
                     "value_type":"int"}
    return lds_ks_sp, lds_sigma_sp, lds_kernel_sp

def get_fds_params(args):
    # KERNEL   = {"gaussian": 0} 
    if args.FDS:
        fds_ks_sp = {"name": "fds_ks",
                     "type": "choice",
                     "values": [5, 9, 15],
                     "value_type": "int",
                     "is_ordered": True}
        fds_sigma_sp = {"name": "fds_sigma",
                        "type": "choice",
                        "values": [1, 2, 3],
                        "value_type": "int",
                        "is_ordered": True}
    else:
        fds_ks_sp = {"name": "fds_ks",
                     "type": "fixed",
                     "value": 5,
                     "value_type": "int"}
        fds_sigma_sp = {"name": "fds_sigma",
                        "type": "fixed",
                        "value": 2,
                        "value_type": "int"}
    fds_kernel_sp = {"name":"fds_kernel_id",
                     "type":"fixed",
                     "value":0,
                     "value_type":"int"}
    return fds_ks_sp, fds_sigma_sp, fds_kernel_sp

def get_sera_params(args):
    # extreme/asym only searched for SERA, fixed otherwise
    # EXTREME  = {"high": 0, "both": 1}
    if args.IR == 'SERA':
        ext_sp = {"name":"extreme_type_id",
                  "type":"choice",
                  "values":[0,1],
                  "value_type":"int"}
        asym_sp = {"name": "asym", "type": "choice",
                   "values": [False, True], "value_type": "bool"}
    else:
        ext_sp = {"name":"extreme_type_id", 
                  "type":"fixed",
                  "value":0,
                  "value_type":"int"}
        asym_sp = {"name": "asym",
                   "type": "fixed",
                   "value": False,
                   "value_type": "bool"}
    return ext_sp, asym_sp

def decode_params(parameters: dict) -> dict:
    p = dict(parameters)
    ID2REWEIGHT = {0:"none", 1:"inverse", 2:"sqrt_inv"}
    ID2LOSSFUNC = {0:"mae", 1:"mse", 2:"huber",
                   3:"focal_mae", 4:"focal_mse",
                   5:"bmse", 6:"sera", 7:"quantile"}
    ID2EXTREME  = {0:"high", 1:"both"}
    ID2KERNEL   = {0:"gaussian"}
    if "reweight_id" in p:     p["reweight"] = ID2REWEIGHT[p.pop("reweight_id")]
    if "loss_func_id" in p:    p["loss_func"] = ID2LOSSFUNC[p.pop("loss_func_id")]
    if "extreme_type_id" in p: p["extreme_type"] = ID2EXTREME[p.pop("extreme_type_id")]
    if "lds_kernel_id" in p:   p["lds_kernel"] = ID2KERNEL[p.pop("lds_kernel_id")]
    if "fds_kernel_id" in p:   p["fds_kernel"] = ID2KERNEL[p.pop("fds_kernel_id")]
    return p


def debug_trial_params(ax_client):
    exp = ax_client.experiment
    for t in exp.trials.values():
        if not t.status.is_completed:
            continue
        params = t.arm.parameters
        bad = {k: (v, type(v)) for k, v in params.items()
               if v is None or isinstance(v, (str, dict, list, tuple, np.ndarray))}
        # also catch numpy scalar objects that might be dtype=object-ish
        bad2 = {k: (v, type(v)) for k, v in params.items()
                if type(v).__module__.startswith("numpy") and getattr(v, "dtype", None) == object}
        if bad or bad2:
            print(f"Trial {t.index} has non-numeric-ish params:", {**bad, **bad2})