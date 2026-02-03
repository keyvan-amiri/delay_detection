# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 09:30:16 2025
@author: Keyvan Amiri Elyasi
"""
import os
from datetime import datetime
import pandas as pd
import torch

from src.utils.HPO import get_hpo_client
from src.utils.optimizer import get_opt_schedule
from src.LSTM.load_dataset import load_DALSTM_data, get_train_params
from src.LSTM.model_DALSTM import get_DALSTM_model
from src.utils.loss_functions import set_loss
from src.LSTM.Train_DALSTM import train_epoch, validate_epoch, DALSTM_inference
from src.utils.utils import add_shots
from src.utils.loss_functions import sera_loss
from src.utils.fds import add_bin_edges

def update_args(args, cfg, parameters):
    args.lr = parameters.get("lr")
    args.reweight = parameters.get("reweight")
    args.loss = parameters.get("loss_func")
    args.lds_kernel = parameters.get("lds_kernel")
    args.lds_ks = parameters.get("lds_ks")
    args.lds_sigma = parameters.get("lds_sigma")
    args.fds_kernel = parameters.get("fds_kernel")
    args.fds_ks = parameters.get("fds_ks")
    args.fds_sigma = parameters.get("fds_sigma")
    args.fds_bucket_num = cfg.get('imbalanced', {}).get('fds_bucket_num', 50)
    args.fds_bucket_start = cfg.get('imbalanced', {}).get('fds_bucket_start', 0)
    args.fds_start_update = cfg.get('imbalanced', {}).get('fds_start_update', 0)
    args.fds_start_smooth = cfg.get('imbalanced', {}).get('fds_start_smooth', 1) 
    args.focal_beta = parameters.get("focal_beta", 0.2)
    args.focal_gamma = parameters.get("focal_gamma", 1.0)
    args.extreme_type = parameters.get("extreme_type")
    args.asym = parameters.get("asym")   
    return args   
    

def conduct_HPO(args, cfg, seed=None, logger=None, gmm_label=None):  
    # set device
    device = f'cuda:{os.environ.get("CUDA_VISIBLE_DEVICES", "0")}' if torch.cuda.is_available() else 'cpu'
    # define HPO AX client
    ax_client, num_trials = get_hpo_client(args)
    # Optimization loop
    for i in range(num_trials):          
        print(f"Running trial {i+1}") 
        if logger is not None:
            logger.info(f'Running trial {i+1}') 
        # Get next parameters from AX, and extract parameters
        parameters, trial_index = ax_client.get_next_trial()
        args = update_args(args, cfg, parameters)      
        # Load data and define training parameters
        if args.model == 'DALSTM':
            if args.IR == 'GMM':
                (train_loader, val_loader, _, _, _, _, _) = load_DALSTM_data(args, cfg, gmm_label=gmm_label)
            else:
                (train_loader, val_loader, _, _, _, _, _) = load_DALSTM_data(args, cfg)
            (num_epochs, early_stop, early_patience, min_delta
             ) = get_train_params(cfg)
            # define model, and FDS configuration
            model, fds_config = get_DALSTM_model(args, cfg, device)
            # Bucketize labels for FDS
            if args.FDS:
                model = add_bin_edges(model, train_loader, val_loader, fds_config, device)
        # define loss function
        criterion = set_loss(args)     
        # Train with these parameters
        raw_data = train_with_hyperparams(
            args, cfg, model, train_loader, val_loader, criterion, 
            num_epochs=num_epochs, early_stop=early_stop,
            early_patience=early_patience, min_delta=min_delta,
            fds_config=fds_config, device=device, seed=seed, logger=logger)               
        # Complete the trial
        ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)    
    # Get best parameters
    best_parameters, values = ax_client.get_best_parameters()
    print(f"Best parameters: {best_parameters}")
    print(f"Best validation loss: {values[0]['valid_loss']}")
    if logger is not None:
        logger.info(f"Best parameters: {best_parameters}") 
        logger.info(f"Best validation loss: {values[0]['valid_loss']}") 
    return best_parameters

def train_with_hyperparams(
        args, cfg, model, train_loader, val_loader, criterion,
        num_epochs=100, early_stop=True, early_patience=30,
        min_delta=0, fds_config=None, device=None, clip_grad_norm=False,
        clip_value=None, seed=None, logger=None):    
    heteroscedastic = args.heteroscedastic
    bmse = args.bmse
    fds_model = args.FDS
    # define optimizer and scheduler
    optimizer, scheduler = get_opt_schedule(args, cfg, model) 
    # Training loop
    current_patience = 0
    best_valid_loss = float('inf')
    val_step = cfg[args.model]['val_step']
    for epoch in range(num_epochs):
        if args.model == 'DALSTM':
            loss = train_epoch(
                model, train_loader, criterion, optimizer, epoch, bmse=bmse,
                fds_model=fds_model, heteroscedastic=heteroscedastic,
                clip_grad_norm=clip_grad_norm, clip_value=clip_value,
                fds_config=fds_config, device=device) 
        if (epoch + 1) % val_step == 0:
            if args.model == 'DALSTM':
                average_valid_loss = validate_epoch(
                    model, val_loader, criterion, epoch, bmse=bmse,
                    fds_model=fds_model, heteroscedastic=heteroscedastic,
                    device=device)   
            print(f'Epoch {epoch + 1}/{num_epochs},', 
                  f'Loss: {loss.item()}, Validation Loss: {average_valid_loss}')
            if logger is not None:
                logger.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {average_valid_loss}') 
            if average_valid_loss < best_valid_loss - min_delta:
                best_valid_loss = average_valid_loss
                current_patience = 0
            else:
                current_patience += val_step
                # Check for early stopping
                if (early_stop and current_patience >= early_patience):    
                    break     
            # Update learning rate if there is any scheduler
            if scheduler is not None:
                scheduler.step(average_valid_loss)
    return {"valid_loss": (best_valid_loss, 0.0)}  


def train_evaluate_best_model(args, cfg, best_params, seed=None, logger=None,
                              clip_grad_norm=False, clip_value=None,
                              val_mode=False, gmm_label=None):
    ##########################################################################
    # Train
    ##########################################################################
    start=datetime.now() # get start time (to compute training time)  
    # set device
    device = f'cuda:{os.environ.get("CUDA_VISIBLE_DEVICES", "0")}' if torch.cuda.is_available() else 'cpu'
    # set checkpoint path
    if args.IR == 'GMM':
        checkpoint_name = args.model_name+'gmm'+str(gmm_label)+'_seed'+str(seed)+'.pt'
    else:
        checkpoint_name = args.model_name+'seed'+str(seed)+'.pt'
    checkpoint_path = os.path.join(args.process_path, checkpoint_name)
    # set to the best hyper-parameters
    args = update_args(args, cfg, best_params)
    # Load data and define training parameters
    if args.model == 'DALSTM':
        if args.IR == 'GMM':
            (train_loader, val_loader, test_loader, test_lengths, test_cases,
             relevance_val, relevance_test) = load_DALSTM_data(args, cfg, gmm_label=gmm_label)
        else:
            (train_loader, val_loader, test_loader, test_lengths, test_cases,
             relevance_val, relevance_test) = load_DALSTM_data(args, cfg)
        (num_epochs, early_stop, early_patience, min_delta
         ) = get_train_params(cfg)
        # define model, and FDS configuration
        model, fds_config = get_DALSTM_model(args, cfg, device)
        # Bucketize labels for FDS
        if args.FDS:
            model = add_bin_edges(model, train_loader, val_loader, fds_config, device)
    # define loss function
    criterion = set_loss(args)  
    # start training
    heteroscedastic = args.heteroscedastic
    bmse = args.bmse
    fds_model = args.FDS
    # define optimizer and scheduler
    optimizer, scheduler = get_opt_schedule(args, cfg, model) 
    current_patience = 0
    best_valid_loss = float('inf')
    val_step = 1
    for epoch in range(num_epochs):
        if args.model == 'DALSTM':
            loss = train_epoch(
                model, train_loader, criterion, optimizer, epoch, bmse=bmse,
                fds_model=fds_model, heteroscedastic=heteroscedastic,
                clip_grad_norm=clip_grad_norm, clip_value=clip_value,
                fds_config=fds_config, device=device)
        if (epoch + 1) % val_step == 0:
            if args.model == 'DALSTM':
                average_valid_loss = validate_epoch(
                    model, val_loader, criterion, epoch, bmse=bmse,
                    fds_model=fds_model, heteroscedastic=heteroscedastic,
                    device=device)   
            print(f'Epoch {epoch + 1}/{num_epochs},', 
                  f'Loss: {loss.item()}, Validation Loss: {average_valid_loss}')
            if logger is not None:
                logger.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {average_valid_loss}') 
            if average_valid_loss < best_valid_loss - min_delta:
                best_valid_loss = average_valid_loss
                current_patience = 0
                # save the best model      
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'best_valid_loss': best_valid_loss            
                    }
                torch.save(checkpoint, checkpoint_path)
            else:
                current_patience += val_step
                # Check for early stopping
                if (early_stop and current_patience >= early_patience):    
                    break  
            # Update learning rate if there is any scheduler
            if scheduler is not None:
                scheduler.step(average_valid_loss)
    training_time = (datetime.now()-start).total_seconds()
    ##########################################################################
    # Inference
    ##########################################################################
    if logger is not None:
        logger.info(f'Training time- in seconds: {training_time}')
    start=datetime.now() # get start time (to compute inference time)
    if heteroscedastic:
        all_results = {'GroundTruth': [], 'Prediction': [],
                       'Epistemic_Uncertainty': [], 'Aleatoric_Uncertainty': [],
                       'Total_Uncertainty': [], 'Absolute_error': []} 
    else:
        all_results = {'GroundTruth': [], 'Prediction': [], 
                       'Absolute_error': []}  
    # on test set, prefix length is added for earliness analysis
    if not val_mode:
        all_results['Case_id'] = [] 
        all_results['Prefix_length'] = []
        inference_loader = test_loader
    else:
        inference_loader = val_loader
    # load checkpoint and set the model to evaluation mode
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    all_results = DALSTM_inference(
        model, checkpoint, inference_loader, all_results, test_lengths,
        test_cases, bmse=bmse, fds_model=fds_model,
        heteroscedastic=heteroscedastic, val_mode=val_mode, device=device)
    inference_time = (datetime.now()-start).total_seconds()
    if not val_mode:
        # inference time is reported in milliseconds.
        instance_t = inference_time/len(test_lengths)*1000
        if logger is not None:
            logger.info(f'Inference time- in seconds: {inference_time}')
            logger.info(f'Inference time for each instance- in miliseconds: {instance_t}')       
        flattened_list = [item for sublist in all_results['Prefix_length'] 
                          for item in sublist]
        all_results['Prefix_length'] = flattened_list  
        all_results['Case_id'] = [item for sublist in all_results['Case_id'] for item in sublist]
    #for key, value in all_results.items():
        #print(f"{key}: {len(value)}")
    results_df = pd.DataFrame(all_results)
    if val_mode:
        if args.IR == 'GMM':
            res_name = args.model_name+'gmm'+str(gmm_label)+'_seed'+str(seed)+'_inference_validation.csv'
        else:
            res_name = args.model_name+'seed'+str(seed)+'_inference_validation.csv'
        res_path = os.path.join(args.process_path, res_name)
    else:
        cols = ['Case_id', 'Prefix_length'] + [c for c in results_df.columns if c not in ['Case_id', 'Prefix_length']]
        results_df = results_df[cols]
        if args.IR == 'GMM':
            res_name = args.model_name+'gmm'+str(gmm_label)+'_seed'+str(seed)+'_inference.csv'
        else:
            res_name = args.model_name+'seed'+str(seed)+'_inference.csv'        
        res_path = os.path.join(args.result_path, res_name)  
        results_df.to_csv(res_path, index=False)
    MAE = results_df["Absolute_error"].mean()
    # get MAE on many, med, and few shots
    df = add_shots(results_df)
    df_many = df[df["many"] == 1]
    df_med  = df[df["med"] == 1]
    df_few  = df[df["few"] == 1]
    MAE_many = df_many["Absolute_error"].mean()
    MAE_med = df_med["Absolute_error"].mean()
    MAE_few = df_few["Absolute_error"].mean()
    preds = torch.tensor(df["Prediction"].values, dtype=torch.float32)
    trues = torch.tensor(df["GroundTruth"].values, dtype=torch.float32)
    if val_mode:
        phi_np = relevance_val
    else:
        phi_np = relevance_test
    phi = torch.tensor(phi_np, dtype=torch.float32)
    new_device = "cpu"
    preds, trues, phi = preds.to(new_device), trues.to(new_device), phi.to(new_device)
    SERA = sera_loss(preds, trues, phi)
    return (MAE, MAE_many, MAE_med, MAE_few, SERA)