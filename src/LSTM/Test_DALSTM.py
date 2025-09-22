import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# function to handle inference with trained model
def test_model(args, model=None, inference_loader=None,
               test_original_lengths=None, test_cases=None,
               val_mode=False, seed=None, device=None, 
               exp_id=None, logger=None): 
    heteroscedastic = args.heteroscedastic
    bmse = args.bmse
    fds_model = args.FDS 
    start=datetime.now()
    print(f'Inference for experiment number: {exp_id}')
    if logger is not None:
        logger.info(f'Inference for experiment number: {exp_id}') 
    checkpoint_name = args.model_name+'seed_'+str(seed)+'_exp_'+str(exp_id)+'.pt'
    checkpoint_path = os.path.join(args.process_path, checkpoint_name) 
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
    # set variabls to zero to collect loss values and length ids
    absolute_error = 0
    length_idx = 0
    # load checkpoint set model to evaluation mode
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()   
    with torch.no_grad():
        for index, test_batch in enumerate(inference_loader):
            inputs = test_batch[0].to(device)
            _y_truth = test_batch[1].to(device)
            batch_size = inputs.shape[0]            
            # get model outputs, and uncertainties if required
            if heteroscedastic or bmse:
                _y_pred, log_var = model(inputs)
                if heteroscedastic:
                    aleatoric_std = torch.sqrt(torch.exp(log_var))
                    epistemic_std = torch.zeros_like(aleatoric_std)
                    total_std = torch.sqrt(epistemic_std**2 + aleatoric_std**2)   
            elif fds_model:
                epoch = checkpoint['epoch']
                _y_pred = model(inputs, _y_truth, epoch)['preds']
            else:            
                _y_pred = model(inputs)
            # Ensure predictions are positive
            epsilon = 1e-8
            _y_pred = torch.maximum(_y_pred, torch.tensor(epsilon))              
            # Compute batch loss
            absolute_error += F.l1_loss(_y_pred, _y_truth).item()
            # Detach predictions and ground truths (np arrays)
            _y_truth = _y_truth.detach().cpu().numpy()
            _y_pred = _y_pred.detach().cpu().numpy()
            mae_batch = np.abs(_y_truth - _y_pred)
            # collect inference result in all_result dict.
            all_results['GroundTruth'].extend(_y_truth.tolist())
            all_results['Prediction'].extend(_y_pred.tolist())
            # for test set we collect prefix lengths
            if not val_mode:
                pre_lengths = test_original_lengths[
                    length_idx:length_idx+batch_size]
                prefix_lengths = (np.array(pre_lengths).reshape(-1, 1)).tolist()
                all_results['Prefix_length'].extend(prefix_lengths)
                pre_cases = test_cases[length_idx:length_idx+batch_size]
                all_results['Case_id'].extend(np.array(pre_cases).reshape(-1, 1).tolist())
                length_idx+=batch_size
            all_results['Absolute_error'].extend(mae_batch.tolist())
            if heteroscedastic:
                epistemic_std = epistemic_std.detach().cpu().numpy()
                aleatoric_std = aleatoric_std.detach().cpu().numpy()
                total_std = total_std.detach().cpu().numpy()                
                all_results['Epistemic_Uncertainty'].extend(epistemic_std.tolist())
                all_results['Aleatoric_Uncertainty'].extend(aleatoric_std.tolist())
                all_results['Total_Uncertainty'].extend(total_std.tolist()) 
        num_test_batches = len(inference_loader)    
        absolute_error /= num_test_batches    
    print('Test - MAE: {:.3f}'.format(round(absolute_error, 3)))         
    inference_time = (datetime.now()-start).total_seconds()     
    if not val_mode:
        # inference time is reported in milliseconds.
        instance_t = inference_time/len(test_original_lengths)*1000
        if logger is not None:
            logger.info(f'Inference time- in seconds: {inference_time}')
            logger.info(f'Inference time for each instance- in miliseconds: {instance_t}')
            logger.info(f'Test - MAE: {absolute_error:.3f}')           
        flattened_list = [item for sublist in all_results['Prefix_length'] 
                          for item in sublist]
        all_results['Prefix_length'] = flattened_list  
        all_results['Case_id'] = [item for sublist in all_results['Case_id'] for item in sublist]
    #for key, value in all_results.items():
        #print(f"{key}: {len(value)}")
    results_df = pd.DataFrame(all_results)
    if val_mode:
        res_name = args.model_name+'seed_'+str(seed)+'_exp_'+str(exp_id)+'_inference_result_validation.csv'
        res_path = os.path.join(args.process_path, res_name)
    else:
        cols = ['Case_id', 'Prefix_length'] + [c for c in results_df.columns if c not in ['Case_id', 'Prefix_length']]
        results_df = results_df[cols]        
        res_name = args.model_name+'seed_'+str(seed)+'_exp_'+str(exp_id)+'_inference_result.csv'
        res_path = os.path.join(args.result_path, res_name)    
    results_df.to_csv(res_path, index=False)
    return results_df