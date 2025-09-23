import numpy as np
import torch
import torch.nn.functional as F

from torch.nn.utils import clip_grad_value_

def train_epoch(model, train_loader, criterion, optimizer, epoch, 
                bmse=False, fds_model=False, heteroscedastic=False,
                clip_grad_norm=False, clip_value=None, fds_config=None, 
                device=None):
    model.train()
    if fds_model:
        # Collect all features and labels from every batch (only for FDS)
        all_features = []
        all_labels = []
    for batch in train_loader:
        # Forward pass
        inputs = batch[0].to(device)
        targets = batch[1].to(device)
        weights = batch[2].to(device)
        optimizer.zero_grad() # Resets the gradients
        # forward pass
        if fds_model and bmse:
            batch_results = model(inputs, targets, epoch)    
            mean = batch_results['preds_mu']
            log_var = batch_results['preds_logvar']
            noise_var = log_var.exp()
            mean = mean.view(-1, 1)
            targets = targets.view(-1, 1)
            noise_var = noise_var.view(-1, 1)
            loss = criterion(mean, targets, noise_var)
            batch_features = batch_results['features'].detach().cpu()
            batch_labels = targets.view(-1).cpu()
            all_features.append(batch_features)
            all_labels.append(batch_labels)
        elif heteroscedastic or bmse:
            mean, log_var = model(inputs)
            if heteroscedastic:
                loss = criterion(mean, targets, log_var) 
            else:
                noise_var = log_var.exp()
                mean = mean.view(-1, 1)
                targets = targets.view(-1, 1)
                noise_var = noise_var.view(-1, 1)
                loss = criterion(mean, targets, noise_var) 
        elif fds_model:
            batch_results = model(inputs, targets, epoch)
            outputs = batch_results['preds']
            loss = criterion(outputs, targets, weights)
            batch_features = batch_results['features'].detach().cpu()                
            batch_labels = targets.cpu()
            all_features.append(batch_features)
            all_labels.append(batch_labels)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets, weights)  
        # Backward pass and optimization
        loss.backward()
        if clip_grad_norm: # if True: clips gradient at specified value
            clip_grad_value_(model.parameters(), clip_value=clip_value)
        optimizer.step()  
    # update FDS statistics after each training epoch
    if fds_model and epoch >= fds_config['start_update']:
        training_features = torch.cat(all_features, dim=0).to(device)
        training_labels = torch.cat(all_labels, dim=0).to(device)
        model.FDS.update_last_epoch_stats(epoch)
        model.FDS.update_running_stats(training_features, training_labels, epoch)        
    return loss

def validate_epoch(model, val_loader, criterion, epoch, 
                   bmse=False, fds_model=False, heteroscedastic=False,
                   device=None):
    model.eval()
    with torch.no_grad():
        total_valid_loss = 0
        for batch in val_loader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            weights = batch[2].to(device)
            if fds_model and bmse:
                batch_results = model(inputs, targets, epoch)    
                mean = batch_results['preds_mu']
                log_var = batch_results['preds_logvar']
                noise_var = log_var.exp()
                mean = mean.view(-1, 1)
                targets = targets.view(-1, 1)
                noise_var = noise_var.view(-1, 1)
                valid_loss = criterion(mean, targets, noise_var)
            elif heteroscedastic or bmse:
                mean, log_var = model(inputs)
                if heteroscedastic:
                    valid_loss = criterion(mean, targets, log_var) 
                else:
                    noise_var = log_var.exp()
                    mean = mean.view(-1, 1)
                    targets = targets.view(-1, 1)
                    noise_var = noise_var.view(-1, 1)
                    valid_loss = criterion(mean, targets, noise_var)  
            elif fds_model:
                outputs = model(inputs, targets, epoch)['preds']
                valid_loss = criterion(outputs, targets, weights)
            else:
                outputs = model(inputs)
                valid_loss = criterion(outputs, targets, weights)                
            total_valid_loss += valid_loss.item()                    
        average_valid_loss = total_valid_loss / len(val_loader)  
    return average_valid_loss

def DALSTM_inference(model, checkpoint, inference_loader, all_results, 
                     test_lengths, test_cases, 
                     bmse=None, fds_model=None, heteroscedastic=None, 
                     val_mode=False, device=None):
    # set variabls to zero to collect loss values and length ids
    absolute_error = 0
    length_idx = 0
    with torch.no_grad():
        for index, test_batch in enumerate(inference_loader):
            inputs = test_batch[0].to(device)
            _y_truth = test_batch[1].to(device)
            batch_size = inputs.shape[0]            
            # get model outputs, and uncertainties if required
            if fds_model and bmse:
                epoch = checkpoint['epoch']
                _y_pred = model(inputs, _y_truth, epoch)['preds_mu']
            elif heteroscedastic or bmse:
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
                pre_lengths = test_lengths[length_idx:length_idx+batch_size]
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
    return all_results