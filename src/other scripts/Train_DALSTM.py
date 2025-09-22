import os
from datetime import datetime
import torch
from torch.nn.utils import clip_grad_value_

from src.utils.optimizer import get_opt_schedule

# method to handle training the model
def train_model(args, cfg, model=None, train_loader=None, val_loader=None,
                train_val_loader=None, criterion=None, optimizer=None,
                scheduler=None, device=None, num_epochs=100, early_stop=True,
                early_patience=50, min_delta=0, fds_config=None,
                clip_grad_norm=False, clip_value=None, 
                seed=None, exp_id=1, logger=None): 
    heteroscedastic = args.heteroscedastic
    bmse = args.bmse
    fds_model = args.FDS
    start=datetime.now() # get start time (to compute training time)  
    print(f'Training for experiment number: {exp_id}')
    if logger is not None:
        logger.info(f'Training will be done for {num_epochs} epochs.') 
    checkpoint_name = args.model_name+'seed_'+str(seed)+'_exp_'+str(exp_id)+'.pt'
    checkpoint_path = os.path.join(args.process_path, checkpoint_name)     
    # Training loop
    current_patience = 0
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        # training
        model.train()
        if fds_model:
            # Lists to collect all features and labels from every batch
            all_features = []
            all_labels = []
        for batch in train_loader:
            # Forward pass
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            weights = batch[2].to(device)
            optimizer.zero_grad() # Resets the gradients
            if heteroscedastic or bmse:
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
        # Validation
        model.eval()
        with torch.no_grad():
            total_valid_loss = 0
            for batch in val_loader:
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                weights = batch[2].to(device)
                if heteroscedastic or bmse:
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
        # print the results       
        print(f'Epoch {epoch + 1}/{num_epochs},',
              f'Loss: {loss.item()}, Validation Loss: {average_valid_loss}')
        if logger is not None:
            logger.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {average_valid_loss}')          
        # save the best model
        if average_valid_loss < best_valid_loss - min_delta:
            best_valid_loss = average_valid_loss
            current_patience = 0
            checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'best_valid_loss': best_valid_loss            
            }
            torch.save(checkpoint, checkpoint_path)
            best_epoch = epoch + 1
        else:
            current_patience += 1
            # Check for early stopping
            if (early_stop and current_patience >= early_patience):
                print('Early stopping No improvement in Val loss for:', 
                      f'{early_patience} epochs.')
                if logger is not None:
                    logger.info(f'Early stopping No improvement in Val loss for: {early_patience} epochs.')       
                break        
        # Update learning rate if there is any scheduler
        if scheduler is not None:
           scheduler.step(average_valid_loss)
    training_time = (datetime.now()-start).total_seconds()
    if logger is not None:
        logger.info(f'Training time- in seconds: {training_time}')     
    if train_val_loader!=None:
        print('Now start retraining on training + validation sets')
        if logger is not None:
            logger.info('Now start retraining on training + validation sets:') 
            logger.info(f'Retraning epochs: {best_epoch}') 
        # get current time (as start) to compute training time
        start=datetime.now()   
        # re-initialize model weights
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        # re-initialize optimizer and scheduler
        optimizer, scheduler = get_opt_schedule(args, cfg, model)
        model.train()
        # Training loop     
        for epoch in range(best_epoch): 
            if fds_model:
                # Lists to collect all features and labels from every batch
                all_features = []
                all_labels = []
            for batch in train_val_loader:
                # Forward pass
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                weights = batch[2].to(device)
                optimizer.zero_grad() # Resets the gradients
                if heteroscedastic or bmse:
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
            # print the results       
            print(f'Epoch {epoch + 1}/{best_epoch},', f'Loss: {loss.item()}')
            if logger is not None:
                logger.info(f'Epoch {epoch + 1}/{best_epoch}: Loss: {loss.item()}')   
            # Update learning rate if there is any scheduler
            if scheduler is not None:
                scheduler.step(average_valid_loss)    
        training_time = (datetime.now()-start).total_seconds()
        if logger is not None:
            logger.info(f'Retraining time- in seconds: {training_time}')   
        checkpoint = {
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
            }
        torch.save(checkpoint, checkpoint_path)    