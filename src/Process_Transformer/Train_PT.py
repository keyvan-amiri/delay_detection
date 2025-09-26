# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 12:30:00 2025
"""
import torch
from torch.nn.utils import clip_grad_norm_


def train_epoch_PT(model, train_loader, criterion, optimizer, epoch,
					 bmse=False, fds_model=False, heteroscedastic=False,
					 clip_grad_norm=False, clip_value=None, fds_config=None,
					 device=None):
    model.train()
    if fds_model:
        all_features, all_labels = [], []
    running_loss = 0.0
    for (tokens, timef), y, w in train_loader:
        if device is not None:
            tokens = tokens.to(device)
            timef = timef.to(device)
            y = y.to(device)
            w = w.to(device)
        optimizer.zero_grad()
        if fds_model and bmse:
            batch_results = model(tokens, timef, y, epoch)
            mean = batch_results['preds_mu']
            log_var = batch_results['preds_logvar']
            noise_var = log_var.exp()
            mean = mean.view(-1, 1)
            y_ = y.view(-1, 1)
            noise_var = noise_var.view(-1, 1)
            loss = criterion(mean, y_, noise_var)
            batch_features = batch_results['features'].detach().cpu()
            batch_labels = y.detach().cpu()
            all_features.append(batch_features)
            all_labels.append(batch_labels)
        elif heteroscedastic or bmse:
            mean, log_var = model(tokens, timef)
            if heteroscedastic:
                loss = criterion(mean, y, log_var)
            else:
                noise_var = log_var.exp()
                mean = mean.view(-1, 1)
                y_ = y.view(-1, 1)
                noise_var = noise_var.view(-1, 1)
                loss = criterion(mean, y_, noise_var)
        elif fds_model:
            batch_results = model(tokens, timef, y, epoch)
            outputs = batch_results['preds']
            loss = criterion(outputs, y, w)
            batch_features = batch_results['features'].detach().cpu()
            batch_labels = y.detach().cpu()
            all_features.append(batch_features)
            all_labels.append(batch_labels)
        else:
            preds = model(tokens, timef)
            loss = criterion(preds, y, w)
        loss.backward()
        if clip_grad_norm:
            clip_grad_norm_(model.parameters(), max_norm=clip_value)
        optimizer.step()
        running_loss += loss.detach().item()
    if fds_model and fds_config is not None and epoch >= fds_config['start_update']:
        training_features = torch.cat(all_features, dim=0).to(device)
        training_labels = torch.cat(all_labels, dim=0).to(device)
        model.FDS.update_last_epoch_stats(epoch)
        model.FDS.update_running_stats(training_features, training_labels, epoch)
    return torch.tensor(running_loss / max(1, len(train_loader)))


def validate_epoch_PT(model, val_loader, criterion, epoch,
					   bmse=False, fds_model=False, heteroscedastic=False,
					   device=None):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (tokens, timef), y, w in val_loader:
            if device is not None:
                tokens = tokens.to(device)
                timef = timef.to(device)
                y = y.to(device)
                w = w.to(device)
            if fds_model and bmse:
                batch_results = model(tokens, timef, y, epoch)
                mean = batch_results['preds_mu']
                log_var = batch_results['preds_logvar']
                noise_var = log_var.exp()
                mean = mean.view(-1, 1)
                y_ = y.view(-1, 1)
                noise_var = noise_var.view(-1, 1)
                loss = criterion(mean, y_, noise_var)
            elif heteroscedastic or bmse:
                mean, log_var = model(tokens, timef)
                if heteroscedastic:
                    loss = criterion(mean, y, log_var)
                else:
                    noise_var = log_var.exp()
                    mean = mean.view(-1, 1)
                    y_ = y.view(-1, 1)
                    noise_var = noise_var.view(-1, 1)
                    loss = criterion(mean, y_, noise_var)
            elif fds_model:
                outputs = model(tokens, timef, y, epoch)['preds']
                loss = criterion(outputs, y, w)
            else:
                preds = model(tokens, timef)
                loss = criterion(preds, y, w)
            val_loss += loss.detach().item()
    return val_loss / max(1, len(val_loader))


def PT_inference(model, checkpoint, loader, all_results, test_lengths, test_cases,
                  val_mode=False, device=None):
	model.eval()
	prefix_len_list = []
	case_id_list = []
	with torch.no_grad():
		for (tokens, timef), y, _ in loader:
			if device is not None:
				tokens = tokens.to(device)
				timef = timef.to(device)
				y = y.to(device)
			preds = model(tokens, timef)
			epsilon = 1e-8
			preds = torch.maximum(preds, torch.tensor(epsilon, device=preds.device))
			abs_err = (preds - y).abs()
			all_results['GroundTruth'].extend(y.detach().cpu().tolist())
			all_results['Prediction'].extend(preds.detach().cpu().tolist())
			all_results['Absolute_error'].extend(abs_err.detach().cpu().tolist())
			# For PT we cannot reconstruct per-sample prefix/case here without auxiliary data
	return all_results


