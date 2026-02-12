# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 09:04:23 2025
@author: Keyvan Amiri Elyasi
"""
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
#from torch.nn.utils import clip_grad_value_

def train_epoch(model, train_loader, criterion, optimizer, epoch,
                bmse=False, fds_model=False, heteroscedastic=False,
                gamma_nll=False,
                clip_grad_norm=False, clip_value=None, fds_config=None,
                device=None):
    model.train()
    distributional = heteroscedastic or gamma_nll
    # Collect for FDS stats only when needed
    all_features = []
    all_labels_bucket = []
    for batch in train_loader:
        inputs  = batch[0].to(device)
        targets = batch[1].to(device)   # [B] continuous
        weights = batch[2].to(device)   # [B] (or broadcastable)
        optimizer.zero_grad(set_to_none=True)
        # -------------------------
        # 1) Distributional: Heteroscedastic (MVE) or Gamma NLL
        # -------------------------
        if distributional:
            mean, log_var = model(inputs)  # both [B]
            loss = criterion(mean, targets, log_var, weights)
        # -------------------------
        # 2) FDS + BMSE (scalar noise_var)
        # -------------------------
        elif fds_model and bmse:
            batch_results = model(inputs, targets, epoch)
            mean = batch_results["preds_mu"]          # [B]
            noise_var = batch_results["noise_var"]    # scalar tensor
            # BMC expects (B,1) for pred/target in your criterion wrapper
            assert noise_var.numel() == 1, f"BMSE expects scalar noise_var, got shape {noise_var.shape}"
            mean_2d = mean.view(-1, 1)
            targets_2d = targets.view(-1, 1)
            loss = criterion(mean_2d, targets_2d, noise_var)  # scalar noise_var
            # Collect features + bucket IDs for FDS running stats
            all_features.append(batch_results["features"].detach())
            all_labels_bucket.append(model.bucketize_for_fds(targets).detach())
        # -------------------------
        # 3) BMSE (no FDS): scalar noise_var
        # -------------------------
        elif bmse:
            mean, noise_var = model(inputs)  # mean [B], noise_var scalar
            mean_2d = mean.view(-1, 1)
            targets_2d = targets.view(-1, 1)
            assert noise_var.numel() == 1, f"BMSE expects scalar noise_var, got shape {noise_var.shape}"
            loss = criterion(mean_2d, targets_2d, noise_var)
        # -------------------------
        # 4) FDS (no BMSE): weighted regression
        # -------------------------
        elif fds_model:
            batch_results = model(inputs, targets, epoch)
            outputs = batch_results["preds"]  # [B]
            loss = criterion(outputs, targets, weights)
            all_features.append(batch_results["features"].detach())
            all_labels_bucket.append(model.bucketize_for_fds(targets).detach())
        # -------------------------
        # 5) Plain (no FDS / no BMSE / no MVE): weighted regression
        # -------------------------
        else:
            outputs = model(inputs)  # [B]
            loss = criterion(outputs, targets, weights)
        loss.backward()
        # Gradient clipping (choose one style)
        if clip_grad_norm:
            clip_grad_norm_(model.parameters(), max_norm=clip_value)
        optimizer.step()
    # -------------------------
    # Update FDS stats after epoch
    # -------------------------
    if fds_model and fds_config is not None and epoch >= fds_config["start_update"]:
        if len(all_features) > 0:
            training_features = torch.cat(all_features, dim=0)
            training_labels_bucket = torch.cat(all_labels_bucket, dim=0)

            model.FDS.update_last_epoch_stats(epoch)
            model.FDS.update_running_stats(
                training_features,
                training_labels_bucket.unsqueeze(1),  # (N,1) for backward compat
                epoch
            )
    return loss


def validate_epoch(model, val_loader, criterion, epoch,
                   bmse=False, fds_model=False, heteroscedastic=False,
                   gamma_nll=False, device=None):
    model.eval()
    distributional = heteroscedastic or gamma_nll
    total_valid_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs  = batch[0].to(device)
            targets = batch[1].to(device)   # [B]
            weights = batch[2].to(device)   # [B] or broadcastable
            # 1) Distributional: Heteroscedastic or Gamma NLL
            if distributional:
                mean, log_var = model(inputs)          # both [B]
                valid_loss = criterion(mean, targets, log_var, weights)
            # 2) FDS + BMSE (scalar noise_var)
            elif fds_model and bmse:
                batch_results = model(inputs, targets, epoch)
                mean = batch_results["preds_mu"]        # [B]
                noise_var = batch_results["noise_var"]  # scalar tensor
                assert noise_var.numel() == 1, f"BMSE expects scalar noise_var, got shape {noise_var.shape}"
                valid_loss = criterion(mean.view(-1, 1), targets.view(-1, 1), noise_var)
            # 3) BMSE (no FDS): scalar noise_var
            elif bmse:
                mean, noise_var = model(inputs)         # mean [B], noise_var scalar
                assert noise_var.numel() == 1, f"BMSE expects scalar noise_var, got shape {noise_var.shape}"
                valid_loss = criterion(mean.view(-1, 1), targets.view(-1, 1), noise_var)
            # 4) FDS (no BMSE): weighted regression
            elif fds_model:
                outputs = model(inputs, targets, epoch)["preds"]  # [B]
                valid_loss = criterion(outputs, targets, weights)
            # 5) Plain weighted regression
            else:
                outputs = model(inputs)                 # [B]
                valid_loss = criterion(outputs, targets, weights)
            total_valid_loss += float(valid_loss.item())
    return total_valid_loss / len(val_loader)


def DALSTM_inference(model, checkpoint, inference_loader, all_results,
                     test_lengths, test_cases,
                     bmse=False, fds_model=False, heteroscedastic=False,
                     gamma_nll=False, val_mode=False, device=None):
    distributional = heteroscedastic or gamma_nll
    length_idx = 0
    with torch.no_grad():
        for test_batch in inference_loader:
            inputs = test_batch[0].to(device)
            y_true = test_batch[1].to(device)
            batch_size = inputs.shape[0]
            # ---- predictions ----
            if fds_model and bmse:
                epoch = checkpoint["epoch"]
                y_pred = model(inputs, y_true, epoch)["preds_mu"]   # [B]
            elif bmse:
                # BMSE: mean + scalar noise_var (ignore noise at inference)
                y_pred, noise_var = model(inputs)                   # y_pred [B], noise_var scalar
            elif gamma_nll:
                y_pred, log_alpha = model(inputs)                   # both [B]
                # Gamma: std = mu / sqrt(alpha)
                alpha = torch.exp(log_alpha).clamp(min=0.1)
                aleatoric_std = y_pred / torch.sqrt(alpha)
                epistemic_std = torch.zeros_like(aleatoric_std)
                total_std = torch.sqrt(epistemic_std**2 + aleatoric_std**2)
            elif heteroscedastic:
                y_pred, log_var = model(inputs)                     # both [B]
                aleatoric_std = torch.sqrt(torch.exp(log_var))
                epistemic_std = torch.zeros_like(aleatoric_std)
                total_std = torch.sqrt(epistemic_std**2 + aleatoric_std**2)
            elif fds_model:
                epoch = checkpoint["epoch"]
                y_pred = model(inputs, y_true, epoch)["preds"]      # [B]
            else:
                y_pred = model(inputs)                              # [B]
            # only keep this if your y is always >= 0
            epsilon = 1e-8
            y_pred = torch.maximum(y_pred, torch.tensor(epsilon, device=device))
            # ---- errors ----
            y_true_np = y_true.detach().cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()
            mae_batch = np.abs(y_true_np - y_pred_np)
            all_results["GroundTruth"].extend(y_true_np.tolist())
            all_results["Prediction"].extend(y_pred_np.tolist())
            all_results["Absolute_error"].extend(mae_batch.tolist())
            if not val_mode:
                pre_lengths = test_lengths[length_idx:length_idx + batch_size]
                pre_cases = test_cases[length_idx:length_idx + batch_size]
                all_results["Prefix_length"].extend(np.array(pre_lengths).reshape(-1, 1).tolist())
                all_results["Case_id"].extend(np.array(pre_cases).reshape(-1, 1).tolist())
                length_idx += batch_size
            if distributional:
                all_results["Epistemic_Uncertainty"].extend(epistemic_std.detach().cpu().numpy().tolist())
                all_results["Aleatoric_Uncertainty"].extend(aleatoric_std.detach().cpu().numpy().tolist())
                all_results["Total_Uncertainty"].extend(total_std.detach().cpu().numpy().tolist())
    return all_results