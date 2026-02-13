# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 09:04:23 2025
@author: Keyvan Amiri Elyasi
"""
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
#from torch.nn.utils import clip_grad_value_

from src.LSTM.load_dataset import inverse_log1p, inverse_boxcox

   
def train_epoch(
        model, train_loader, criterion, optimizer, epoch, bmse=False,
        fds_model=False, heteroscedastic=False, clip_grad_norm=False,
        clip_value=None, fds_config=None, quantile_regression=False,
        quantiles=(0.1, 0.5, 0.6, 0.9, 0.95, 0.99), device=None):    
    model.train()
    # Collect for FDS stats only when needed
    all_features = []
    all_labels_bucket = []
    total_loss = 0.0
    n_batches = 0
    for batch in train_loader:
        inputs  = batch[0].to(device)
        targets = batch[1].to(device)   # [B] continuous
        weights = batch[2].to(device)   # [B] (or broadcastable)
        optimizer.zero_grad(set_to_none=True)
        # -------------------------
        # 1) Quantile Regression
        # -------------------------
        if quantile_regression:
            # model outputs (B, K)
            q_pred = model(inputs)
            # criterion should be quantile_pinball_loss(y_true, y_pred, quantiles, sample_weight=...)
            loss = criterion(targets, q_pred, quantiles, sample_weight=weights)
        # -------------------------
        # 2) Heteroscedastic (MVE)
        # -------------------------
        elif heteroscedastic:
            mean, log_var = model(inputs)  # both [B]
            loss = criterion(mean, targets, log_var)
        # -------------------------
        # 3) FDS + BMSE (scalar noise_var)
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
        # 4) BMSE (no FDS): scalar noise_var
        # -------------------------
        elif bmse:
            mean, noise_var = model(inputs)  # mean [B], noise_var scalar
            mean_2d = mean.view(-1, 1)
            targets_2d = targets.view(-1, 1)
            assert noise_var.numel() == 1, f"BMSE expects scalar noise_var, got shape {noise_var.shape}"
            loss = criterion(mean_2d, targets_2d, noise_var)
        # -------------------------
        # 5) FDS (no BMSE): weighted regression
        # -------------------------
        elif fds_model:
            batch_results = model(inputs, targets, epoch)
            outputs = batch_results["preds"]  # [B]
            loss = criterion(outputs, targets, weights)
            all_features.append(batch_results["features"].detach())
            all_labels_bucket.append(model.bucketize_for_fds(targets).detach())
        # -------------------------
        # 6) Vanilla (no FDS / no BMSE / no MVE): weighted regression
        # -------------------------
        else:
            outputs = model(inputs)  # [B]
            loss = criterion(outputs, targets, weights)
        loss.backward()
        # Gradient clipping (choose one style)
        if clip_grad_norm:
            clip_grad_norm_(model.parameters(), max_norm=clip_value)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
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
    return torch.tensor(total_loss / max(n_batches, 1), device=device)


def validate_epoch(
        model, val_loader, criterion, epoch, hpo_mode=False, 
        bmse=False, fds_model=False, heteroscedastic=False,
        quantile_regression=False, quantiles=(0.1, 0.5, 0.6, 0.9, 0.95, 0.99),
        device=None):
    if quantile_regression:
        q_to_idx = {float(q): i for i, q in enumerate(quantiles)}
    model.eval()
    total_valid_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs  = batch[0].to(device)
            targets = batch[1].to(device)   # [B]
            weights = batch[2].to(device)   # [B] or broadcastable
            # 1) Quantile Regression
            if quantile_regression:
                q_pred = model(inputs)  # (B,K)
                if hpo_mode:
                    # point prediction = median
                    y_pred = q_pred[:, q_to_idx[0.5]]  # (B,)
                    valid_loss = criterion(y_pred, targets, weights)
                else:
                    valid_loss = criterion(targets, q_pred, quantiles, sample_weight=weights)
            # 2) Heteroscedastic (per-sample logvar)
            elif heteroscedastic:
                mean, log_var = model(inputs)       # both [B]
                if hpo_mode:
                    valid_loss = criterion(mean, targets, weights)
                else:
                    valid_loss = criterion(mean, targets, log_var)
            # 3) FDS + BMSE (scalar noise_var)
            elif fds_model and bmse:
                batch_results = model(inputs, targets, epoch)
                mean = batch_results["preds_mu"]        # [B]
                noise_var = batch_results["noise_var"]  # scalar tensor
                assert noise_var.numel() == 1, f"BMSE expects scalar noise_var, got shape {noise_var.shape}"
                if hpo_mode:
                    valid_loss = criterion(mean, targets, weights)
                else:
                    valid_loss = criterion(mean.view(-1, 1), targets.view(-1, 1), noise_var)
            # 4) BMSE (no FDS): scalar noise_var
            elif bmse:
                mean, noise_var = model(inputs)         # mean [B], noise_var scalar
                assert noise_var.numel() == 1, f"BMSE expects scalar noise_var, got shape {noise_var.shape}"
                if hpo_mode:
                    valid_loss = criterion(mean, targets, weights)
                else:
                    valid_loss = criterion(mean.view(-1, 1), targets.view(-1, 1), noise_var)
            # 5) FDS (no BMSE): weighted regression
            elif fds_model:
                outputs = model(inputs, targets, epoch)["preds"]  # [B]
                valid_loss = criterion(outputs, targets, weights)
            # 6) Plain weighted regression
            else:
                outputs = model(inputs)                 # [B]
                valid_loss = criterion(outputs, targets, weights)
            total_valid_loss += float(valid_loss.item())
    return total_valid_loss / len(val_loader)


def DALSTM_inference(
        args, model, checkpoint, inference_loader, all_results, test_lengths,
        test_cases, meta, bmse=False, fds_model=False, heteroscedastic=False,
        val_mode=False, quantiles=(0.1, 0.5, 0.6, 0.9, 0.95, 0.99), device=None):
    if args.IR == 'quantile':
        quantile_regression=True
        # helpful index mapping for quantiles
        q_to_idx = {float(q): i for i, q in enumerate(quantiles)}
        has_q05 = 0.5 in q_to_idx
        has_q01 = 0.1 in q_to_idx
        has_q09 = 0.9 in q_to_idx
    else:
        quantile_regression=False
    length_idx = 0
    with torch.no_grad():
        for test_batch in inference_loader:
            inputs = test_batch[0].to(device)
            y_true = test_batch[1].to(device)
            batch_size = inputs.shape[0]
            # ---- predictions ----
            if quantile_regression:
                # model outputs (B, K)
                q_pred = model(inputs)
                if not has_q05:
                    raise ValueError("Quantile regression inference expects q=0.5 in quantiles for point prediction.")
                # point prediction = median
                y_pred = q_pred[:, q_to_idx[0.5]]  # (B,)
                # store intervals and/or all quantiles
                if has_q01:
                    q10 = q_pred[:, q_to_idx[0.1]]
                else:
                    q10 = None
                if has_q09:
                    q90 = q_pred[:, q_to_idx[0.9]]
                else:
                    q90 = None
            if fds_model and bmse:
                epoch = checkpoint["epoch"]
                y_pred = model(inputs, y_true, epoch)["preds_mu"]   # [B]
            elif bmse:
                # BMSE: mean + scalar noise_var (ignore noise at inference)
                y_pred, noise_var = model(inputs)  # y_pred [B], noise_var scalar
            elif heteroscedastic:
                y_pred, log_var = model(inputs)    # both [B]
                aleatoric_std = torch.sqrt(torch.exp(log_var))
                epistemic_std = torch.zeros_like(aleatoric_std)
                total_std = torch.sqrt(epistemic_std**2 + aleatoric_std**2)
            elif fds_model:
                epoch = checkpoint["epoch"]
                y_pred = model(inputs, y_true, epoch)["preds"]      # [B]
            else:
                y_pred = model(inputs)   # [B]
            # only keep this if your y is always >= 0
            epsilon = 1e-8
            if quantile_regression:
                # clamp ALL quantiles consistently (preserves monotonicity)
                q_pred = torch.maximum(q_pred, torch.tensor(epsilon, device=device))
                y_pred = q_pred[:, q_to_idx[0.5]]
                if has_q01:
                    q10 = q_pred[:, q_to_idx[0.1]]
                if has_q09:
                    q90 = q_pred[:, q_to_idx[0.9]]
            else:
                y_pred = torch.maximum(y_pred, torch.tensor(epsilon, device=device))
            # ---- errors ----
            y_true_np = y_true.detach().cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()
            # If storing intervals/quantiles, prepare numpy too
            if quantile_regression:
                q_pred_np = q_pred.detach().cpu().numpy()  # (B,K)
                if has_q01:
                    q10_np = q10.detach().cpu().numpy()
                if has_q09:
                    q90_np = q90.detach().cpu().numpy()
            if args.log_trans:
                y_pred_np = inverse_log1p(y_pred_np)
                y_true_np = inverse_log1p(y_true_np)
                if quantile_regression:
                    q_pred_np = inverse_log1p(q_pred_np)
                    if has_q01:
                        q10_np = inverse_log1p(q10_np)
                    if has_q09:
                        q90_np = inverse_log1p(q90_np)
            elif args.box_cox:
                y_pred_np = inverse_boxcox(y_pred_np, meta)  
                y_true_np = inverse_boxcox(y_true_np, meta)
                if quantile_regression:
                    q_pred_np = inverse_boxcox(q_pred_np, meta)
                    if has_q01:
                        q10_np = inverse_boxcox(q10_np, meta)
                    if has_q09:
                        q90_np = inverse_boxcox(q90_np, meta)
            mae_batch = np.abs(y_true_np - y_pred_np)
            all_results["GroundTruth"].extend(y_true_np.tolist())
            all_results["Prediction"].extend(y_pred_np.tolist())
            all_results["Absolute_error"].extend(mae_batch.tolist()) 
            # ---- add quantile outputs to results
            if quantile_regression:
                all_results = add_quantile_results(
                    all_results, quantiles, y_true_np, q_pred_np, q10_np,
                    q90_np, has_q01, has_q09)
            if not val_mode:
                pre_lengths = test_lengths[length_idx:length_idx + batch_size]
                pre_cases = test_cases[length_idx:length_idx + batch_size]
                all_results["Prefix_length"].extend(np.array(pre_lengths).reshape(-1, 1).tolist())
                all_results["Case_id"].extend(np.array(pre_cases).reshape(-1, 1).tolist())
                length_idx += batch_size
            if heteroscedastic:
                all_results["Epistemic_Uncertainty"].extend(epistemic_std.detach().cpu().numpy().tolist())
                all_results["Aleatoric_Uncertainty"].extend(aleatoric_std.detach().cpu().numpy().tolist())
                all_results["Total_Uncertainty"].extend(total_std.detach().cpu().numpy().tolist())
    return all_results

def add_quantile_results(all_results, quantiles, y_true_np, 
                         q_pred_np, q10_np, q90_np, has_q01, has_q09):
    # store each quantile as its own column
    for qi, q in enumerate(quantiles):
        key = f"Q{str(q).replace('.', '_')}"  # e.g. Q0_95
        if key not in all_results:
            all_results[key] = []
        all_results[key].extend(q_pred_np[:, qi].tolist())
    # store interval (if available)
    if has_q01:
        if "PI10" not in all_results:
            all_results["PI10"] = []
        all_results["PI10"].extend(q10_np.tolist())
    if has_q09:
        if "PI90" not in all_results:
            all_results["PI90"] = []
        all_results["PI90"].extend(q90_np.tolist())
    if has_q01 and has_q09:
        if "PI_Width_10_90" not in all_results:
            all_results["PI_Width_10_90"] = []
        all_results["PI_Width_10_90"].extend((q90_np - q10_np).tolist())
        if "PI_Coverage_10_90" not in all_results:
            all_results["PI_Coverage_10_90"] = []
        cov = ((y_true_np >= q10_np) & (y_true_np <= q90_np)).astype(np.float32)
        all_results["PI_Coverage_10_90"].extend(cov.tolist())
    return all_results