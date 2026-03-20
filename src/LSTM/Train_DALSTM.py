# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 09:04:23 2025
"""
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
#from torch.nn.utils import clip_grad_value_

from src.LSTM.load_dataset import inverse_log1p, inverse_boxcox
from src.utils.loss_functions import (
    survival_hazard_nll,
    hazard_logits_to_survival_summary,
    survival_distribution_stats,
    pmf_to_quantile,
    hazard_logits_to_event_probs,
    event_probs_to_time,
)


   
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

def train_epoch_survival(model, train_loader, optimizer, bin_edges, device,
                         clip_grad_norm=True, clip_value=1.0):
    model.train()
    running_loss = 0.0
    n_batches = 0
    for inputs, targets, weights in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = survival_hazard_nll(logits, targets, bin_edges, reduction='mean')
        if torch.isnan(loss) or torch.isinf(loss):
            return float("nan")
        loss.backward()
        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(n_batches, 1)

@torch.no_grad()
def validate_epoch_survival(model, val_loader, bin_edges, device):
    model.eval()
    running_loss = 0.0
    n_batches = 0
    for inputs, targets, weights in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1)
        weights = weights.to(device)
        logits = model(inputs)
        loss = survival_hazard_nll(logits, targets, bin_edges, reduction='mean')
        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(n_batches, 1)

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

def get_reporting_bin_edges(args, surv_bin_edges, meta, device):
    edges_np = surv_bin_edges.detach().cpu().numpy()
    if args.log_trans:
        edges_np = inverse_log1p(edges_np)
    elif args.box_cox:
        edges_np = inverse_boxcox(edges_np, meta)
    return torch.tensor(edges_np, dtype=surv_bin_edges.dtype, device=device)

def survival_inference(
        args, model, inference_loader, all_results, surv_bin_edges,
        test_cases, test_lengths, val_mode=False, device=None, meta=None):

    length_idx = 0
    epsilon = 1e-8

    # IMPORTANT: compute reported summaries on original-scale bin edges
    report_bin_edges = get_reporting_bin_edges(
        args=args,
        surv_bin_edges=surv_bin_edges,
        meta=meta,
        device=device
    )

    with torch.no_grad():
        for inputs, targets, weights in inference_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1)

            logits = model(inputs)

            # full predictive distribution
            hazards, event_probs, cdf, tail_mass = hazard_logits_to_survival_summary(logits)

            # summary stats on ORIGINAL scale
            pred_mean, pred_std, _ = survival_distribution_stats(
                event_probs=event_probs,
                bin_edges=report_bin_edges,
                tail_mass=tail_mass
            )

            pred_median = pmf_to_quantile(
                event_probs=event_probs,
                bin_edges=report_bin_edges,
                q=0.5
            )

            pi80_low = pmf_to_quantile(
                event_probs=event_probs,
                bin_edges=report_bin_edges,
                q=0.10
            )
            pi80_high = pmf_to_quantile(
                event_probs=event_probs,
                bin_edges=report_bin_edges,
                q=0.90
            )

            pi90_low = pmf_to_quantile(
                event_probs=event_probs,
                bin_edges=report_bin_edges,
                q=0.05
            )
            pi90_high = pmf_to_quantile(
                event_probs=event_probs,
                bin_edges=report_bin_edges,
                q=0.95
            )

            # choose the exported point prediction
            if args.surv_pred_type == "mean":
                preds = pred_mean
            elif args.surv_pred_type == "median":
                preds = pred_median
            else:
                raise ValueError(f"Unknown surv_pred_type: {args.surv_pred_type}")

            preds = torch.maximum(preds, torch.tensor(epsilon, device=device))
            pred_mean = torch.maximum(pred_mean, torch.tensor(epsilon, device=device))
            pred_median = torch.maximum(pred_median, torch.tensor(epsilon, device=device))
            pred_std = torch.clamp(pred_std, min=0.0)

            # ground truth still needs inverse transform exactly as before
            y_true_np = targets.detach().cpu().numpy()
            if args.log_trans:
                y_true_np = inverse_log1p(y_true_np)
            elif args.box_cox:
                y_true_np = inverse_boxcox(y_true_np, meta)
                
            # prediction width
            pi80_width = pi80_high - pi80_low
            pi90_width = pi90_high - pi90_low

            # move predictions to numpy
            y_pred_np = preds.detach().cpu().numpy()
            pred_mean_np = pred_mean.detach().cpu().numpy()
            pred_median_np = pred_median.detach().cpu().numpy()
            pred_std_np = pred_std.detach().cpu().numpy()
            pi80_low_np = pi80_low.detach().cpu().numpy()
            pi80_high_np = pi80_high.detach().cpu().numpy()
            pi90_low_np = pi90_low.detach().cpu().numpy()
            pi90_high_np = pi90_high.detach().cpu().numpy()
            tail_mass_np = tail_mass.detach().cpu().numpy()
            pi80_width_np = pi80_width.detach().cpu().numpy()
            pi90_width_np = pi90_width.detach().cpu().numpy()

            mae_batch = np.abs(y_true_np - y_pred_np)

            all_results["GroundTruth"].extend(y_true_np.tolist())
            all_results["Prediction"].extend(y_pred_np.tolist())
            all_results["Absolute_error"].extend(mae_batch.tolist())

            all_results["Prediction_mean"].extend(pred_mean_np.tolist())
            all_results["Prediction_median"].extend(pred_median_np.tolist())
            all_results["PredStd"].extend(pred_std_np.tolist())
            all_results["PI80_low"].extend(pi80_low_np.tolist())
            all_results["PI80_high"].extend(pi80_high_np.tolist())
            all_results["PI90_low"].extend(pi90_low_np.tolist())
            all_results["PI90_high"].extend(pi90_high_np.tolist())
            all_results["Tail_mass"].extend(tail_mass_np.tolist())
            all_results["PI80_width"].extend(pi80_width_np.tolist())
            all_results["PI90_width"].extend(pi90_width_np.tolist())
            if "Used_New_Inference" in all_results:
                all_results["Used_New_Inference"].extend([0] * len(y_true_np))

            if not val_mode:
                batch_size = len(y_true_np)
                pre_lengths = test_lengths[length_idx:length_idx + batch_size]
                pre_cases = test_cases[length_idx:length_idx + batch_size]

                all_results["Prefix_length"].extend(
                    np.array(pre_lengths).reshape(-1, 1).tolist()
                )
                all_results["Case_id"].extend(
                    np.array(pre_cases).reshape(-1, 1).tolist()
                )
                length_idx += batch_size

    return all_results

def survival_inference_heuristic(
        args, model, inference_loader, all_results, surv_bin_edges,
        test_cases, test_lengths, val_mode=False, device=None, meta=None):
    length_idx = 0
    epsilon = 1e-8
    report_bin_edges = get_reporting_bin_edges(
        args=args,
        surv_bin_edges=surv_bin_edges,
        meta=meta,
        device=device)
    with torch.no_grad():
        for inputs, targets, weights in inference_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1)
            logits = model(inputs)
            # old heuristic prediction
            event_probs = hazard_logits_to_event_probs(logits)
            preds = event_probs_to_time(
                event_probs=event_probs,
                bin_edges=report_bin_edges,
                pred_type=args.surv_pred_type)
            preds = torch.maximum(preds, torch.tensor(epsilon, device=device))
            # extra columns from full survival summary
            hazards, event_probs_full, cdf, tail_mass = hazard_logits_to_survival_summary(logits)
            pred_mean, pred_std, _ = survival_distribution_stats(
                event_probs=event_probs_full,
                bin_edges=report_bin_edges,
                tail_mass=tail_mass)
            pred_median = pmf_to_quantile(
                event_probs=event_probs_full,
                bin_edges=report_bin_edges,
                q=0.5)
            pi80_low = pmf_to_quantile(event_probs_full, report_bin_edges, 0.10)
            pi80_high = pmf_to_quantile(event_probs_full, report_bin_edges, 0.90)
            pi90_low = pmf_to_quantile(event_probs_full, report_bin_edges, 0.05)
            pi90_high = pmf_to_quantile(event_probs_full, report_bin_edges, 0.95)
            pred_mean = torch.maximum(pred_mean, torch.tensor(epsilon, device=device))
            pred_median = torch.maximum(pred_median, torch.tensor(epsilon, device=device))
            pred_std = torch.clamp(pred_std, min=0.0)
            y_true_np = targets.detach().cpu().numpy()
            if args.log_trans:
                y_true_np = inverse_log1p(y_true_np)
            elif args.box_cox:
                y_true_np = inverse_boxcox(y_true_np, meta)
            y_pred_np = preds.detach().cpu().numpy()
            pred_mean_np = pred_mean.detach().cpu().numpy()
            pred_median_np = pred_median.detach().cpu().numpy()
            pred_std_np = pred_std.detach().cpu().numpy()
            pi80_low_np = pi80_low.detach().cpu().numpy()
            pi80_high_np = pi80_high.detach().cpu().numpy()
            pi90_low_np = pi90_low.detach().cpu().numpy()
            pi90_high_np = pi90_high.detach().cpu().numpy()
            tail_mass_np = tail_mass.detach().cpu().numpy()
            mae_batch = np.abs(y_true_np - y_pred_np)
            all_results["GroundTruth"].extend(y_true_np.tolist())
            all_results["Prediction"].extend(y_pred_np.tolist())
            all_results["Absolute_error"].extend(mae_batch.tolist())
            if "Prediction_mean" in all_results:
                all_results["Prediction_mean"].extend(pred_mean_np.tolist())
            if "Prediction_median" in all_results:
                all_results["Prediction_median"].extend(pred_median_np.tolist())
            if "PredStd" in all_results:
                all_results["PredStd"].extend(pred_std_np.tolist())
            if "PI80_low" in all_results:
                all_results["PI80_low"].extend(pi80_low_np.tolist())
            if "PI80_high" in all_results:
                all_results["PI80_high"].extend(pi80_high_np.tolist())
            if "PI90_low" in all_results:
                all_results["PI90_low"].extend(pi90_low_np.tolist())
            if "PI90_high" in all_results:
                all_results["PI90_high"].extend(pi90_high_np.tolist())
            if "PI80_width" in all_results:
                all_results["PI80_width"].extend((pi80_high_np - pi80_low_np).tolist())
            if "PI90_width" in all_results:
                all_results["PI90_width"].extend((pi90_high_np - pi90_low_np).tolist())
            if "Tail_mass" in all_results:
                all_results["Tail_mass"].extend(tail_mass_np.tolist())
            if "Used_New_Inference" in all_results:
                all_results["Used_New_Inference"].extend([0] * len(y_true_np))
            if not val_mode:
                batch_size = len(y_true_np)
                pre_lengths = test_lengths[length_idx:length_idx + batch_size]
                pre_cases = test_cases[length_idx:length_idx + batch_size]
                all_results["Prefix_length"].extend(
                    np.array(pre_lengths).reshape(-1, 1).tolist())
                all_results["Case_id"].extend(
                    np.array(pre_cases).reshape(-1, 1).tolist())
                length_idx += batch_size
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

def survival_collect_dual_predictions(
        args, model, inference_loader, surv_bin_edges,
        device=None, meta=None):

    report_bin_edges = get_reporting_bin_edges(
        args=args,
        surv_bin_edges=surv_bin_edges,
        meta=meta,
        device=device
    )

    out = {
        "GroundTruth": [],
        "Pred_Old": [],
        "Pred_New": [],
        "PI80_width": [],
        "Tail_mass": []
    }

    epsilon = 1e-8

    with torch.no_grad():
        for inputs, targets, weights in inference_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1)

            logits = model(inputs)

            # old heuristic prediction
            event_probs_old = hazard_logits_to_event_probs(logits)
            pred_old = event_probs_to_time(
                event_probs=event_probs_old,
                bin_edges=report_bin_edges,
                pred_type=args.surv_pred_type
            )
            pred_old = torch.maximum(pred_old, torch.tensor(epsilon, device=device))

            # new prediction + width
            hazards, event_probs, cdf, tail_mass = hazard_logits_to_survival_summary(logits)

            pred_mean_new, pred_std, _ = survival_distribution_stats(
                event_probs=event_probs,
                bin_edges=report_bin_edges,
                tail_mass=tail_mass
            )
            pred_median_new = pmf_to_quantile(
                event_probs=event_probs,
                bin_edges=report_bin_edges,
                q=0.5
            )

            if args.surv_pred_type == "mean":
                pred_new = pred_mean_new
            elif args.surv_pred_type == "median":
                pred_new = pred_median_new
            else:
                raise ValueError(f"Unknown surv_pred_type: {args.surv_pred_type}")

            pi80_low = pmf_to_quantile(event_probs, report_bin_edges, 0.10)
            pi80_high = pmf_to_quantile(event_probs, report_bin_edges, 0.90)
            pi80_width = pi80_high - pi80_low

            y_true_np = targets.detach().cpu().numpy()
            if args.log_trans:
                y_true_np = inverse_log1p(y_true_np)
            elif args.box_cox:
                y_true_np = inverse_boxcox(y_true_np, meta)

            out["GroundTruth"].extend(y_true_np.tolist())
            out["Pred_Old"].extend(pred_old.detach().cpu().numpy().tolist())
            out["Pred_New"].extend(pred_new.detach().cpu().numpy().tolist())
            out["PI80_width"].extend(pi80_width.detach().cpu().numpy().tolist())
            out["Tail_mass"].extend(tail_mass.detach().cpu().numpy().tolist())

    return out

def learn_pi80_width_threshold(val_dual_results, grid_size=41):
    y_true = np.asarray(val_dual_results["GroundTruth"], dtype=np.float64)
    pred_old = np.asarray(val_dual_results["Pred_Old"], dtype=np.float64)
    pred_new = np.asarray(val_dual_results["Pred_New"], dtype=np.float64)
    pi80_width = np.asarray(val_dual_results["PI80_width"], dtype=np.float64)

    if len(pi80_width) == 0:
        return None

    w_min = float(np.min(pi80_width))
    w_max = float(np.max(pi80_width))

    if np.isclose(w_min, w_max):
        return w_min

    candidates = np.linspace(w_min, w_max, grid_size)

    best_thr = candidates[0]
    best_mae = float("inf")

    for thr in candidates:
        use_new = pi80_width >= thr
        pred_hybrid = np.where(use_new, pred_new, pred_old)
        mae = np.mean(np.abs(y_true - pred_hybrid))
        if mae < best_mae:
            best_mae = mae
            best_thr = thr

    return float(best_thr)

def survival_inference_pi80_hybrid(
        args, model, inference_loader, all_results, surv_bin_edges,
        pi80_width_threshold, test_cases, test_lengths,
        val_mode=False, device=None, meta=None):

    length_idx = 0
    epsilon = 1e-8

    report_bin_edges = get_reporting_bin_edges(
        args=args,
        surv_bin_edges=surv_bin_edges,
        meta=meta,
        device=device
    )

    with torch.no_grad():
        for inputs, targets, weights in inference_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1)

            logits = model(inputs)

            # old heuristic prediction
            event_probs_old = hazard_logits_to_event_probs(logits)
            pred_old = event_probs_to_time(
                event_probs=event_probs_old,
                bin_edges=report_bin_edges,
                pred_type=args.surv_pred_type
            )

            # new prediction + intervals
            hazards, event_probs, cdf, tail_mass = hazard_logits_to_survival_summary(logits)

            pred_mean, pred_std, _ = survival_distribution_stats(
                event_probs=event_probs,
                bin_edges=report_bin_edges,
                tail_mass=tail_mass
            )
            pred_median = pmf_to_quantile(event_probs, report_bin_edges, 0.5)

            if args.surv_pred_type == "mean":
                pred_new = pred_mean
            elif args.surv_pred_type == "median":
                pred_new = pred_median
            else:
                raise ValueError(f"Unknown surv_pred_type: {args.surv_pred_type}")

            pi80_low = pmf_to_quantile(event_probs, report_bin_edges, 0.10)
            pi80_high = pmf_to_quantile(event_probs, report_bin_edges, 0.90)
            pi90_low = pmf_to_quantile(event_probs, report_bin_edges, 0.05)
            pi90_high = pmf_to_quantile(event_probs, report_bin_edges, 0.95)

            pi80_width = pi80_high - pi80_low
            pi90_width = pi90_high - pi90_low

            use_new = pi80_width >= pi80_width_threshold
            preds = torch.where(use_new, pred_new, pred_old)
            preds = torch.maximum(preds, torch.tensor(epsilon, device=device))

            y_true_np = targets.detach().cpu().numpy()
            if args.log_trans:
                y_true_np = inverse_log1p(y_true_np)
            elif args.box_cox:
                y_true_np = inverse_boxcox(y_true_np, meta)

            y_pred_np = preds.detach().cpu().numpy()
            mae_batch = np.abs(y_true_np - y_pred_np)

            all_results["GroundTruth"].extend(y_true_np.tolist())
            all_results["Prediction"].extend(y_pred_np.tolist())
            all_results["Absolute_error"].extend(mae_batch.tolist())

            all_results["Prediction_mean"].extend(pred_mean.detach().cpu().numpy().tolist())
            all_results["Prediction_median"].extend(pred_median.detach().cpu().numpy().tolist())
            all_results["PredStd"].extend(pred_std.detach().cpu().numpy().tolist())
            all_results["PI80_low"].extend(pi80_low.detach().cpu().numpy().tolist())
            all_results["PI80_high"].extend(pi80_high.detach().cpu().numpy().tolist())
            all_results["PI90_low"].extend(pi90_low.detach().cpu().numpy().tolist())
            all_results["PI90_high"].extend(pi90_high.detach().cpu().numpy().tolist())
            all_results["PI80_width"].extend(pi80_width.detach().cpu().numpy().tolist())
            all_results["PI90_width"].extend(pi90_width.detach().cpu().numpy().tolist())
            all_results["Tail_mass"].extend(tail_mass.detach().cpu().numpy().tolist())

            if "Used_New_Inference" in all_results:
                all_results["Used_New_Inference"].extend(
                    use_new.detach().cpu().numpy().astype(np.int32).tolist()
                )

            if not val_mode:
                batch_size = len(y_true_np)
                pre_lengths = test_lengths[length_idx:length_idx + batch_size]
                pre_cases = test_cases[length_idx:length_idx + batch_size]

                all_results["Prefix_length"].extend(
                    np.array(pre_lengths).reshape(-1, 1).tolist()
                )
                all_results["Case_id"].extend(
                    np.array(pre_cases).reshape(-1, 1).tolist()
                )
                length_idx += batch_size

    return all_results

def quantile_inference(
        args, model, all_results,
        inference_loader, inference_lengths, inference_cases,
        quantiles=(0.1, 0.5, 0.6, 0.9, 0.95, 0.99), device=None):
    # helpful index mapping for quantiles
    q_to_idx = {float(q): i for i, q in enumerate(quantiles)}
    has_q05 = 0.5 in q_to_idx
    has_q01 = 0.1 in q_to_idx
    has_q09 = 0.9 in q_to_idx
    length_idx = 0
    with torch.no_grad():
        for batch in inference_loader:
            inputs = batch[0].to(device)
            y_true = batch[1].to(device)
            batch_size = inputs.shape[0]
            # ---- predictions ----
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
            # only keep this if your y is always >= 0
            epsilon = 1e-8
            # clamp ALL quantiles consistently (preserves monotonicity)
            q_pred = torch.maximum(q_pred, torch.tensor(epsilon, device=device))
            y_pred = q_pred[:, q_to_idx[0.5]]
            if has_q01:
                q10 = q_pred[:, q_to_idx[0.1]]
            if has_q09:
                q90 = q_pred[:, q_to_idx[0.9]]
            # ---- errors ----
            y_true_np = y_true.detach().cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()
            # If storing intervals/quantiles, prepare numpy too
            q_pred_np = q_pred.detach().cpu().numpy()  # (B,K)
            if has_q01:
                q10_np = q10.detach().cpu().numpy()
            if has_q09:
                q90_np = q90.detach().cpu().numpy()
            if args.log_trans:
                y_pred_np = inverse_log1p(y_pred_np)
                y_true_np = inverse_log1p(y_true_np)
                q_pred_np = inverse_log1p(q_pred_np)
                if has_q01:
                    q10_np = inverse_log1p(q10_np)
                if has_q09:
                    q90_np = inverse_log1p(q90_np)
            mae_batch = np.abs(y_true_np - y_pred_np)
            all_results["GroundTruth"].extend(y_true_np.tolist())
            all_results["Prediction"].extend(y_pred_np.tolist())
            all_results["Absolute_error"].extend(mae_batch.tolist()) 
            # ---- add quantile outputs to results
            all_results = add_quantile_results(
                all_results, quantiles, y_true_np, q_pred_np, q10_np,
                q90_np, has_q01, has_q09)
            pre_lengths = inference_lengths[length_idx:length_idx + batch_size]
            pre_cases = inference_cases[length_idx:length_idx + batch_size]
            all_results["Prefix_length"].extend(np.array(pre_lengths).reshape(-1, 1).tolist())
            all_results["Case_id"].extend(np.array(pre_cases).reshape(-1, 1).tolist())
            length_idx += batch_size
    return all_results