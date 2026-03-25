# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:16:54 2026
"""
import os, argparse, yaml, pickle, copy, random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, ParameterGrid
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

from src.LSTM.load_dataset import get_train_params
from src.LSTM.model_DALSTM import DALSTMClassifier, DALSTMSurvivalModel
from src.utils.optimizer import set_optimizer
from src.LSTM.Preprocess_DALSTM import compute_survival_bin_edges
from src.LSTM.Train_DALSTM import survival_inference_heuristic
    

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def format_quantile_for_name(quantile):
    q_str = str(quantile).replace(".", "p")
    return f"q{q_str}"    
    
def get_result_name(result_dir, dataset, model_name, seed, quantile=None):
    q_part = f"_{format_quantile_for_name(quantile)}" if quantile is not None else ""
    csv_name = (
        dataset + '_' + model_name + 'classification' +
        q_part + '_seed_' + str(seed) + '_inference.csv'
    )
    return os.path.join(result_dir, csv_name)

def get_ml_result_name(result_dir, dataset, model_name, ml_name, seed, quantile=None):
    q_part = f"_{format_quantile_for_name(quantile)}" if quantile is not None else ""
    csv_name = (
        dataset + '_' + model_name + 'classification_' + ml_name +
        q_part + '_seed_' + str(seed) + '_inference.csv'
    )
    return os.path.join(result_dir, csv_name)

def get_ml_summary_name(result_dir, dataset, model_name, ml_name, quantile=None):
    q_part = f"_{format_quantile_for_name(quantile)}" if quantile is not None else ""
    summary_name = (
        dataset + '_' + model_name + 'classification_' + ml_name +
        q_part + '_summary.csv'
    )
    return os.path.join(result_dir, summary_name)

def get_summary_name(result_dir, dataset, model_name, quantile=None):
    q_part = f"_{format_quantile_for_name(quantile)}" if quantile is not None else ""
    summary_name = (
        dataset + '_' + model_name + 'classification' +
        q_part + '_summary.csv'
    )
    return os.path.join(result_dir, summary_name)

def get_ml_cv_result_name(result_dir, dataset, model_name, ml_name, seed, quantile=None):
    q_part = f"_{format_quantile_for_name(quantile)}" if quantile is not None else ""
    csv_name = (
        dataset + '_' + model_name + '_classification_' + ml_name +
        q_part + '_seed_' + str(seed) + '_cv_results.csv'
    )
    return os.path.join(result_dir, csv_name)


def get_ml_feature_name(result_dir, dataset, model_name, ml_name, seed, split, quantile=None):
    q_part = f"_{format_quantile_for_name(quantile)}" if quantile is not None else ""
    csv_name = (
        dataset + '_' + model_name + '_classification_' + ml_name +
        q_part + '_seed_' + str(seed) + f'_{split}_features.csv'
    )
    return os.path.join(result_dir, csv_name)
    
def load_case_ids(temp_dir, dataset, model_name):    
    train_case_lst_path = os.path.join(temp_dir, model_name+'_train_cases_'+dataset+'.pkl') 
    val_case_lst_path = os.path.join(temp_dir, model_name+'_val_cases_'+dataset+'.pkl') 
    test_case_lst_path = os.path.join(temp_dir, model_name+'_test_cases_'+dataset+'.pkl') 
    with open(train_case_lst_path, "rb") as f:
        train_cases = pickle.load(f)
    with open(val_case_lst_path, "rb") as f:
        val_cases = pickle.load(f)
    with open(test_case_lst_path, "rb") as f:
        test_cases = pickle.load(f)
    return (train_cases, val_cases, test_cases)

def load_case_lengths(temp_dir, dataset, model_name):    
    train_case_length_path = os.path.join(temp_dir, model_name+'_train_length_list_'+dataset+'.pkl') 
    val_case_length_path = os.path.join(temp_dir, model_name+'_val_length_list_'+dataset+'.pkl') 
    test_case_length_path = os.path.join(temp_dir, model_name+'_test_length_list_'+dataset+'.pkl') 
    with open(train_case_length_path, "rb") as f:
        train_lengths = pickle.load(f)
    with open(val_case_length_path, "rb") as f:
        val_lengths = pickle.load(f)
    with open(test_case_length_path, "rb") as f:
        test_lengths = pickle.load(f)
    return (train_lengths, val_lengths, test_lengths)


def get_dataset(temp_dir, model_name, dataset):
    y_train_path = os.path.join(temp_dir, model_name+'_y_train_'+dataset+'.pt')
    y_val_path = os.path.join(temp_dir, model_name+'_y_val_'+dataset+'.pt')
    y_test_path = os.path.join(temp_dir, model_name+'_y_test_'+dataset+'.pt')
    y_train = torch.load(y_train_path)
    y_val = torch.load(y_val_path)
    y_test = torch.load(y_test_path)
    X_train_path = os.path.join(temp_dir, model_name+'_X_train_'+dataset+'.pt')
    X_val_path = os.path.join(temp_dir, model_name+'_X_val_'+dataset+'.pt')
    X_test_path = os.path.join(temp_dir, model_name+'_X_test_'+dataset+'.pt')
    X_train = torch.load(X_train_path)
    X_val = torch.load(X_val_path)
    X_test = torch.load(X_test_path)
    return (y_train, y_val, y_test, X_train, X_val, X_test)  

def get_delayed_cases(train_cases, val_cases, test_cases,
                      y_train, y_val, y_test, quantile):
    """
    Compute quantile threshold τ using train + validation ground truth.
    Return dictionary {case_id: delayed_label}.    
    delayed_label:
        1 -> delayed
        0 -> not delayed
    """
    train_cases = np.asarray(train_cases)
    val_cases = np.asarray(val_cases)
    test_cases = np.asarray(test_cases)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)
    df_train = pd.DataFrame({"Case_id": train_cases, "GT": y_train})
    df_val = pd.DataFrame({"Case_id": val_cases, "GT": y_val})
    df_test = pd.DataFrame({"Case_id": test_cases, "GT": y_test})
    # Compute threshold using train + validation
    df_tv = pd.concat([df_train, df_val], ignore_index=True)
    df_tv = df_tv.groupby("Case_id").first()
    true_total_tv = df_tv["GT"].astype(float)
    tau = float(true_total_tv.quantile(quantile))
    # Get total duration for all cases
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    df_all = df_all.groupby("Case_id").first()
    true_total_all = df_all["GT"].astype(float)
    # Create delayed dictionary
    delayed_dict = (true_total_all > tau).astype(int).to_dict()
    return tau, delayed_dict

def get_loaders(cfg, X_train, X_val, X_test, y_train_cls, y_val_cls, y_test_cls):
    batch_size = cfg['DALSTM']['batch_size']
    test_batch_size = cfg['DALSTM']['test_batch_size']
    train_dataset = TensorDataset(X_train, y_train_cls)
    val_dataset = TensorDataset(X_val, y_val_cls)
    test_dataset = TensorDataset(X_test, y_test_cls)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return (train_loader, val_loader, test_loader)

def get_model_config(cfg):
    n_layers = cfg['DALSTM']['n_layers']
    hidden_size = cfg['DALSTM']['hidden_size']
    dropout = cfg['DALSTM']['dropout']
    dropout_prob = cfg['DALSTM']['dropout_prob']
    return (n_layers, hidden_size, dropout, dropout_prob)

def get_opt_schedule(cfg, model, learning_rate=0.001): 
    optimizer_type = cfg['DALSTM']['optimizer']  
    eps = cfg['DALSTM']['eps']
    weight_decay = cfg['DALSTM']['weight_decay']
    # define optimizer
    optimizer = set_optimizer(
        model, optimizer_type, learning_rate, eps, weight_decay)
    # define scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)    
    return optimizer, scheduler  

def train_dalstm_classifier(
        model, train_loader, val_loader, test_loader, optimizer, 
        scheduler=None, num_epochs=300, early_stop=True, early_patience=30,
        min_delta=0.0, validate_every=5, device=None,):
    """
    Train DALSTMClassifier for binary delay detection.
    Parameters
    ----------
    model : nn.Module
        DALSTMClassifier model.
    train_loader, val_loader, test_loader : DataLoader
        Data loaders.
    optimizer : torch.optim.Optimizer
        Optimizer.
    scheduler : optional
        Learning rate scheduler.
    num_epochs : int
        Maximum number of epochs.
    early_stop : bool
        Whether to use early stopping.
    early_patience : int
        Number of epochs with no sufficient improvement before stopping.
    min_delta : float
        Minimum improvement in validation loss to reset patience.
    validate_every : int
        Run validation every N epochs.
    device : str or torch.device, optional
        Device to use.
    Returns
    -------
    results : dict
        Dictionary containing:
        - best_model_state
        - history
        - best_val_loss
        - test_loss
        - test_probs
        - test_preds
        - test_targets
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_epoch": [],
    }
    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_train_loss = 0.0
        n_train_samples = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            optimizer.zero_grad()
            logits = model(X_batch)  # shape: [batch]
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            batch_size = X_batch.size(0)
            running_train_loss += loss.item() * batch_size
            n_train_samples += batch_size
        epoch_train_loss = running_train_loss / max(n_train_samples, 1)
        history["train_loss"].append(epoch_train_loss)
        print(f"Epoch {epoch:03d}/{num_epochs:03d} | Train Loss: {epoch_train_loss:.6f}")
        # Validate every `validate_every` epochs
        if epoch % validate_every == 0:
            val_loss = evaluate_binary_classifier_loss(
                model=model, data_loader=val_loader, criterion=criterion,
                device=device,)
            history["val_loss"].append(val_loss)
            history["val_epoch"].append(epoch)
            print(f"Epoch {epoch:03d}/{num_epochs:03d} | Val Loss:   {val_loss:.6f}")
            if scheduler is not None:
                scheduler.step(val_loss)
            improved = (best_val_loss - val_loss) > min_delta
            if improved:
                best_epoch = epoch
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"  -> New best model saved (val_loss={best_val_loss:.6f})")
            else:
                if early_stop and (epoch - best_epoch >= early_patience):
                    print("Early stopping triggered.")
                    break
    # Load best model before test evaluation
    model.load_state_dict(best_model_state)
    test_loss, test_probs, test_preds, test_targets = evaluate_binary_classifier_full(
        model=model, data_loader=test_loader, criterion=criterion, device=device,)
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Test Loss: {test_loss:.6f}")
    results = {
        "best_model_state": best_model_state,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_probs": test_probs,
        "test_preds": test_preds,
        "test_targets": test_targets,
        "model": model,}
    return results


def evaluate_binary_classifier_loss(model, data_loader, criterion, device):
    """Compute average loss on a dataset."""
    model.eval()
    running_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            batch_size = X_batch.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size
    return running_loss / max(n_samples, 1)


def evaluate_binary_classifier_full(model, data_loader, criterion, device, threshold=0.5):
    """
    Evaluate model and return loss, probabilities, predictions, and targets.
    """
    model.eval()
    running_loss = 0.0
    n_samples = 0
    all_probs = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()
            batch_size = X_batch.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu().long())
    avg_loss = running_loss / max(n_samples, 1)
    all_probs = torch.cat(all_probs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return avg_loss, all_probs, all_preds, all_targets

def build_classification_inference_df_with_probs(
        test_cases, test_prefix_lengths, y_test_cls, test_preds, test_probs,):
    if torch.is_tensor(y_test_cls):
        y_test_cls = y_test_cls.detach().cpu().numpy()
    if torch.is_tensor(test_preds):
        test_preds = test_preds.detach().cpu().numpy()
    if torch.is_tensor(test_probs):
        test_probs = test_probs.detach().cpu().numpy()
    df = pd.DataFrame({
        "Case_id": list(test_cases),
        "Prefix_length": list(test_prefix_lengths),
        "GroundTruth": y_test_cls.astype(int),
        "Prediction": test_preds.astype(int),
        "Probability": test_probs.astype(float),})
    return df

def compute_classification_metrics_with_auc(y_true, y_pred, y_prob):
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_prob):
        y_prob = y_prob.detach().cpu().numpy()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    # AUROC requires both classes to be present
    if len(set(y_true.tolist())) > 1:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["auroc"] = float("nan")
    return metrics

def run_single_seed_classification(
        cfg, seed, X_train, X_val, X_test, y_train_cls, y_val_cls, y_test_cls,
        test_cases, test_prefix_lengths, get_loaders, get_train_params, 
        get_model_config, get_opt_schedule, DALSTMClassifier,
        train_dalstm_classifier, device=None,):
    """
    Run the whole train/validation/test flow for one seed.
    """
    set_all_seeds(seed)
    # loaders
    train_loader, val_loader, test_loader = get_loaders(
        cfg, X_train, X_val, X_test, y_train_cls, y_val_cls, y_test_cls)
    # training params
    num_epochs, early_stop, early_patience, min_delta = get_train_params(cfg)
    # model config
    input_size = X_train.shape[-1]
    n_layers, hidden_size, dropout, dropout_prob = get_model_config(cfg)
    model = DALSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout=dropout,
        p_fix=dropout_prob,)
    optimizer, scheduler = get_opt_schedule(cfg, model)
    # train
    results = train_dalstm_classifier(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=num_epochs,
    early_stop=early_stop,
    min_delta=min_delta,
    validate_every=5,
    device=device,)
    # metrics
    test_metrics = compute_classification_metrics_with_auc(
        results["test_targets"],
        results["test_preds"],
        results["test_probs"],)
    # inference dataframe
    inference_df = build_classification_inference_df_with_probs(
        test_cases=test_cases,
        test_prefix_lengths=test_prefix_lengths,
        y_test_cls=y_test_cls,
        test_preds=results["test_preds"],
        test_probs=results["test_probs"],)
    
    return {
        "seed": seed,
        "model": results["model"],
        "history": results["history"],
        "best_epoch": results["best_epoch"],
        "best_val_loss": results["best_val_loss"],
        "test_loss": results["test_loss"],
        "test_metrics": test_metrics,
        "inference_df": inference_df,
        "test_preds": results["test_preds"],
        "test_probs": results["test_probs"],
        "test_targets": results["test_targets"],}

def run_multi_seed_classification(
        cfg, seeds, X_train, X_val, X_test, y_train_cls, y_val_cls, y_test_cls,
        test_cases, test_prefix_lengths, get_loaders, get_train_params,
        get_model_config, get_opt_schedule, DALSTMClassifier,
        train_dalstm_classifier, dataset, model_name,
        device=None, save_dir=None, quantile=None):
    """
    Run classification training/evaluation for multiple seeds.
    Returns:
    - per_seed_results
    - summary_metrics
    - concatenated inference dataframe
    """
    per_seed_results = []
    summary_rows = []
    inference_dfs = []
    for seed in seeds:
        print(f"\n{'='*20} Seed {seed} {'='*20}")
        out = run_single_seed_classification(
            cfg=cfg,
            seed=seed,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train_cls=y_train_cls,
            y_val_cls=y_val_cls,
            y_test_cls=y_test_cls,
            test_cases=test_cases,
            test_prefix_lengths=test_prefix_lengths,
            get_loaders=get_loaders,
            get_train_params=get_train_params,
            get_model_config=get_model_config,
            get_opt_schedule=get_opt_schedule,
            DALSTMClassifier=DALSTMClassifier,
            train_dalstm_classifier=train_dalstm_classifier,
            device=device,)
        per_seed_results.append(out)
        row = {
            "seed": seed,
            "best_epoch": out["best_epoch"],
            "best_val_loss": out["best_val_loss"],
            "test_loss": out["test_loss"],
            **out["test_metrics"],}
        summary_rows.append(row)
        df_seed = out["inference_df"].copy()
        df_seed["seed"] = seed
        inference_dfs.append(df_seed)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            csv_path = get_result_name(
                result_dir=save_dir,
                dataset=dataset,
                model_name=model_name,
                seed=seed,
                quantile=quantile)
            df_seed.to_csv(csv_path, index=False)
    summary_df = pd.DataFrame(summary_rows)
    all_inference_df = pd.concat(inference_dfs, axis=0, ignore_index=True)
    metric_cols = [c for c in summary_df.columns if c != "seed"]
    summary_metrics = {}
    for col in metric_cols:
        summary_metrics[col] = {
            "mean": summary_df[col].mean(),
            "std": summary_df[col].std(),}
    if save_dir is not None:
        summary_path = get_summary_name(
            result_dir=save_dir,
            dataset=dataset,
            model_name=model_name,
            quantile=quantile
            )
        summary_df.to_csv(summary_path, index=False)
    return {
        "per_seed_results": per_seed_results,
        "summary_df": summary_df,
        "summary_metrics": summary_metrics,
        "all_inference_df": all_inference_df,}

def get_survival_model(
        args, cfg, temp_dir, result_dir, dataset, model_name, seed, X_train, y_train,
        device="cuda" if torch.cuda.is_available() else "cpu"):
    checkpoint_name = dataset+'_'+model_name+'_survival_wos_seed'+str(seed)+'.pt'
    result_dict_name = dataset+'_'+model_name+'_overall_results.pkl'   
    result_dict_path = os.path.join(result_dir, result_dict_name)
    with open(result_dict_path, "rb") as f:
        result_dict = pickle.load(f)
    checkpoint_path = os.path.join(temp_dir, checkpoint_name)
    input_size = X_train.shape[-1]
    (n_layers, hidden_size, dropout, dropout_prob) = get_model_config(cfg)
    survival_params = result_dict[('survival', 'wos')]['best_params'][0]
    surv_num_bins = survival_params['surv_num_bins']
    surv_tail_frac = survival_params['surv_tail_frac']
    surv_tail_bin_frac = survival_params['surv_tail_bin_frac']
    surv_binning = survival_params['surv_binning']
    surv_pred_type = survival_params['surv_pred_type']
    checkpoint = torch.load(checkpoint_path, map_location=device)    
    model = DALSTMSurvivalModel(
        input_size=input_size, hidden_size=hidden_size, n_layers=n_layers,
        dropout=dropout, p_fix=dropout_prob,
        num_bins=surv_num_bins).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    surv_bin_edges = compute_survival_bin_edges(
        y_train=y_train, num_bins=surv_num_bins, method=surv_binning,
        tail_frac=surv_tail_frac, tail_bin_frac=surv_tail_bin_frac).to(device)
    args.surv_pred_type = surv_pred_type
    args.log_trans = False
    args.box_cox = False
    return model, surv_bin_edges, args
        
def inference_with_survival(
        args, cfg, model, surv_bin_edges,
        X_train, X_val, X_test, y_train, y_val, y_test,
        train_cases, val_cases, test_cases,
        train_lengths, val_lengths, test_lengths,
        device="cuda" if torch.cuda.is_available() else "cpu"):
    batch_size = cfg['DALSTM']['batch_size']
    test_batch_size = cfg['DALSTM']['test_batch_size']
    all_results_train_val = {
        'Case_id': [], 'Prefix_length': [], 'GroundTruth': [], 'Prediction': [],
        'Absolute_error': [], 'Prediction_mean': [], 'Prediction_median': [],
        'PredStd': [], 'PI80_low': [], 'PI80_high': [], 'PI90_low': [],
        'PI90_high': [], 'PI80_width': [], 'PI90_width': [], 'Tail_mass': [], 
        'Used_New_Inference': [],}
    all_results_test = {
        'Case_id': [], 'Prefix_length': [], 'GroundTruth': [], 'Prediction': [],
        'Absolute_error': [], 'Prediction_mean': [], 'Prediction_median': [],
        'PredStd': [], 'PI80_low': [], 'PI80_high': [], 'PI90_low': [],
        'PI90_high': [], 'PI80_width': [], 'PI90_width': [], 'Tail_mass': [], 
        'Used_New_Inference': [],}
    model.eval()
    weights_train = torch.ones(X_train.shape[0], dtype=torch.float32)
    weights_val   = torch.ones(X_val.shape[0], dtype=torch.float32)
    weights_test  = torch.ones(X_test.shape[0], dtype=torch.float32)
    train_dataset = TensorDataset(X_train, y_train, weights_train)
    val_dataset   = TensorDataset(X_val, y_val, weights_val)
    test_dataset  = TensorDataset(X_test, y_test, weights_test)
    trainval_dataset = ConcatDataset([train_dataset, val_dataset])
    trainval_loader = DataLoader(trainval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    train_val_cases = train_cases + val_cases
    train_val_lengths = train_lengths + val_lengths
    all_results_train_val = survival_inference_heuristic(
                args, model, trainval_loader, all_results_train_val, surv_bin_edges,
                train_val_cases, train_val_lengths, val_mode=False, device=device)    
    flattened_list = [item for sublist in all_results_train_val['Prefix_length'] 
                      for item in sublist]
    all_results_train_val['Prefix_length'] = flattened_list  
    all_results_train_val['Case_id'] = [item for sublist in all_results_train_val['Case_id']
                                        for item in sublist]
    train_val_df = pd.DataFrame(all_results_train_val) 
    cols = ['Case_id', 'Prefix_length'] + [c for c in train_val_df.columns if c not in ['Case_id', 'Prefix_length']]
    train_val_df = train_val_df[cols]
    all_results_test = survival_inference_heuristic(
                args, model, test_loader, all_results_test, surv_bin_edges,
                test_cases, test_lengths, val_mode=False, device=device) 
    flattened_list = [item for sublist in all_results_test['Prefix_length'] 
                      for item in sublist]
    all_results_test['Prefix_length'] = flattened_list  
    all_results_test['Case_id'] = [item for sublist in all_results_test['Case_id']
                                        for item in sublist]
    test_df = pd.DataFrame(all_results_test)
    cols = ['Case_id', 'Prefix_length'] + [c for c in test_df.columns if c not in ['Case_id', 'Prefix_length']]
    test_df = test_df[cols]
    train_val_df = train_val_df.sort_values(["Case_id", "Prefix_length"]).copy()
    train_val_df["Elapsed_time"] = (
        train_val_df.groupby("Case_id")["GroundTruth"].transform("first") - train_val_df["GroundTruth"])
    train_val_df["Time_since_last_event"] = (
        train_val_df.groupby("Case_id")["GroundTruth"].shift(1) - train_val_df["GroundTruth"]).fillna(0)
    train_val_df["Case_id"] = train_val_df["Case_id"].astype(str)
    test_df = test_df.sort_values(["Case_id", "Prefix_length"]).copy()
    test_df["Elapsed_time"] = (
        test_df.groupby("Case_id")["GroundTruth"].transform("first") - test_df["GroundTruth"])
    test_df["Time_since_last_event"] = (
        test_df.groupby("Case_id")["GroundTruth"].shift(1) - test_df["GroundTruth"]).fillna(0)  
    test_df["Case_id"] = test_df["Case_id"].astype(str)
    cols_to_drop = ['GroundTruth', 'Absolute_error', 'Used_New_Inference']
    train_val_df = train_val_df.drop(cols_to_drop, axis=1)
    test_df = test_df.drop(cols_to_drop, axis=1)
    train_val_df = train_val_df.rename(columns={'Prediction': 'Prediction_time'})
    test_df = test_df.rename(columns={'Prediction': 'Prediction_time'})
    return train_val_df, test_df
    
def get_seed_feature_tables(
        args, cfg, temp_dir, result_dir, dataset, model_name, seed,
        X_train, X_val, X_test, y_train, y_val, y_test,
        train_cases, val_cases, test_cases,
        train_lengths, val_lengths, test_lengths,
        delayed_dict,
        device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Build train_val_df and test_df for one seed using survival-model outputs
    as features and delayed_dict as labels.
    """

    model, surv_bin_edges, args = get_survival_model(
        args=args, cfg=cfg, temp_dir=temp_dir, result_dir=result_dir,
        dataset=dataset, model_name=model_name, seed=seed,
        X_train=X_train, y_train=y_train, device=device
    )

    train_val_df, test_df = inference_with_survival(
        args=args, cfg=cfg, model=model, surv_bin_edges=surv_bin_edges,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        train_cases=train_cases, val_cases=val_cases, test_cases=test_cases,
        train_lengths=train_lengths, val_lengths=val_lengths, test_lengths=test_lengths,
        device=device
    )

    delayed_dict = {str(k): int(v) for k, v in delayed_dict.items()}

    train_val_df["Case_id"] = train_val_df["Case_id"].astype(str)
    test_df["Case_id"] = test_df["Case_id"].astype(str)

    train_val_df["GroundTruth"] = train_val_df["Case_id"].map(delayed_dict).astype(int)
    test_df["GroundTruth"] = test_df["Case_id"].map(delayed_dict).astype(int)

    return train_val_df, test_df

def get_feature_columns(df):
    return [c for c in df.columns if c not in ["Case_id", "GroundTruth"]]

def cross_validate_catboost(
        train_val_df,
        param_grid=None,
        n_splits=5,
        random_seed=42,
        score_name="auroc"):
    """
    Grouped cross-validation on train_val_df using Case_id as groups.
    Returns best params and a CV summary dataframe.
    """

    feature_cols = get_feature_columns(train_val_df)

    X = train_val_df[feature_cols]
    y = train_val_df["GroundTruth"].astype(int).to_numpy()
    groups = train_val_df["Case_id"].astype(str).to_numpy()

    if param_grid is None:
        param_grid = {
            "depth": [4, 6],
            "learning_rate": [0.03, 0.05],
            "iterations": [200, 300],
            "l2_leaf_reg": [3.0, 5.0],
        }

    cv = GroupKFold(n_splits=n_splits)
    all_rows = []

    for params in ParameterGrid(param_grid):
        fold_scores = []

        for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
            X_tr = X.iloc[tr_idx]
            y_tr = y[tr_idx]
            X_va = X.iloc[va_idx]
            y_va = y[va_idx]

            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=random_seed,
                verbose=False,
                allow_writing_files=False,
                auto_class_weights="Balanced",
                **params
            )

            model.fit(X_tr, y_tr)

            va_prob = model.predict_proba(X_va)[:, 1]

            if score_name == "auroc":
                if len(set(y_va.tolist())) > 1:
                    score = roc_auc_score(y_va, va_prob)
                else:
                    score = float("nan")
            else:
                raise ValueError(f"Unsupported score_name: {score_name}")

            fold_scores.append(score)

        row = dict(params)
        row["cv_mean_" + score_name] = np.nanmean(fold_scores)
        row["cv_std_" + score_name] = np.nanstd(fold_scores)
        all_rows.append(row)

    cv_results_df = pd.DataFrame(all_rows)
    best_idx = cv_results_df["cv_mean_" + score_name].idxmax()
    best_params = {
        k: cv_results_df.loc[best_idx, k]
        for k in param_grid.keys()
    }

    return best_params, cv_results_df

def fit_catboost_classifier_with_cv(
        train_val_df,
        test_df,
        random_seed=42,
        param_grid=None,
        n_splits=5):
    """
    Run grouped CV on train_val_df, refit best CatBoost on full train_val_df,
    then infer on test_df.
    """

    feature_cols = get_feature_columns(train_val_df)

    best_params, cv_results_df = cross_validate_catboost(
        train_val_df=train_val_df,
        param_grid=param_grid,
        n_splits=n_splits,
        random_seed=random_seed,
        score_name="auroc"
    )

    X_trainval = train_val_df[feature_cols]
    y_trainval = train_val_df["GroundTruth"].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df["GroundTruth"].astype(int)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=False,
        auto_class_weights="Balanced",
        **best_params
    )

    model.fit(X_trainval, y_trainval)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    metrics = compute_classification_metrics_with_auc(
        y_true=y_test.to_numpy(),
        y_pred=test_pred,
        y_prob=test_prob,
    )

    inference_df = pd.DataFrame({
        "Case_id": test_df["Case_id"].values,
        "Prefix_length": test_df["Prefix_length"].values,
        "GroundTruth": y_test.values,
        "Prediction": test_pred,
        "Probability": test_prob,
    })

    return {
        "model": model,
        "feature_cols": feature_cols,
        "best_params": best_params,
        "cv_results_df": cv_results_df,
        "metrics": metrics,
        "inference_df": inference_df,
    }

def run_single_seed_ml_classifier(
        args, cfg, temp_dir, result_dir, dataset, model_name, seed,
        X_train, X_val, X_test, y_train, y_val, y_test,
        train_cases, val_cases, test_cases,
        train_lengths, val_lengths, test_lengths,
        delayed_dict,
        ml_name="CAT",
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir=None,
        save_feature_tables=False,
        quantile=None):
    """
    For one seed:
    1) build feature tables
    2) run grouped CV on train_val_df
    3) refit best model on full train_val_df
    4) predict on test_df
    """

    set_all_seeds(seed)

    train_val_df, test_df = get_seed_feature_tables(
        args=args, cfg=cfg, temp_dir=temp_dir, result_dir=result_dir,
        dataset=dataset, model_name=model_name, seed=seed,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        train_cases=train_cases, val_cases=val_cases, test_cases=test_cases,
        train_lengths=train_lengths, val_lengths=val_lengths, test_lengths=test_lengths,
        delayed_dict=delayed_dict,
        device=device
    )

    if ml_name == "CAT":
        out = fit_catboost_classifier_with_cv(
            train_val_df=train_val_df,
            test_df=test_df,
            random_seed=seed,
            param_grid={
                "depth": [4, 6],
                "learning_rate": [0.03, 0.05],
                "iterations": [200, 300],
                "l2_leaf_reg": [3.0, 5.0],
            },
            n_splits=5
        )
    else:
        raise ValueError(f"Unsupported ml_name: {ml_name}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        inf_path = get_ml_result_name(
            result_dir=save_dir,
            dataset=dataset,
            model_name=model_name,
            ml_name=ml_name,
            seed=seed,
            quantile=quantile
        )
        out["inference_df"].to_csv(inf_path, index=False)
        cv_path = get_ml_cv_result_name(
            result_dir=save_dir,
            dataset=dataset,
            model_name=model_name,
            ml_name=ml_name,
            seed=seed,
            quantile=quantile
            )
        out["cv_results_df"].to_csv(cv_path, index=False)

        if save_feature_tables:
            train_val_path = get_ml_feature_name(
                result_dir=save_dir,
                dataset=dataset,
                model_name=model_name,
                ml_name=ml_name,
                seed=seed,
                split="trainval",
                quantile=quantile
            )
            test_path = get_ml_feature_name(
                result_dir=save_dir,
                dataset=dataset,
                model_name=model_name,
                ml_name=ml_name,
                seed=seed,
                split="test",
                quantile=quantile
            )
            train_val_path = os.path.join(
                save_dir,
                dataset + '_' + model_name + '_classification_' +
                ml_name + '_seed_' + str(seed) + '_trainval_features.csv'
            )
            test_path = os.path.join(
                save_dir,
                dataset + '_' + model_name + '_classification_' +
                ml_name + '_seed_' + str(seed) + '_test_features.csv'
            )
            train_val_df.to_csv(train_val_path, index=False)
            test_df.to_csv(test_path, index=False)

    return {
        "seed": seed,
        "train_val_df": train_val_df,
        "test_df": test_df,
        "metrics": out["metrics"],
        "inference_df": out["inference_df"],
        "model": out["model"],
        "feature_cols": out["feature_cols"],
        "best_params": out["best_params"],
        "cv_results_df": out["cv_results_df"],
    }

def run_multi_seed_ml_classifier(
        args, cfg, temp_dir, result_dir, dataset, model_name, seeds,
        X_train, X_val, X_test, y_train, y_val, y_test,
        train_cases, val_cases, test_cases,
        train_lengths, val_lengths, test_lengths,
        delayed_dict,
        ml_name="CAT",
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir=None,
        save_feature_tables=False,
        quantile=None):

    all_seed_outputs = []
    summary_rows = []

    for seed in seeds:
        print(f"\n{'='*20} {ml_name} Seed {seed} {'='*20}")

        out = run_single_seed_ml_classifier(
            args=args, cfg=cfg, temp_dir=temp_dir, result_dir=result_dir,
            dataset=dataset, model_name=model_name, seed=seed,
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            train_cases=train_cases, val_cases=val_cases, test_cases=test_cases,
            train_lengths=train_lengths, val_lengths=val_lengths, test_lengths=test_lengths,
            delayed_dict=delayed_dict,
            ml_name=ml_name,
            device=device,
            save_dir=save_dir,
            save_feature_tables=save_feature_tables,
            quantile=quantile
        )

        all_seed_outputs.append(out)

        row = {
            "seed": seed,
            **out["metrics"],
            **{f"best_{k}": v for k, v in out["best_params"].items()}
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        summary_path = get_ml_summary_name(
            result_dir=save_dir,
            dataset=dataset,
            model_name=model_name,
            ml_name=ml_name,
            quantile=quantile
            )
        summary_df.to_csv(summary_path, index=False)

    metric_cols = [c for c in summary_df.columns if c != "seed" and not c.startswith("best_")]
    summary_metrics = {
        c: {
            "mean": summary_df[c].mean(),
            "std": summary_df[c].std()
        }
        for c in metric_cols
    }

    return {
        "per_seed_results": all_seed_outputs,
        "summary_df": summary_df,
        "summary_metrics": summary_metrics,
    }

def main():
    parser = argparse.ArgumentParser(
        description='Delay detection: classification vs. regression')
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--dataset', type=str, default='P2P')
    parser.add_argument('--quantile', type=float, default=0.8,
                    help='Quantile threshold for defining delays')
    args = parser.parse_args()
    root_path = os.getcwd()
    cfg_file = args.cfg if args.cfg is not None else args.dataset + '.yaml' 
    with open(os.path.join(root_path, 'cfg', cfg_file) , 'r') as f:
        cfg = yaml.safe_load(f)
    seeds = [409, 1824, 3657, 4012, 4506]
    model_name = 'DALSTM'
    result_dir = os.path.join(root_path, "results", model_name, args.dataset)    
    temp_dir = os.path.join(root_path, "temp", model_name, args.dataset)
    (train_cases, val_cases, test_cases) = load_case_ids(
        temp_dir, args.dataset, model_name)
    (train_lengths, val_lengths, test_lengths) = load_case_lengths(
        temp_dir, args.dataset, model_name)
    (y_train, y_val, y_test, X_train, X_val, X_test) = get_dataset(
        temp_dir, model_name, args.dataset)
    tau, delayed_dict = get_delayed_cases(train_cases, val_cases, test_cases,
                          y_train, y_val, y_test, args.quantile)
    y_train_cls = torch.tensor([delayed_dict[c] for c in train_cases], dtype=torch.float32)
    y_val_cls   = torch.tensor([delayed_dict[c] for c in val_cases], dtype=torch.float32)
    y_test_cls  = torch.tensor([delayed_dict[c] for c in test_cases], dtype=torch.float32)
    multi_seed_out = run_multi_seed_classification(
        cfg=cfg, seeds=seeds, X_train=X_train, X_val=X_val, X_test=X_test,
        y_train_cls=y_train_cls, y_val_cls=y_val_cls, y_test_cls=y_test_cls,
        test_cases=test_cases, test_prefix_lengths=test_lengths,
        get_loaders=get_loaders, get_train_params=get_train_params,
        get_model_config=get_model_config, get_opt_schedule=get_opt_schedule,
        DALSTMClassifier=DALSTMClassifier, 
        train_dalstm_classifier=train_dalstm_classifier,
        dataset=args.dataset, model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir=result_dir, quantile=args.quantile,)
    summary_df = multi_seed_out["summary_df"]
    summary_metrics = multi_seed_out["summary_metrics"]
    all_inference_df = multi_seed_out["all_inference_df"]
    print(summary_df)
    print(summary_metrics)
    print(all_inference_df.head()) 
    ml_name = "CAT"
    multi_seed_ml_out = run_multi_seed_ml_classifier(
        args=args, cfg=cfg, temp_dir=temp_dir, result_dir=result_dir,
        dataset=args.dataset, model_name=model_name, seeds=seeds,
        X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train,
        y_val=y_val, y_test=y_test, train_cases=train_cases,
        val_cases=val_cases, test_cases=test_cases, train_lengths=train_lengths,
        val_lengths=val_lengths, test_lengths=test_lengths, 
        delayed_dict=delayed_dict, ml_name=ml_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir=result_dir, save_feature_tables=True,
        quantile=args.quantile,)
    summary_df = multi_seed_ml_out["summary_df"]
    summary_metrics = multi_seed_ml_out["summary_metrics"]
    print(summary_df)
    print(summary_metrics)

if __name__ == "__main__":
    main()