# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 07:52:18 2025
@author: Keyvan Amiri Elyasi
"""
import os
import logging
import argparse
import pandas as pd
import numpy as np

def get_second_logger( result_dir):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_par = logging.getLogger('Second_model_Logger')
    logger_par.setLevel(logging.INFO)
    # Clear previous handlers
    if logger_par.hasHandlers():
        logger_par.handlers.clear()
    logger_par_name = 'tabular_report.log'
    logger_par_path = os.path.join(result_dir, logger_par_name)
    file_handler_par = logging.FileHandler(logger_par_path)
    file_handler_par.setLevel(logging.INFO)
    file_handler_par.setFormatter(formatter)
    logger_par.addHandler(file_handler_par)
    return logger_par

def add_shots_quantile(
    df_inp,
    trg_col="GroundTruth",
    many_frac=0.6,
    med_frac=0.3,
    tail="high",   # "high", "low", or "both"
):
    """
    Quantile-based grouping for regression targets.

    Groups samples into many / med / few using quantiles.

    Parameters
    ----------
    df_inp : pandas DataFrame
    trg_col : str
        Target column name
    many_frac : float
        Fraction assigned to many-shot region
    med_frac : float
        Fraction assigned to medium-shot region
    tail : str
        Tail definition:
            "high"  -> few = highest targets (default)
            "low"   -> few = lowest targets
            "both"  -> few = both extreme tails (symmetric)
    Returns
    -------
    DataFrame with added columns: many, med, few
    """
    if many_frac <= 0 or med_frac < 0 or many_frac + med_frac >= 1:
        raise ValueError("Require: many_frac > 0, med_frac >= 0, and many_frac + med_frac < 1.")
    if tail not in {"high", "low", "both"}:
        raise ValueError("tail must be one of: 'high', 'low', 'both'")
    df = df_inp.copy()
    y = df[trg_col]
    few_frac = 1.0 - (many_frac + med_frac)
    df[["many", "med", "few"]] = 0
    # ---------- ONE-SIDED: HIGH ----------
    if tail == "high":
        q_many = y.quantile(many_frac)
        q_med  = y.quantile(many_frac + med_frac)
        df.loc[y <= q_many, "many"] = 1
        df.loc[(y > q_many) & (y <= q_med), "med"] = 1
        df.loc[y > q_med, "few"] = 1
    # ---------- ONE-SIDED: LOW ----------
    elif tail == "low":
        q_few = y.quantile(few_frac)
        q_med = y.quantile(few_frac + med_frac)
        df.loc[y <= q_few, "few"] = 1
        df.loc[(y > q_few) & (y <= q_med), "med"] = 1
        df.loc[y > q_med, "many"] = 1
    # ---------- TWO-SIDED ----------
    else:  # both
        q1 = few_frac / 2
        q2 = q1 + med_frac / 2
        q3 = 1 - q2
        q4 = 1 - q1
        b1, b2, b3, b4 = y.quantile([q1, q2, q3, q4]).to_list()
        few_mask  = (y <= b1) | (y >= b4)
        many_mask = (y > b2) & (y < b3)
        med_mask  = ~(few_mask | many_mask)
        df.loc[few_mask, "few"] = 1
        df.loc[many_mask, "many"] = 1
        df.loc[med_mask, "med"] = 1
    return df

def train_eval_xgb_catboost(
        df: pd.DataFrame,
        id_cols=("Case_id", "Prefix_length"),
        target_col="GroundTruth",
        test_frac=0.2,
        feature_cols=None,
        xgb_params=None,
        cat_params=None,
        random_state=42,
        ):
    """
    Uses first (1-test_frac) rows for training and last test_frac rows for testing (by current row order).
    Trains XGBoost + CatBoost regressors and returns two result DataFrames with identical columns:
      [Case_id, Prefix_length, GroundTruth, Prediction, Absolute_error]
    """
    # ---- basic checks ----
    missing = [c for c in list(id_cols) + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    # ---- feature selection ----
    if feature_cols is None:
        # Default: all numeric cols except target, excluding Case_id; keep Prefix_length as a feature.
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col and c != "Case_id"]
    if not feature_cols:
        raise ValueError("No feature columns selected.")
    # ---- split by row order (first 80% train, last 20% test) ----
    n = len(df)
    if n < 2:
        raise ValueError("DataFrame must have at least 2 rows.")
    split_idx = int(np.floor(n * (1 - test_frac)))
    if split_idx <= 0 or split_idx >= n:
        raise ValueError("Split results in empty train or test set; adjust test_frac or provide more rows.")
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test  = test_df[feature_cols]
    y_test  = test_df[target_col]
    # ---- models ----
    # XGBoost
    try:
        from xgboost import XGBRegressor
    except ImportError as e:
        raise ImportError("xgboost is not installed. Install it via `pip install xgboost`.") from e
    default_xgb_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    if xgb_params:
        default_xgb_params.update(xgb_params)
    xgb = XGBRegressor(**default_xgb_params)
    xgb.fit(X_train, y_train)
    # CatBoost
    try:
        from catboost import CatBoostRegressor
    except ImportError as e:
        raise ImportError("catboost is not installed. Install it via `pip install catboost`.") from e
    default_cat_params = dict(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=False,
    )
    if cat_params:
        default_cat_params.update(cat_params)
    cat = CatBoostRegressor(**default_cat_params)
    cat.fit(X_train, y_train)
    # ---- predictions + outputs ----
    pred_xgb = np.asarray(xgb.predict(X_test), dtype=float)
    pred_cat = np.asarray(cat.predict(X_test), dtype=float)
    def make_out(pred):
        out = test_df.loc[:, list(id_cols) + [target_col]].copy()
        out["Prediction"] = pred
        out["Absolute_error"] = (out[target_col] - out["Prediction"]).abs()
        # Ensure exact column order requested
        return out.loc[:, [id_cols[0], id_cols[1], target_col, "Prediction", "Absolute_error"]]
    return make_out(pred_xgb), make_out(pred_cat)

def add_many_med_few(
        df: pd.DataFrame,
        df_all: pd.DataFrame,
        id_cols=("Case_id", "Prefix_length"),
        cols_to_add=("many", "med", "few"),
        validate=True,
        ):
    """
    Adds specified columns from df_all to df based on matching id_cols.

    Assumes:
        - Every row in df has exactly one counterpart in df_all.
        - Matching is done on id_cols.
    """
    # --- checks ---
    for c in list(id_cols) + list(cols_to_add):
        if c not in df_all.columns:
            raise ValueError(f"{c} not found in df_all")
    for c in id_cols:
        if c not in df.columns:
            raise ValueError(f"{c} not found in df")
    # --- select only needed columns from df_all ---
    df_all_subset = df_all[list(id_cols) + list(cols_to_add)].copy()
    # --- merge ---
    merged = df.merge(
        df_all_subset,
        on=list(id_cols),
        how="left",
        validate="many_to_one" if validate else None,
    )
    if validate:
        if merged[list(cols_to_add)].isnull().any().any():
            raise ValueError("Some rows in df did not find a match in df_all.")
    return merged

def main():
    parser = argparse.ArgumentParser(
        description='Imbalanced Regression for Remaining Time Prediction')
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    args.model = 'DALSTM'
    root_path = os.getcwd()
    result_dir = os.path.join(root_path, 'results', args.model, args.dataset)
    model_dir = os.path.join(root_path, 'temp', args.model, args.dataset)    
    logger = get_second_logger(result_dir) 
    logger.info("Now Start training Tabular models")
    seeds = [4012, 4506, 409, 1824, 3657]
    xgb_mae_list, xgb_mae_many_list, xgb_mae_med_list, xgb_mae_few_list = [], [], [], []
    cat_mae_list, cat_mae_many_list, cat_mae_med_list, cat_mae_few_list = [], [], [], []
    for seed in seeds:
        quantile_name = args.dataset+'_'+args.model+'_quantile_wos_seed'+str(seed)+'_inference.csv'  
        model_name = args.dataset+'_'+args.model+'_quantile_wos_seed'+str(seed)+'.pt'
        quantile_df = pd.read_csv(os.path.join(result_dir, quantile_name))
        
        xgp_df, cat_df = train_eval_xgb_catboost(quantile_df)
        df_all = add_shots_quantile(quantile_df)
        xgp_df = add_many_med_few(xgp_df, df_all)
        cat_df = add_many_med_few(cat_df, df_all)
        df_many = xgp_df[xgp_df["many"] == 1]
        df_med  = xgp_df[xgp_df["med"] == 1]
        df_few  = xgp_df[xgp_df["few"] == 1]
        xgb_mae_list.append(xgp_df["Absolute_error"].mean())
        xgb_mae_many_list.append(df_many["Absolute_error"].mean())
        xgb_mae_med_list.append(df_med["Absolute_error"].mean())
        xgb_mae_few_list.append(df_few["Absolute_error"].mean())
        df_many = cat_df[cat_df["many"] == 1]
        df_med  = cat_df[cat_df["med"] == 1]
        df_few  = cat_df[cat_df["few"] == 1]
        cat_mae_list.append(cat_df["Absolute_error"].mean())
        cat_mae_many_list.append(df_many["Absolute_error"].mean())
        cat_mae_med_list.append(df_med["Absolute_error"].mean())
        cat_mae_few_list.append(df_few["Absolute_error"].mean())
    MAE_mean, MAE_std = np.mean(xgb_mae_list), np.std(xgb_mae_list)
    MAE_many_mean, MAE_many_std = np.mean(xgb_mae_many_list), np.std(xgb_mae_many_list)
    MAE_med_mean,  MAE_med_std  = np.mean(xgb_mae_med_list),  np.std(xgb_mae_med_list)
    MAE_few_mean,  MAE_few_std  = np.mean(xgb_mae_few_list),  np.std(xgb_mae_few_list)
    print(f"MAE (XGB): Average: {MAE_mean:.4f} std: {MAE_std:.4f}")
    logger.info(f"MAE (XGB): Average: {MAE_mean:.4f} std: {MAE_std:.4f}")
    print(f"MAE_many (XGB): Average: {MAE_many_mean:.4f} std: {MAE_many_std:.4f}")
    logger.info(f"MAE_many (XGB):  Average: {MAE_many_mean:.4f} std: {MAE_many_std:.4f}")
    print(f"MAE_med (XGB): Average: {MAE_med_mean:.4f} std: {MAE_med_std:.4f}")
    logger.info(f"MAE_med (XGB): Average: {MAE_med_mean:.4f} std: {MAE_med_std:.4f}")
    print(f"MAE_few (XGB): Average: {MAE_few_mean:.4f} std: {MAE_few_std:.4f}")
    logger.info(f"MAE_few (XGB): Average: {MAE_few_mean:.4f} std: {MAE_few_std:.4f}")
    MAE_mean, MAE_std = np.mean(cat_mae_list), np.std(cat_mae_list)
    MAE_many_mean, MAE_many_std = np.mean(cat_mae_many_list), np.std(cat_mae_many_list)
    MAE_med_mean,  MAE_med_std  = np.mean(cat_mae_med_list),  np.std(cat_mae_med_list)
    MAE_few_mean,  MAE_few_std  = np.mean(cat_mae_few_list),  np.std(cat_mae_few_list)
    print(f"MAE (CAT): Average: {MAE_mean:.4f} std: {MAE_std:.4f}")
    logger.info(f"MAE (CAT): Average: {MAE_mean:.4f} std: {MAE_std:.4f}")
    print(f"MAE_many (CAT): Average: {MAE_many_mean:.4f} std: {MAE_many_std:.4f}")
    logger.info(f"MAE_many (CAT): Average: {MAE_many_mean:.4f} std: {MAE_many_std:.4f}")
    print(f"MAE_med (CAT): Average: {MAE_med_mean:.4f} std: {MAE_med_std:.4f}")
    logger.info(f"MAE_med (CAT): Average: {MAE_med_mean:.4f} std: {MAE_med_std:.4f}")
    print(f"MAE_few (CAT): Average: {MAE_few_mean:.4f} std: {MAE_few_std:.4f}")
    logger.info(f"MAE_few (CAT): Average: {MAE_few_mean:.4f} std: {MAE_few_std:.4f}")
        

if __name__ == '__main__':
    main() 