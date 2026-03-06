# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 13:25:33 2026

@author: kamirel
"""
from __future__ import annotations
from typing import Optional, Tuple, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool

@dataclass
class MixtureResidualModel:
    low_model: CatBoostRegressor
    high_model: CatBoostRegressor
    gate_threshold: float
    low_alpha: float
    high_alpha: float
    feature_cols: List[str]
    cat_idx: List[int]
    w_col: str


def _detect_cat_features(x: pd.DataFrame, feature_cols: List[str]) -> List[int]:
    cat_cols = [
        c for c in feature_cols
        if pd.api.types.is_object_dtype(x[c]) or pd.api.types.is_categorical_dtype(x[c])
    ]
    return [feature_cols.index(c) for c in cat_cols]


def _split_by_case_late_validation(
    df: pd.DataFrame,
    case_col: str,
    val_case_fraction: float,
    *,
    order_by: str = "case_end_index",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (train_case_ids, val_case_ids) using a "late cases" strategy.

    order_by:
      - "case_end_index": uses the max row index per case as a proxy for time/order in df
      - "max_prefix": uses max Prefix_length per case (often correlates with case progress/length)
    """
    if not (0.0 < val_case_fraction < 1.0):
        raise ValueError("val_case_fraction must be in (0,1).")

    case_groups = df.groupby(case_col, sort=False)

    if order_by == "case_end_index":
        # stable: last occurrence index of each case in df
        case_stat = case_groups.apply(lambda g: g.index.max())
    elif order_by == "max_prefix":
        if "Prefix_length" not in df.columns:
            raise ValueError("Prefix_length missing but order_by='max_prefix' was requested.")
        case_stat = case_groups["Prefix_length"].max()
    else:
        raise ValueError("order_by must be one of: {'case_end_index', 'max_prefix'}")

    case_stat = case_stat.sort_values()  # "earlier" -> smaller, "later" -> larger
    case_ids_sorted = case_stat.index.to_numpy()

    n_val = max(1, int(len(case_ids_sorted) * val_case_fraction))
    val_case_ids = case_ids_sorted[-n_val:]
    train_case_ids = case_ids_sorted[:-n_val]
    return train_case_ids, val_case_ids


def _best_alpha_mae(y_true: np.ndarray, base_pred: np.ndarray, res_hat: np.ndarray) -> float:
    """
    Find alpha in [0, 1] (coarse grid) that minimizes MAE of: base_pred + alpha * res_hat
    """
    # coarse but robust; you can refine if you want
    grid = np.linspace(0.0, 1.0, 21)
    maes = []
    for a in grid:
        pred = base_pred + a * res_hat
        maes.append(np.mean(np.abs(pred - y_true)))
    return float(grid[int(np.argmin(maes))])


def catboost_residual_boosting_moe_mae(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    # columns
    case_col: str = "Case_id",
    w_col: str = "W10_90",
    # gating
    high_uncertainty_quantile: float = 0.80,   # top 20% go to high expert by default
    # validation split by cases
    use_eval_set: bool = True,
    val_case_fraction: float = 0.15,
    val_order_by: str = "case_end_index",      # or "max_prefix"
    # catboost params
    iterations: int = 5000,
    learning_rate: float = 0.03,
    depth: int = 8,
    l2_leaf_reg: float = 5.0,
    random_seed: int = 42,
    early_stopping_rounds: int = 200,
    verbose: int = 200,
    # misc
    allow_alpha_shrinkage: bool = True,
) -> Tuple[pd.DataFrame, MixtureResidualModel]:
    """
    Mixture-of-experts residual correction with:
      1) Case-level "late" validation split for early stopping
      2) MAE optimization (CatBoost loss = MAE)
      3) Two experts split by W10_90 threshold

    Returns:
      results_df with columns: Case_id, Prefix_length, GroundTruth, Prediction, Absolute_error
      + fitted MixtureResidualModel bundle
    """
    required_train = {case_col, "Prefix_length", "GroundTruth", "Prediction", "residual", w_col}
    required_test = {case_col, "Prefix_length", "GroundTruth", "Prediction", w_col}

    missing_train = required_train - set(train_df.columns)
    missing_test = required_test - set(test_df.columns)
    if missing_train:
        raise ValueError(f"train_df missing required columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"test_df missing required columns: {sorted(missing_test)}")

    # features: all except identifiers/labels
    exclude = {case_col, "GroundTruth", "Prediction", "residual"}
    feature_cols = [c for c in train_df.columns if c not in exclude]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding case_col/GroundTruth/Prediction/residual.")

    x_train_all = train_df[feature_cols].copy()
    y_train_all = train_df["residual"].astype(float).to_numpy()

    # cat features detection
    cat_idx = _detect_cat_features(x_train_all, feature_cols)

    # gating threshold learned from train distribution of W10_90
    gate_threshold = float(np.nanquantile(train_df[w_col].to_numpy(dtype=float), high_uncertainty_quantile))

    # helper to build pools
    def make_pool(df_subset: pd.DataFrame) -> Pool:
        x = df_subset[feature_cols]
        y = df_subset["residual"].astype(float) if "residual" in df_subset.columns else None
        return Pool(x, y, cat_features=cat_idx if cat_idx else None)

    # split by cases for eval set (optional)
    eval_case_ids = None
    if use_eval_set and 0.0 < val_case_fraction < 1.0 and train_df[case_col].nunique() >= 10:
        train_case_ids, val_case_ids = _split_by_case_late_validation(
            train_df, case_col, val_case_fraction, order_by=val_order_by
        )
        eval_case_ids = set(val_case_ids.tolist())
        train_case_ids = set(train_case_ids.tolist())

        tr_mask = train_df[case_col].isin(train_case_ids)
        va_mask = train_df[case_col].isin(eval_case_ids)

        train_part = train_df.loc[tr_mask].copy()
        eval_part = train_df.loc[va_mask].copy()
    else:
        train_part = train_df.copy()
        eval_part = None

    # split into regimes
    def split_regime(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        high_mask = df_in[w_col].astype(float) >= gate_threshold
        return df_in.loc[~high_mask].copy(), df_in.loc[high_mask].copy()

    tr_low, tr_high = split_regime(train_part)
    ev_low, ev_high = (None, None)
    if eval_part is not None:
        ev_low, ev_high = split_regime(eval_part)

    # build models (MAE)
    def fit_model(train_df_reg: pd.DataFrame, eval_df_reg: Optional[pd.DataFrame]) -> CatBoostRegressor:
        model = CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_seed,
            od_type="Iter",
            od_wait=early_stopping_rounds,
            verbose=verbose,
        )
        train_pool = make_pool(train_df_reg)
        eval_pool = make_pool(eval_df_reg) if eval_df_reg is not None and len(eval_df_reg) > 0 else None
        model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
        return model

    # Edge case: if one regime is tiny, fall back to single model duplicated
    min_regime_rows = 50
    if len(tr_low) < min_regime_rows or len(tr_high) < min_regime_rows:
        # train single model on all train_part
        single = fit_model(train_part, eval_part)
        low_model = single
        high_model = single
    else:
        low_model = fit_model(tr_low, ev_low if eval_part is not None else None)
        high_model = fit_model(tr_high, ev_high if eval_part is not None else None)

    # learn per-regime alpha shrinkage on eval set (or on a slice of train if no eval)
    low_alpha = 1.0
    high_alpha = 1.0

    if allow_alpha_shrinkage:
        calib_df = eval_part if eval_part is not None and len(eval_part) > 0 else train_part

        cal_low, cal_high = split_regime(calib_df)

        def alpha_for(df_reg: pd.DataFrame, model: CatBoostRegressor) -> float:
            if df_reg is None or len(df_reg) == 0:
                return 1.0
            x = df_reg[feature_cols]
            pool = Pool(x, cat_features=cat_idx if cat_idx else None)
            res_hat = model.predict(pool)
            y_true = df_reg["GroundTruth"].astype(float).to_numpy()
            base_pred = df_reg["Prediction"].astype(float).to_numpy()
            return _best_alpha_mae(y_true, base_pred, np.asarray(res_hat, dtype=float))

        low_alpha = alpha_for(cal_low, low_model)
        high_alpha = alpha_for(cal_high, high_model)

    # --- Predict on test with routing
    x_test = test_df.reindex(columns=feature_cols).copy()
    test_pool = Pool(x_test, cat_features=cat_idx if cat_idx else None)

    # We'll predict in two batches for efficiency
    high_mask_test = test_df[w_col].astype(float).to_numpy() >= gate_threshold

    residual_hat = np.zeros(len(test_df), dtype=float)

    if np.any(~high_mask_test):
        idx = np.where(~high_mask_test)[0]
        pool = Pool(x_test.iloc[idx], cat_features=cat_idx if cat_idx else None)
        residual_hat[idx] = low_model.predict(pool) * low_alpha

    if np.any(high_mask_test):
        idx = np.where(high_mask_test)[0]
        pool = Pool(x_test.iloc[idx], cat_features=cat_idx if cat_idx else None)
        residual_hat[idx] = high_model.predict(pool) * high_alpha

    updated_pred = test_df["Prediction"].astype(float).to_numpy() + residual_hat
    abs_err = np.abs(updated_pred - test_df["GroundTruth"].astype(float).to_numpy())

    results_df = pd.DataFrame({
        case_col: test_df[case_col].values,
        "Prefix_length": test_df["Prefix_length"].values,
        "GroundTruth": test_df["GroundTruth"].astype(float).values,
        "Prediction": updated_pred,
        "Absolute_error": abs_err,
    }).rename(columns={case_col: "Case_id"})  # output schema requested

    bundle = MixtureResidualModel(
        low_model=low_model,
        high_model=high_model,
        gate_threshold=gate_threshold,
        low_alpha=low_alpha,
        high_alpha=high_alpha,
        feature_cols=feature_cols,
        cat_idx=cat_idx,
        w_col=w_col,
    )

    return results_df, bundle