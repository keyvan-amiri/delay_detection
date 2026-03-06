# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 15:06:07 2026

@author: kamirel
"""
from __future__ import annotations
from typing import Tuple, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

# =========================
# 5) REGION CLASSIFIER (0/1/2) + COST WEIGHTS
# =========================
@dataclass
class OrdinalRegionBundle:
    clf60: CatBoostClassifier
    clf90: CatBoostClassifier
    feature_cols: List[str]
    cat_idx: List[int]
    thresholds: Tuple[float, float]  # (q60, q90)


def train_region_classifier_ordinal_catboost(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    case_col: str = "Case_id",
    region_target_col: str = "GroundTruth",
    q1: float = 0.60,
    q2: float = 0.90,
    use_eval_set: bool = True,
    val_case_fraction: float = 0.15,
    val_order_by: str = "case_end_index",
    iterations: int = 3000,
    learning_rate: float = 0.05,
    depth: int = 8,
    l2_leaf_reg: float = 5.0,
    random_seed: int = 42,
    early_stopping_rounds: int = 200,
    verbose: int = 200,
) -> Tuple[pd.DataFrame, OrdinalRegionBundle]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    y = train_df[region_target_col].astype(float).to_numpy()
    t1 = float(np.quantile(y, q1))
    t2 = float(np.quantile(y, q2))

    # labels for the two thresholds
    train_df["gt_gt_q60"] = (train_df[region_target_col].astype(float) > t1).astype(int)
    train_df["gt_gt_q90"] = (train_df[region_target_col].astype(float) > t2).astype(int)

    exclude = {case_col, region_target_col, "residual", "log_residual", "gt_gt_q60", "gt_gt_q90"}
    feature_cols = [c for c in train_df.columns if c not in exclude]
    if not feature_cols:
        raise ValueError("No feature columns available for ordinal region classifier.")

    cat_idx = _detect_cat_features(train_df[feature_cols], feature_cols)

    # case-level eval split
    if use_eval_set and train_df[case_col].nunique() >= 10:
        tr_cases, va_cases = _split_by_case_late_validation(train_df, case_col, val_case_fraction, order_by=val_order_by)
        tr_mask = train_df[case_col].isin(set(tr_cases.tolist()))
        va_mask = train_df[case_col].isin(set(va_cases.tolist()))
        tr = train_df.loc[tr_mask].copy()
        va = train_df.loc[va_mask].copy()
    else:
        tr, va = train_df, None

    def fit_binary(label_col: str, pos_weight: float) -> CatBoostClassifier:
        # pos_weight emphasizes catching the tail threshold
        clf = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_seed,
            od_type="Iter",
            od_wait=early_stopping_rounds,
            verbose=verbose,
        )

        def pool(df_subset: pd.DataFrame) -> Pool:
            X = df_subset[feature_cols]
            yb = df_subset[label_col].astype(int)
            w = np.where(yb.to_numpy() == 1, pos_weight, 1.0)
            return Pool(X, yb, weight=w, cat_features=cat_idx if cat_idx else None)

        clf.fit(pool(tr), eval_set=pool(va) if va is not None else None, use_best_model=True)
        return clf

    # heavier weight for q90 (rarer, more important)
    clf60 = fit_binary("gt_gt_q60", pos_weight=2.0)
    clf90 = fit_binary("gt_gt_q90", pos_weight=6.0)

    # predict on test
    test_pool = Pool(test_df[feature_cols], cat_features=cat_idx if cat_idx else None)
    p60 = clf60.predict_proba(test_pool)[:, 1]
    p90 = clf90.predict_proba(test_pool)[:, 1]

    # infer ordinal region
    # You can tune these cutoffs; start with 0.5
    region_pred = np.zeros(len(test_df), dtype=int)
    region_pred[p60 >= 0.5] = 1
    region_pred[p90 >= 0.5] = 2

    out = pd.DataFrame({
        "Case_id": test_df[case_col].values,
        "Prefix_length": test_df["Prefix_length"].values,
        "region_pred": region_pred,
        "p_gt_q60": p60,
        "p_gt_q90": p90,
    })

    bundle = OrdinalRegionBundle(
        clf60=clf60,
        clf90=clf90,
        feature_cols=feature_cols,
        cat_idx=cat_idx,
        thresholds=(t1, t2),
    )
    return out, bundle