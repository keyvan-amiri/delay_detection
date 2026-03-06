# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 12:54:27 2026

@author: kamirel
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool


def catboost_residual_boosting(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    huber_delta: float = 1.0,
    iterations: int = 2000,
    learning_rate: float = 0.05,
    depth: int = 8,
    l2_leaf_reg: float = 3.0,
    random_seed: int = 42,
    early_stopping_rounds: int = 100,
    use_eval_set: bool = True,
    eval_fraction: float = 0.15,
    verbose: int = 200,
) -> Tuple[pd.DataFrame, CatBoostRegressor]:
    """
    Train a CatBoost residual model (Huber loss) on train_df['residual'] and apply it to test_df.
    Updates test_df['Prediction'] by adding predicted residual, then returns inference results.

    Expected columns:
      - train_df: 'Case_id', 'Prefix_length', 'GroundTruth', 'Prediction', 'residual', + feature columns
      - test_df:  'Case_id', 'Prefix_length', 'GroundTruth', 'Prediction', + feature columns

    Feature columns:
      All columns except: {'Case_id', 'GroundTruth', 'Prediction', 'residual'}.

    Returns:
      (results_df, fitted_model)
      results_df columns: Case_id, Prefix_length, GroundTruth, Prediction, Absolute_error
    """
    required_train = {"Case_id", "Prefix_length", "GroundTruth", "Prediction", "residual"}
    required_test = {"Case_id", "Prefix_length", "GroundTruth", "Prediction"}

    missing_train = required_train - set(train_df.columns)
    missing_test = required_test - set(test_df.columns)
    if missing_train:
        raise ValueError(f"train_df missing required columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"test_df missing required columns: {sorted(missing_test)}")

    # --- Select feature columns (allow Prefix_length etc.; exclude identifiers/labels)
    exclude = {"Case_id", "GroundTruth", "Prediction", "residual"}
    feature_cols = [c for c in train_df.columns if c not in exclude]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding Case_id/GroundTruth/Prediction/residual.")

    # Ensure test has the same feature columns (fill missing with NaN; drop extras)
    x_train = train_df[feature_cols].copy()
    y_train = train_df["residual"].astype(float).copy()

    x_test = test_df.reindex(columns=feature_cols).copy()

    # --- Detect categorical columns (future-proofing)
    # CatBoost accepts indices of categorical features
    cat_cols = [
        c for c in feature_cols
        if pd.api.types.is_object_dtype(x_train[c]) or pd.api.types.is_categorical_dtype(x_train[c])
    ]
    cat_idx = [feature_cols.index(c) for c in cat_cols]

    # --- Optional train/valid split for early stopping
    train_pool: Pool
    eval_pool: Optional[Pool] = None

    if use_eval_set and 0.0 < eval_fraction < 1.0 and len(train_df) >= 50:
        # Simple random split (you can swap this with GroupKFold on Case_id if desired)
        rng = np.random.default_rng(random_seed)
        idx = np.arange(len(train_df))
        rng.shuffle(idx)
        n_eval = max(1, int(len(idx) * eval_fraction))
        eval_idx = idx[:n_eval]
        tr_idx = idx[n_eval:]

        train_pool = Pool(
            x_train.iloc[tr_idx],
            y_train.iloc[tr_idx],
            cat_features=cat_idx if cat_idx else None,
        )
        eval_pool = Pool(
            x_train.iloc[eval_idx],
            y_train.iloc[eval_idx],
            cat_features=cat_idx if cat_idx else None,
        )
    else:
        train_pool = Pool(x_train, y_train, cat_features=cat_idx if cat_idx else None)

    # --- Model
    model = CatBoostRegressor(
        loss_function=f"Huber:delta={huber_delta}",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_seed,
        od_type="Iter",
        od_wait=early_stopping_rounds,
        verbose=verbose,
    )

    model.fit(train_pool, eval_set=eval_pool if eval_pool is not None else None, use_best_model=True)

    # --- Predict residuals on test, update predictions
    test_pool = Pool(x_test, cat_features=cat_idx if cat_idx else None)
    residual_hat = model.predict(test_pool)

    updated_pred = test_df["Prediction"].astype(float).to_numpy() + np.asarray(residual_hat, dtype=float)
    abs_err = np.abs(updated_pred - test_df["GroundTruth"].astype(float).to_numpy())

    results_df = pd.DataFrame({
        "Case_id": test_df["Case_id"].values,
        "Prefix_length": test_df["Prefix_length"].values,
        "GroundTruth": test_df["GroundTruth"].astype(float).values,
        "Prediction": updated_pred,
        "Absolute_error": abs_err,
    })

    return results_df, model