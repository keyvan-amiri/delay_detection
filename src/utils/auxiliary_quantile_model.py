# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 16:28:13 2026
"""
from __future__ import annotations
import os
import pickle
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype, is_bool_dtype
import numpy as np
import torch
from catboost import CatBoostRegressor, Pool

from src.LSTM.Train_DALSTM import quantile_inference
from src.LSTM.model_DALSTM import DALSTMQuantileModel

def get_quantile_model(args, cfg, device=None):
    with open(args.input_size_path, 'rb') as f:
        input_size = pickle.load(f)
    n_layers = cfg['DALSTM']['n_layers'] or 2
    hidden_size = cfg['DALSTM']['hidden_size'] or 150
    dropout = cfg['DALSTM']['dropout']
    if dropout is None:
        dropout = True
    dropout_prob = cfg['DALSTM']['dropout_prob'] or 0.1
    model = DALSTMQuantileModel(
            input_size=input_size, hidden_size=hidden_size, n_layers=n_layers,
            dropout=dropout, p_fix=dropout_prob,
            quantiles=(0.1, 0.5, 0.6, 0.9, 0.95, 0.99)).to(device)        
    return model

def get_quantile_structure(mode='train', quantiles=(0.1, 0.5, 0.6, 0.9, 0.95, 0.99)):
    all_results = {'Case_id': [], 'Prefix_length': [],'GroundTruth': [], 
                   'Prediction': [], 'Absolute_error': []}   
    if mode=='train':
        for q in quantiles:
            key = f"Q{str(q).replace('.', '_')}"  # e.g. Q0_95
            all_results[key] = []
        all_results["PI10"] = []
        all_results["PI90"] = []
        all_results["PI_Width_10_90"] = []
        all_results["PI_Coverage_10_90"] = []
    return all_results

def get_training_dataframe(
        args, cfg, quantile_loader=None, quantile_lengths=None, quantile_cases=None, 
        quantiles=None, seed=None, device=None):
    # load pre-trained quantile regression model
    model = get_quantile_model(args, cfg, device)
    checkpoint_name = args.model_name+'seed'+str(seed)+'.pt'
    checkpoint_path = os.path.join(args.process_path, checkpoint_name)
    try:
        checkpoint = torch.load(checkpoint_path)
    except:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    all_results = get_quantile_structure()
    all_results = quantile_inference(
        args, model, all_results, quantile_loader, quantile_lengths,
        quantile_cases, quantiles=quantiles, device=device)
    flattened_list = [item for sublist in all_results['Prefix_length'] 
                      for item in sublist]
    all_results['Prefix_length'] = flattened_list  
    all_results['Case_id'] = [item for sublist in all_results['Case_id'] for item in sublist]
    results_df = pd.DataFrame(all_results) 
    cols = ['Case_id', 'Prefix_length'] + [c for c in results_df.columns if c not in ['Case_id', 'Prefix_length']]
    results_df = results_df[cols]
    if args.log_trans:
        res_name = args.model_name+'logtrans_seed'+str(seed)+'_quantile_train_val.csv'
    else:
        res_name = args.model_name+'seed'+str(seed)+'_quantile_train_val.csv'        
    res_path = os.path.join(args.result_path, res_name)
    results_df.to_csv(res_path, index=False)
    return results_df

def get_test_dataframe(args, seed=None):
    quantile_name = args.dataset+'_'+args.model+'_quantile_wos_seed'+str(seed)+'_inference.csv'        
    quantile_df = pd.read_csv(os.path.join(args.result_path, quantile_name)) 
    return quantile_df

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

# =========================
# 1) FEATURE ENGINEERING
# =========================
def extract_features(
        df: pd.DataFrame,
        *,
        train_flag: bool = True,
        eps: float = 1e-6,
        # history features
        add_history: bool = True,
        roll_k: int = 3,
        # which point prediction is the "base" (median by default)
        base_point_col: str = "Q0_5",
        # PI width column name in input (will map -> W10_90)
        width_col_in: str = "PI_Width_10_90",
        ) -> pd.DataFrame:
    """
    Adds distribution-shape + uncertainty features + (optionally) history features
    that are leakage-safe (only use smaller Prefix_length within the same Case_id).
    Requirements in df for current features:
      Q0_1,Q0_5,Q0_6,Q0_9,Q0_95,Q0_99, Case_id, Prefix_length
    """
    df = df.copy()
    # --- core distribution features
    df["local_slope"] = df["Q0_6"] - df["Q0_5"]
    df["lower_spread"] = df["Q0_5"] - df["Q0_1"]
    df["upper_spread"] = df["Q0_9"] - df["Q0_5"]
    df["Skew_10_90"] = df["Q0_9"] + df["Q0_1"] - 2 * df["Q0_5"]
    df["Skew_ratio"] = (df["Q0_9"] - df["Q0_5"]) / (df["Q0_5"] - df["Q0_1"] + eps)
    df["Tail_ratio"] = (df["Q0_99"] - df["Q0_95"]) / (df["Q0_9"] - df["Q0_5"] + eps)
    df["Upper_tail_share"] = (df["Q0_99"] - df["Q0_5"]) / (df["Q0_9"] - df["Q0_1"] + eps)
    df["Median_position"] = (df["Q0_5"] - df["Q0_1"]) / (df["Q0_9"] - df["Q0_1"] + eps)
    # --- width features
    if width_col_in in df.columns and "W10_90" not in df.columns:
        df.rename(columns={width_col_in: "W10_90"}, inplace=True)
    if "W10_90" not in df.columns:
        df["W10_90"] = df["Q0_9"] - df["Q0_1"]
    df["W50_95"] = df["Q0_95"] - df["Q0_5"]
    df["W50_90"] = df["Q0_9"] - df["Q0_5"]
    df["W10_50"] = df["Q0_5"] - df["Q0_1"]
    df["W95_99"] = df["Q0_99"] - df["Q0_95"]
    df["W90_99"] = df["Q0_99"] - df["Q0_9"]
    df["RelW10_90"] = (df["Q0_9"] - df["Q0_1"]) / (df["Q0_5"].abs() + eps)
    df["RelTail99"] = (df["Q0_99"] - df["Q0_95"]) / (df["Q0_95"] - df["Q0_9"] + eps)
    # --- optional history features: safe online (only smaller Prefix_length)
    if add_history:
        # sort so shift/rolling only uses earlier prefixes
        df.sort_values(["Case_id", "Prefix_length"], inplace=True)
        # we use base_point_col as the point forecast time-series per case
        if base_point_col not in df.columns:
            raise ValueError(f"base_point_col='{base_point_col}' not found in df.")
        # previous-step deltas
        df["prev_" + base_point_col] = df.groupby("Case_id")[base_point_col].shift(1)
        df["prev_W10_90"] = df.groupby("Case_id")["W10_90"].shift(1)
        df["Delta_" + base_point_col] = df[base_point_col] - df["prev_" + base_point_col]
        df["Delta_W10_90"] = df["W10_90"] - df["prev_W10_90"]
        # rolling stats of deltas / volatility (use shift(1) so current step not included)
        # rolling mean of delta (trend)
        df[f"RollMean_Delta_{base_point_col}_{roll_k}"] = (
            df.groupby("Case_id")["Delta_" + base_point_col]
            .apply(lambda s: s.shift(1).rolling(roll_k, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        # rolling std of base point (volatility)
        df[f"RollStd_{base_point_col}_{roll_k}"] = (
            df.groupby("Case_id")[base_point_col]
            .apply(lambda s: s.shift(1).rolling(roll_k, min_periods=2).std())
            .reset_index(level=0, drop=True)
        ).fillna(0.0)
        # rolling std of deltas (jumpiness)
        df[f"RollStd_Delta_{base_point_col}_{roll_k}"] = (
            df.groupby("Case_id")["Delta_" + base_point_col]
            .apply(lambda s: s.shift(1).rolling(roll_k, min_periods=2).std())
            .reset_index(level=0, drop=True)
        ).fillna(0.0)
        # clean helper cols
        df.drop(columns=["prev_" + base_point_col, "prev_W10_90"], inplace=True, errors="ignore")
    # --- train-time residual target(s)
    if train_flag:
        # Signed residual in original scale:
        df["residual"] = df["GroundTruth"] - df[base_point_col]
        # Log-residual target (more stable for heavy tails)
        # NOTE: base must be non-negative for log1p. If your times can be negative, handle separately.
        base_pos = np.maximum(df[base_point_col].astype(float).to_numpy(), 0.0)
        gt_pos = np.maximum(df["GroundTruth"].astype(float).to_numpy(), 0.0)
        df["log_residual"] = np.log1p(gt_pos) - np.log1p(base_pos)
    # --- drop known-leak / eval-only cols if present
    df.drop(columns=["Absolute_error", "PI10", "PI90", "PI_Coverage_10_90"], inplace=True, errors="ignore")
    return df

def build_prefix_feature_table(
    log: pd.DataFrame,
    log_ids,
    *,
    window: int = 5,
    max_prefix: Optional[int] = None,
    include_case_attrs: bool = True,
    include_event_attrs: bool = True,
) -> pd.DataFrame:
    """
    Seed-independent prefix feature table (one row per (Case_id, Prefix_length)),
    aligned with the SAME sorting policy used by read_event_log(). :contentReference[oaicite:1]{index=1}

    Output:
      Case_id, Prefix_length,
      act_lag_1..W, res_lag_1..W,
      last-event (event_num/cat + case_num/cat),
      time_since_prev_end_day, time_since_case_start_day, end_dow, end_hour
    """
    case_col = log_ids.case
    act_col = log_ids.activity
    res_col = log_ids.resource
    start_col = getattr(log_ids, "start_time", None)
    enabled_col = getattr(log_ids, "enabled_time", None)
    end_col = log_ids.end_time

    log2 = log.copy()
    log2[case_col] = log2[case_col].astype(object)

    # timezone-safe datetime conversion
    if end_col in log2.columns and not is_datetime64_any_dtype(log2[end_col]):
        log2[end_col] = pd.to_datetime(log2[end_col], utc=True, errors="coerce")
    if start_col and start_col in log2.columns and not is_datetime64_any_dtype(log2[start_col]):
        log2[start_col] = pd.to_datetime(log2[start_col], utc=True, errors="coerce")
    if enabled_col and enabled_col in log2.columns and not is_datetime64_any_dtype(log2[enabled_col]):
        log2[enabled_col] = pd.to_datetime(log2[enabled_col], utc=True, errors="coerce")

    # --- EXACT sorting policy from read_event_log() :contentReference[oaicite:2]{index=2}
    if start_col and start_col in log2.columns and enabled_col and enabled_col in log2.columns:
        sort_cols = [start_col, end_col, enabled_col]
    elif start_col and start_col in log2.columns:
        sort_cols = [start_col, end_col]
    else:
        sort_cols = [end_col]

    # IMPORTANT: must sort within each case (otherwise _evt_idx spans cases)
    # So we include case_col as the first key to keep per-case order stable.
    log2 = log2.sort_values([case_col] + sort_cols).reset_index(drop=True)

    # prefix index within case (1-based)
    log2["_evt_idx"] = log2.groupby(case_col).cumcount() + 1

    if max_prefix is not None:
        log2 = log2.loc[log2["_evt_idx"] <= int(max_prefix)].copy()

    # --- time features (based on end_col only, as you requested)
    log2["_prev_end"] = log2.groupby(case_col)[end_col].shift(1)
    log2["_case_start_end"] = log2.groupby(case_col)[end_col].transform("min")

    log2["time_since_prev_end_day"] = (log2[end_col] - log2["_prev_end"]).dt.total_seconds() / 3600.0 / 24.0
    log2["time_since_case_start_day"] = (log2[end_col] - log2["_case_start_end"]).dt.total_seconds() / 3600.0 / 24.0
    log2["end_dow"] = log2[end_col].dt.dayofweek
    log2["end_hour"] = log2[end_col].dt.hour

    # --- last W activities/resources at this prefix end
    for k in range(1, window + 1):
        log2[f"act_lag_{k}"] = log2.groupby(case_col)[act_col].shift(k - 1)
        if res_col in log2.columns:
            log2[f"res_lag_{k}"] = log2.groupby(case_col)[res_col].shift(k - 1)
        else:
            log2[f"res_lag_{k}"] = "__MISSING__"

    # --- last-event attributes
    event_num_cols = list(getattr(log_ids, "event_num_features", []) or [])
    event_cat_cols = list(getattr(log_ids, "event_cat_features", []) or [])
    case_num_cols  = list(getattr(log_ids, "case_num_features", []) or [])
    case_cat_cols  = list(getattr(log_ids, "case_cat_features", []) or [])

    def present(cols: List[str]) -> List[str]:
        return [c for c in cols if c in log2.columns]

    keep_cols = [
        case_col, "_evt_idx",
        "time_since_prev_end_day", "time_since_case_start_day", "end_dow", "end_hour",
    ]
    keep_cols += [f"act_lag_{k}" for k in range(1, window + 1)]
    keep_cols += [f"res_lag_{k}" for k in range(1, window + 1)]

    if include_event_attrs:
        keep_cols += present(event_num_cols + event_cat_cols)
    if include_case_attrs:
        keep_cols += present(case_num_cols + case_cat_cols)

    feat = log2[keep_cols].copy()

    # rename keys to match your prediction dfs
    feat.rename(columns={case_col: "Case_id", "_evt_idx": "Prefix_length"}, inplace=True)

    # make categorical as object with None for missing
    cat_cols = [c for c in feat.columns if c.startswith("act_lag_") or c.startswith("res_lag_")]
    cat_cols += present(event_cat_cols + case_cat_cols)
    MISSING_CAT = "__MISSING__"
    for c in cat_cols:
        feat[c] = feat[c].astype(object)
        feat[c] = feat[c].where(~feat[c].isna(), MISSING_CAT)  # string sentinel
    return feat


def merge_prefix_features(
        df: pd.DataFrame,
        prefix_feat: pd.DataFrame,
        *,
        key_cols: Tuple[str, str] = ("Case_id", "Prefix_length"),
        ) -> pd.DataFrame:
    out = df.copy()
    out["Case_id"] = out["Case_id"].astype(object)
    out["Prefix_length"] = out["Prefix_length"].astype(int)
    # avoid duplicate columns if called multiple times
    feat_cols = [c for c in prefix_feat.columns if c not in key_cols]
    # if any of these already exist, suffix them
    overlap = set(feat_cols).intersection(out.columns)
    if overlap:
        out = out.merge(prefix_feat, on=list(key_cols), how="left", suffixes=("", "_log"))
    else:
        out = out.merge(prefix_feat, on=list(key_cols), how="left")
    return out


def _sanitize_for_catboost(
    df: pd.DataFrame,
    feature_cols: List[str],
    cat_idx: List[int],
    *,
    missing_cat: str = "__MISSING__",
) -> pd.DataFrame:
    """
    CatBoost does NOT like Python None in categorical/object columns.
    Replace None/NaN in categorical cols with a sentinel string.
    Leave numeric cols as-is (NaN is fine).
    """
    out = df.copy()
    if not cat_idx:
        return out

    cat_cols = [feature_cols[i] for i in cat_idx]
    for c in cat_cols:
        if c not in out.columns:
            continue
        # ensure object dtype
        out[c] = out[c].astype(object)
        # replace both NaN and None
        out[c] = out[c].where(out[c].notna(), missing_cat)
        out[c] = out[c].replace({None: missing_cat})
    return out

# =========================
# 2) CONDITIONAL SHRINKAGE
# =========================
def _fit_alpha_isotonic(
        w: np.ndarray,
        y_true: np.ndarray,
        base_pred: np.ndarray,
        res_hat: np.ndarray,
        *,
        n_bins: int = 10,
        clip: Tuple[float, float] = (0.0, 1.0),
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Conditional shrinkage alpha(w) learned via binning (robust and dependency-free).
    Returns (bin_edges, bin_alphas).

    We search alpha in [0,1] per bin to minimize MAE of base_pred + alpha*res_hat.
    """
    w = np.asarray(w, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    base_pred = np.asarray(base_pred, dtype=float)
    res_hat = np.asarray(res_hat, dtype=float)
    # quantile bin edges
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(w[~np.isnan(w)], qs)
    # make strictly increasing
    edges = np.unique(edges)
    if len(edges) < 3:
        # not enough variety -> fall back to global alpha
        edges = np.array([-np.inf, np.inf])
    # assign bins
    bin_ids = np.digitize(w, edges[1:-1], right=True)
    grid = np.linspace(clip[0], clip[1], 21)
    bin_alphas = np.zeros(int(bin_ids.max()) + 1, dtype=float)
    for b in range(len(bin_alphas)):
        m = bin_ids == b
        if m.sum() < 20:
            # too few -> alpha=0 is safest
            bin_alphas[b] = 0.0
            continue
        maes = []
        for a in grid:
            pred = base_pred[m] + a * res_hat[m]
            maes.append(np.mean(np.abs(pred - y_true[m])))
        bin_alphas[b] = float(grid[int(np.argmin(maes))])
    return edges, bin_alphas

def _apply_alpha_binned(w: np.ndarray, edges: np.ndarray, bin_alphas: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    bin_ids = np.digitize(w, edges[1:-1], right=True) if len(edges) > 2 else np.zeros_like(w, dtype=int)
    bin_ids = np.clip(bin_ids, 0, len(bin_alphas) - 1)
    return bin_alphas[bin_ids]

# =========================
# 3) CASE-LEVEL VALIDATION SPLIT
# =========================
def _split_by_case_late_validation(
        df: pd.DataFrame,
        case_col: str,
        val_case_fraction: float,
        *,
        order_by: str = "case_end_index",
        ) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < val_case_fraction < 1.0):
        raise ValueError("val_case_fraction must be in (0,1).")
    case_groups = df.groupby(case_col, sort=False)
    if order_by == "case_end_index":
        case_stat = case_groups.apply(lambda g: g.index.max())
    elif order_by == "max_prefix":
        if "Prefix_length" not in df.columns:
            raise ValueError("Prefix_length missing but order_by='max_prefix' was requested.")
        case_stat = case_groups["Prefix_length"].max()
    else:
        raise ValueError("order_by must be one of: {'case_end_index', 'max_prefix'}")
    case_stat = case_stat.sort_values()
    case_ids_sorted = case_stat.index.to_numpy()
    n_val = max(1, int(len(case_ids_sorted) * val_case_fraction))
    val_case_ids = case_ids_sorted[-n_val:]
    train_case_ids = case_ids_sorted[:-n_val]
    return train_case_ids, val_case_ids

def _detect_cat_features(x: pd.DataFrame, feature_cols: List[str]) -> List[int]:
    """
    Treat any non-numeric (incl. strings, categories, mixed objects) as categorical.
    Bool is also treated as categorical by default (often safer).
    """
    cat_idx = []
    for i, c in enumerate(feature_cols):
        s = x[c]
        if is_bool_dtype(s):
            cat_idx.append(i)
        elif not is_numeric_dtype(s):
            cat_idx.append(i)
    return cat_idx


@dataclass
class TabularPreprocessor:
    numeric_cols: List[str]
    categorical_cols: List[str]
    cat_top_values: Dict[str, Set[str]]
    missing_token: str = "__MISSING__"
    other_token: str = "__OTHER__"


def fit_tabular_preprocessor(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    *,
    top_k: int = 20,
    missing_token: str = "__MISSING__",
    other_token: str = "__OTHER__",
) -> TabularPreprocessor:
    """
    Fit ONLY categorical top-k filtering (no numeric scaling).

    - numeric_cols are detected but untouched
    - categorical_cols include non-numeric and bool (bool treated as categorical by default)
    """
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for c in feature_cols:
        if c not in train_df.columns:
            continue
        s = train_df[c]
        # treat bool as categorical (safer for CatBoost / avoids numeric coercion surprises)
        if is_bool_dtype(s):
            categorical_cols.append(c)
        elif is_numeric_dtype(s):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    cat_top_values: Dict[str, Set[str]] = {}
    for c in categorical_cols:
        # normalize to string, with explicit missing token
        col = train_df[c].astype("string").fillna(missing_token)
        vc = col.value_counts(dropna=False)
        cat_top_values[c] = set(vc.head(top_k).index.astype(str).tolist())

    return TabularPreprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        cat_top_values=cat_top_values,
        missing_token=missing_token,
        other_token=other_token,
    )


def transform_tabular_preprocessor(
    df: pd.DataFrame,
    prep: TabularPreprocessor,
) -> pd.DataFrame:
    """
    Apply ONLY categorical top-k filtering (no numeric scaling).

    For each categorical column:
      - cast to pandas string
      - fill missing with prep.missing_token
      - map rare categories to prep.other_token
      - cast to object (CatBoost-friendly)
    """
    out = df.copy()
    for c in prep.categorical_cols:
        if c not in out.columns:
            continue
        allowed = prep.cat_top_values.get(c, None)
        if not allowed:
            # if somehow missing, just standardize missing + dtype
            col = out[c].astype("string").fillna(prep.missing_token)
            out[c] = col.astype(object)
            continue
        col = out[c].astype("string").fillna(prep.missing_token)
        col = col.where(col.isin(allowed), prep.other_token)
        out[c] = col.astype(object)
    return out


# =========================
# 4) RESIDUAL MODEL (LOG TARGET) + CONDITIONAL SHRINKAGE
# =========================
@dataclass
class ResidualCorrector:
    model: CatBoostRegressor
    feature_cols: List[str]
    cat_idx: List[int]
    w_col: str
    alpha_edges: np.ndarray
    alpha_bins: np.ndarray
    base_point_col: str  # usually "Prediction" or "Q0_5"


def catboost_log_residual_with_conditional_shrinkage(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        case_col: str = "Case_id",
        w_col: str = "W10_90",
        base_point_col: str = "Prediction",
        use_log: bool = False,

        # validation / CV
        use_eval_set: bool = True,
        val_case_fraction: float = 0.20,       # kept for backward compatibility (used only if k_folds<=1)
        val_order_by: str = "case_end_index",  # kept for backward compatibility

        # NEW: K-fold case CV
        k_folds: int = 5,
        shuffle_cases: bool = True,

        # catboost params
        iterations: int = 6000,
        learning_rate: float = 0.03,
        depth: int = 8,
        l2_leaf_reg: float = 5.0,
        random_seed: int = 42,
        early_stopping_rounds: int = 250,
        verbose: int = 200,

        # alpha(w) params
        alpha_bins: int = 5,
        min_bin: int = 100,
        alpha_cap: float = 1.0,  # set to 0.7 or 0.5 if you want to prevent overcorrection
) -> Tuple[pd.DataFrame, "ResidualCorrector"]:
    """
    Case-level K-fold CV for early stopping + out-of-fold alpha(W) calibration.

    - Train target:
        if use_log: log_residual = log1p(GT) - log1p(base_point)
        else:       residual     = GT - base_point
    - Apply correction:
        if use_log:
            updated = expm1(log1p(base_point) + alpha(W) * log_res_hat)
        else:
            updated = base_point + alpha(W) * res_hat
    """

    # -------------------------
    # checks
    # -------------------------
    required_train = {case_col, "Prefix_length", "GroundTruth", base_point_col, w_col}
    required_test = {case_col, "Prefix_length", "GroundTruth", base_point_col, w_col}
    target_col = "log_residual" if use_log else "residual"
    required_train.add(target_col)

    miss_tr = required_train - set(train_df.columns)
    miss_te = required_test - set(test_df.columns)
    if miss_tr:
        raise ValueError(f"train_df missing required columns: {sorted(miss_tr)}")
    if miss_te:
        raise ValueError(f"test_df missing required columns: {sorted(miss_te)}")

    # -------------------------
    # features
    # -------------------------
    exclude = {case_col, "GroundTruth", "residual", "log_residual"}
    feature_cols = [c for c in train_df.columns if c not in exclude]
    if not feature_cols:
        raise ValueError("No feature columns found after exclusions.")

    x_train_all = train_df[feature_cols].copy()
    cat_idx = _detect_cat_features(x_train_all, feature_cols)

    # sanitize once (important for CatBoost categorical handling)
    train_df = _sanitize_for_catboost(train_df, feature_cols, cat_idx)
    test_df  = _sanitize_for_catboost(test_df,  feature_cols, cat_idx)

    def make_pool(df_subset: pd.DataFrame, with_label: bool = True) -> Pool:
        x = df_subset[feature_cols]
        if with_label:
            y = df_subset[target_col].astype(float)
            return Pool(x, y, cat_features=cat_idx if cat_idx else None)
        return Pool(x, cat_features=cat_idx if cat_idx else None)

    # -------------------------
    # K-fold CV by cases (OOF predictions for alpha calibration)
    # -------------------------
    unique_cases = pd.Index(train_df[case_col].astype(object).unique())
    n_cases = len(unique_cases)

    # If CV disabled or too few cases, fall back to your old single split behavior
    do_cv = bool(use_eval_set and k_folds and k_folds >= 2 and n_cases >= k_folds * 2)

    if do_cv:
        rng = np.random.default_rng(random_seed)
        cases = unique_cases.to_numpy().copy()
        if shuffle_cases:
            rng.shuffle(cases)

        folds = np.array_split(cases, k_folds)

        oof_hat = np.full(len(train_df), np.nan, dtype=float)
        best_iters: List[int] = []

        for fi in range(k_folds):
            val_cases = set(folds[fi].tolist())
            tr_mask = ~train_df[case_col].isin(val_cases)
            va_mask = train_df[case_col].isin(val_cases)

            tr_part = train_df.loc[tr_mask].copy()
            va_part = train_df.loc[va_mask].copy()

            # sanitize (safe even if already sanitized)
            tr_part = _sanitize_for_catboost(tr_part, feature_cols, cat_idx)
            va_part = _sanitize_for_catboost(va_part, feature_cols, cat_idx)

            model = CatBoostRegressor(
                loss_function="MAE",
                eval_metric="MAE",
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                l2_leaf_reg=l2_leaf_reg,
                random_seed=random_seed + fi,
                od_type="Iter",
                od_wait=early_stopping_rounds,
                verbose=verbose,
            )

            model.fit(
                make_pool(tr_part, with_label=True),
                eval_set=make_pool(va_part, with_label=True),
                use_best_model=True
            )

            # best iteration for final fit
            bi = int(getattr(model, "best_iteration_", None) or model.get_best_iteration() or iterations)
            best_iters.append(max(1, bi))

            # OOF preds on val fold
            va_pool = make_pool(va_part, with_label=False)
            oof_hat[va_mask.to_numpy()] = np.asarray(model.predict(va_pool), dtype=float)

        # build calibration df (OOF only)
        calib_df = train_df.copy()
        calib_df["_oof_hat"] = oof_hat
        calib_df = calib_df[np.isfinite(calib_df["_oof_hat"].to_numpy())].copy()

        # Choose final iteration count (robust)
        final_iterations = int(np.median(best_iters)) if best_iters else iterations
        final_iterations = max(50, min(final_iterations, iterations))

    else:
        # old split path
        if use_eval_set and 0.0 < val_case_fraction < 1.0 and train_df[case_col].nunique() >= 10:
            tr_cases, va_cases = _split_by_case_late_validation(
                train_df, case_col, val_case_fraction, order_by=val_order_by
            )
            tr_mask = train_df[case_col].isin(set(tr_cases.tolist()))
            va_mask = train_df[case_col].isin(set(va_cases.tolist()))
            train_part = train_df.loc[tr_mask].copy()
            eval_part = train_df.loc[va_mask].copy()
        else:
            train_part = train_df.copy()
            eval_part = None

        train_part = _sanitize_for_catboost(train_part, feature_cols, cat_idx)
        if eval_part is not None:
            eval_part = _sanitize_for_catboost(eval_part, feature_cols, cat_idx)

        model_tmp = CatBoostRegressor(
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
        model_tmp.fit(
            make_pool(train_part, with_label=True),
            eval_set=make_pool(eval_part, with_label=True) if eval_part is not None else None,
            use_best_model=True
        )

        calib_df = eval_part if eval_part is not None and len(eval_part) > 0 else train_part
        calib_df = _sanitize_for_catboost(calib_df, feature_cols, cat_idx)

        # predictions used for alpha
        calib_pool = make_pool(calib_df, with_label=False)
        calib_df = calib_df.copy()
        calib_df["_oof_hat"] = np.asarray(model_tmp.predict(calib_pool), dtype=float)

        final_iterations = int(getattr(model_tmp, "best_iteration_", None) or model_tmp.get_best_iteration() or iterations)
        final_iterations = max(50, min(final_iterations, iterations))

    # -------------------------
    # Fit FINAL model on ALL train_df using final_iterations
    # -------------------------
    final_model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=final_iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_seed,
        verbose=verbose,
    )
    final_model.fit(make_pool(train_df, with_label=True), use_best_model=False)

    # -------------------------
    # Learn alpha(W) on calib_df using its predicted residuals (OOF if CV path)
    # -------------------------
    w_cal = calib_df[w_col].astype(float).to_numpy()
    y_true_cal = calib_df["GroundTruth"].astype(float).to_numpy()
    base_cal = calib_df[base_point_col].astype(float).to_numpy()
    res_hat_cal = calib_df["_oof_hat"].astype(float).to_numpy()

    ok = ~np.isnan(w_cal)
    if ok.sum() < 10:
        edges = np.array([-np.inf, np.inf])
        bin_alphas = np.array([0.0], dtype=float)
    else:
        edges = np.unique(np.quantile(w_cal[ok], np.linspace(0, 1, alpha_bins + 1)))
        if len(edges) < 3:
            edges = np.array([-np.inf, np.inf])
        bin_ids = np.digitize(w_cal, edges[1:-1], right=True)

        grid = np.linspace(0.0, 1.0, 21)
        bin_alphas = np.zeros(int(bin_ids.max()) + 1, dtype=float)

        for b in range(len(bin_alphas)):
            m = (bin_ids == b) & ok
            if m.sum() < min_bin:
                bin_alphas[b] = 0.0
                continue

            best_a, best_mae = 0.0, np.inf
            for a in grid:
                if use_log:
                    pred = np.expm1(np.log1p(np.maximum(base_cal[m], 0.0)) + a * res_hat_cal[m])
                else:
                    pred = base_cal[m] + a * res_hat_cal[m]
                mae = np.mean(np.abs(pred - y_true_cal[m]))
                if mae < best_mae:
                    best_mae = mae
                    best_a = float(a)
            bin_alphas[b] = best_a

        if alpha_cap < 1.0:
            bin_alphas = np.clip(bin_alphas, 0.0, float(alpha_cap))

    # -------------------------
    # Predict on test with FINAL model
    # -------------------------
    x_test = test_df.reindex(columns=feature_cols).copy()
    x_test = _sanitize_for_catboost(x_test, feature_cols, cat_idx)
    test_pool = Pool(x_test, cat_features=cat_idx if cat_idx else None)

    res_hat = np.asarray(final_model.predict(test_pool), dtype=float)
    alpha = _apply_alpha_binned(test_df[w_col].astype(float).to_numpy(), edges, bin_alphas)
    if alpha_cap < 1.0:
        alpha = np.clip(alpha, 0.0, float(alpha_cap))

    base_test = test_df[base_point_col].astype(float).to_numpy()
    if use_log:
        updated_pred = np.expm1(np.log1p(np.maximum(base_test, 0.0)) + alpha * res_hat)
    else:
        updated_pred = base_test + alpha * res_hat

    abs_err = np.abs(updated_pred - test_df["GroundTruth"].astype(float).to_numpy())
    results_df = pd.DataFrame({
        "Case_id": test_df[case_col].values,
        "Prefix_length": test_df["Prefix_length"].values,
        "GroundTruth": test_df["GroundTruth"].astype(float).values,
        "Prediction": updated_pred,
        "Absolute_error": abs_err,
    })

    bundle = ResidualCorrector(
        model=final_model,
        feature_cols=feature_cols,
        cat_idx=cat_idx,
        w_col=w_col,
        alpha_edges=edges,
        alpha_bins=bin_alphas,
        base_point_col=base_point_col,
    )

    return results_df, bundle


def catboost_log_residual_with_conditional_shrinkage2(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        case_col: str = "Case_id",
        w_col: str = "W10_90",
        base_point_col: str = "Prediction",
        use_log: bool = False,
        # validation
        use_eval_set: bool = True,
        val_case_fraction: float = 0.20,
        val_order_by: str = "case_end_index",
        # catboost params
        iterations: int = 6000,
        learning_rate: float = 0.03,
        depth: int = 8,
        l2_leaf_reg: float = 5.0,
        random_seed: int = 42,
        early_stopping_rounds: int = 250,
        verbose: int = 200,
        # alpha(w) params
        alpha_bins: int = 5,
        min_bin: int=100,
        ) -> Tuple[pd.DataFrame, ResidualCorrector]:
    """
    Trains CatBoost to predict log_residual = log1p(GT) - log1p(base_point),
    then applies inverse transform:
        updated = expm1(log1p(base_point) + alpha(W)*log_res_hat)

    Alpha is learned as a binned function of W10_90 on validation cases (case-level split).
    """
    required_train = {case_col, "Prefix_length", "GroundTruth", base_point_col, w_col}
    if use_log:
        required_train.add("log_residual")
    else:
        required_train.add("residual")
    required_test = {case_col, "Prefix_length", "GroundTruth", base_point_col, w_col}
    miss_tr = required_train - set(train_df.columns)
    miss_te = required_test - set(test_df.columns)
    if miss_tr:
        raise ValueError(f"train_df missing required columns: {sorted(miss_tr)}")
    if miss_te:
        raise ValueError(f"test_df missing required columns: {sorted(miss_te)}")
    # --- feature set: exclude only label and identifiers; keep base_point_col as a feature!
    exclude = {case_col, "GroundTruth", "residual", "log_residual"}  # keep Prediction/Q0_5 in features
    feature_cols = [c for c in train_df.columns if c not in exclude]
    if not feature_cols:
        raise ValueError("No feature columns found after exclusions.")
    x_train_all = train_df[feature_cols].copy()
    target_col = "log_residual" if use_log else "residual"
    y_train_all = train_df[target_col].astype(float).to_numpy()
    cat_idx = _detect_cat_features(x_train_all, feature_cols)
    # sanitize the full train/test frames once (safe even if eval_part=None later)
    train_df = _sanitize_for_catboost(train_df, feature_cols, cat_idx)
    test_df  = _sanitize_for_catboost(test_df,  feature_cols, cat_idx)
    
    def make_pool(df_subset: pd.DataFrame) -> Pool:
        x = df_subset[feature_cols]
        y = df_subset[target_col].astype(float) if target_col in df_subset.columns else None
        return Pool(x, y, cat_features=cat_idx if cat_idx else None)

    # --- case-level eval split
    if use_eval_set and 0.0 < val_case_fraction < 1.0 and train_df[case_col].nunique() >= 10:
        tr_cases, va_cases = _split_by_case_late_validation(train_df, case_col, val_case_fraction, order_by=val_order_by)
        tr_mask = train_df[case_col].isin(set(tr_cases.tolist()))
        va_mask = train_df[case_col].isin(set(va_cases.tolist()))
        train_part = train_df.loc[tr_mask].copy()
        eval_part = train_df.loc[va_mask].copy()
    else:
        train_part = train_df.copy()
        eval_part = None
        
    train_part = _sanitize_for_catboost(train_part, feature_cols, cat_idx)
    if eval_part is not None:
        eval_part = _sanitize_for_catboost(eval_part, feature_cols, cat_idx)

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

    model.fit(make_pool(train_part), eval_set=make_pool(eval_part) if eval_part is not None else None, use_best_model=True)

    # --- learn alpha(W) on eval cases (fallback to train_part if no eval)
    calib_df = eval_part if eval_part is not None and len(eval_part) > 0 else train_part
    calib_df = _sanitize_for_catboost(calib_df, feature_cols, cat_idx)
    calib_pool = Pool(calib_df[feature_cols], cat_features=cat_idx if cat_idx else None)
    res_hat_cal = np.asarray(model.predict(calib_pool), dtype=float)

    w_cal = calib_df[w_col].astype(float).to_numpy()
    y_true_cal = calib_df["GroundTruth"].astype(float).to_numpy()
    base_cal = calib_df[base_point_col].astype(float).to_numpy()

    # compute alpha bins to minimize MAE after inverse transform
    # We do this in *original space* (after expm1), not in log space.
    # In binned optimization, we simulate corrected preds.
    # We'll reuse _fit_alpha_isotonic but with a modified res_hat: "delta in log space".
    # The formula is:
    #   pred = expm1(log1p(base) + alpha*log_res_hat)
    # We'll optimize alpha per bin by grid-search directly here.
    edges = np.unique(np.quantile(w_cal[~np.isnan(w_cal)], np.linspace(0, 1, alpha_bins + 1)))
    if len(edges) < 3:
        edges = np.array([-np.inf, np.inf])
    bin_ids = np.digitize(w_cal, edges[1:-1], right=True)
    grid = np.linspace(0.0, 1.0, 21)
    bin_alphas = np.zeros(int(bin_ids.max()) + 1, dtype=float)

    for b in range(len(bin_alphas)):
        m = bin_ids == b
        if m.sum() < min_bin:
            bin_alphas[b] = 0.0
            continue
        best_a, best_mae = 0.0, np.inf
        for a in grid:
            if use_log:
                pred = np.expm1(np.log1p(np.maximum(base_cal[m], 0.0)) + a * res_hat_cal[m])
            else:
                pred = base_cal[m] + a * res_hat_cal[m]
            mae = np.mean(np.abs(pred - y_true_cal[m]))
            if mae < best_mae:
                best_mae = mae
                best_a = float(a)
        bin_alphas[b] = best_a

    # --- predict on test
    x_test = test_df.reindex(columns=feature_cols).copy()
    x_test = _sanitize_for_catboost(x_test, feature_cols, cat_idx)
    test_pool = Pool(x_test, cat_features=cat_idx if cat_idx else None)
    res_hat = np.asarray(model.predict(test_pool), dtype=float)

    alpha = _apply_alpha_binned(test_df[w_col].astype(float).to_numpy(), edges, bin_alphas)

    base_test = test_df[base_point_col].astype(float).to_numpy()
    if use_log:
        updated_pred = np.expm1(np.log1p(np.maximum(base_test, 0.0)) + alpha * res_hat)
    else:
        updated_pred = base_test + alpha * res_hat
    abs_err = np.abs(updated_pred - test_df["GroundTruth"].astype(float).to_numpy())
    results_df = pd.DataFrame({
        "Case_id": test_df[case_col].values,
        "Prefix_length": test_df["Prefix_length"].values,
        "GroundTruth": test_df["GroundTruth"].astype(float).values,
        "Prediction": updated_pred,
        "Absolute_error": abs_err,
    })

    bundle = ResidualCorrector(
        model=model,
        feature_cols=feature_cols,
        cat_idx=cat_idx,
        w_col=w_col,
        alpha_edges=edges,
        alpha_bins=bin_alphas,
        base_point_col=base_point_col,
    )
    return results_df, bundle