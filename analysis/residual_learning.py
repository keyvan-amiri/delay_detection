# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 12:41:32 2026
@author: kamirel
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

#from src.utils.auxiliary_quantile_model import extract_features

def get_train_test_dataframes(result_dir=None, dataset=None, model=None, seed=None):
    train_name = f"{dataset}_{model}_quantile_wos_seed{seed}_quantile_train_val.csv"
    test_name  = f"{dataset}_{model}_quantile_wos_seed{seed}_inference.csv"
    train_df = pd.read_csv(os.path.join(result_dir, train_name))
    test_df  = pd.read_csv(os.path.join(result_dir, test_name))
    return train_df, test_df

def get_best_thresh(metric_list, thresholds):
    i = int(np.nanargmin(np.asarray(metric_list, dtype=float)))
    return thresholds[i]

def train_corrective_model(train_few: pd.DataFrame, feature_lst, critertion="mae"):
    df = train_few.dropna(subset=list(feature_lst) + ["GroundTruth", "Prediction"]).copy()
    # do-nothing model if too few samples
    if len(df) < 5:
        #m = LinearRegression()
        m = make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0)
        )
        m.coef_ = np.zeros(len(feature_lst), dtype=float)
        m.intercept_ = 0.0
        m.n_features_in_ = len(feature_lst)
        return m
    X = df[feature_lst]  # keep as DataFrame (column order preserved)
    y_res = (df["GroundTruth"] - df["Prediction"]).to_numpy()
    #m = LinearRegression()
    m = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    )
    m.fit(X, y_res)
    return m

def gated_mae(df, model, feature_lst, thresh_val):
    d = df.dropna(subset=list(feature_lst) + ["GroundTruth", "Prediction", "PI_Width_10_90"]).copy()
    # start from base predictions
    d["y_pred"] = d["Prediction"]
    # correct only the 'few'
    few_mask = d["PI_Width_10_90"] > thresh_val
    if few_mask.any():
        d.loc[few_mask, "y_pred"] = (
            d.loc[few_mask, "Prediction"] + model.predict(d.loc[few_mask, feature_lst])
        )
    return float(np.mean(np.abs(d["GroundTruth"] - d["y_pred"])))


def get_best_thresh_cv_on_few(
    train_df,
    thresholds,
    feature_lst,
    n_splits=5,
    seed=42,
    min_few_total=50,     # minimum size of FEW set to even consider this threshold
    min_few_fold=10,      # minimum size of FEW in a fold to score it
):
    """
    For each threshold q:
      1) compute thresh_val on FULL train_df (validation set)
      2) subset FEW = rows with PI_Width_10_90 > thresh_val
      3) run KFold CV ONLY on FEW
         - train corrective model on few_train
         - evaluate MAE on few_val using corrected prediction (Prediction + residual_hat)
      4) average fold MAEs -> score(q)
    Return: best_q, scores_dict
    """
    df = train_df.dropna(subset=list(feature_lst) + ["GroundTruth", "Prediction", "PI_Width_10_90"]).copy()

    scores = {}
    for q in thresholds:
        thresh_val = float(df["PI_Width_10_90"].quantile(q))
        few = df[df["PI_Width_10_90"] > thresh_val].copy()

        # If FEW set is too small, mark as invalid / very bad
        if len(few) < min_few_total:
            scores[q] = np.inf
            continue

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_scores = []

        for tr_idx, va_idx in kf.split(few):
            few_tr = few.iloc[tr_idx]
            few_va = few.iloc[va_idx]

            if len(few_tr) < min_few_fold or len(few_va) < min_few_fold:
                continue

            model = train_corrective_model(few_tr, feature_lst=feature_lst)

            # MAE ONLY on FEW validation fold, using corrected prediction everywhere in that fold
            d = few_va.dropna(subset=list(feature_lst) + ["GroundTruth", "Prediction"]).copy()
            y_true = d["GroundTruth"].to_numpy()
            y_pred = d["Prediction"].to_numpy() + model.predict(d[feature_lst])
            fold_scores.append(float(np.mean(np.abs(y_true - y_pred))))

        scores[q] = float(np.mean(fold_scores)) if len(fold_scores) > 0 else np.inf

    best_q = min(scores, key=scores.get)
    return best_q, scores


def get_best_thresh_cv_overall(
    train_df,
    thresholds,
    feature_lst,
    n_splits=5,
    seed=42,
    min_few_train=30,
):
    df = train_df.dropna(subset=list(feature_lst) + ["GroundTruth", "Prediction", "PI_Width_10_90"]).copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scores = {}
    for q in thresholds:
        fold_scores = []
        for tr_idx, va_idx in kf.split(df):
            tr = df.iloc[tr_idx]
            va = df.iloc[va_idx]

            thresh_val = float(tr["PI_Width_10_90"].quantile(q))
            tr_few = tr[tr["PI_Width_10_90"] > thresh_val].copy()

            # if too few to train correction -> no correction (baseline gated = baseline)
            if len(tr_few) < min_few_train:
                fold_scores.append(float(np.mean(np.abs(va["GroundTruth"] - va["Prediction"]))))
                continue

            model = train_corrective_model(tr_few, feature_lst=feature_lst)
            fold_scores.append(gated_mae(va, model, feature_lst, thresh_val))

        scores[q] = float(np.mean(fold_scores))

    best_q = min(scores, key=scores.get)
    return best_q, scores


def main():
    # device = f'cuda:{os.environ.get("CUDA_VISIBLE_DEVICES", "0")}' if torch.cuda.is_available() else 'cpu'
    #quantiles=(0.1, 0.5, 0.6, 0.9, 0.95, 0.99)
    #datsets = ['P2P', 'BPIC15_1']
    dataset =  'BPIC_2017_W' # 'Sepsis', 'BPIC20ID', 'BPIC20DD', 'BPIC20PTC' 'BPIC_2017_W'
    seeds = [409, 1824, 3657, 4012, 4506]
    min_few_pct = 0.20
    max_few_pct = 0.50
    n_candidates = 11  # grid size
    q_min = 1.0 - max_few_pct
    q_max = 1.0 - min_few_pct
    thresholds = np.linspace(q_min, q_max, n_candidates).round(4).tolist()
    root_path = os.getcwd()
    result_dir = os.path.join(root_path, 'results', 'DALSTM', dataset)
    feature_lst = ['Prefix_length', 'Q0_1','Q0_5','Q0_6','Q0_9','Q0_95','Q0_99']
    for seed in seeds:
        train_df, test_df = get_train_test_dataframes(
            result_dir=result_dir, dataset=dataset, model='DALSTM', seed=seed)
        val_mae_before = np.mean(np.abs(train_df["GroundTruth"] - train_df["Prediction"]))
        print("val_mae_before", val_mae_before)
        # --- pick best threshold via CV on validation (train_df) with gated MAE ---
        """
        best_q, cv_scores = get_best_thresh_cv_on_few(
            train_df=train_df,
            thresholds=thresholds,
            feature_lst=feature_lst,
            n_splits=5,
            seed=seed,
            min_few_total=50,
            min_few_fold=10
            )
        """
        best_q, cv_scores = get_best_thresh_cv_overall(
            train_df=train_df,
            thresholds=thresholds,
            feature_lst=feature_lst,
            n_splits=5,
            seed=seed,
            min_few_train=30
            )
        # --- train final corrective model on FULL validation-few using best threshold ---
        thresh_val = float(train_df["PI_Width_10_90"].quantile(best_q))
        train_few = train_df[train_df["PI_Width_10_90"] > thresh_val].copy()
        model = train_corrective_model(train_few, feature_lst=feature_lst)
        # --- evaluate on test with same gating ---
        mae_before = float(np.mean(test_df["Absolute_error"]))
        mae_after = gated_mae(test_df, model, feature_lst, thresh_val)
        # report many/few stats on test
        test_many = test_df[test_df["PI_Width_10_90"] <= thresh_val].copy()
        test_few  = test_df[test_df["PI_Width_10_90"] >  thresh_val].copy()
        frac_many = len(test_many) / max(len(test_df), 1)
        frac_few  = len(test_few)  / max(len(test_df), 1)
        print(
            dataset,
            f"seed={seed}",
            f"best_q={best_q}",
            f"thresh_val={thresh_val:.6f}",
            f"cv={cv_scores}",
            f"test_split(many/few)={frac_many:.3f}/{frac_few:.3f}",
            f"MAE_before={mae_before:.4f}",
            f"MAE_after={mae_after:.4f}",
            )        


   
if __name__ == '__main__':
    main() 