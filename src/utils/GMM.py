# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 10:33:04 2026
@author: Keyvan Amiri Elyasi
"""

from __future__ import annotations
import os
import math
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def fit_label_gmm(
    y_train: torch.Tensor,
    y_val: torch.Tensor,
    *,
    min_fraction_per_component: float = 0.10,
    max_components: int = 10,
    covariance_type: str = "full",
    n_init: int = 10,
    reg_covar: float = 1e-6,
    random_state: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a 1D Gaussian Mixture Model (GMM) to the concatenated (train+val) labels,
    choose the number of components by BIC among candidates that satisfy:
      - each component has at least `min_fraction_per_component` of (train+val) examples
        *after assignment by argmax posterior responsibility*.
    Returns:
      z_train: Long tensor, same shape as y_train, mixture-component IDs (0..K-1)
      z_val:   Long tensor, same shape as y_val, mixture-component IDs (0..K-1)
    """
    if not (0.0 < min_fraction_per_component <= 1.0):
        raise ValueError("min_fraction_per_component must be in (0, 1].")
    if max_components < 1:
        raise ValueError("max_components must be >= 1.")
    device = f'cuda:{os.environ.get("CUDA_VISIBLE_DEVICES", "0")}' if torch.cuda.is_available() else 'cpu'
    # --- Prepare y as 2D float64 numpy arrays for sklearn ---
    def _to_2d_numpy(y: torch.Tensor) -> np.ndarray:
        if not torch.is_tensor(y):
            raise TypeError("y_train/y_val must be torch tensors.")
        #y_ = y.detach().reshape(-1, 1).to(dtype=torch.float64, device=device).numpy()
        y_ = (y.detach().reshape(-1, 1).to(dtype=torch.float64).cpu().numpy())
        return y_

    y_tr_np = _to_2d_numpy(y_train)
    y_va_np = _to_2d_numpy(y_val)
    y_all = np.vstack([y_tr_np, y_va_np])
    n_all = y_all.shape[0]
    if n_all < 2:
        # Degenerate: only one sample total -> 1 component
        z_train = torch.zeros_like(y_train, dtype=torch.long)
        z_val = torch.zeros_like(y_val, dtype=torch.long)
        return z_train, z_val

    # Candidate K upper bound implied by min fraction constraint
    # (e.g., min_fraction=0.10 => K <= 10)
    max_k_by_fraction = int(math.floor(1.0 / min_fraction_per_component + 1e-12))
    max_k = max(1, min(max_components, max_k_by_fraction, n_all))  # cannot exceed n_all

    min_count = int(math.ceil(min_fraction_per_component * n_all))

    best_gmm: Optional[GaussianMixture] = None
    best_bic: float = float("inf")
    best_k: int = 1

    # --- Model selection: minimize BIC subject to min-count-per-component constraint ---
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            n_init=n_init,
            reg_covar=reg_covar,
            random_state=random_state,
        )
        try:
            gmm.fit(y_all)
        except Exception:
            continue  # skip pathological fits

        # Hard assignment by max posterior responsibility
        labels_all = gmm.predict(y_all)
        counts = np.bincount(labels_all, minlength=k)

        if np.min(counts) < min_count:
            # violates your ">=10% in each component" requirement
            continue

        bic = gmm.bic(y_all)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_k = k

    # If nothing satisfied the constraint, fall back to 1 component
    if best_gmm is None:
        best_gmm = GaussianMixture(
            n_components=1,
            covariance_type=covariance_type,
            n_init=n_init,
            reg_covar=reg_covar,
            random_state=random_state,
        ).fit(y_all)
        best_k = 1
        
    # --- Predict component IDs for train/val separately ---
    z_tr_np = best_gmm.predict(y_tr_np).astype(np.int64)
    z_va_np = best_gmm.predict(y_va_np).astype(np.int64)

    z_train = torch.as_tensor(z_tr_np, dtype=torch.long, device=y_train.device).reshape(y_train.shape)
    z_val = torch.as_tensor(z_va_np, dtype=torch.long, device=y_val.device).reshape(y_val.shape)
        
    return z_train, z_val


class LSTMComponentClassifier(nn.Module):
    """
    2-layer LSTM (hidden=150) with dropout=0.1 between layers (PyTorch LSTM dropout),
    followed by a linear head on the last time step to predict component id.
    """
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 150, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = x.float()
        out, _ = self.lstm(x)          # out: [B, T, H]
        last = out[:, -1, :]           # last time step: [B, H]
        logits = self.head(last)       # [B, C]
        return logits


@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 100
    patience: int = 10              # early stopping patience on val accuracy
    min_delta: float = 1e-4        # required improvement
    grad_clip: float = 1.0
    device: Optional[str] = None   # e.g. "cuda" or "cpu"


@torch.no_grad()
def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train_lstm_and_predict_test_components(
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    X_test: torch.Tensor,
    z_train: torch.Tensor,
    z_val: torch.Tensor,
    y_test: torch.Tensor,
    *,
    cfg: TrainConfig = TrainConfig(),
) -> Tuple[torch.Tensor, LSTMComponentClassifier, Dict[str, Any]]:
    """
    Train an LSTM classifier to predict mixture component ids (z) from sequences (X),
    using validation for early stopping. Then predict z_test for X_test.

    Inputs:
      X_*: [N, T, F] float tensors
      z_train, z_val: integer class labels (same shape as y_train/y_val; typically [N])
      y_test: used only for shaping the returned z_test (same shape as y_test)

    Returns:
      z_test: Long tensor, same shape as y_test
      model: trained model (best val accuracy checkpoint)
      info: training history (best epoch, best val acc, etc.)
    """
    if X_train.ndim != 3 or X_val.ndim != 3 or X_test.ndim != 3:
        raise ValueError("Expected X_train/X_val/X_test to have shape [N, T, F].")

    # Flatten z to [N] for CrossEntropyLoss if needed
    z_train_1d = z_train.view(-1).long()
    z_val_1d = z_val.view(-1).long()

    # Determine number of classes from train+val labels
    num_classes = int(torch.max(torch.cat([z_train_1d, z_val_1d])).item()) + 1
    input_size = X_train.shape[-1]

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMComponentClassifier(input_size=input_size, num_classes=num_classes).to(device)

    train_ds = TensorDataset(X_train, z_train_1d)
    val_ds = TensorDataset(X_val, z_val_1d)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(cfg.max_epochs):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            opt.step()

            bs = xb.size(0)
            running_loss += loss.item() * bs
            n_seen += bs

        train_loss = running_loss / max(1, n_seen)

        # Validation accuracy
        model.eval()
        all_logits = []
        all_y = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                all_logits.append(model(xb))
                all_y.append(yb)
        val_logits = torch.cat(all_logits, dim=0)
        val_y = torch.cat(all_y, dim=0)
        val_acc = _accuracy(val_logits, val_y)

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)

        # Early stopping on val accuracy
        if val_acc > best_val_acc + cfg.min_delta:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict z_test
    model.eval()
    test_loader = DataLoader(TensorDataset(X_test), batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    preds = []
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu())
    z_test_1d = torch.cat(preds, dim=0).long()

    # Match y_test shape (e.g., [N] or [N,1] etc.)
    z_test = z_test_1d.to(device=y_test.device).reshape(y_test.shape)

    info = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "num_classes": num_classes,
        "device": device,
        "history": history,
    }
    print('Best validation accuracy for GMM classification:', best_val_acc)
    return z_test, model, info