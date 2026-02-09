# -*- coding: utf-8 -*-
"""
Classify prefixes into many / med / few subsets based on remaining time
using a small LSTM classifier.

Usage:
    python classify_subsets.py --dataset P2P
    python classify_subsets.py --dataset Sepsis --hidden_size 128
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report


def load_tensors(dataset, model="DALSTM", root_path=None):
    """Load the preprocessed X and y tensors for train, val, and test."""
    if root_path is None:
        root_path = os.path.dirname(os.path.abspath(__file__))
    process_path = os.path.join(root_path, "temp", model, dataset)

    paths = {
        "X_train": os.path.join(process_path, f"DALSTM_X_train_{dataset}.pt"),
        "X_val":   os.path.join(process_path, f"DALSTM_X_val_{dataset}.pt"),
        "X_test":  os.path.join(process_path, f"DALSTM_X_test_{dataset}.pt"),
        "y_train": os.path.join(process_path, f"DALSTM_y_train_{dataset}.pt"),
        "y_val":   os.path.join(process_path, f"DALSTM_y_val_{dataset}.pt"),
        "y_test":  os.path.join(process_path, f"DALSTM_y_test_{dataset}.pt"),
    }

    data = {}
    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing preprocessed file: {path}\n"
                f"Run preprocessing first via main.py --dataset {dataset}"
            )
        data[key] = torch.load(path, weights_only=True)

    return data


def compute_quantile_thresholds(y, many_frac=0.6, med_frac=0.3):
    """Compute the two quantile boundaries from a target array."""
    q_many = np.quantile(y, many_frac)
    q_med  = np.quantile(y, many_frac + med_frac)
    return q_many, q_med


def assign_labels_with_thresholds(y, q_many, q_med):
    """Assign many (0) / med (1) / few (2) using pre-computed thresholds."""
    labels = np.full(len(y), -1, dtype=int)
    labels[y <= q_many] = 0
    labels[(y > q_many) & (y <= q_med)] = 1
    labels[y > q_med] = 2
    return labels


def compute_prefix_lengths(X):
    """Return actual (non-padding) length per sample. X: (N, T, F)."""
    non_pad = (X.abs().sum(dim=2) > 0)   # (N, T) bool
    return non_pad.sum(dim=1).long()      # (N,)


# ---------- LSTM classifier ----------

class LSTMClassifier(nn.Module):
    """Small LSTM that reads a variable-length prefix and predicts 3 classes."""

    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2,
                 num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        """
        x       : (batch, seq_len, features)
        lengths : (batch,)  actual prefix lengths
        """
        # Pack padded sequences so the LSTM ignores padding
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)       # h_n: (num_layers, batch, hidden)
        out = self.drop(h_n[-1])              # last layer hidden state
        return self.fc(out)                   # (batch, num_classes)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for X_batch, y_batch, len_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        len_batch = len_batch.to(device)

        logits = model(X_batch, len_batch)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        n += len(y_batch)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for X_batch, y_batch, len_batch in loader:
        X_batch = X_batch.to(device)
        len_batch = len_batch.to(device)
        logits = model(X_batch, len_batch)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(y_batch)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def make_loader(X, y, lengths, batch_size, shuffle):
    """Build a DataLoader."""
    ds = TensorDataset(
        X.float(),
        torch.tensor(y, dtype=torch.long),
        lengths,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Train a small LSTM classifier to predict many/med/few subset"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of the dataset (must be preprocessed already)")
    parser.add_argument("--many_frac", type=float, default=0.6)
    parser.add_argument("--med_frac", type=float, default=0.3)
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="LSTM hidden size (default: 64)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of LSTM layers (default: 2)")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early-stopping patience (default: 10)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 1. Load preprocessed tensors ----
    print(f"Loading preprocessed data for '{args.dataset}' ...")
    data = load_tensors(args.dataset)

    X_train_t = data["X_train"].float()
    X_val_t   = data["X_val"].float()
    X_test_t  = data["X_test"].float()
    y_train_t = data["y_train"]
    y_val_t   = data["y_val"]
    y_test_t  = data["y_test"]

    # ---- 2. Assign many/med/few labels ----
    y_all = torch.cat([y_train_t, y_val_t, y_test_t]).numpy()
    q_many, q_med = compute_quantile_thresholds(y_all, args.many_frac, args.med_frac)
    print(f"Quantile thresholds: many <= {q_many:.4f}, "
          f"med <= {q_med:.4f}, few > {q_med:.4f}")

    labels_train = assign_labels_with_thresholds(y_train_t.numpy(), q_many, q_med)
    labels_val   = assign_labels_with_thresholds(y_val_t.numpy(), q_many, q_med)
    labels_test  = assign_labels_with_thresholds(y_test_t.numpy(), q_many, q_med)

    # ---- 3. Compute prefix lengths ----
    len_train = compute_prefix_lengths(X_train_t)
    len_val   = compute_prefix_lengths(X_val_t)
    len_test  = compute_prefix_lengths(X_test_t)

    # ---- 4. Class weights (inverse frequency) ----
    label_names = ["many", "med", "few"]
    all_train_labels = np.concatenate([labels_train, labels_val])
    class_counts = np.bincount(all_train_labels, minlength=3).astype(float)
    class_weights_np = class_counts.sum() / (3.0 * class_counts)
    class_weights_t = torch.tensor(class_weights_np, dtype=torch.float)

    print(f"\nInput size: {X_train_t.shape[2]}  |  Seq length: {X_train_t.shape[1]}")
    print(f"Train: {len(labels_train)}  Val: {len(labels_val)}  Test: {len(labels_test)}")
    for i, name in enumerate(label_names):
        n_tr = np.sum(labels_train == i) + np.sum(labels_val == i)
        n_te = np.sum(labels_test == i)
        print(f"  {name}: train+val={n_tr}, test={n_te}  (weight={class_weights_np[i]:.2f})")

    # ---- 5. Data loaders ----
    train_loader = make_loader(X_train_t, labels_train, len_train,
                               args.batch_size, shuffle=True)
    val_loader   = make_loader(X_val_t, labels_val, len_val,
                               args.batch_size, shuffle=False)
    test_loader  = make_loader(X_test_t, labels_test, len_test,
                               args.batch_size, shuffle=False)

    # ---- 6. Model ----
    input_size = X_train_t.shape[2]
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t.to(device))

    # ---- 7. Training with early stopping ----
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    print(f"\nTraining LSTM classifier (epochs={args.epochs}, patience={args.patience}) ...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_preds, val_labels = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        if epoch <= 5 or epoch % 5 == 0 or marker:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}  loss={train_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  lr={current_lr:.1e}{marker}")

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # ---- 8. Evaluate best model on test ----
    model.load_state_dict(best_state)
    model.to(device)
    y_pred, y_true = evaluate(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  Best val acc : {best_val_acc:.4f}")
    print(f"  Test accuracy: {acc:.4f}")
    print(f"{'='*50}\n")
    print(classification_report(y_true, y_pred, target_names=label_names))


if __name__ == "__main__":
    main()
