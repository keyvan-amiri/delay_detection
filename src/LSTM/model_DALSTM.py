# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 09:04:23 2025
@author: Keyvan Amiri Elyasi
"""
import pickle
import torch
import torch.nn as nn

from src.utils.fds import FDS
from src.LSTM.model_stochasticDALSTM import DALSTMModelMve, DALSTMModelGamma

def get_DALSTM_model(args, cfg, device=None):
    with open(args.input_size_path, 'rb') as f:
        input_size = pickle.load(f)
    n_layers = cfg['DALSTM']['n_layers'] or 2
    hidden_size = cfg['DALSTM']['hidden_size'] or 150
    dropout = cfg['DALSTM']['dropout']
    if dropout is None:
        dropout = True
    dropout_prob = cfg['DALSTM']['dropout_prob'] or 0.1
    fds_config = dict(
        feature_dim=hidden_size,
        bucket_num=args.fds_bucket_num,
        bucket_start=args.fds_bucket_start,
        start_update=args.fds_start_update,
        start_smooth=args.fds_start_smooth,
        kernel=args.fds_kernel,
        ks=args.fds_ks,
        sigma=args.fds_sigma,
    )
    if args.bmse and args.FDS:
        model = DALSTMFDSModelBMC(
            input_size=input_size, hidden_size=hidden_size,
            n_layers=n_layers, dropout=dropout, p_fix=dropout_prob,
            **fds_config
        ).to(device)
    elif args.FDS:
        model = DALSTMFDSModel(
            input_size=input_size, hidden_size=hidden_size,
            n_layers=n_layers, dropout=dropout, p_fix=dropout_prob,
            **fds_config
        ).to(device)
    elif args.bmse:
        model = DALSTMModelBMC(
            input_size=input_size, hidden_size=hidden_size,
            n_layers=n_layers, dropout=dropout, p_fix=dropout_prob
        ).to(device)
    elif args.gamma_nll:
        model = DALSTMModelGamma(
            input_size=input_size, hidden_size=hidden_size,
            n_layers=n_layers, dropout=dropout, p_fix=dropout_prob
        ).to(device)
    elif args.heteroscedastic:
        model = DALSTMModelMve(
            input_size=input_size, hidden_size=hidden_size,
            n_layers=n_layers, dropout=dropout, p_fix=dropout_prob
        ).to(device)
    else:
        model = DALSTMModel(
            input_size=input_size, hidden_size=hidden_size,
            n_layers=n_layers, dropout=dropout, p_fix=dropout_prob
        ).to(device)
    return model, fds_config

##############################################################################
# Backbone Data-aware LSTM model for remaining time prediction
##############################################################################
class DALSTMModel(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, n_layers=None,
                 dropout=True, p_fix=0.2, exclude_last_layer=False,
                 return_squeezed=True):
        super(DALSTMModel, self).__init__()         

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.exclude_last_layer = exclude_last_layer
        self.return_squeezed = return_squeezed
        self.dropout = dropout
        # Create LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(n_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.lstm_layers.append(
                nn.LSTM(input_dim, hidden_size, batch_first=True)
            )
        # Layer normalization layers (replacing BatchNorm)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(n_layers)
        ])
        # Recurrent dropout
        if self.dropout:
            self.recurrent_dropout = nn.Dropout(p_fix)
        if not self.exclude_last_layer:
            self.linear1 = nn.Linear(hidden_size, 1)

    def _apply_lstm_with_dropout(self, lstm, x, h_prev=None, c_prev=None):
        """Apply LSTM with recurrent dropout to hidden states"""
        batch_size = x.size(0)

        if h_prev is None:
            h = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        else:
            h, c = h_prev, c_prev

        if self.training and self.dropout:
            h = self.recurrent_dropout(h)
            c = self.recurrent_dropout(c)
        return lstm(x, (h, c))

    def forward(self, x):
        x = x.float()
        """
        for i, (lstm, ln) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            if i == 0:
                x, (h, c) = self._apply_lstm_with_dropout(lstm, x)
            else:
                x, (h, c) = self._apply_lstm_with_dropout(lstm, x, h, c)
            # LayerNorm over feature dimension (B, T, H)
            x = ln(x)
        """
        for i, (lstm, ln) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            x, (h, c) = self._apply_lstm_with_dropout(lstm, x)  # each layer has its own (h,c)
            x = ln(x)
        # Last timestep
        last_output = x[:, -1, :]
        if not self.exclude_last_layer:
            yhat = self.linear1(last_output)
            return yhat.squeeze(dim=1) if self.return_squeezed else yhat
        else:
            return last_output
   
##############################################################################
# Data-aware LSTM model with feature distribution smoothing
############################################################################## 
class DALSTMFDSModel(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, n_layers=None,
                 dropout=True, p_fix=0.2, return_squeezed=True, **config):
        super(DALSTMFDSModel, self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.return_squeezed = return_squeezed
        self.config = config

        # Create LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(n_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.lstm_layers.append(
                nn.LSTM(input_dim, hidden_size, batch_first=True)
            )

        # Layer normalization layers (replacing BatchNorm)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(n_layers)
        ])

        # Recurrent dropout
        if self.dropout:
            self.recurrent_dropout = nn.Dropout(p_fix)

        self.regressor = nn.Linear(config['feature_dim'], 1)
        self.FDS = FDS(**config)

        # Empty registration for label buckets
        self.register_buffer("fds_bin_edges", torch.empty(0))

    def _apply_lstm_with_dropout(self, lstm, x, h_prev=None, c_prev=None):
        """Apply LSTM with recurrent dropout to hidden states"""
        batch_size = x.size(0)
        if h_prev is None:
            h = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        else:
            h, c = h_prev, c_prev

        if self.training and self.dropout:
            h = self.recurrent_dropout(h)
            c = self.recurrent_dropout(c)

        return lstm(x, (h, c))

    def forward(self, x, y, epoch):
        x = x.float()

        # Process through each LSTM layer
        """
        for i, (lstm, ln) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            if i == 0:
                x, (h, c) = self._apply_lstm_with_dropout(lstm, x)
            else:
                x, (h, c) = self._apply_lstm_with_dropout(lstm, x, h, c)
            # LayerNorm over feature dimension (B, T, H)
            x = ln(x)
        """
        for i, (lstm, ln) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            x, (h, c) = self._apply_lstm_with_dropout(lstm, x)  # each layer has its own (h,c)
            x = ln(x)


        # Last timestep features
        features = x[:, -1, :]

        # Smooth feature distributions over the target space
        smoothed_features = features.clone()
        if self.training and epoch >= self.config['start_smooth']:
            y_bucket = self.bucketize_for_fds(y)  # (B,)
            smoothed_features = self.FDS.smooth(
                smoothed_features,
                y_bucket.unsqueeze(1),  # (B, 1) for backward compatibility
                epoch
            )

        preds = self.regressor(smoothed_features)
        if self.return_squeezed:
            return {'preds': preds.squeeze(dim=1), 'features': features}
        else:
            return {'preds': preds, 'features': features}

    @torch.no_grad()
    def set_fds_bin_edges(self, bin_edges: torch.Tensor):
        self.fds_bin_edges = bin_edges.to(self.fds_bin_edges.device)

    def bucketize_for_fds(self, y_cont: torch.Tensor) -> torch.Tensor:
        if self.fds_bin_edges.numel() == 0:
            raise RuntimeError("FDS bin edges not set. Call model.set_fds_bin_edges(...)")

        bucket_start = self.config["bucket_start"]
        bucket_num = self.config["bucket_num"]

        y = y_cont.detach().float()
        inner_edges = self.fds_bin_edges[1:-1]  # length num_bins - 1
        b = torch.bucketize(y, inner_edges)     # in [0..num_bins-1]

        return (b + bucket_start).clamp(bucket_start, bucket_num - 1).long()

##############################################################################
# Data-aware LSTM model to support BMSE loss
##############################################################################  
class DALSTMModelBMC(nn.Module):
    """
    DALSTM for BMC/BMSE:
      - predicts mean mu per sample
      - has ONE learnable scalar log_noise (shared across all samples)
    """
    def __init__(self, input_size=None, hidden_size=None, n_layers=None,
                 dropout=True, p_fix=0.2, return_squeezed=True,
                 init_log_noise=0.0):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.return_squeezed = return_squeezed

        self.lstm_layers = nn.ModuleList()
        for i in range(n_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.lstm_layers.append(nn.LSTM(input_dim, hidden_size, batch_first=True))

        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(n_layers)])

        if self.dropout:
            self.recurrent_dropout = nn.Dropout(p_fix)

        self.linear_mu = nn.Linear(hidden_size, 1)

        # <<< the key part: scalar, learnable
        self.log_noise = nn.Parameter(torch.tensor(float(init_log_noise)))

    def _apply_lstm_with_dropout(self, lstm, x):
        batch_size = x.size(0)
        h = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=x.device)

        if self.training and self.dropout:
            h = self.recurrent_dropout(h)
            c = self.recurrent_dropout(c)

        return lstm(x, (h, c))

    def forward(self, x):
        x = x.float()
        for lstm, ln in zip(self.lstm_layers, self.layer_norms):
            x, _ = self._apply_lstm_with_dropout(lstm, x)
            x = ln(x)

        last_output = x[:, -1, :]
        mu = self.linear_mu(last_output)  # [B,1]

        # scalar noise_var (positive)
        noise_var = torch.exp(self.log_noise).clamp(min=1e-6)  # scalar tensor

        if self.return_squeezed:
            return mu.squeeze(dim=1), noise_var
        else:
            return mu, noise_var

##############################################################################
# Data-aware LSTM model with BMSE loss and feature distribution smoothing
##############################################################################         
class DALSTMFDSModelBMC(nn.Module):
    """
    FDS + BMC:
      - predicts mu per sample
      - scalar learnable log_noise
    """
    def __init__(self, input_size=None, hidden_size=None, n_layers=None,
                 dropout=True, p_fix=0.2, return_squeezed=True,
                 init_log_noise=0.0, **config):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.return_squeezed = return_squeezed
        self.config = config

        self.lstm_layers = nn.ModuleList()
        for i in range(n_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.lstm_layers.append(nn.LSTM(input_dim, hidden_size, batch_first=True))

        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(n_layers)])

        if self.dropout:
            self.recurrent_dropout = nn.Dropout(p_fix)

        self.regressor_mu = nn.Linear(config['feature_dim'], 1)
        self.FDS = FDS(**config)

        self.register_buffer("fds_bin_edges", torch.empty(0))

        # <<< scalar learnable
        self.log_noise = nn.Parameter(torch.tensor(float(init_log_noise)))

    def _apply_lstm_with_dropout(self, lstm, x):
        batch_size = x.size(0)
        h = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=x.device)

        if self.training and self.dropout:
            h = self.recurrent_dropout(h)
            c = self.recurrent_dropout(c)
        return lstm(x, (h, c))

    def forward(self, x, y, epoch):
        x = x.float()
        for lstm, ln in zip(self.lstm_layers, self.layer_norms):
            x, _ = self._apply_lstm_with_dropout(lstm, x)
            x = ln(x)

        features = x[:, -1, :]
        smoothed_features = features.clone()
        if self.training and epoch >= self.config['start_smooth']:
            y_bucket = self.bucketize_for_fds(y)
            smoothed_features = self.FDS.smooth(smoothed_features, y_bucket.unsqueeze(1), epoch)

        mu = self.regressor_mu(smoothed_features)  # [B,1]
        noise_var = torch.exp(self.log_noise).clamp(min=1e-6)  # scalar

        if self.return_squeezed:
            return {
                "preds_mu": mu.squeeze(dim=1),
                "noise_var": noise_var,
                "features": features
            }
        else:
            return {
                "preds_mu": mu,
                "noise_var": noise_var,
                "features": features
            }

    @torch.no_grad()
    def set_fds_bin_edges(self, bin_edges: torch.Tensor):
        self.fds_bin_edges = bin_edges.to(self.fds_bin_edges.device)

    def bucketize_for_fds(self, y_cont: torch.Tensor) -> torch.Tensor:
        if self.fds_bin_edges.numel() == 0:
            raise RuntimeError("FDS bin edges not set. Call model.set_fds_bin_edges(...)")

        bucket_start = self.config["bucket_start"]
        bucket_num = self.config["bucket_num"]

        y = y_cont.detach().float()
        inner_edges = self.fds_bin_edges[1:-1]
        b = torch.bucketize(y, inner_edges)
        return (b + bucket_start).clamp(bucket_start, bucket_num - 1).long()