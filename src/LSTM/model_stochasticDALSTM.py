# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 10:00:59 2026
@author: Keyvan Amiri Elyasi
"""
import torch
import torch.nn as nn

from src.utils.fds import FDS

##############################################################################
# Stochastic Data-aware LSTM model for remaining time prediction (mean & variance)
##############################################################################
class DALSTMModelMve(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, n_layers=None,
                 dropout=True, p_fix=0.2, return_squeezed=True):
        '''
        ARGUMENTS:
        input_size: number of features
        hidden_size: number of neurons in LSTM layers
        n_layers: number of LSTM layers
        dropout: apply dropout if True
        p_fix: dropout probability
        '''
        super(DALSTMModelMve, self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.return_squeezed = return_squeezed

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

        # Linear layers for mean and variance (MVE)
        self.linear_mu = nn.Linear(hidden_size, 1)
        self.linear_logvar = nn.Linear(hidden_size, 1)

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
        # Last timestep representation
        last_output = x[:, -1, :]

        # Predict mean and log-variance
        mu = self.linear_mu(last_output)
        logvar = self.linear_logvar(last_output)

        if self.return_squeezed:
            return mu.squeeze(dim=1), logvar.squeeze(dim=1)
        else:
            return mu, logvar
        
##############################################################################
# Stochastic Data-aware LSTM model with feature distribution smoothing
############################################################################## 
class DALSTMFDSModelMve(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, n_layers=None,
                 dropout=True, p_fix=0.2, return_squeezed=True, **config):
        '''
        ARGUMENTS:
        input_size: number of features
        hidden_size: number of neurons in LSTM layers
        n_layers: number of LSTM layers
        dropout: apply dropout if True
        p_fix: dropout probability
        '''
        super(DALSTMFDSModelMve, self).__init__()

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

        # Linear layers for mean and variance (MVE)
        self.regressor_mu = nn.Linear(config['feature_dim'], 1)
        self.regressor_logvar = nn.Linear(config['feature_dim'], 1)

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

        # Smooth the feature distributions over the target space
        smoothed_features = features.clone()
        if self.training and epoch >= self.config['start_smooth']:
            y_bucket = self.bucketize_for_fds(y)  # (B,)
            smoothed_features = self.FDS.smooth(
                smoothed_features,
                y_bucket.unsqueeze(1),  # (B, 1) for backward compatibility
                epoch
            )

        preds_mu = self.regressor_mu(smoothed_features)
        preds_logvar = self.regressor_logvar(smoothed_features)

        if self.return_squeezed:
            return {
                'preds_mu': preds_mu.squeeze(dim=1),
                'preds_logvar': preds_logvar.squeeze(dim=1),
                'features': features
            }
        else:
            return {
                'preds_mu': preds_mu,
                'preds_logvar': preds_logvar,
                'features': features
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
        inner_edges = self.fds_bin_edges[1:-1]  # length num_bins - 1
        b = torch.bucketize(y, inner_edges)     # in [0..num_bins-1]

        return (b + bucket_start).clamp(bucket_start, bucket_num - 1).long()    