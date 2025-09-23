import pickle
import torch
import torch.nn as nn

from src.utils.fds import FDS

def get_DALSTM_model(args, cfg, device=None):
    # define model parameters
    with open(args.input_size_path, 'rb') as f:
        input_size =  pickle.load(f) # input_size corresponds to vocab_size
    n_layers = cfg['DALSTM']['n_layers'] or 2
    hidden_size = cfg['DALSTM']['hidden_size'] or 150
    dropout = cfg['DALSTM']['dropout']
    if dropout is None:
        dropout = True
    dropout_prob = cfg['DALSTM']['dropout_prob'] or 0.1
    fds_config= dict(
        feature_dim=hidden_size, start_update=args.fds_start_update,
        start_smooth=args.fds_start_smooth, kernel=args.fds_kernel,
        ks=args.fds_ks, sigma=args.fds_sigma) 
    if args.bmse and args.FDS:
        model = DALSTMFDSModelMve(
                input_size=input_size, hidden_size=hidden_size,
                n_layers=n_layers, dropout=dropout,
                p_fix=dropout_prob, **fds_config).to(device)
    elif args.FDS:
        model = DALSTMFDSModel(
            input_size=input_size, hidden_size=hidden_size,
            n_layers=n_layers, dropout=dropout,
            p_fix=dropout_prob, **fds_config).to(device)         
    elif args.heteroscedastic or args.bmse:
        model = DALSTMModelMve(
                input_size=input_size, hidden_size=hidden_size, 
                n_layers=n_layers, dropout=dropout, 
                p_fix=dropout_prob).to(device)   
    else:
        model = DALSTMModel(
                input_size=input_size, hidden_size=hidden_size, 
                n_layers=n_layers, dropout=dropout, 
                p_fix=dropout_prob).to(device) 
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
            lstm_layer = nn.LSTM(
                input_dim, 
                hidden_size, 
                batch_first=True
            )
            self.lstm_layers.append(lstm_layer)        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for _ in range(n_layers)
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
        # Apply recurrent dropout to hidden states during training
        if self.training and self.dropout:
            h = self.recurrent_dropout(h)
            c = self.recurrent_dropout(c)        
        return lstm(x, (h, c))
    
    def forward(self, x):
        x = x.float()        
        # Process through each LSTM layer
        for i, (lstm, bn) in enumerate(zip(self.lstm_layers, self.batch_norms)):
            # Apply LSTM with recurrent dropout
            if i == 0:
                x, (h, c) = self._apply_lstm_with_dropout(lstm, x)
            else:
                x, (h, c) = self._apply_lstm_with_dropout(lstm, x, h, c)            
            # Apply batch normalization
            x = x.transpose(1, 2)
            x = bn(x)
            x = x.transpose(1, 2)        
        # Get the last output in the sequence
        last_output = x[:, -1, :]        
        if not self.exclude_last_layer:
            yhat = self.linear1(last_output)
            if self.return_squeezed:
                return yhat.squeeze(dim=1)
            else:
                return yhat
        else:
            return last_output
    
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
        dropout: apply dropout if "True", otherwise no dropout
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
            lstm_layer = nn.LSTM(
                input_dim, 
                hidden_size, 
                batch_first=True
            )
            self.lstm_layers.append(lstm_layer)        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for _ in range(n_layers)
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
        # Apply recurrent dropout to hidden states during training
        if self.training and self.dropout:
            h = self.recurrent_dropout(h)
            c = self.recurrent_dropout(c)        
        return lstm(x, (h, c))
    
    def forward(self, x):
        x = x.float()        
        # Process through each LSTM layer
        for i, (lstm, bn) in enumerate(zip(self.lstm_layers, self.batch_norms)):
            # Apply LSTM with recurrent dropout
            if i == 0:
                x, (h, c) = self._apply_lstm_with_dropout(lstm, x)
            else:
                x, (h, c) = self._apply_lstm_with_dropout(lstm, x, h, c)            
            # Apply batch normalization to HIDDEN FEATURES dimension
            # Transpose: (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size, seq_len)
            x = x.transpose(1, 2)
            x = bn(x)
            x = x.transpose(1, 2)  # Back to (batch_size, seq_len, hidden_size)        
        # Get the last output in the sequence for MVE prediction
        last_output = x[:, -1, :]
        # Predict mean and variance
        mu = self.linear_mu(last_output)
        logvar = self.linear_logvar(last_output)
        if self.return_squeezed:
            return mu.squeeze(dim=1), logvar.squeeze(dim=1)
        else:
            return mu, logvar

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
            lstm_layer = nn.LSTM(
                input_dim, 
                hidden_size, 
                batch_first=True
            )
            self.lstm_layers.append(lstm_layer)        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for _ in range(n_layers)
        ]) 
        # Recurrent dropout 
        if self.dropout:        
            self.recurrent_dropout = nn.Dropout(p_fix)        
        self.regressor = nn.Linear(config['feature_dim'], 1)
        self.FDS = FDS(**config)
    
    def _apply_lstm_with_dropout(self, lstm, x, h_prev=None, c_prev=None):
        """Apply LSTM with recurrent dropout to hidden states"""
        batch_size = x.size(0)        
        if h_prev is None:
            h = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        else:
            h, c = h_prev, c_prev        
        # Apply recurrent dropout to hidden states during training
        if self.training and self.dropout:
            h = self.recurrent_dropout(h)
            c = self.recurrent_dropout(c)        
        return lstm(x, (h, c))
    
    def forward(self, x, y, epoch):
        x = x.float()        
        # Process through each LSTM layer
        for i, (lstm, bn) in enumerate(zip(self.lstm_layers, self.batch_norms)):
            # Apply LSTM with recurrent dropout
            if i == 0:
                x, (h, c) = self._apply_lstm_with_dropout(lstm, x)
            else:
                x, (h, c) = self._apply_lstm_with_dropout(lstm, x, h, c)            
            # Apply batch normalization
            x = x.transpose(1, 2)
            x = bn(x)
            x = x.transpose(1, 2)        
        # Get the last output in the sequence
        features = x[:, -1, :]
        # smooth the feature distributions over the target space
        smoothed_features = features
        if self.training and epoch >= self.config['start_smooth']:
            y_reshaped = y.unsqueeze(1)
            smoothed_features = self.FDS.smooth(smoothed_features, y_reshaped, epoch)
        preds = self.regressor(smoothed_features)
        if self.return_squeezed:
            return {'preds': preds.squeeze(dim=1), 'features': features}
        else:
            return {'preds': preds, 'features': features}

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
        dropout: apply dropout if "True", otherwise no dropout
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
            lstm_layer = nn.LSTM(
                input_dim, 
                hidden_size, 
                batch_first=True
            )
            self.lstm_layers.append(lstm_layer)        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for _ in range(n_layers)
        ])        
        # Recurrent dropout
        if self.dropout:  
            self.recurrent_dropout = nn.Dropout(p_fix)        
        # Linear layers for mean and variance (MVE)
        self.regressor_mu = nn.Linear(config['feature_dim'], 1)
        self.regressor_logvar = nn.Linear(config['feature_dim'], 1)
        self.FDS = FDS(**config)
        
    def _apply_lstm_with_dropout(self, lstm, x, h_prev=None, c_prev=None):
        """Apply LSTM with recurrent dropout to hidden states"""
        batch_size = x.size(0)        
        if h_prev is None:
            h = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        else:
            h, c = h_prev, c_prev        
        # Apply recurrent dropout to hidden states during training
        if self.training and self.dropout:
            h = self.recurrent_dropout(h)
            c = self.recurrent_dropout(c)        
        return lstm(x, (h, c))
    
    def forward(self, x, y, epoch):
        x = x.float()        
        # Process through each LSTM layer
        for i, (lstm, bn) in enumerate(zip(self.lstm_layers, self.batch_norms)):
            # Apply LSTM with recurrent dropout
            if i == 0:
                x, (h, c) = self._apply_lstm_with_dropout(lstm, x)
            else:
                x, (h, c) = self._apply_lstm_with_dropout(lstm, x, h, c)            
            # Apply batch normalization to HIDDEN FEATURES dimension
            # Transpose: (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size, seq_len)
            x = x.transpose(1, 2)
            x = bn(x)
            x = x.transpose(1, 2)  # Back to (batch_size, seq_len, hidden_size)        
        # Get the last output in the sequence for MVE prediction
        features = x[:, -1, :]
        # smooth the feature distributions over the target space
        smoothed_features = features
        if self.training and epoch >= self.config['start_smooth']:
            y_reshaped = y.unsqueeze(1)
            smoothed_features = self.FDS.smooth(smoothed_features, y_reshaped, epoch)
        preds_mu = self.regressor_mu(smoothed_features)
        preds_logvar = self.regressor_logvar(smoothed_features)
        if self.return_squeezed:
            return {'preds_mu': preds_mu.squeeze(dim=1),
                    'preds_logvar': preds_logvar.squeeze(dim=1),
                    'features': features}
        else:
            return {'preds_mu': preds_mu,
                    'preds_logvar': preds_logvar,
                    'features': features}