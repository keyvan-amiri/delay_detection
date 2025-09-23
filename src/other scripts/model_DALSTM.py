import torch.nn as nn

##############################################################################
# Backbone Data-aware LSTM model for remaining time prediction
##############################################################################
class DALSTMModel(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, n_layers=None,
                 max_len=None, dropout=True, p_fix=0.2, 
                 exclude_last_layer=False, return_squeezed=True):
        '''
        ARGUMENTS:
        input_size: number of features
        hidden_size: number of neurons in LSTM layers
        n_layers: number of LSTM layers
        max_len: maximum length for prefixes in the dataset
        dropout: apply dropout if "True", otherwise no dropout
        p_fix: dropout probability
        '''
        super(DALSTMModel, self).__init__()
        
        self.n_layers = n_layers 
        self.dropout = dropout
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(p=p_fix)
        self.batch_norm1 = nn.BatchNorm1d(max_len)
        # whether to drop the last layer or not (for embedding-based UQ)
        self.exclude_last_layer = exclude_last_layer
        self.return_squeezed = return_squeezed
        if not self.exclude_last_layer:
            self.linear1 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = x.float() # if tensors are saved in a different format
        x, (hidden_state,cell_state) = self.lstm1(x)
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.batch_norm1(x)
        if self.n_layers > 1:
            for i in range(self.n_layers - 1):
                x, (hidden_state,cell_state) = self.lstm2(
                    x, (hidden_state,cell_state))
                if self.dropout:
                    x = self.dropout_layer(x)
                x = self.batch_norm1(x)
        if not self.exclude_last_layer:
            # only the last one in the sequence is used by dense layer
            yhat = self.linear1(x[:, -1, :]) 
            if self.return_squeezed:
                return yhat.squeeze(dim=1)
            else:
                return yhat 
        else:
            # Return the output without applying the last linear layer
            return x[:, -1, :]  
    
##############################################################################
# Stochastic Data-aware LSTM modelL remaining time prediction (mean & variance)
##############################################################################

class DALSTMModelMve(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, n_layers=None,
                 max_len=None, dropout=True, p_fix=0.2, return_squeezed=True):
        '''
        ARGUMENTS:
        input_size: number of features
        hidden_size: number of neurons in LSTM layers
        n_layers: number of LSTM layers
        max_len: maximum length for prefixes in the dataset
        dropout: apply dropout if "True", otherwise no dropout
        p_fix: dropout probability
        '''
        super(DALSTMModelMve, self).__init__()
        
        self.n_layers = n_layers 
        self.dropout = dropout
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(p=p_fix)
        self.batch_norm1 = nn.BatchNorm1d(max_len)
        
        # Linear layers for mean and variance
        self.linear_mu = nn.Linear(hidden_size, 1)
        self.linear_logvar = nn.Linear(hidden_size, 1)
        self.return_squeezed = return_squeezed
        
    def forward(self, x):
        x = x.float() # if tensors are saved in a different format
        x, (hidden_state, cell_state) = self.lstm1(x)
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.batch_norm1(x)
        if self.n_layers > 1:
            for i in range(self.n_layers - 1):
                x, (hidden_state, cell_state) = self.lstm2(
                    x, (hidden_state, cell_state))
                if self.dropout:
                    x = self.dropout_layer(x)
                x = self.batch_norm1(x)
        
        # Predict mean and variance
        mu = self.linear_mu(x[:, -1, :]) # mean prediction       
        logvar = self.linear_logvar(x[:, -1, :]) # log-variance prediction
        
        if self.return_squeezed:
            return mu.squeeze(dim=1), logvar.squeeze(dim=1)
        else:
            return mu, logvar
        

# Custom weight initialization function for ensembles
def dalstm_init_weights(m):
    if isinstance(m, nn.Linear):
        # Xavier uniform initialization for Linear layers
        nn.init.xavier_uniform_(m.weight) 
        # Initialize linear biases to zero
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                # Xavier uniform for input-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Orthogonal for hidden-hidden weights
                nn.init.orthogonal_(param.data)  
            elif 'bias' in name:
                # Initialize LSTM biases to zero
                nn.init.zeros_(param.data)  
    elif isinstance(m, nn.BatchNorm1d):
        # BatchNorm weight should start at 1
        nn.init.ones_(m.weight) 
        # BatchNorm bias should start at 0
        nn.init.zeros_(m.bias)  