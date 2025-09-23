import torch
import torch.nn as nn


def set_loss(loss_func=None, heteroscedastic=False): 
    if heteroscedastic:
        criterion = heteroscedastic_loss(metric=loss_func)
    else:
        if loss_func == 'mae':
            criterion = nn.L1Loss()
        elif loss_func == 'LogCoshLoss':
            criterion = LogCoshLoss()
        elif loss_func == 'mse':
            criterion = nn.MSELoss()
        elif loss_func == 'Huber':
            criterion = nn.HuberLoss()
        elif loss_func == 'smooth_mae':
            criterion = nn.SmoothL1Loss()
        elif loss_func == 'rmse':
            criterion = RMSELoss()        
    return criterion

# Custom class for Root Mean Squared Error (RMSE)
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        mse = self.mse_loss(y_pred, y_true)
        rmse = torch.sqrt(mse)
        return rmse
    
class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))
    
# Custom class for heteroscedastic loss.
class heteroscedastic_loss(nn.Module):
    # metric: "mae" or "rmse" (default is "rmse")
    def __init__(self, metric='mse'):
        super(heteroscedastic_loss, self).__init__()
        self.metric = metric
        
    def forward(self, mean, true, log_var):
        '''
        ARGUMENTS:
        true: target values shape of: batch_size
        mean: predictions with shape of: batch_size
        log_var: Logaritms of uncertainty estimates. shape: batch_size
        OUTPUTS:
        loss: Tensor (0)
        '''
        precision = torch.exp(-log_var)
        if self.metric == 'mae':
            # based on L1-loss and its relation to Laplace distribution
            loss = torch.mean(precision**0.5 * torch.abs(true - mean) + log_var)
        elif self.metric == 'mse':
            loss = 0.5 * torch.mean(precision * (true - mean) ** 2 + log_var)
        else:
            raise ValueError("Metric has to be 'mse' or 'mae'")            
        return loss
    
def mape(outputs, targets, epsilon=1e-8, threshold=1e-6):
    """
    # Custom function for Mean Absolute Percentage Error (MAPE)
    def mape(outputs, targets, epsilon=1e-8):
        # Add epsilon to the targets to avoid division by zero
        return torch.mean(torch.abs((targets - outputs) / (targets + epsilon))) * 100
    """
    # Create a mask for non-zero targets
    non_zero_mask = torch.abs(targets) > threshold    
    # Compute the absolute percentage error only for non-zero targets
    absolute_percentage_error = torch.abs((targets - outputs) / (targets + epsilon))    
    # Apply the mask to ignore errors where targets are zero or close to zero
    mape_value = torch.mean(absolute_percentage_error[non_zero_mask])    
    return mape_value * 100