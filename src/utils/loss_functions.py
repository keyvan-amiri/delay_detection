import torch
import torch.nn as nn
import torch.nn.functional as F

def set_loss(args): 
    loss_func = args.loss
    if args.heteroscedastic:
        criterion = heteroscedastic_loss(metric=loss_func)
    else:
        if loss_func == 'bmse':
            criterion = bmc_loss
        elif loss_func == 'sera':
            criterion = sera_loss
        elif loss_func == 'mae':
            criterion = weighted_l1_loss
        elif loss_func == 'mse':
            criterion = weighted_mse_loss
        elif loss_func == 'focal_mae':
            criterion = weighted_focal_l1_loss
        elif loss_func == 'focal_mse':
            criterion = weighted_focal_mse_loss
        elif loss_func == 'huber':    
            criterion = weighted_huber_loss      
    return criterion

def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_focal_l1_loss(inputs, targets, weights=None,
                           activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_focal_mse_loss(inputs, targets, weights=None,
                            activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_huber_loss(inputs, targets, weights=None, beta=1.):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def bmc_loss(pred, target, noise_var):
    """
    #The original implementation is adjusted.The original implementation:
    def bmc_loss(pred, target, noise_var):
        logits = - 0.5 * (pred - target.T).pow(2) / noise_var
        loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))
        loss = loss * (2 * noise_var)
        return loss
    """
    logits = -0.5 * (pred - target.T).pow(2) / noise_var
    labels = torch.arange(pred.shape[0], device=logits.device, dtype=torch.long)
    loss = F.cross_entropy(logits, labels)
    loss = loss * (2 * noise_var.mean())
    return loss

def sera_loss(preds, trues, phi_trues, step=0.001, norm=False):
    device = trues.device
    th = torch.arange(0, 1 + step, step, device=device)  # [T]    
    # Expand for broadcasting
    phi_b   = phi_trues.unsqueeze(0)    # [1,B]
    # mask: [T,B]
    mask = (phi_b >= th.unsqueeze(1))   # for each threshold, which samples are included
    # squared errors: [B]
    se = (trues - preds) ** 2     
    # errors per threshold: [T]
    # sum only where mask==True
    errors = torch.where(mask, se.unsqueeze(0), torch.zeros_like(se).unsqueeze(0)).sum(dim=1)
    if norm and errors[0] != 0:
        errors = errors / errors[0]
    # trapezoidal integration to get SERA
    sera = 0.5 * step * torch.sum(errors[:-1] + errors[1:])  # scalar
    return sera


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