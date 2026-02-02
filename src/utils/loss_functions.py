import torch
import torch.nn as nn
import torch.nn.functional as F


def set_loss(args):
    loss_func = args.loss
    if args.heteroscedastic:
        return heteroscedastic_loss(metric=loss_func)
    if loss_func == 'bmse':
        return bmc_loss
    if loss_func == 'sera':
        return sera_loss
    if loss_func == 'mae':
        return weighted_l1_loss
    if loss_func == 'mse':
        return weighted_mse_loss
    if loss_func == 'huber':
        return weighted_huber_loss
    if loss_func == 'focal_mae':
        beta = getattr(args, "focal_beta", 0.2)
        gamma = getattr(args, "focal_gamma", 1.0)
        return lambda inputs, targets, weights=None: weighted_focal_l1_loss(
            inputs, targets, weights=weights, activate='sigmoid', beta=beta, gamma=gamma
        )
    if loss_func == 'focal_mse':
        beta = getattr(args, "focal_beta", 0.2)
        gamma = getattr(args, "focal_gamma", 1.0)
        return lambda inputs, targets, weights=None: weighted_focal_mse_loss(
            inputs, targets, weights=weights, activate='sigmoid', beta=beta, gamma=gamma
        )
    raise ValueError(f"Unknown loss: {loss_func}")

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
    # per-sample absolute error
    err = torch.abs(inputs - targets)
    # focal scaling in [0, 1]
    if activate == 'tanh':
        scale = torch.tanh(beta * err).pow(gamma)
    else:  # 'sigmoid' (matches your description)
        scale = torch.sigmoid(beta * err).pow(gamma)
    loss = scale * err  # Focal-R L1
    if weights is not None:
        loss = loss * weights.expand_as(loss)
    return loss.mean()

def weighted_focal_mse_loss(inputs, targets, weights=None,
                            activate='sigmoid', beta=.2, gamma=1):
    # per-sample squared error
    err = torch.abs(inputs - targets)        # use abs for the scaling term
    se  = (inputs - targets) ** 2            # base regression error
    # focal scaling in [0, 1]
    if activate == 'tanh':
        scale = torch.tanh(beta * err).pow(gamma)
    else:  # 'sigmoid' (matches your description)
        scale = torch.sigmoid(beta * err).pow(gamma)
    loss = scale * se  # Focal-R MSE
    if weights is not None:
        loss = loss * weights.expand_as(loss)
    return loss.mean()

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

def sera_loss(preds, trues, phi_trues, step=0.001, norm=False,
              method="exact", reduction="mean"):
    """
    SERA = ∫_0^1 SER_t dt,  where SER_t = sum_{i: phi(y_i) >= t} (yhat_i - y_i)^2
    Paper approximates this integral with trapezoidal rule on a grid (step=0.001).
    But given phi in [0,1], the integral also has a closed form:
        SERA = sum_i (yhat_i - y_i)^2 * phi_i
    (derivation below).
    method: "exact" or "trapz"
    norm: if True, normalize by SER_{t=0} (= sum squared errors).
    reduction: mean or sum
    """
    # ensure 1D
    preds = preds.view(-1)
    trues = trues.view(-1)
    phi_trues = phi_trues.view(-1).clamp(0.0, 1.0).to(trues.device)
    se = (trues - preds) ** 2  # [B]
    ser0 = se.sum()
    if method == "exact":
        # closed form: sum se_i * ∫_0^1 1(phi_i >= t) dt = sum se_i * phi_i
        sera = (se * phi_trues).sum()
    elif method == "trapz":
        # trapezoidal rule on uniform grid in [0,1]
        # paper uses step=0.001 => T=1000, and SERA ≈ (1/T) * [0.5 SER_0 + Σ SER_k + 0.5 SER_T]
        T = int(round(1.0 / step))
        step_eff = 1.0 / T  # enforce consistent step so 1/T matches grid
        th = torch.linspace(0.0, 1.0, T + 1, device=trues.device)  # [T+1]
        # SER at each threshold: SER_tk = sum_{i: phi_i >= th_k} se_i
        # mask shape [T+1, B]
        mask = (phi_trues.unsqueeze(0) >= th.unsqueeze(1))
        ser = torch.where(mask, se.unsqueeze(0), torch.zeros_like(se).unsqueeze(0)).sum(dim=1)  # [T+1]
        # trapezoid: step * (0.5*ser[0] + ser[1:-1].sum() + 0.5*ser[-1])
        sera = step_eff * (0.5 * ser[0] + ser[1:-1].sum() + 0.5 * ser[-1])
    else:
        raise ValueError("method must be 'exact' or 'trapz'")
    if norm and ser0.item() != 0.0:
        sera = sera / ser0
    if reduction == "mean":
        sera = sera / se.numel()
    elif reduction != "sum":
        raise ValueError("reduction must be 'mean' or 'sum'")
    return sera

def sera_trapezoidal_loss(preds, trues, phi_trues, step=0.001, norm=False):
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