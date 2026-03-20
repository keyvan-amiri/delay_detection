import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_loss(args, loss_func=None):
    if loss_func is None: 
        loss_func = args.loss
    if loss_func == 'bmse':
        return bmc_loss
    elif loss_func == 'sera':
        return sera_loss
    elif loss_func == 'mae':
        return weighted_l1_loss
    elif loss_func == 'mse':
        return weighted_mse_loss
    elif loss_func == 'huber':
        return weighted_huber_loss
    elif loss_func == 'focal_mae':
        beta = getattr(args, "focal_beta", 0.2)
        gamma = getattr(args, "focal_gamma", 1.0)
        return lambda inputs, targets, weights=None: weighted_focal_l1_loss(
            inputs, targets, weights=weights, activate='sigmoid', beta=beta, gamma=gamma
        )
    elif loss_func == 'focal_mse':
        beta = getattr(args, "focal_beta", 0.2)
        gamma = getattr(args, "focal_gamma", 1.0)
        return lambda inputs, targets, weights=None: weighted_focal_mse_loss(
            inputs, targets, weights=weights, activate='sigmoid', beta=beta, gamma=gamma
        )
    elif loss_func == 'quantile':
        return quantile_loss
    if args.heteroscedastic:
        return heteroscedastic_loss(metric=loss_func)
    
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

def weighted_focal_l1_loss(
        inputs, targets, weights=None, activate='sigmoid', beta=0.2, gamma=1,
        use_relative=False, rel_eps=1e-6):
    # per-sample absolute error
    abs_err = torch.abs(inputs - targets)
    if use_relative:
        scale_err = abs_err / (torch.abs(targets) + rel_eps)
    else:
        scale_err = abs_err
    # focal scaling
    if activate == 'tanh':
        scale = torch.tanh(beta * scale_err).pow(gamma)
    else:
        scale = torch.sigmoid(beta * scale_err).pow(gamma)
    # base loss stays absolute L1
    loss = scale * abs_err
    if weights is not None:
        loss = loss * weights.expand_as(loss)
    return loss.mean()

def weighted_focal_mse_loss(
        inputs, targets, weights=None, activate='sigmoid', beta=0.2, gamma=1,
        use_relative=False, rel_eps=1e-6):
    # per-sample squared error
    abs_err = torch.abs(inputs - targets)
    se = (inputs - targets) ** 2
    if use_relative:
        scale_err = abs_err / (torch.abs(targets) + rel_eps)
    else:
        scale_err = abs_err
    # focal scaling
    if activate == 'tanh':
        scale = torch.tanh(beta * scale_err).pow(gamma)
    else:
        scale = torch.sigmoid(beta * scale_err).pow(gamma)
    loss = scale * se
    if weights is not None:
        loss = loss * weights.expand_as(loss)
    return loss.mean()


def weighted_huber_loss(inputs, targets, weights=None, beta=1.0):
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
    # ensure scalar + stable
    noise_var = noise_var.clamp(min=1e-6)
    logits = -0.5 * (pred - target.T).pow(2) / noise_var
    labels = torch.arange(pred.shape[0], device=logits.device, dtype=torch.long)
    loss = F.cross_entropy(logits, labels)
    # scalar τ → no mean()
    loss = loss * (2.0 * noise_var)
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


def quantile_pinball_loss(y_true: torch.Tensor,
                          y_pred: torch.Tensor,
                          quantiles,
                          sample_weight: torch.Tensor = None,
                          reduction: str = "mean") -> torch.Tensor:
    """
    y_true: (B,) or (B,1)
    y_pred: (B,K)
    quantiles: list/tuple of length K
    sample_weight: (B,) optional 
    """
    if y_true.dim() == 2 and y_true.size(1) == 1:
        y_true = y_true.squeeze(1)
    y_true = y_true.unsqueeze(1)  # (B,1)
    q = torch.as_tensor(quantiles, device=y_pred.device, dtype=y_pred.dtype).view(1, -1)  # (1,K)
    e = y_true - y_pred  # (B,K)
    loss = torch.maximum(q * e, (q - 1.0) * e)  # (B,K)
    if sample_weight is not None:
        sw = sample_weight.view(-1, 1).to(loss.dtype)
        loss = loss * sw
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    
def quantile_loss(
        y_true, y_pred, quantiles = (0.1, 0.5, 0.6, 0.9, 0.95, 0.99),
        q_weights = torch.tensor([1.0, 1.0, 1.2, 2.0, 3.0, 4.0]),
        sample_weight=None):
    base = quantile_pinball_loss(y_true, y_pred, quantiles,
                                 sample_weight=sample_weight, reduction="none")  # (B,K)
    qw = q_weights.to(base.device, base.dtype).view(1, -1)
    return (base * qw).mean()

def discretize_time_targets(y, bin_edges):
    """
    Map continuous targets y to bin indices in [0, num_bins-1].
    bin_edges shape: [num_bins + 1]
    """
    y = y.view(-1)
    edges = bin_edges.to(y.device)
    # torch.bucketize returns index in [0, len(edges)]
    bin_idx = torch.bucketize(y, edges[1:-1], right=False)
    return bin_idx.long()

 # TODO: remove the following funcitons if everything works!
def survival_hazard_nll(logits, y, bin_edges, reduction='mean', eps = 1e-6):
    """
    logits: [B, K] hazard logits
    y: [B] continuous remaining time
    bin_edges: [K+1]
    """    
    bin_idx = discretize_time_targets(y, bin_edges)   # [B]
    hazards = torch.sigmoid(logits).clamp(min=eps, max=1 - eps)   # [B, K]
    B, K = hazards.shape
    arange_k = torch.arange(K, device=hazards.device).unsqueeze(0)  # [1, K]
    j = bin_idx.unsqueeze(1)                                         # [B, 1]
    survive_mask = (arange_k < j).float()
    event_mask = (arange_k == j).float()
    loglik = survive_mask * torch.log(1.0 - hazards) + event_mask * torch.log(hazards)
    nll = -loglik.sum(dim=1)
    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    return nll

def hazard_logits_to_event_probs(logits):
    """
    Convert hazard logits [B, K] to event probabilities over bins [B, K].
    """
    eps = 1e-8
    hazards = torch.sigmoid(logits).clamp(min=eps, max=1 - eps)  # [B, K]
    surv_prev = torch.cumprod(
        torch.cat([torch.ones(hazards.size(0), 1, device=hazards.device),
                   1.0 - hazards[:, :-1]], dim=1),
        dim=1
    )
    event_probs = surv_prev * hazards
    return event_probs

def event_probs_to_time(event_probs, bin_edges, pred_type='mean'):
    """
    Convert event probability mass over bins to scalar remaining-time prediction.
    """
    edges = bin_edges.to(event_probs.device)
    mids = 0.5 * (edges[:-1] + edges[1:])  # [K]
    if pred_type == 'mean':
        return (event_probs * mids.unsqueeze(0)).sum(dim=1)
    elif pred_type == 'median':
        cdf = torch.cumsum(event_probs, dim=1)
        med_bin = (cdf >= 0.5).float().argmax(dim=1)
        return mids[med_bin]
    else:
        raise ValueError(f"Unknown pred_type: {pred_type}")

def hazard_logits_to_survival_summary(logits):
    eps = 1e-8
    hazards = torch.sigmoid(logits).clamp(min=eps, max=1 - eps)  # [B, K]
    surv_prev = torch.cumprod(
        torch.cat([
            torch.ones(hazards.size(0), 1, device=hazards.device),
            1.0 - hazards[:, :-1]
        ], dim=1),
        dim=1
    )  # [B, K]
    event_probs = surv_prev * hazards                     # P(T in bin k)
    cdf = torch.cumsum(event_probs, dim=1)               # cumulative bin mass
    tail_mass = torch.prod(1.0 - hazards, dim=1)         # P(T beyond last bin)
    return hazards, event_probs, cdf, tail_mass

def survival_distribution_stats(event_probs, bin_edges, tail_mass=None):
    edges = bin_edges.to(event_probs.device)
    left = edges[:-1]
    right = edges[1:]
    widths = right - left
    # proxy for the open-ended last bin
    # simplest practical choice: reuse previous width
    right_eff = right.clone()
    if len(widths) > 1:
        right_eff[-1] = left[-1] + widths[-2]
    else:
        right_eff[-1] = right[-1]

    mids = 0.5 * (left + right_eff)  # [K]
    # truncated mean over represented bins
    mean = (event_probs * mids.unsqueeze(0)).sum(dim=1)
    if tail_mass is not None:
        tail_expectation = left[-1] + 2 * widths[-2]
        mean = mean + tail_mass * tail_expectation
    # second moment under uniform-within-bin assumption
    bin_second_moment = (left**2 + left * right_eff + right_eff**2) / 3.0
    second_moment = (event_probs * bin_second_moment.unsqueeze(0)).sum(dim=1)
    var = torch.clamp(second_moment - mean**2, min=0.0)
    std = torch.sqrt(var)
    return mean, std, right_eff

def pmf_to_quantile(event_probs, bin_edges, q):
    edges = bin_edges.to(event_probs.device)
    left = edges[:-1]
    right = edges[1:]
    widths = right - left
    right_eff = right.clone()
    if len(widths) > 1:
        right_eff[-1] = left[-1] + widths[-2]
    else:
        right_eff[-1] = right[-1]
    cdf = torch.cumsum(event_probs, dim=1)
    mask = cdf >= q
    idx = mask.float().argmax(dim=1)
    idx[mask.sum(dim=1) == 0] = event_probs.size(1) - 1
    cdf_prev = torch.zeros(event_probs.size(0), device=event_probs.device)
    nonzero = idx > 0
    cdf_prev[nonzero] = cdf[nonzero, idx[nonzero] - 1]
    pk = event_probs[torch.arange(event_probs.size(0), device=event_probs.device), idx].clamp_min(1e-12)
    l = left[idx]
    r = right_eff[idx]
    frac = ((q - cdf_prev) / pk).clamp(0.0, 1.0)
    return l + frac * (r - l)

def compute_tail_blend_score(results_dict, alpha=0.5, tail_q=0.9):
    y_true = np.asarray(results_dict["GroundTruth"], dtype=np.float64).reshape(-1)
    y_pred = np.asarray(results_dict["Prediction"], dtype=np.float64).reshape(-1)

    abs_err = np.abs(y_true - y_pred)
    mae_all = float(abs_err.mean())

    tail_threshold = np.quantile(y_true, tail_q)
    tail_mask = y_true >= tail_threshold

    if tail_mask.sum() == 0:
        mae_tail = mae_all
    else:
        mae_tail = float(abs_err[tail_mask].mean())

    score = alpha * mae_all + (1.0 - alpha) * mae_tail

    return {
        "score": score,
        "mae_all": mae_all,
        "mae_tail": mae_tail,
        "tail_threshold": float(tail_threshold),
        "tail_count": int(tail_mask.sum())
    }
        
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