# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 13:37:26 2025
@author: kamirel
"""
import torch.optim as optim

def get_opt_schedule(args, cfg, model): 
    learning_rate = args.lr
    optimizer_type = cfg[args.model]['optimizer']  
    eps = cfg[args.model]['eps']
    weight_decay = cfg[args.model]['weight_decay']
    # get number of model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    # define optimizer
    optimizer = set_optimizer(
        model, optimizer_type, learning_rate, eps, weight_decay)
    # define scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)    
    print(f'Total model parameters: {total_params}')    
    return optimizer, scheduler  

def set_optimizer (model, optimizer_type, base_lr, eps, weight_decay):
    eps = float(eps) #ensure to having a floating number
    if optimizer_type == 'NAdam':
        optimizer = optim.NAdam(model.parameters(), lr=base_lr, eps=eps,
                                weight_decay=weight_decay)
    elif optimizer_type == 'AdamW':   
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, eps=eps,
                                weight_decay=weight_decay)
    elif optimizer_type == 'Adam':   
        optimizer = optim.Adam(model.parameters(), lr=base_lr, eps=eps,
                               weight_decay=weight_decay) 
    elif optimizer_type == 'RAdam':
        optimizer = optim.RAdam(model.parameters(), lr=base_lr, eps=eps,
                               weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=base_lr,
                              weight_decay=weight_decay)
    else:
        print(f'The optimizer {optimizer_type} is not supported')
    return optimizer