# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 09:04:23 2025
@author: Keyvan Amiri Elyasi
"""
import pickle
import torch
from torch.utils.data import DataLoader

from src.LSTM.dataset_class import DALSTM_dataset
from src.utils.relevance_scores import phi_control, phi

def get_train_params(cfg):
    # get training parameters
    max_epochs = cfg['DALSTM']['max_epochs'] or 200
    early_stop = cfg['DALSTM']['early_stop']
    if early_stop is None:
        early_stop = True
    patience = cfg['DALSTM']['patience'] or 30
    min_delta = cfg['DALSTM']['min_delta'] or 0   
    return (max_epochs, early_stop, patience, min_delta)

def load_DALSTM_data(args, cfg):
    # load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args)
    y_train_val = torch.cat([y_train, y_val], dim=0)
    # compute relevance scores for SERA
    ph = phi_control(y_train_val, extr_type=args.extreme_type, asym=args.asym)
    relevance_train_val = phi(y_train_val, ph)
    relevance_train = relevance_train_val[:len(y_train)]
    relevance_val = relevance_train_val[len(y_train):]
    relevance_test = phi(y_test, ph)
    if args.IR == 'SERA':
        train_dataset = DALSTM_dataset(X_train, y_train, weights=relevance_train)
        val_dataset = DALSTM_dataset(X_val, y_val, weights=relevance_val)
        test_dataset = DALSTM_dataset(X_test, y_test, weights=relevance_test) 
        train_val_dataset = DALSTM_dataset(
            X=torch.cat([X_train, X_val], dim=0), y=y_train_val,
            weights=relevance_train_val)
    else:
        # compute weights for train+val
        train_val_dataset = DALSTM_dataset(
            X=torch.cat([X_train, X_val], dim=0),
            y=y_train_val, args=args)
        trainval_weights = train_val_dataset.weights
        labels_trainval = y_train_val.cpu().numpy()
        train_weights = trainval_weights[:len(y_train)]
        val_weights = trainval_weights[len(y_train):]
        # Create train/val/test datasets
        train_dataset = DALSTM_dataset(X_train, y_train, weights=train_weights)
        val_dataset = DALSTM_dataset(X_val, y_val, weights=val_weights)
        test_dataset = DALSTM_dataset(
            X_test, y_test,
            labels_trainval=labels_trainval,
            trainval_weights=trainval_weights)
    batch_size = cfg['DALSTM']['batch_size']
    test_batch_size = cfg['DALSTM']['test_batch_size']  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)   
    test_lengths, test_cases = load_test_lenght_and_ids(args)
    return (train_loader, val_loader, test_loader, test_lengths, test_cases, relevance_test)

def load_data(args):
    X_train = torch.load(args.X_train_path, weights_only=True)
    X_val = torch.load(args.X_val_path, weights_only=True)
    X_test = torch.load(args.X_test_path, weights_only=True)
    y_train = torch.load(args.y_train_path, weights_only=True)
    y_val = torch.load(args.y_val_path, weights_only=True)
    y_test = torch.load(args.y_test_path, weights_only=True)
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_test_lenght_and_ids(args):
    with open(args.test_length_path, 'rb') as f:
        test_lengths  =  pickle.load(f)
    with open(args.test_cases_path, 'rb') as f:
        test_cases  =  pickle.load(f)            
    return test_lengths, test_cases

