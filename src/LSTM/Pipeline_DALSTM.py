# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 12:37:06 2025
@author: Keyvan Amiri Elyasi
"""
import os
import pickle
import torch
from torch.utils.data import DataLoader

from src.LSTM.dataset_class import DALSTM_dataset
from src.LSTM.model_DALSTM import get_DALSTM_model
from src.LSTM.Train_DALSTM import train_model
from src.LSTM.Test_DALSTM import test_model
from src.utils.loss_functions import set_loss
from src.utils.relevance_scores import phi_control, phi

class DALSTM_train_evaluate ():
    def __init__ (self, args, cfg): 
        self.args = args
        self.cfg = cfg
        # set device
        self.device = f'cuda:{os.environ.get("CUDA_VISIBLE_DEVICES", "0")}' if torch.cuda.is_available() else 'cpu'
        # load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        y_train_val = torch.cat([y_train, y_val], dim=0)
        # compute relevance scores for SERA
        ph = phi_control(y_train_val, extr_type=args.extreme_type, asym=args.asym)
        self.relevance_train_val = phi(y_train_val, ph)
        self.relevance_train = self.relevance_train_val[:len(y_train)]
        self.relevance_val = self.relevance_train_val[len(y_train):]
        self.relevance_test = phi(y_test, ph)
        if self.args.sera:
            train_dataset = DALSTM_dataset(X_train, y_train,
                                           weights=self.relevance_train)
            val_dataset = DALSTM_dataset(X_val, y_val,
                                         weights=self.relevance_val)
            test_dataset = DALSTM_dataset(X_test, y_test,
                                          weights=self.relevance_test) 
            train_val_dataset = DALSTM_dataset(
                X=torch.cat([X_train, X_val], dim=0),
                y=y_train_val, 
                weights=self.relevance_train_val)
        else:            
            # compute weights for train+val
            train_val_dataset = DALSTM_dataset(
                X=torch.cat([X_train, X_val], dim=0),
                y=y_train_val, 
                args=args)
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
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False)   
        self.test_lengths, self.test_cases = self.load_test_lenght_and_ids()
        # get training parameters
        self.max_epochs = cfg['DALSTM']['max_epochs'] or 200
        self.early_stop = cfg['DALSTM']['early_stop']
        if self.early_stop is None:
            self.early_stop = True
        self.patience = cfg['DALSTM']['patience'] or 50
        self.min_delta = cfg['DALSTM']['min_delta'] or 0        
        # define loss function
        self.criterion = set_loss(self.args)
        # define model, and FDS configuration
        self.fds_config, self.model = get_DALSTM_model(
            self.args, self.cfg, self.device)
   
    def train(self, seed, exp_id=1, logger=None):
        train_model(
            self.args, self.cfg, self.model, self.train_loader, self.val_loader,
            self.criterion, num_epochs=self.max_epochs,
            early_stop=self.early_stop, early_patience=self.patience,
            min_delta=self.min_delta, fds_config=self.fds_config,
            device=self.device, seed=seed, exp_id=exp_id, logger=logger)      
        
    def inference(self, seed, exp_id=1, val_mode=False, logger=None): 
        if val_mode:
            inference_loader = self.val_loader
        else:
            inference_loader = self.test_loader
        results = test_model(
            self.args, model=self.model, inference_loader=inference_loader,
            test_original_lengths=self.test_lengths, test_cases=self.test_cases,
            val_mode=val_mode, seed=seed, device=self.device,
            exp_id=exp_id, logger=logger)
        return results

    def load_data(self):
        X_train = torch.load(self.args.X_train_path, weights_only=True)
        X_val = torch.load(self.args.X_val_path, weights_only=True)
        X_test = torch.load(self.args.X_test_path, weights_only=True)
        y_train = torch.load(self.args.y_train_path, weights_only=True)
        y_val = torch.load(self.args.y_val_path, weights_only=True)
        y_test = torch.load(self.args.y_test_path, weights_only=True)
        return X_train, X_val, X_test, y_train, y_val, y_test
   
    def load_test_lenght_and_ids(self):
        with open(self.args.test_length_path, 'rb') as f:
            test_lengths  =  pickle.load(f)
        with open(self.args.test_cases_path, 'rb') as f:
            test_cases  =  pickle.load(f)            
        return test_lengths, test_cases