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
from src.LSTM.model_DALSTM import DALSTMModel, DALSTMModelMve, DALSTMFDSModel
from src.LSTM.Train_DALSTM import train_model
from src.LSTM.Test_DALSTM import test_model
from src.utils.optimizer import get_opt_schedule
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
        batch_size = cfg['DALSTM']['batch_size'] or self.max_len
        test_batch_size = cfg['DALSTM']['test_batch_size'] or self.max_len        
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False)
        self.train_val_loader = DataLoader(
            train_val_dataset, batch_size=batch_size, shuffle=True)         
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
        # define model
        input_size, max_len = self.load_dimensions()
        n_layers = cfg['DALSTM']['n_layers'] or 2
        hidden_size = cfg['DALSTM']['hidden_size'] or 150
        dropout = cfg['DALSTM']['dropout']
        if dropout is None:
            dropout = True
        dropout_prob = cfg['DALSTM']['dropout_prob'] or 0.1
        # FDS configuration
        self.config= dict(
            feature_dim=hidden_size, start_update=args.fds_start_update,
            start_smooth=args.fds_start_smooth, kernel=args.fds_kernel,
            ks=args.fds_ks, sigma=args.fds_sigma)
        if args.heteroscedastic or args.bmse:
            self.model = DALSTMModelMve(
                input_size=input_size, hidden_size=hidden_size, 
                n_layers=n_layers, max_len=max_len, dropout=dropout, 
                p_fix=dropout_prob).to(self.device) 
        elif args.FDS:
            self.model = DALSTMFDSModel(
                input_size=input_size, hidden_size=hidden_size,
                n_layers=n_layers, max_len=max_len, dropout=dropout,
                p_fix=dropout_prob, **self.config).to(self.device)               
        else:
            self.model = DALSTMModel(
                input_size=input_size, hidden_size=hidden_size, 
                n_layers=n_layers, max_len=max_len, dropout=dropout, 
                p_fix=dropout_prob).to(self.device) 
    
    def train(self, seed, exp_id=1, logger=None):
        # define optimizer and scheduler
        optimizer, scheduler = get_opt_schedule(self.args, self.cfg, self.model)
        # train
        train_model(
            self.args, self.cfg, model=self.model, 
            train_loader=self.train_loader, val_loader=self.val_loader,
            train_val_loader=self.train_val_loader,
            criterion=self.criterion, optimizer=optimizer,
            scheduler=scheduler, device=self.device, num_epochs=self.max_epochs,
            early_stop=self.early_stop, early_patience=self.patience,
            min_delta=self.min_delta, fds_config=self.config,
            seed=seed, exp_id=exp_id, logger=logger)
        
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

    def load_dimensions(self):   
        # input_size corresponds to vocab_size
        with open(self.args.input_size_path, 'rb') as f:
            input_size =  pickle.load(f)
        with open(self.args.max_len_path, 'rb') as f:
            max_len =  pickle.load(f) 
        return input_size, max_len
    
    def load_test_lenght_and_ids(self):
        with open(self.args.test_length_path, 'rb') as f:
            test_lengths  =  pickle.load(f)
        with open(self.args.test_cases_path, 'rb') as f:
            test_cases  =  pickle.load(f)            
        return test_lengths, test_cases