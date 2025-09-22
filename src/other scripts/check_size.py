# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 12:05:13 2025

@author: kamirel
"""
import os
import argparse
import pickle
import torch

def add_DALSTM_arguments(args):
    args.process_path = os.path.join(args.root_path, 'temp', args.dataset)
    args.result_path = os.path.join(args.root_path, 'results', args.dataset)
    args.X_train_path = os.path.join(
        args.process_path, "DALSTM_X_train_"+args.dataset+".pt")
    args.X_val_path = os.path.join(
        args.process_path, "DALSTM_X_val_"+args.dataset+".pt")
    args.X_test_path = os.path.join(
        args.process_path, "DALSTM_X_test_"+args.dataset+".pt")
    args.y_train_path = os.path.join(
        args.process_path, "DALSTM_y_train_"+args.dataset+".pt")
    args.y_val_path = os.path.join(
        args.process_path, "DALSTM_y_val_"+args.dataset+".pt")
    args.y_test_path = os.path.join(
        args.process_path, "DALSTM_y_test_"+args.dataset+".pt") 
    args.test_length_path = os.path.join(
        args.process_path, "DALSTM_test_length_list_"+args.dataset+".pkl")  
    args.test_cases_path = os.path.join(
        args.process_path, "DALSTM_test_cases_"+args.dataset+".pkl")
    args.input_size_path = os.path.join(
        args.process_path, "DALSTM_input_size_"+args.dataset+".pkl")
    args.max_len_path = os.path.join(
        args.process_path, "DALSTM_max_len_"+args.dataset+".pkl")  
    args.delay_plot_path = os.path.join(
        args.result_path, "DALSTM_delay_plot_"+args.dataset+".pdf") 
    args.dist_plot_path = os.path.join(
        args.result_path, "DALSTM_distribution_plot_"+args.dataset+".pdf") 
    return args

def main():
    parser = argparse.ArgumentParser(
        description='Imbalanced Regression for Remaining Time Prediction')
    parser.add_argument('--dataset', type=str, default='BPIC20PTC')
    args = parser.parse_args()
    args.root_path = os.getcwd()
    args = add_DALSTM_arguments(args)
    X_train = torch.load(args.X_train_path, weights_only=True)
    X_val = torch.load(args.X_val_path, weights_only=True)
    X_test = torch.load(args.X_test_path, weights_only=True)
    y_train = torch.load(args.y_train_path, weights_only=True)
    y_val = torch.load(args.y_val_path, weights_only=True)
    y_test = torch.load(args.y_test_path, weights_only=True)
    with open(args.input_size_path, 'rb') as f:
        input_size =  pickle.load(f)
    with open(args.max_len_path, 'rb') as f:
        max_len =  pickle.load(f) 
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)
    print(input_size)
    print(max_len)
    
if __name__ == '__main__':
    main()  
    
