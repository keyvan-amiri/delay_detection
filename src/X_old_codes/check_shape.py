# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 12:32:05 2026

@author: kamirel
"""
import os
import numpy as np
import torch

def get_label(root_path, dataset, part='train'):
    if part=='train':
        name = 'DALSTM_y_train_' + dataset + '.pt'
    elif part=='val':
        name = 'DALSTM_y_val_' + dataset + '.pt'
    else:
        name = 'DALSTM_y_test_' + dataset + '.pt'    
    path=os.path.join(root_path, 'temp', 'DALSTM', dataset, name)
    y = torch.load(path, weights_only=True)
    return y

def main():
    root_path = os.getcwd()    
    dataset = 'HelpDesk'
    dataset = 'BPIC15_1'
    dataset = 'BPIC20PTC'    
    y_train = get_label(root_path, dataset, part='train')
    y_val = get_label(root_path, dataset, part='val')
    y_test = get_label(root_path, dataset, part='test')  
    y_train_val = torch.cat((y_train, y_val), dim=0)
    #print(y_test.shape)   
    #print(y_train_val.shape)    
    y = y_train_val.detach().cpu().numpy()
    frac = np.mean(np.abs(y - np.floor(y)) > 1e-6)
    print("fraction non-integer:", frac)
    print("min/max:", y.min(), y.max(), "range:", y.max()-y.min())

    bins = np.floor(y).astype(int)   # matches int(label) behavior for positive labels
    unique, counts = np.unique(bins, return_counts=True)
    print("occupied bins:", len(unique))
    print("bins with count=1:", np.sum(counts==1))
    print("bins with count<=3:", np.sum(counts<=3))
    print("max bin count:", counts.max())
    
    
if __name__ == '__main__':
    main()  