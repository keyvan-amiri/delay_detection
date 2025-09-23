# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 10:39:15 2025
@author: Keyvan Amiri Elyasi
"""
import os
import numpy as np
from datetime import datetime
import time
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing import sequence

def check_processed_tensors(args):
    file_list = [args.X_train_path, args.X_val_path, args.X_test_path, 
                 args.y_train_path, args.y_val_path, args.y_test_path,
                 args.test_length_path, args.input_size_path]    
    all_files_exist = all(os.path.exists(f) for f in file_list)
    return all_files_exist

def remove_small_values(X_train, X_val, y_train, y_val, ratio):
    # filter out train, validation prefixes with very small remaining time
    y_combined = np.concatenate((y_train, y_val))
    y_combined_median = np.median(y_combined)
    threshold = ratio * y_combined_median
    train_mask = y_train >= threshold
    val_mask = y_val >= threshold
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]
    return X_train, X_val, y_train, y_val

def pad_arrays(X_train, X_val, X_test, dataset_name):
    # execute padding, and error handling for BPIC13I
    if dataset_name == 'BPI_2013_I':
        X_train = sequence.pad_sequences(X_train, dtype="int16")
        X_test = sequence.pad_sequences(X_test, maxlen=X_train.shape[1], 
                                            dtype="int16")
        X_val = sequence.pad_sequences(X_val, maxlen=X_train.shape[1],
                                           dtype="int16")
    else:
        X_train = sequence.pad_sequences(X_train)
        X_test = sequence.pad_sequences(X_test, maxlen=X_train.shape[1])
        X_val = sequence.pad_sequences(X_val, maxlen=X_train.shape[1])
    return X_train, X_val, X_test
    
def normalize_tensors(X_train, X_val, X_test):
    # normalize input data
    # compute the normalization values only on training set
    max = [0] * len(X_train[0][0])
    for a1 in X_train:
        for s in a1:
            for i in range(len(s)):
                if s[i] > max[i]:
                    max[i] = s[i]
    # normalization for train, validation, and test sets
    for a1 in X_train:
        for s in a1:
            for i in range(len(s)):
                if (max[i] > 0):
                    s[i] = s[i] / max[i]
    for a1 in X_val:
        for s in a1:
            for i in range(len(s)):
                if (max[i] > 0):
                    s[i] = s[i] / max[i]
    for a1 in X_test:
        for s in a1:
            for i in range(len(s)):
                if (max[i] > 0):
                    s[i] = s[i] / max[i]
    return X_train, X_val, X_test      


# Auxiliary method for preprocessing   
def buildOHE(index=None, n=None):
    L = [0] * n
    L[index] = 1
    return L

# A method for DALSTM preprocessing (output: Pytorch tensors for training)
def dalstm_load_dataset(dataframe, prev_values=None, 
                        time_format="%Y-%m-%d %H:%M:%S"):
    
    dataframe = dataframe.replace(r's+', 'empty', regex=True)
    dataframe = dataframe.replace("-", "UNK")
    dataframe = dataframe.fillna(0)
    dataset = dataframe.values
    
    if prev_values is None:
        values = []
        for i in range(dataset.shape[1]):
            try:
                values.append(len(np.unique(dataset[:, i])))  # +1
            except:
                dataset[:, i] = dataset[:, i].astype(str)       
                values.append(len(np.unique(dataset[:, i])))  # +1
        return (None, None, None), values 
    else:
        values = prev_values
        
    datasetTR = dataset
    
    def generate_set(dataset):
        data = []
        # To collect prefix lengths (required for earliness analysis)
        original_lengths = []  
        newdataset = []
        temptarget = []            
        # analyze first dataset line
        caseID = dataset[0][0]
        starttime = datetime.fromtimestamp(
            time.mktime(time.strptime(dataset[0][2], time_format)))
        lastevtime = datetime.fromtimestamp(
            time.mktime(time.strptime(dataset[0][2], time_format)))
        t = time.strptime(dataset[0][2], time_format)
        midnight = datetime.fromtimestamp(
            time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = (
            datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
        n = 1
        temptarget.append(
            datetime.fromtimestamp(time.mktime(
                time.strptime(dataset[0][2], time_format))))
        a = [(datetime.fromtimestamp(
            time.mktime(time.strptime(
                dataset[0][2], time_format))) - starttime).total_seconds()]
        a.append((datetime.fromtimestamp(
            time.mktime(time.strptime(
                dataset[0][2], time_format))) - lastevtime).total_seconds())
        a.append(timesincemidnight)
        a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)
        a.extend(
            buildOHE(
                index=one_hot(dataset[0][1], values[1], split="|")[0],
                n=values[1]))
        field = 3
        for i in dataset[0][3:]:
            if not np.issubdtype(dataframe.dtypes[field], np.number):
                a.extend(
                    buildOHE(
                        index=one_hot(str(i), values[field], split="|")[0],
                        n=values[field]))
                #print(field, values[field])
            else:
                #print('numerical', field)
                a.append(i)
            field += 1
        newdataset.append(a)
        #line_counter = 1
        for line in dataset[1:, :]:
            #print(line_counter)
            case = line[0]
            if case == caseID:
                # continues the current case
                t = time.strptime(line[2], time_format)
                midnight = datetime.fromtimestamp(time.mktime(t)).replace(
                        hour=0, minute=0, second=0, microsecond=0)
                timesincemidnight = (datetime.fromtimestamp(
                        time.mktime(t)) - midnight).total_seconds()
                temptarget.append(datetime.fromtimestamp(
                        time.mktime(time.strptime(line[2], time_format))))
                a = [(datetime.fromtimestamp(
                        time.mktime(time.strptime(
                            line[2], time_format))) - starttime).total_seconds()]
                a.append((datetime.fromtimestamp(
                        time.mktime(time.strptime(
                            line[2], time_format))) - lastevtime).total_seconds())
                a.append(timesincemidnight)
                a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)

                lastevtime = datetime.fromtimestamp(
                        time.mktime(time.strptime(line[2], time_format)))

                a.extend(
                        buildOHE(
                            index=one_hot(line[1], values[1], filters=[],
                                          split="|")[0], n=values[1]))

                field = 3
                for i in line[3:]:
                    if not np.issubdtype(
                                dataframe.dtypes[field], np.number):
                        a.extend(
                                buildOHE(
                                    index= one_hot(str(i), values[field],
                                                   filters=[],split="|")[0],
                                    n=values[field]))
                    else:
                        a.append(i)
                    field += 1
                newdataset.append(a)
                n += 1
                finishtime = datetime.fromtimestamp(
                        time.mktime(time.strptime(line[2], time_format)))
            else:
                caseID = case
                # Exclude prefix of length one: the loop range is changed.
                # +1 not adding last case. target is 0, not interesting. era 1
                #for i in range(2, len(newdataset)): 
                for i in range(1, len(newdataset)): 
                    data.append(newdataset[:i])
                    # Keep track of prefix lengths (earliness analysis)
                    original_lengths.append(i) 
                    # print newdataset[:i]
                newdataset = []
                starttime = datetime.fromtimestamp(
                        time.mktime(time.strptime(line[2], time_format)))
                lastevtime = datetime.fromtimestamp(
                        time.mktime(time.strptime(line[2], time_format)))

                t = time.strptime(line[2], time_format)
                midnight = datetime.fromtimestamp(
                        time.mktime(t)).replace(
                            hour=0, minute=0, second=0, microsecond=0)
                timesincemidnight = (
                        datetime.fromtimestamp(
                            time.mktime(t)) - midnight).total_seconds()

                a = [(datetime.fromtimestamp(
                        time.mktime(time.strptime(
                            line[2], time_format))) - starttime).total_seconds()]
                a.append((datetime.fromtimestamp(
                        time.mktime(time.strptime(
                            line[2], time_format))) - lastevtime).total_seconds())
                a.append(timesincemidnight)
                a.append(datetime.fromtimestamp(time.mktime(t)).weekday() + 1)

                a.extend(
                        buildOHE(
                            index=one_hot(line[1], values[1], split="|")[0],
                            n=values[1]))

                field = 3
                for i in line[3:]:
                    if not np.issubdtype(dataframe.dtypes[field], np.number):
                        a.extend(
                                buildOHE(
                                    index=one_hot(str(i), values[field],
                                                  split="|")[0], n=values[field]))
                    else:
                        a.append(i)
                    field += 1
                newdataset.append(a)
                for i in range(n):  
                    # try-except: error handling of the original implementation.
                    try:
                        temptarget[-(i + 1)] = (
                                finishtime - temptarget[-(i + 1)]).total_seconds()
                    except UnboundLocalError:
                        # Set target value to zero if finishtime is not defined
                        # The effect is negligible as only for one dataset,
                        # this exception is for one time executed
                        print('one error in loading dataset is observed', i, n)
                        temptarget[-(i + 1)] = 0
                # Remove the target attribute for the prefix of length one
                #if n > 1:
                    #temptarget.pop(0-n)
                temptarget.pop()  # remove last element with zero target
                temptarget.append(
                        datetime.fromtimestamp(
                            time.mktime(time.strptime(
                                line[2], time_format))))
                finishtime = datetime.fromtimestamp(
                        time.mktime(time.strptime(line[2], time_format)))

                n = 1
            #line_counter += 1
        # last case
        # To exclude prefix of length 1: the loop range is adjusted.
        # + 1 not adding last event, target is 0 in that case. era 1
        #for i in range(2, len(newdataset)):
        for i in range(1, len(newdataset)):  
            data.append(newdataset[:i])
            original_lengths.append(i) # Keep track of prefix lengths
            # print newdataset[:i]
        for i in range(n):  # era n.
            temptarget[-(i + 1)] = (
                    finishtime - temptarget[-(i + 1)]).total_seconds()
            # print temptarget[-(i + 1)]
        # Remove the target attribute for the prefix of length one
        #if n > 1:
            #temptarget.pop(0-n)
        temptarget.pop()  # remove last element with zero target

        # print temptarget
        print("Generated dataset with n_samples:", len(temptarget))
        assert (len(temptarget) == len(data))
        return data, temptarget, original_lengths 

    return generate_set(datasetTR), values