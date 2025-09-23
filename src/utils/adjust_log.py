# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 15:10:37 2025
@author: Keyvan Amiri Elyasi
"""

def align_column_names(df, log_ids, model='LSTM'):
    if model == 'LSTM':
        df = df.rename(columns={log_ids.case: 'caseid'})
        log_ids.case = 'caseid'
        df = df.rename(columns={log_ids.activity: 'task'})
        log_ids.activity = 'task'
        df = df.rename(columns={log_ids.resource: 'user'})
        log_ids.resource = 'user'
        if log_ids.start_time in df.columns:
            df = df.rename(columns={log_ids.start_time: 'start_timestamp'})
            log_ids.start_time = 'start_timestamp'
        df = df.rename(columns={log_ids.end_time: 'end_timestamp'})  
        log_ids.end_time = 'end_timestamp'
        if log_ids.transition in df.columns:
            df = df.rename(columns={log_ids.transition: 'event_type'})  
            log_ids.transition = 'event_type'        
    return df, log_ids



        