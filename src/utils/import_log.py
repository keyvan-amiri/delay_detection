# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 13:32:43 2025
"""
import os
import pandas as pd
import pm4py
from dataclasses import dataclass, field, fields
from typing import Optional

@dataclass
class EventLogIDs:
    # General
    case: str = "case"  # Case ID
    activity: str = "activity"  # Activity label
    # Resource who performed this activity instance
    resource: str = "resource" 
    # Start time of the activity instance
    start_time: str = "start_time"
    # End time of the activity instance
    end_time: str = "end_time"
    # life_cycle transition
    transition: str = "lifecycle:transition"
    # Enablement time of the activity instance
    enabled_time: str = "enabled_time" 
    # Label of the activity instance enabling the current one
    enabling_activity: str = "enabling_activity"
    # Last availability time of the resource who performed this activity instance
    available_time: str = ("available_time")
    # Estimated start time of the activity instance
    estimated_start_time: str = "estimated_start_time" 
    # ID of the batch instance this activity instance belongs to, if any
    batch_id: str = "batch_instance_id"  
    # Type of the batch instance this activity instance belongs to, if any
    batch_type: str = "batch_instance_type"  
    event_num_features: list[str] = field(default_factory=list)
    event_cat_features: list[str] = field(default_factory=list)
    case_num_features: list[str] = field(default_factory=list)
    case_cat_features: list[str] = field(default_factory=list)    

    @staticmethod
    def from_dict(config: dict) -> "EventLogIDs":
        return EventLogIDs(**config)

    def to_dict(self) -> dict:
        return {attr.name: getattr(self, attr.name) for attr in fields(self.__class__)}
    
def get_event_log(args, cfg):
    data_path = cfg['data']['path']
    log_format = cfg['data']['format']
    num_TS = cfg['data']['num_TS']
    use_data = cfg['data']['use_data_attributes']
    case_col = cfg['data']['case_col']
    act_col = cfg['data']['act_col']
    res_col = cfg['data']['res_col']
    start_col = cfg['data']['start_time']
    end_col = cfg['data']['end_time']
    trans_col = cfg['data']['trans_col']
    if use_data:
        case_num_cols = cfg['data']['case_num_feat']
        case_cat_cols = cfg['data']['case_cat_feat']
        event_num_cols = cfg['data']['event_num_feat']
        event_cat_cols = cfg['data']['event_cat_feat'] 
    else:
        case_num_cols, case_cat_cols = [], []
        event_num_cols, event_cat_cols = [], []
    if num_TS > 1:
        log_ids = EventLogIDs(
            case=case_col, activity=act_col, resource=res_col,
            start_time=start_col, end_time=end_col, transition=trans_col,            
            event_num_features=event_num_cols, event_cat_features=event_cat_cols,
            case_num_features=case_num_cols, case_cat_features=case_cat_cols)
    else:
        log_ids = EventLogIDs(
            case=case_col, activity=act_col, resource=res_col,
            end_time=end_col, transition=trans_col,               
            event_num_features=event_num_cols, event_cat_features=event_cat_cols,
            case_num_features=case_num_cols, case_cat_features=case_cat_cols)   
    log = read_event_log(
        os.path.join(data_path, args.dataset + '.' + log_format),
        log_ids, ext=log_format) 
    return log, log_ids

def read_event_log(
    log_path,
    log_ids: EventLogIDs,
    missing_resource: Optional[str] = "NOT_SET",
    sort=True,
    ext: str = "csv"
) -> pd.DataFrame:
    """
    Read an event log from a CSV/XES file given the column IDs in [log_ids].
    Set the enabled_time, start_time, and end_time columns to date,
    set the NA resource cells to [missing_value] if not None, and 
    sort by [end, start, enabled].

    :param log_path: path to the CSV/XES log file.
    :param log_ids: IDs of the columns of the event log.
    :param missing_resource: string to set as NA value for the resource column (not set if None).
    :param sort: if true, sort event log by start, end, enabled (if available).

    :return: the read event log,
    """
    # Read log
    if ext == 'csv':
        event_log = pd.read_csv(log_path)
    elif ext == 'xes':
        event_log = pm4py.read_xes(log_path)
    else:
        raise ValueError(f"Invalid extension type for an event log: {ext}.")
    # Set case id as object
    event_log = event_log.astype({log_ids.case: object})
    # remove artificial start and end activities
    event_log = event_log[~event_log[log_ids.activity].str.lower().isin(['start', 'end'])]
    # Fix missing resources (don't do it if [missing_resources] is set to None)
    if missing_resource:
        if log_ids.resource not in event_log.columns:
            event_log[log_ids.resource] = missing_resource
        else:
            event_log[log_ids.resource] = event_log[log_ids.resource].fillna(missing_resource)
    # Set resource type to string if numeric
    if log_ids.resource in event_log.columns:
        event_log[log_ids.resource] = event_log[log_ids.resource].apply(str)
    # Convert timestamp value to pd.Timestamp (setting timezone to UTC)
    event_log[log_ids.end_time] = pd.to_datetime(event_log[log_ids.end_time], utc=True, errors="coerce")
    failed_ts = event_log[log_ids.end_time].isna().sum()
    if failed_ts > 0:
        print(f'Number of failures for importing end timestamps: {failed_ts}')
    if log_ids.start_time in event_log.columns:
        event_log[log_ids.start_time] = pd.to_datetime(event_log[log_ids.start_time], utc=True, errors="coerce")
        failed_ts = event_log[log_ids.start_time].isna().sum()
        if failed_ts > 0:
            print(f'Number of failures for importing start timestamps: {failed_ts}')
    if log_ids.enabled_time in event_log.columns:
        event_log[log_ids.enabled_time] = pd.to_datetime(event_log[log_ids.enabled_time], utc=True, errors="coerce")
        failed_ts = event_log[log_ids.enabled_time].isna().sum()
        if failed_ts > 0:
            print(f'Number of failures for importing enabled timestamps: {failed_ts}')
    # Sort by end time
    if sort:
        if log_ids.start_time in event_log.columns and log_ids.enabled_time in event_log.columns:
            event_log = event_log.sort_values([log_ids.start_time, log_ids.end_time, log_ids.enabled_time])
        elif log_ids.start_time in event_log.columns:
            event_log = event_log.sort_values([log_ids.start_time, log_ids.end_time])
        else:
            event_log = event_log.sort_values(log_ids.end_time)
    # Return parsed event log
    return event_log