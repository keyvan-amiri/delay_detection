# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 12:10:00 2025
"""
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import time
import torch

from src.utils.data_split import split_cases


class PT_preprocessing():
	def __init__(self, log, log_ids, args, overwrite=False, time_format="%Y-%m-%d %H:%M:%S"):
		self.log = log.copy()
		self.log_ids = log_ids
		self.args = args
		self.time_format = time_format
		files_exist = self._check_files()
		if overwrite or not files_exist:
			self.execute_pipeline()
		else:
			print(f"For '{args.dataset}' PT preprocessing is already done.")

	def _check_files(self):
		file_list = [self.args.Xtok_train_path, self.args.Xtok_val_path, self.args.Xtok_test_path,
					 self.args.Xtime_train_path, self.args.Xtime_val_path, self.args.Xtime_test_path,
					 self.args.y_train_path, self.args.y_val_path, self.args.y_test_path,
					 self.args.test_length_path, self.args.test_cases_path,
					 self.args.vocab_size_path, self.args.max_len_path]
		return all(os.path.exists(f) for f in file_list)

	def execute_pipeline(self):
		# select necessary columns; include start time if available in the log even if not configured
		start_col = self.log_ids.start_time if getattr(self.log_ids, 'start_time', None) else None
		if start_col is None:
			# heuristic: common XES column name
			if 'start_timestamp' in self.log.columns:
				start_col = 'start_timestamp'
			elif 'time:start' in self.log.columns:
				start_col = 'time:start'
		cols = [self.log_ids.case, self.log_ids.activity, self.log_ids.end_time]
		if start_col is not None and start_col in self.log.columns:
			cols.append(start_col)
		pd_log = self.log[cols].copy()
		pd_log[self.log_ids.end_time] = pd.to_datetime(pd_log[self.log_ids.end_time], utc=True, errors='coerce')
		pd_log[self.log_ids.end_time] = pd_log[self.log_ids.end_time].dt.strftime(self.time_format)
		if start_col is not None and start_col in pd_log.columns:
			pd_log[start_col] = pd.to_datetime(pd_log[start_col], utc=True, errors='coerce')
			pd_log[start_col] = pd_log[start_col].dt.strftime(self.time_format)
		# split data by cases (use end timestamp column, with validation)
		_, train_df, val_df, test_df, _ = split_cases(
			pd_log, self.args, self.log_ids,
			time_col=self.log_ids.end_time, validation=True,
			train_ratio=self.args.train_ratio, val_ratio=self.args.val_ratio)
		# build vocabulary on train+val+test (consistent across splits)
		all_acts = pd.concat([train_df, val_df, test_df])[self.log_ids.activity].astype(str).str.lower().str.replace(" ", "-").unique().tolist()
		# index 0 reserved for PAD
		act_to_idx = {act: idx+1 for idx, act in enumerate(all_acts)}
		vocab_size = len(act_to_idx) + 1
		# generate samples (features in SECONDS to match DALSTM logic)
		Xtok_train, Xtime_train, y_train, _, _ = self._generate_set(train_df, act_to_idx, start_col=start_col)
		Xtok_val, Xtime_val, y_val, _, _ = self._generate_set(val_df, act_to_idx, start_col=start_col)
		Xtok_test, Xtime_test, y_test, test_lengths, test_cases = self._generate_set(test_df, act_to_idx, start_col=start_col, keep_lengths=True)
		# Convert targets from seconds to days to match DALSTM
		y_train = (np.asarray(y_train, dtype=np.float32) / (24*3600)).tolist()
		y_val = (np.asarray(y_val, dtype=np.float32) / (24*3600)).tolist()
		y_test = (np.asarray(y_test, dtype=np.float32) / (24*3600)).tolist()
		# normalize time features per-column by train maxima to match DALSTM style
		Xtime_train = np.asarray(Xtime_train, dtype=np.float32)
		Xtime_val = np.asarray(Xtime_val, dtype=np.float32)
		Xtime_test = np.asarray(Xtime_test, dtype=np.float32)
		# compute max per feature on train (avoid zeros)
		time_max = Xtime_train.max(axis=0)
		time_max[time_max == 0] = 1.0
		Xtime_train = (Xtime_train / time_max).astype(np.float32)
		Xtime_val = (Xtime_val / time_max).astype(np.float32)
		Xtime_test = (Xtime_test / time_max).astype(np.float32)
		# pad token sequences to train max length
		max_len = max(max([len(s) for s in Xtok_train]) if len(Xtok_train) else 0,
					 max([len(s) for s in Xtok_val]) if len(Xtok_val) else 0,
					 max([len(s) for s in Xtok_test]) if len(Xtok_test) else 0)
		def pad(seqs, pad_val=0):
			arr = np.full((len(seqs), max_len), pad_val, dtype=np.int64)
			for i, s in enumerate(seqs):
				arr[i, :len(s)] = s
			return arr
		Xtok_train = torch.tensor(pad(Xtok_train), dtype=torch.long)
		Xtok_val = torch.tensor(pad(Xtok_val), dtype=torch.long)
		Xtok_test = torch.tensor(pad(Xtok_test), dtype=torch.long)
		Xtime_train = torch.tensor(Xtime_train, dtype=torch.float32)
		Xtime_val = torch.tensor(Xtime_val, dtype=torch.float32)
		Xtime_test = torch.tensor(Xtime_test, dtype=torch.float32)
		y_train = torch.tensor(np.asarray(y_train, dtype=np.float32), dtype=torch.float32)
		y_val = torch.tensor(np.asarray(y_val, dtype=np.float32), dtype=torch.float32)
		y_test = torch.tensor(np.asarray(y_test, dtype=np.float32), dtype=torch.float32)
		# persist normalization stats for reproducibility
		with open(os.path.join(self.args.process_path, 'PT_time_max.pkl'), 'wb') as f:
			pickle.dump(time_max, f)
		# save tensors and metadata
		torch.save(Xtok_train, self.args.Xtok_train_path)
		torch.save(Xtok_val, self.args.Xtok_val_path)
		torch.save(Xtok_test, self.args.Xtok_test_path)
		torch.save(Xtime_train, self.args.Xtime_train_path)
		torch.save(Xtime_val, self.args.Xtime_val_path)
		torch.save(Xtime_test, self.args.Xtime_test_path)
		torch.save(y_train, self.args.y_train_path)
		torch.save(y_val, self.args.y_val_path)
		torch.save(y_test, self.args.y_test_path)
		with open(self.args.test_length_path, 'wb') as f:
			pickle.dump(test_lengths, f)
		with open(self.args.test_cases_path, 'wb') as f:
			pickle.dump(test_cases, f)
		with open(self.args.vocab_size_path, 'wb') as f:
			pickle.dump(vocab_size, f)
		with open(self.args.max_len_path, 'wb') as f:
			pickle.dump(max_len, f)

	def _generate_set(self, df, act_to_idx, start_col=None, keep_lengths=False):
		case_col = self.log_ids.case
		act_col = self.log_ids.activity
		time_col = self.log_ids.end_time
		processed_tokens, processed_time, targets = [], [], []
		prefix_lengths, case_ids = [], []
		for case_id, group in df.groupby(case_col):
			acts = group[act_col].astype(str).str.lower().str.replace(" ", "-").tolist()
			times = group[time_col].astype(str).str[:19].tolist()
			# No start timestamps in general
			starts = None
			if len(acts) < 2:
				continue
			starttime = datetime.fromtimestamp(time.mktime(time.strptime(times[0], self.time_format)))
			lastevtime = starttime
			time_passed = 0
			for i in range(len(acts)):
				# prefix tokens up to i
				prefix_tokens = [act_to_idx[a] for a in acts[:i+1]]
				# compute time features in SECONDS to match DALSTM
				cur_end = datetime.fromtimestamp(time.mktime(time.strptime(times[i], self.time_format)))
				if i > 0:
					prev_end = datetime.fromtimestamp(time.mktime(time.strptime(times[i-1], self.time_format)))
					latest_time = int((cur_end - prev_end).total_seconds())
					if i > 1:
						prev2_end = datetime.fromtimestamp(time.mktime(time.strptime(times[i-2], self.time_format)))
						recent_time = int((cur_end - prev2_end).total_seconds())
					else:
						recent_time = 0
				else:
					latest_time = 0
					recent_time = 0
				time_passed += latest_time
				# target remaining time to last event, in seconds
				ttc = int((datetime.fromtimestamp(time.mktime(time.strptime(times[-1], self.time_format))) - cur_end).total_seconds())
				processed_tokens.append(prefix_tokens)
				processed_time.append([recent_time, latest_time, time_passed])
				targets.append(ttc)
				if keep_lengths:
					prefix_lengths.append(i+1)
					case_ids.append(case_id)
		return processed_tokens, processed_time, targets, prefix_lengths, case_ids

