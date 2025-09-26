# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 12:20:00 2025
"""
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.ndimage import convolve1d

from src.utils.relevance_scores import phi_control, phi
from src.utils.LDS import get_lds_kernel_window


def get_train_params_PT(cfg):
	max_epochs = cfg['PT']['max_epochs'] if 'PT' in cfg else 200
	early_stop = cfg['PT']['early_stop'] if 'PT' in cfg else True
	patience = cfg['PT']['patience'] if 'PT' in cfg else 30
	min_delta = cfg['PT']['min_delta'] if 'PT' in cfg else 0
	return (max_epochs, early_stop, patience, min_delta)


class PT_dataset(Dataset):
	def __init__(self, Xtok, Xtime, y, args=None, weights=None,
				 labels_trainval=None, trainval_weights=None):
		assert Xtok.shape[0] == y.shape[0] == Xtime.shape[0]
		self.Xtok = Xtok
		self.Xtime = Xtime
		self.y = y
		if weights is not None:
			self.weights = weights
		elif labels_trainval is not None and trainval_weights is not None:
			y_np = self.y.cpu().numpy()
			self.weights = np.array([
				trainval_weights[np.argmin(np.abs(labels_trainval - val))]
				for val in y_np
			], dtype=np.float32)
			self.weights *= len(self.weights) / self.weights.sum()
		else:
			self.weights = self._prepare_weights(args)

	def __len__(self):
		return self.y.shape[0]

	def __getitem__(self, idx):
		tokens = self.Xtok[idx]
		timef = self.Xtime[idx]
		y = self.y[idx]
		w = torch.tensor(self.weights[idx], dtype=torch.float32) if self.weights is not None else torch.tensor(1.0, dtype=torch.float32)
		return (tokens, timef), y, w

	def _prepare_weights(self, args):
		labels = self.y.cpu().numpy().tolist()
		max_target = int(max(labels)) + 1 if len(labels) else 1
		value_dict = {x: 0 for x in range(max_target)}
		for label in labels:
			value_dict[min(max_target - 1, int(label))] += 1
		if args.reweight == 'sqrt_inv':
			value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
		elif args.reweight == 'inverse':
			value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}
		num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
		if not len(num_per_label) or args.reweight == 'none':
			weights = np.ones(len(labels), dtype=np.float32)
			return weights
		if args.LDS:
			lds_kernel_window = get_lds_kernel_window(args.lds_kernel, args.lds_ks, args.lds_sigma)
			smoothed_value = convolve1d(
				np.asarray([v for _, v in value_dict.items()]),
				weights=lds_kernel_window, mode='constant')
			num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]
		weights = np.array([1.0 / x for x in num_per_label], dtype=np.float32)
		scaling = len(weights) / weights.sum()
		return weights * scaling


def load_PT_data(args, cfg):
	# load tensors
	Xtok_train = torch.load(args.Xtok_train_path, weights_only=True)
	Xtok_val = torch.load(args.Xtok_val_path, weights_only=True)
	Xtok_test = torch.load(args.Xtok_test_path, weights_only=True)
	Xtime_train = torch.load(args.Xtime_train_path, weights_only=True)
	Xtime_val = torch.load(args.Xtime_val_path, weights_only=True)
	Xtime_test = torch.load(args.Xtime_test_path, weights_only=True)
	y_train = torch.load(args.y_train_path, weights_only=True)
	y_val = torch.load(args.y_val_path, weights_only=True)
	y_test = torch.load(args.y_test_path, weights_only=True)
	# relevance scores
	y_train_val = torch.cat([y_train, y_val], dim=0)
	ph = phi_control(y_train_val, extr_type=args.extreme_type, asym=args.asym)
	relevance_train_val = phi(y_train_val, ph)
	relevance_train = relevance_train_val[:len(y_train)]
	relevance_val = relevance_train_val[len(y_train):]
	relevance_test = phi(y_test, ph)
	if args.IR == 'SERA':
		train_dataset = PT_dataset(Xtok_train, Xtime_train, y_train, weights=relevance_train)
		val_dataset = PT_dataset(Xtok_val, Xtime_val, y_val, weights=relevance_val)
		test_dataset = PT_dataset(Xtok_test, Xtime_test, y_test, weights=relevance_test)
		train_val_dataset = PT_dataset(
			Xtok=torch.cat([Xtok_train, Xtok_val], dim=0),
			Xtime=torch.cat([Xtime_train, Xtime_val], dim=0),
			y=y_train_val, weights=relevance_train_val)
	else:
		train_val_dataset = PT_dataset(
			Xtok=torch.cat([Xtok_train, Xtok_val], dim=0),
			Xtime=torch.cat([Xtime_train, Xtime_val], dim=0),
			y=y_train_val, args=args)
		trainval_weights = train_val_dataset.weights
		labels_trainval = y_train_val.cpu().numpy()
		train_weights = trainval_weights[:len(y_train)]
		val_weights = trainval_weights[len(y_train):]
		train_dataset = PT_dataset(Xtok_train, Xtime_train, y_train, weights=train_weights)
		val_dataset = PT_dataset(Xtok_val, Xtime_val, y_val, weights=val_weights)
		test_dataset = PT_dataset(
			Xtok_test, Xtime_test, y_test,
			labels_trainval=labels_trainval,
			trainval_weights=trainval_weights)
	batch_size = cfg['PT']['batch_size'] if 'PT' in cfg else 64
	test_batch_size = cfg['PT']['test_batch_size'] if 'PT' in cfg else 1024
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
	with open(args.test_length_path, 'rb') as f:
		test_lengths = pickle.load(f)
	with open(args.test_cases_path, 'rb') as f:
		test_cases = pickle.load(f)
	return (train_loader, val_loader, test_loader, test_lengths, test_cases, relevance_test)


