from torchvision import transforms
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import math
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import wandb
import random
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time

# MOSI/MOSEI Training
def mosi_label(y_batch):
	res = copy.deepcopy(y_batch)
	res[y_batch >= 0] = 1
	res[y_batch < 0] = 0
	return res

# Sarcasm/Humor Training
def sarcasm_label(y_batch):
	res = copy.deepcopy(y_batch)
	res[y_batch == -1] = 0
	return res

def evaluate_raw_data(config, ds_name='mosi'):
	raw_data = {'train': {}, 'val': {}, 'test': {}}
	label = {'train': {}, 'val': {}, 'test': {}}

	for type in ['train', 'val', 'test']:
		if ds_name == 'mimic':
			raw_data[type] = [[data_[0], data_[1]] for data_ in config[type]] # TODO
		else:
			# data: processed_input, processed_input_lengths, inds, labels
			raw_data[type] = [[data_[0][0], data_[0][2], data_[1][0], data_[1][2]] for data_ in config[type]]
		label[type] = np.concatenate([data_[3].detach().cpu().numpy() for data_ in config[type]])
		if ds_name == 'mosi' or ds_name == 'mosei':
			label[type] = mosi_label(label[type])
		elif ds_name == 'sarcasm' or ds_name == 'humor':
			label[type] = sarcasm_label(label[type])
		else:
			raise NotImplementedError('Dataset not implemented yet')
		label[type] = np.asarray(label[type]).reshape(-1).astype(int)

	# Train Logistic Classifier on raw data X (for baseline)
	if ds_name == 'mimic':
		raw_train_embeds_x = np.concatenate([x.numpy().mean(axis=1) for x,y in raw_data['train']], axis=0)
		raw_val_embeds_x = np.concatenate([x.numpy().mean(axis=1) for x,y in raw_data['val']], axis=0)
		raw_test_embeds_x = np.concatenate([x.numpy().mean(axis=1) for x,y in raw_data['test']], axis=0)
	else:
		raw_train_embeds_x = np.concatenate([x.numpy().mean(axis=1) for x,y,lx,ly in raw_data['train']], axis=0)
		raw_val_embeds_x = np.concatenate([x.numpy().mean(axis=1) for x,y,lx,ly in raw_data['val']], axis=0)
		raw_test_embeds_x = np.concatenate([x.numpy().mean(axis=1) for x,y,lx,ly in raw_data['test']], axis=0)
	clf = make_pipeline(StandardScaler(with_mean=True, with_std=True), LogisticRegression(max_iter=1000, solver='liblinear')) if ds_name == 'mosi' else LogisticRegression(max_iter=200)
	clf.fit(raw_train_embeds_x, label['train'])
	score_x_raw = clf.score(raw_test_embeds_x, label['test'])
	val_score_x_raw = clf.score(raw_val_embeds_x, label['val'])

	# Train Logistic Classifier on raw data Y (for baseline)
	if ds_name == 'mimic':
		raw_train_embeds_y = np.concatenate([y.numpy().mean(axis=1) for x,y in raw_data['train']], axis=0)
		raw_val_embeds_y = np.concatenate([y.numpy().mean(axis=1) for x,y in raw_data['val']], axis=0)
		raw_test_embeds_y = np.concatenate([y.numpy().mean(axis=1) for x,y in raw_data['test']], axis=0)
	else:
		raw_train_embeds_y = np.concatenate([y.numpy().mean(axis=1) for x,y,lx,ly in raw_data['train']], axis=0)
		raw_val_embeds_y = np.concatenate([y.numpy().mean(axis=1) for x,y,lx,ly in raw_data['val']], axis=0)
		raw_test_embeds_y = np.concatenate([y.numpy().mean(axis=1) for x,y,lx,ly in raw_data['test']], axis=0)
	clf = make_pipeline(StandardScaler(with_mean=True, with_std=True), LogisticRegression(max_iter=1000, solver='liblinear')) if ds_name == 'mosi' else LogisticRegression(max_iter=200)
	clf.fit(raw_train_embeds_y, label['train'])
	score_y_raw = clf.score(raw_test_embeds_y, label['test'])
	val_score_y_raw = clf.score(raw_val_embeds_y, label['val'])

	# Train Logistic Classifier on raw data XY together (for baseline)
	train_raw_embeds = np.concatenate([raw_train_embeds_x, raw_train_embeds_y], axis=1)
	val_raw_embeds = np.concatenate([raw_val_embeds_x, raw_val_embeds_y], axis=1)
	test_raw_embeds = np.concatenate([raw_test_embeds_x, raw_test_embeds_y], axis=1)
	clf = make_pipeline(StandardScaler(with_mean=True, with_std=True), LogisticRegression(max_iter=1000, solver='liblinear')) if ds_name == 'mosi' else LogisticRegression(max_iter=200)
	clf.fit(train_raw_embeds, label['train'])
	score_xy_raw = clf.score(test_raw_embeds, label['test'])
	val_score_xy_raw = clf.score(val_raw_embeds, label['val'])

	results = {"val/score_x_raw": val_score_x_raw, "val/score_y_raw": val_score_y_raw,
		"test/score_x_raw": score_x_raw, "test/score_y_raw": score_y_raw, 
		"test/score_xy_raw": score_xy_raw, "val/score_xy_raw": val_score_xy_raw}

	return results

@torch.no_grad()
def evaluate(model, config, ds_name='mosi'):

	def make_classifier(classifier_type, ds_name):
		if classifier_type == 'logistic':
			return make_pipeline(StandardScaler(with_mean=True, with_std=True), LogisticRegression(max_iter=1000, solver='liblinear')) if ds_name == 'mosi' else LogisticRegression(max_iter=200)
		elif classifier_type == 'knn':
			return KNeighborsClassifier()
		else:
			raise ValueError(f"Unsupported classifier type: {classifier_type}")

	separate = False
	embds = {'train': {}, 'val': {}, 'test': {}}
	raw_data = {'train': {}, 'val': {}, 'test': {}}
	results = {}

	modality_separate = []
	model.eval()
	start_time = time.time()
	for type in ['train', 'val', 'test']:
		if ds_name == 'mimic':
			raw_data[type] = [[data_[0], data_[1]] for data_ in config[type]]
		else:
			# data: processed_input, processed_input_lengths, inds, labels
			raw_data[type] = [[data_[0][0], data_[0][2], data_[1][0], data_[1][2]] for data_ in config[type]]
		embds_info = [model(x.cuda(), y.cuda(), x_lengths=lx.cuda(), y_lengths=ly.cuda()) for x,y,lx,ly in raw_data[type]]
		for i in range(len(raw_data[type])):
			lx, ly = raw_data[type][i][2], raw_data[type][i][3]
			mask_x = torch.arange(raw_data[type][i][0].shape[1], device=lx.device).unsqueeze(0) < lx.unsqueeze(1) # [batch_size, seq_len]
			mask_x_expanded = mask_x.unsqueeze(-1).expand_as(embds_info[i]['zx']).float().cuda() # [batch_size, seq_len, zdim]
			zx_i_mean = (embds_info[i]['zx'] * mask_x_expanded).sum(dim=1) / mask_x_expanded.sum(dim=1) # [batch_size, zdim]
			mask_y = torch.arange(raw_data[type][i][1].shape[1], device=ly.device).unsqueeze(0) < ly.unsqueeze(1) # [batch_size, seq_len]
			mask_y_expanded = mask_y.unsqueeze(-1).expand_as(embds_info[i]['zy']).float().cuda() # [batch_size, seq_len, zdim]
			zy_i_mean = (embds_info[i]['zy'] * mask_y_expanded).sum(dim=1) / mask_y_expanded.sum(dim=1) # [batch_size, zdim]
			embds[type].setdefault('x1', []).append(zx_i_mean)
			embds[type].setdefault('x2', []).append(zy_i_mean)
			
		embds[type]['x1'] = np.concatenate([x.detach().cpu().numpy() for x in embds[type]['x1']]) # [num_sample, zdim]
		embds[type]['x2'] = np.concatenate([x.detach().cpu().numpy() for x in embds[type]['x2']]) # [num_sample, zdim]
		# if separate:
		# 	embds[type]['x1_private'] = np.concatenate([emb_info['x_private'].mean(dim=1).detach().cpu().numpy() for emb_info in embds_info])
		# 	embds[type]['x2_private'] = np.concatenate([emb_info['y_private'].mean(dim=1).detach().cpu().numpy() for emb_info in embds_info])
		embds[type]['loss_x1'] = np.array([emb_info['loss_x'].item() for emb_info in embds_info])
		embds[type]['loss_x2'] = np.array([emb_info['loss_y'].item() for emb_info in embds_info])
		embds[type]['labels'] = np.concatenate([data[3].detach().cpu().numpy() for data in config[type]])

		if ds_name == 'mosi' or ds_name == 'mosei':
			embds[type]['labels'] = mosi_label(embds[type]['labels'])
		elif ds_name == 'sarcasm' or ds_name == 'humor':
			embds[type]['labels'] = sarcasm_label(embds[type]['labels'])
		else:
			raise NotImplementedError('Dataset not implemented yet')
		
		embds[type]['labels'] = np.asarray(embds[type]['labels']).reshape(-1).astype(int)

		random_idx = np.random.permutation(embds[type]['x1'].shape[0]+embds[type]['x2'].shape[0])
		embds[type]['x1_x2'] = np.concatenate([embds[type]['x1'], embds[type]['x2']], axis=0)[random_idx]
		embds[type]['x1_x2_labels'] = np.array(list([0]*embds[type]['x1'].shape[0]) + list([1]*embds[type]['x2'].shape[0]))[random_idx]
		clf = make_classifier('logistic', ds_name)
		clf.fit(embds[type]['x1_x2'], embds[type]['x1_x2_labels'])
		modality_separate.append(clf.score(embds[type]['x1_x2'], embds[type]['x1_x2_labels']))
	results[f'val/modality_separate'] = np.array(modality_separate).mean()

	end_time = time.time()
	# print(f"Evaluation embedding extraction time: {end_time-start_time:.4f}s")

	start_time = time.time()

	val_loss_x = np.mean(embds['val']['loss_x1'])
	loss_x = np.mean(embds['test']['loss_x1'])
	val_loss_y = np.mean(embds['val']['loss_x2'])
	loss_y = np.mean(embds['test']['loss_x2'])
	for classifier_type in ['logistic']:
		# Train Classifier on X alone
		clf = make_classifier(classifier_type, ds_name)
		clf.fit(embds['train']['x1'], embds['train']['labels'])
		val_score_x = clf.score(embds['val']['x1'], embds['val']['labels'])
		score_x = clf.score(embds['test']['x1'], embds['test']['labels'])

		# Train Classifier on Y alone
		clf = make_classifier(classifier_type, ds_name)
		clf.fit(embds['train']['x2'], embds['train']['labels'])
		val_score_y = clf.score(embds['val']['x2'], embds['val']['labels'])
		score_y = clf.score(embds['test']['x2'], embds['test']['labels'])
		
		# Train Classifier on XY together
		train_embeds = np.concatenate([embds['train']['x1'], embds['train']['x2']], axis=1)
		val_embeds = np.concatenate([embds['val']['x1'], embds['val']['x2']], axis=1)
		test_embeds = np.concatenate([embds['test']['x1'], embds['test']['x2']], axis=1)
		
		clf = make_classifier(classifier_type, ds_name)
		clf.fit(train_embeds, embds['train']['labels'])
		score_xy = clf.score(test_embeds, embds['test']['labels'])
		val_score_xy = clf.score(val_embeds, embds['val']['labels'])

		if separate:
			# x private
			clf = make_classifier(classifier_type, ds_name)
			clf.fit(embds['train']['x1_private'], embds['train']['labels'])
			val_score_x_private = clf.score(embds['val']['x1_private'], embds['val']['labels'])
			score_x_private = clf.score(embds['test']['x1_private'], embds['test']['labels'])

			# x complete
			clf = make_classifier(classifier_type, ds_name)
			clf.fit(np.concatenate([embds['train']['x1'], embds['train']['x1_private']], axis=1), embds['train']['labels'])
			val_score_x_complete = clf.score(np.concatenate([embds['val']['x1'], embds['val']['x1_private']], axis=1), embds['val']['labels'])
			score_x_complete = clf.score(np.concatenate([embds['test']['x1'], embds['test']['x1_private']], axis=1), embds['test']['labels'])

			# y private
			clf = make_classifier(classifier_type, ds_name)
			clf.fit(embds['train']['x2_private'], embds['train']['labels'])
			val_score_y_private = clf.score(embds['val']['x2_private'], embds['val']['labels'])
			score_y_private = clf.score(embds['test']['x2_private'], embds['test']['labels'])

			# y complete
			clf = make_classifier(classifier_type, ds_name)
			clf.fit(np.concatenate([embds['train']['x2'], embds['train']['x2_private']], axis=1), embds['train']['labels'])
			val_score_y_complete = clf.score(np.concatenate([embds['val']['x2'], embds['val']['x2_private']], axis=1), embds['val']['labels'])
			score_y_complete = clf.score(np.concatenate([embds['test']['x2'], embds['test']['x2_private']], axis=1), embds['test']['labels'])

			# xy complete
			train_embeds_complete = np.concatenate([embds['train']['x1'], embds['train']['x1_private'], embds['train']['x2'], embds['train']['x2_private']], axis=1)
			clf = make_classifier(classifier_type, ds_name)
			clf.fit(train_embeds_complete, embds['train']['labels'])
			val_embeds_complete = np.concatenate([embds['val']['x1'], embds['val']['x1_private'], embds['val']['x2'], embds['val']['x2_private']], axis=1)
			test_embeds_complete = np.concatenate([embds['test']['x1'], embds['test']['x1_private'], embds['test']['x2'], embds['test']['x2_private']], axis=1)
			score_xy_complete = clf.score(test_embeds_complete, embds['test']['labels'])
			val_score_xy_complete = clf.score(val_embeds_complete, embds['val']['labels'])
		else:
			val_score_x_private = score_x_private = torch.nan
			val_score_x_complete = score_x_complete = torch.nan
			val_score_y_private = score_y_private = torch.nan
			val_score_y_complete = score_y_complete = torch.nan
			score_xy_complete = val_score_xy_complete = torch.nan
		
		results.update({f"test/score_x": score_x, f"test/score_y": score_y, f"test/score_xy": score_xy, 
			f"val/score_x": val_score_x, f"val/score_y": val_score_y, f"val/score_xy": val_score_xy, 
			f"test/score_x_private": score_x_private, f"val/score_x_private": val_score_x_private,
			f"test/score_y_private": score_y_private, f"val/score_y_private": val_score_y_private,
			f"test/score_x_complete": score_x_complete, f"val/score_x_complete": val_score_x_complete,
			f"test/score_y_complete": score_y_complete, f"val/score_y_complete": val_score_y_complete,
			f"test/score_xy_complete": score_xy_complete, f"val/score_xy_complete": val_score_xy_complete})

	results.update({'val/loss_x': val_loss_x, 'test/loss_x': loss_x, 'val/loss_y': val_loss_y, 'test/loss_y': loss_y})
	end_time = time.time()

	return results

# spectral
import matplotlib.pyplot as plt
import os
import torch.fft
def analyze_spectral_bias(ground_truth, prediction, loss_value, iteration, modality_name='modality', postfix = ""):
	gt = ground_truth.detach().cpu()
	pred = prediction.detach().cpu()
	fft_gt = torch.fft.rfft(gt, dim=1)
	fft_pred = torch.fft.rfft(pred, dim=1)
	mag_gt = torch.abs(fft_gt).mean(dim=(0, 2))
	mag_pred = torch.abs(fft_pred).mean(dim=(0, 2))

	freqs = torch.arange(mag_gt.shape[0])
	plt.figure(figsize=(10, 6))
	plt.plot(freqs, mag_gt, label='Ground Truth (High Freq Source)', alpha=0.8, color='black')
	plt.plot(freqs, mag_pred, label='Prediction (Transformer Output)', alpha=0.8, color='red', linestyle='--')

	plt.xscale('log')
	plt.yscale('log')
	plt.title(f'Spectral Analysis: {modality_name} (Log-Log Scale)')
	plt.xlabel('Frequency (Log scale)')
	plt.ylabel('Magnitude (Log scale)')
	plt.legend()
	plt.grid(True, which="both", ls="-", alpha=0.2)
	os.makedirs(f'spectral_analysis{postfix}', exist_ok=True)
	plt.savefig(f'spectral_analysis{postfix}/spectral_analysis_{modality_name}_{iteration}_loss_{loss_value:.4f}.png', dpi=300)

def rollout(model, x, y, steps=10):
	# x: (B, 1, D)
	# y: (B, 1, D)
	model.eval()
	with torch.no_grad():
		pred_x, pred_y = None, None
		if x is not None:
			pred_x = [x]
			x_recon_t = x
			for t in range(1,steps+1):
				x_proj_t = model.xproj_in(x_recon_t[:,-1,:].unsqueeze(1))
				zx_t = model.encoder(x_proj_t)
				x_recon_t = model.decoders[0](zx_t)
				pred_x.append(x_recon_t)
			pred_x = torch.cat(pred_x, dim=1)
		if y is not None:
			pred_y = [y]
			y_recon_t = y
			for t in range(1,steps+1):
				y_proj_t = model.yproj_in(y_recon_t[:, -1, :].unsqueeze(1))
				zy_t = model.encoder(y_proj_t)
				y_recon_t = model.decoders[1](zy_t)
				pred_y.append(y_recon_t)
			pred_y = torch.cat(pred_y, dim=1)
	return pred_x, pred_y

from utilis import compute_effective_rank, cka, mknn

def train(model, train_mode, train_loader_1, train_loader_2, optimizer, modalities=[0,2], num_epoch=100, step_k=30, ds_name='mosi', eval_config = {}, alpha_x=1.0, alpha_y=1.0, capture_embeddings_during_training=False, augment=False, debug=False, args=None):
	model.train()
	progress_bar = tqdm(total=num_epoch, desc='Training')

	# modification: for embedding capture
	if capture_embeddings_during_training:
		train_loader_1_clone = copy.deepcopy(train_loader_1)
		train_loader_2_clone = copy.deepcopy(train_loader_2)
		batch_size = train_loader_1.batch_size
		fixed_samples = {'x1': [], 'x2': [], 'lx1': [], 'lx2': []}
		embeddings = {'x1': [], 'x2': [], 'x1_label': [], 'x2_label': []}
		# collect fixed samples for embedding capture
		n_samples = 1000
		for i, (data_batch_1, data_batch_2) in enumerate(zip(train_loader_1_clone, train_loader_2_clone)):
			if ds_name != 'mimic':
				x1_batch = data_batch_1[0][modalities[0]].float() # shape: (B, T, D)
				x2_batch = data_batch_2[0][modalities[1]].float() # shape: (B, T, D)
				x1_batch_lengths = data_batch_1[1][modalities[0]] # shape: (B,)
				x2_batch_lengths = data_batch_2[1][modalities[1]] # shape: (B,)
				fixed_samples['x1'].append(x1_batch[:batch_size if (i+1)*batch_size <= n_samples else n_samples - i*batch_size])
				fixed_samples['x2'].append(x2_batch[:batch_size if (i+1)*batch_size <= n_samples else n_samples - i*batch_size])
				fixed_samples['lx1'].append(x1_batch_lengths[:batch_size if (i+1)*batch_size <= n_samples else n_samples - i*batch_size])
				fixed_samples['lx2'].append(x2_batch_lengths[:batch_size if (i+1)*batch_size <= n_samples else n_samples - i*batch_size])
				embeddings['x1_label'].append(data_batch_1[3][:batch_size if (i+1)*batch_size <= n_samples else n_samples - i*batch_size])
				embeddings['x2_label'].append(data_batch_2[3][:batch_size if (i+1)*batch_size <= n_samples else n_samples - i*batch_size])
			else:
				x1_batch = data_batch_1[0].float()
				x2_batch = data_batch_2[1].float()
				fixed_samples['x1'].append(x1_batch[:batch_size if (i+1)*batch_size <= n_samples else n_samples - i*batch_size])
				fixed_samples['x2'].append(x2_batch[:batch_size if (i+1)*batch_size <= n_samples else n_samples - i*batch_size])
				embeddings['x1_label'].append(data_batch_1[2][:batch_size if (i+1)*batch_size <= n_samples else n_samples - i*batch_size])
				embeddings['x2_label'].append(data_batch_2[2][:batch_size if (i+1)*batch_size <= n_samples else n_samples - i*batch_size])
			if (i+1)*batch_size >= n_samples:
				break
		embeddings['x1_label'] = torch.cat(embeddings['x1_label'], dim=0)
		embeddings['x2_label'] = torch.cat(embeddings['x2_label'], dim=0)
		
		# deal with variable-length sequences
		flattened_fixed_samples_x1 = []
		for i in range(len(fixed_samples['x1'])):
			for j in range(fixed_samples['x1'][i].shape[0]):
				x1_len = fixed_samples['lx1'][i][j].item()
				flattened_fixed_samples_x1.append(fixed_samples['x1'][i][j,:x1_len,:])
		flattened_fixed_samples_x1 = torch.cat(flattened_fixed_samples_x1, dim = 0)
		flattened_fixed_samples_x2 = []
		for i in range(len(fixed_samples['x2'])):
			for j in range(fixed_samples['x2'][i].shape[0]):
				x2_len = fixed_samples['lx2'][i][j].item()
				flattened_fixed_samples_x2.append(fixed_samples['x2'][i][j,:x2_len,:])
		flattened_fixed_samples_x2 = torch.cat(flattened_fixed_samples_x2, dim = 0)
		cka_raw = min(max(cka(flattened_fixed_samples_x1, flattened_fixed_samples_x2), 0.0), 1.0)
		mknn_raw = min(max(mknn(flattened_fixed_samples_x1, flattened_fixed_samples_x2), 0.0), 1.0)
		# print(f"Collected {embeddings['x1_label'].shape[0]} fixed samples for embedding capture during training.")

	# evaluate raw features
	raw_results = evaluate_raw_data(eval_config, ds_name=ds_name)

	private_weight = 0
	for _iter in range(num_epoch):
		alphas = [alpha_x, alpha_y]
		if _iter <= step_k and train_mode == 'xy':
			print(f"Training only on y, step: [{_iter}/{step_k}]; total steps: {num_epoch}")
			alphas[0] = 0.0 # train only on y for warmup when training on unpaired x+y data

		for i_batch, (data_batch_1, data_batch_2) in enumerate(zip(train_loader_1, train_loader_2)):
			if ds_name != 'mimic':
				x1_batch = data_batch_1[0][modalities[0]].float().cuda()
				x2_batch = data_batch_2[0][modalities[1]].float().cuda()
				x1_batch_lengths = data_batch_1[1][modalities[0]].cuda() # shape: (B,)
				x2_batch_lengths = data_batch_2[1][modalities[1]].cuda() # shape: (B,)
			else:
				x1_batch = data_batch_1[0].float().cuda()
				x2_batch = data_batch_2[1].float().cuda()
				# TODO
				x1_batch_lengths = data_batch_1[2].cuda() # shape: (B,)
				x2_batch_lengths = data_batch_2[3].cuda() # shape: (B,)

			if 'x' not in train_mode:
				x1_batch = None
			if 'y' not in train_mode:
				x2_batch = None
			start_time = time.time()
			out = model(x1_batch, x2_batch, x1_batch_lengths, x2_batch_lengths)
			loss_x, loss_y, loss_private = out['loss_x'], out['loss_y'], out['loss_private']
			x2_batch_nex = x2_batch[:,1:,:] # next-token prediction
			y_recon = out['y_recon'][:,:-1,:]

			mask_x2 = torch.arange(x2_batch_nex.shape[1], device=x2_batch_lengths.device).unsqueeze(0) < (x2_batch_lengths-1).unsqueeze(1) # [batch_size, seq_len]
			flattened_valid_x2_batch = x2_batch_nex[mask_x2] # [(num of valid time steps), feature_dim]
			flattened_valid_y_recon = y_recon[mask_x2] # [(num of valid time steps), feature_dim]
			gt_covariance = (flattened_fixed_samples_x2 - flattened_fixed_samples_x2.mean(dim=0)).T @ (flattened_fixed_samples_x2 - flattened_fixed_samples_x2.mean(dim=0)) / flattened_fixed_samples_x2.shape[0]
			gt_effective_rank = compute_effective_rank(flattened_fixed_samples_x2.unsqueeze(0)).item()
			pred_covariance = (flattened_valid_y_recon - flattened_valid_y_recon.mean(dim=0)).T @ (flattened_valid_y_recon - flattened_valid_y_recon.mean(dim=0)) / flattened_valid_y_recon.shape[0]
			pred_effective_rank = compute_effective_rank(flattened_valid_y_recon.unsqueeze(0)).item()

			diff_next_x, diff_next_y = out['diff_next_x'], out['diff_next_y']
			# loss_x_k, loss_y_k = out['k_step_loss_x'], out['k_step_loss_y']
			# loss_trivial_x_k, loss_trivial_y_k = out['k_step_trivial_loss_x'], out['k_step_trivial_loss_y']
			loss = alphas[0] * loss_x + alphas[1] * loss_y
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			end_time = time.time()

			# print(f"backward: {end_time-start_time:.4f}s")

			# trivial reconstruction loss
			if x1_batch is not None:
				if x1_batch.shape[1] == 1:
					trivial_loss_x = x1_batch[:,0,:] - x1_batch[:,0,:]
				else:
					trivial_loss_x = x1_batch[:,:-1,:] - x1_batch[:,1:,:]
				mask = torch.arange(x1_batch.shape[1], device=x1_batch_lengths.device).unsqueeze(0) < x1_batch_lengths.unsqueeze(1) # [batch_size, seq_len]
				mask_expanded = mask.unsqueeze(-1).expand_as(x1_batch)
				trivial_loss_x = (trivial_loss_x**2) * mask_expanded[:,:-1,:].float()
				trivial_loss_x = trivial_loss_x.sum() / (mask_expanded[:,:-1,:].float().sum()+1e-8)
			else:
				trivial_loss_x = torch.tensor(0.0)
			if x2_batch is not None:
				if x2_batch.shape[1] == 1:
					trivial_loss_y = x2_batch[:,0,:] - x2_batch[:,0,:]
				else:
					trivial_loss_y = x2_batch[:,:-1,:] - x2_batch[:,1:,:]
				mask = torch.arange(x2_batch.shape[1], device=x2_batch_lengths.device).unsqueeze(0) < x2_batch_lengths.unsqueeze(1) # [batch_size, seq_len]
				mask_expanded = mask.unsqueeze(-1).expand_as(x2_batch)
				trivial_loss_y = (trivial_loss_y**2) * mask_expanded[:,:-1,:].float()
				trivial_loss_y = trivial_loss_y.sum() / (mask_expanded[:,:-1,:].float().sum()+1e-8)
				recon_y_loss = ((out['y_recon'][:,:-1,:] - x2_batch[:,1:,:])**2 * mask_expanded[:,1:,:].float()).sum() / (mask_expanded[:,1:,:].float().sum()+1e-8)
			else:
				trivial_loss_y = torch.tensor(0.0)

			wandb.log({'train/loss_x': loss_x.item(), 'train/loss_y': loss_y.item(), 'train/loss': loss.item(), 
					'train/recon_y_loss': recon_y_loss.item(),
					'train/loss_x_norm': torch.norm(loss_x).item(), 'train/loss_y_norm': torch.norm(loss_y).item(), 'train/loss_private': loss_private.item(), 
					'train/trivial_loss_x': trivial_loss_x.item(), 'train/trivial_loss_y': trivial_loss_y.item(),
					'train/diff_next_x': diff_next_x.item() if diff_next_x is not None else 0.0,
					'train/diff_next_y': diff_next_y.item() if diff_next_y is not None else 0.0,
					# 'train/loss_x_k': loss_x_k.item() if loss_x_k is not None else 0.0,
					# 'train/loss_y_k': loss_y_k.item() if loss_y_k is not None else 0.0,
					# 'train/trivial_loss_x_k': loss_trivial_x_k.item() if loss_trivial_x_k is not None else 0.0,
					# 'train/trivial_loss_y_k': loss_trivial_y_k.item() if loss_trivial_y_k is not None else 0.0,
					'train/pred_effective_rank_y': pred_effective_rank, 'train/gt_effective_rank_y': gt_effective_rank
					})
			if eval_config and i_batch % eval_config['freq'] == 0:
				model.eval()
				start_time = time.time()
				score = evaluate(model, eval_config, ds_name)
				end_time = time.time()
				# print(f"evaluation: {end_time-start_time:.4f}s")
				progress_bar.set_postfix({'iter': _iter, 'loss_x': loss_x.item(), 'loss_y': loss_y.item(), 'loss': loss.item()})

				val_results = {}
				for k in score:
					if not ('private' in k or 'complete' in k):
						val_results.update({k: score[k]})

				val_results.update({'val/score_x_raw': raw_results['val/score_x_raw'], 'val/score_y_raw': raw_results['val/score_y_raw'], 'val/score_xy_raw': raw_results['val/score_xy_raw'],
							'test/score_x_raw': raw_results['test/score_x_raw'], 'test/score_y_raw': raw_results['test/score_y_raw'], 'test/score_xy_raw': raw_results['test/score_xy_raw'],})

				# modification: for embedding capture
				if capture_embeddings_during_training:
					start_time = time.time()
					# capture embeddings and calculate alignment metrics (cka/mknn/cosine similarity)
					with torch.no_grad():
						embeddings_this_epoch = {'x1': [], 'x2': [], 'x_proj': [], 'y_proj': [], 'x_recon': [], 'y_recon': []}
						for i in range(len(fixed_samples['x1'])):
							out = model(fixed_samples['x1'][i].float().cuda(), fixed_samples['x2'][i].float().cuda(), fixed_samples['lx1'][i].cuda(), fixed_samples['lx2'][i].cuda())
							for j in range(fixed_samples['x1'][i].shape[0]):
								x1_len = fixed_samples['lx1'][i][j].item()
								x2_len = fixed_samples['lx2'][i][j].item()
								embeddings_this_epoch['x1'].append(out['zx'][j, :x1_len, :])
								embeddings_this_epoch['x2'].append(out['zy'][j, :x2_len, :])
								embeddings_this_epoch['x_proj'].append(out['x_proj'][j, :x1_len, :])
								embeddings_this_epoch['y_proj'].append(out['y_proj'][j, :x2_len, :])
								embeddings_this_epoch['x_recon'].append(out['x_recon'][j, :x1_len, :])
								embeddings_this_epoch['y_recon'].append(out['y_recon'][j, :x2_len, :])
					
						# import random
						# sample_idx = random.randint(0, len(fixed_samples['x1'])-1)
						# sample_idx = 0
						# x_pred_roll, _ = rollout(model, fixed_samples['x1'][sample_idx][:,0,:].unsqueeze(1).float().cuda(), None, steps=fixed_samples['x1'][sample_idx].shape[1]-1)
						# _, y_pred_roll = rollout(model, None, fixed_samples['x2'][sample_idx][:,0,:].unsqueeze(1).float().cuda(), steps=fixed_samples['x2'][sample_idx].shape[1]-1)
						# print(f"fixed_samples['x1'][{sample_idx}] shape: {fixed_samples['x1'][sample_idx].shape}, fixed_samples['x2'][{sample_idx}] shape: {fixed_samples['x2'][sample_idx].shape}")
						# print(f"x_pred_roll shape: {x_pred_roll.shape}, y_pred_roll shape: {y_pred_roll.shape}")
						# analyze_spectral_bias(fixed_samples['x1'][sample_idx].float().cuda(), x_pred_roll, loss_x.item(), iteration=_iter, modality_name='vision', postfix=f"_{ds_name}_{args.infoNCE_loss}")
						# analyze_spectral_bias(fixed_samples['x2'][sample_idx].float().cuda(), y_pred_roll, loss_y.item(), iteration=_iter, modality_name='text', postfix=f"_{ds_name}_{args.infoNCE_loss}")
						
					embeddings_this_epoch['x1'] = torch.cat(embeddings_this_epoch['x1'], dim=0) # shape: (n_samples, embed_dim)
					embeddings_this_epoch['x2'] = torch.cat(embeddings_this_epoch['x2'], dim=0) # shape: (n_samples, embed_dim)
					embeddings_this_epoch['x_proj'] = torch.cat(embeddings_this_epoch['x_proj'], dim=0) # shape: (n_samples, proj_dim)
					embeddings_this_epoch['y_proj'] = torch.cat(embeddings_this_epoch['y_proj'], dim=0) # shape: (n_samples, proj_dim)
					embeddings_this_epoch['x_recon'] = torch.cat(embeddings_this_epoch['x_recon'], dim=0) # shape: (n_samples, input_dim)
					embeddings_this_epoch['y_recon'] = torch.cat(embeddings_this_epoch['y_recon'], dim=0) # shape: (n_samples, input_dim)
					
					# between projected x and projected y
					cka_proj = max(min(cka(embeddings_this_epoch['x_proj'], embeddings_this_epoch['y_proj']), 1.0), 0.0)
					mknn_proj = mknn(embeddings_this_epoch['x_proj'], embeddings_this_epoch['y_proj'])
					cos_sim_proj = F.cosine_similarity(embeddings_this_epoch['x_proj'], embeddings_this_epoch['y_proj'], dim=1).mean().item()
					
					# between embeddings x and embeddings y
					cka_score = max(min(cka(embeddings_this_epoch['x1'], embeddings_this_epoch['x2']), 1.0), 0.0)
					mknn_score = mknn(embeddings_this_epoch['x1'], embeddings_this_epoch['x2'])
					cos_sim = F.cosine_similarity(embeddings_this_epoch['x1'], embeddings_this_epoch['x2'], dim=1).mean().item()

					# between reconstructed x and reconstructed y
					cka_out = max(min(cka(embeddings_this_epoch['x_recon'], embeddings_this_epoch['y_recon']), 1.0), 0.0)
					mknn_out = mknn(embeddings_this_epoch['x_recon'], embeddings_this_epoch['y_recon'])

					embeddings['x1'].append(embeddings_this_epoch['x1'])
					embeddings['x2'].append(embeddings_this_epoch['x2'])
					cka_text_embeddings_features = max(min(cka(embeddings_this_epoch['x2'], flattened_fixed_samples_x2.cuda()), 1.0), 0.0)
					val_results.update({'val/cka_proj': cka_proj, 'val/mknn_proj': mknn_proj, 'val/cos_sim_proj': cos_sim_proj, 
						'val/cka_embed': cka_score, 'val/mknn_embed': mknn_score, 'val/cos_sim_embed': cos_sim,
						'val/cka_out': cka_out, 'val/mknn_out': mknn_out,
						'val/cka_text_embeddings_features': cka_text_embeddings_features,
						'val/cka_raw': cka_raw, 'val/mknn_raw': mknn_raw})

					end_time = time.time()
				model.train()
				wandb.log(val_results)
		progress_bar.update(1)

		if eval_config and _iter == num_epoch-1:
			print('Final evaluation...')
			model.eval()
			score = evaluate(model, eval_config, ds_name)
			model.train()
			progress_bar.close()
			# print({'Final score_x': score['test/score_x'], 'Final score_y': score['test/score_y'], 'Final score_xy': score['test/score_xy'], 'Final val_score_x': score['val/score_x'], 'Final val_score_y': score['val/score_y'], 'Final val_score_xy': score['val/score_xy']})
			for k in score:
				if not ('private' in k or 'complete' in k):
					wandb.log({f"final_{k}": score[k]})
			# wandb.log({'final_test/score_x': score['test/score_x'], 'final_test/score_y': score['test/score_y'], 'final_test/score_xy': score['test/score_xy']})
			# wandb.log({'final_val/score_x': score['val/score_x'], 'final_val/score_y': score['val/score_y'], 'final_val/score_xy': score['val/score_xy']})

	# modification: for embedding capture
	if capture_embeddings_during_training:
		embeddings['x1'] = torch.stack(embeddings['x1'], dim=0)  # (num_epochs, n_samples, embed_dim)
		embeddings['x2'] = torch.stack(embeddings['x2'], dim=0)  # (num_epochs, n_samples, embed_dim)
		return score, embeddings
	return score
			

