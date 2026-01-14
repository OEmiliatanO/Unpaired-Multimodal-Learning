import torch
import numpy as np
import random
import os
from metrics import AlignmentMetrics

# Set seed
def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False  # Critical: disable cuDNN benchmarking
	os.environ['PYTHONHASHSEED'] = str(seed)

# CKA and mKNN metrics
def cka(feats_A, feats_B):
	kwargs = {'kernel_metric': 'ip'}
	return AlignmentMetrics.measure('cka', feats_A, feats_B, **kwargs)

def mknn(feats_A, feats_B):
	kwargs = {'topk': 10}
	return AlignmentMetrics.measure('mutual_knn', feats_A, feats_B, **kwargs)

def compute_effective_rank(A, eps=1e-6):
	"""
	A: (B, N, D) tensor
	returns: (B,) tensor of effective ranks
	"""
	singular_values = torch.linalg.svdvals(A)
	p = singular_values / singular_values.sum(dim=-1, keepdim=True)
	entropy = -torch.sum(p * torch.log(p + eps), dim=-1)
	erank = torch.exp(entropy)
	return erank

####

# Simple Augmentations 
def permute(x):
	# shuffle the sequence order
	idx = torch.randperm(x.shape[0])
	return x[idx]

def noise(x):
	noise = torch.randn(x.shape) * 0.1
	return x + noise.to(x.device)

def drop(x):
	# drop 20% of the sequences
	drop_num = x.shape[0] // 5
	
	x_aug = torch.clone(x)
	drop_idxs = np.random.choice(x.shape[0], drop_num, replace=False)
	x_aug[drop_idxs] = 0.0
	return x_aug  

def mixup(x, alpha=1.0):
	indices = torch.randperm(x.shape[0])
	lam = np.random.beta(alpha, alpha)
	aug_x = x * lam + x[indices] * (1 - lam)

	return aug_x

def identity(x):
	return x

def augment(x_batch):
	v1 = x_batch
	v2 = torch.clone(v1)
	transforms = [permute, noise, drop, identity]

	for i in range(x_batch.shape[0]):
		t_idxs = np.random.choice(4, 2, replace=False)
		t1 = transforms[t_idxs[0]]
		t2 = transforms[t_idxs[1]]
		v1[i] = t1(v1[i])
		v2[i] = t2(v2[i])
	
	return v1, v2

def augment_single(x_batch):
	v1 = x_batch
	v2 = torch.clone(v1)
	transforms = [permute, noise, drop, identity]

	for i in range(x_batch.shape[0]):
		t_idxs = np.random.choice(4, 1, replace=False)
		t = transforms[t_idxs[0]]
		v2[i] = t(v2[i])
	
	return v2


def augment_embed_single(x_batch):
	v1 = x_batch
	v2 = torch.clone(v1)
	transforms = [noise, mixup, identity]

	t_idxs = np.random.choice(3, 1, replace=False)
	t = transforms[t_idxs[0]]
	v2 = t(v2)

	return v2


def augment_mimic(x_batch):
	if x_batch.dim() == 2:
		return augment_embed_single(x_batch)
	else:
		return augment_single(x_batch)