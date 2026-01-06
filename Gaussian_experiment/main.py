import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from dataset import UnpairedDataset
from tqdm import tqdm
from metrics import AlignmentMetrics
import wandb
import random
import argparse
from model import SharedAutoencoder
from utils import make_reproducible
from data import generate_data
import yaml
import os
import sys
from itertools import product

def cka(feats_A, feats_B):
	kwargs = {'kernel_metric': 'ip'}
	return AlignmentMetrics.measure('cka', feats_A, feats_B, **kwargs)

def mknn(feats_A, feats_B):
	kwargs = {'topk': 10}
	return AlignmentMetrics.measure('mutual_knn', feats_A, feats_B, **kwargs)

EVAL_EVERY = 1

def train_model_steps(model, data_loader, optimizer, num_steps, val_data_x, val_data_y, device, args):
    mode = args.mode
    logger = wandb.init(entity="unpaired_multimodal", project="Gaussian_experiments", tags=[mode, args.tag], config={**vars(args)}, reinit="finish_previous")
    model.train()
    data_iter = iter(data_loader)
    alpha_x = args.alpha_x
    alpha_y = args.alpha_y

    tqdm_range = tqdm(range(num_steps), desc=f"Training ({mode})")
    for step in tqdm_range:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        optimizer.zero_grad()
        
        x, y = batch['x'].to(device), batch['y'].to(device)
        
        if mode == 'xy':
            loss_x, loss_y, _, _ = model(x, y)
            loss = alpha_x * loss_x + alpha_y * loss_y
        elif mode == 'x':
            loss_x, _, _, _ = model(x=x, y=None)
            loss_y = torch.tensor(0.0)
            loss = loss_x
        
        loss.backward()
        optimizer.step()

        logger.log({
            'train/loss_x': loss_x.item(),
            'train/loss_y': loss_y.item(),
            'train/loss': loss.item()
        })

        if (step + 1) % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                _, _, recon_val_x, recon_val_y = model(x=val_data_x, y=val_data_y)
                val_loss_x = model.loss_fn(recon_val_x, val_data_x)
                val_loss_y = model.loss_fn(recon_val_y, val_data_y)
                logger.log({
                    'val/loss_x': val_loss_x.item(),
                    'val/loss_y': val_loss_y.item(),
                    'val/loss': val_loss_x.item() + val_loss_y.item()
                })
                embeddings_x, embeddings_y = model.get_embeddings(x=val_data_x, y=val_data_y)
                cka_score = cka(embeddings_x, embeddings_y)
                mknn_score = mknn(embeddings_x, embeddings_y)
                logger.log({
                    'val/cka': cka_score,
                    'val/mknn': mknn_score
                })
            model.train()
            tqdm_range.set_description(f"Training ({mode}), Val Loss X: {val_loss_x.item():.6f}, Val Loss Y: {val_loss_y.item():.6f}, Train Loss X: {loss_x.item():.6f}, Train Loss Y: {loss_y.item():.6f}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    unpaired_train_data = generate_data({
        'seed': 42,
        'num_samples': args.train_num_samples,
        'dim_c': args.data_dim_common,
        'dim_x': args.data_dim_x,
        'dim_y': args.data_dim_y,
        'dim_obs': args.dim_obs,
        'noise_std': args.noise_std,
        'attenuate_x': True,
        'attenuation': args.attenuation
    })
    val_data = generate_data({
        'seed': 43,
        'num_samples': args.val_num_samples,
        'dim_c': args.data_dim_common,
        'dim_x': args.data_dim_x,
        'dim_y': args.data_dim_y,
        'dim_obs': args.dim_obs,
        'noise_std': args.noise_std,
        'attenuate_x': False,
        'attenuation': args.attenuation
    })
    val_data_x = val_data['x'].to(device)
    val_data_y = val_data['y'].to(device)

    if args.mode == 'xy':
        # use both x and y, so the lengths are half of args.train_num_samples
        dataset = UnpairedDataset(unpaired_train_data['x'][:args.train_num_samples//2], unpaired_train_data['y'][:args.train_num_samples-args.train_num_samples//2])
    else:
        # use only x, so the length is args.train_num_samples
        dataset = UnpairedDataset(unpaired_train_data['x'], unpaired_train_data['y']) 
    g_origin = torch.Generator()
    g_origin.manual_seed(42)
    loader_unpaired = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=g_origin)

    make_reproducible(args.seed)
    model = SharedAutoencoder(dim_obs = args.dim_obs, dim_common=args.dim_common, dim_latent=args.dim_latent).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_model_steps(
        model, loader_unpaired, optimizer, args.num_steps, val_data_x, val_data_y, device, args
    )

if __name__ == "__main__":
    outer_parser = argparse.ArgumentParser(description="Synthetic Search Experiment")
    outer_parser.add_argument("-c", "--config", type=str, default="train.yaml", help="Configuration file")
    outer_parser.add_argument("-s", "--slurm", action="store_true", help="Launched with slurm")
    outer_parser.add_argument("-r", "--run", action="store_true", help="run the experiments")
    outer_args = outer_parser.parse_args()

    with open(outer_args.config, "r") as f:
        sweep_args = yaml.load(f, Loader=yaml.FullLoader)
    
    keys, values = zip(*sweep_args.items())
    combinations = [dict(zip(keys, v)) for v in product(*[v if isinstance(v, list) else [v] for v in values])]

    print("Total combinations:", len(combinations))
    for i, combo in enumerate(combinations):
        print(f"Combination {i}: {combo}")

    if not outer_args.run:
        print("use -r to run experiments")
        exit(0)

    parser = argparse.ArgumentParser(description="Synthetic Search Experiment")
    parser.add_argument('--dim_obs', type=int, default=50, help='Dimension of observed data')
    parser.add_argument('--dim_common', type=int, default=100, help='Dimension of common representation')
    parser.add_argument('--dim_latent', type=int, default=128, help='Dimension of latent space')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--num_steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data_dim_common', type=int, default=5, help='Dimension of common latent variable in data generation')
    parser.add_argument('--data_dim_x', type=int, default=10, help='Dimension of X-specific latent variable in data generation')
    parser.add_argument('--data_dim_y', type=int, default=10, help='Dimension of Y-specific latent variable in data generation')
    parser.add_argument('--noise_std', type=float, default=0.1, help='Standard deviation of noise in data generation')
    parser.add_argument('--train_num_samples', type=int, default=100000, help='Number of training samples')
    parser.add_argument('--val_num_samples', type=int, default=2000, help='Number of validation samples')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--alpha_x', type=float, default=1.0, help='Alpha value for X')
    parser.add_argument('--alpha_y', type=float, default=1.0, help='Alpha value for Y')
    parser.add_argument('--mode', type=str, default='xy', help='Training mode: xy or x')
    parser.add_argument('--tag', type=str, default='default', help='Tag for the experiment')
    parser.add_argument('--attenuation', type=float, default=0.05, help='Attenuation factor for X generation')
    # args = parser.parse_args()

    if outer_args.slurm:
        job_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "-1"))
        if job_id < 0 or job_id >= len(combinations):
            print("Invalid SLURM_ARRAY_TASK_ID")
            sys.exit(1)
        combination = combinations[job_id]
        print(f"=> Running combination {job_id}: {combination}")
        args = parser.parse_args([], argparse.Namespace(**combination))
        main(args)
    else:
        for i, combo in enumerate(combinations):
            print(f"=> Running job {i}")
            args = parser.parse_args([], argparse.Namespace(**combo))
            print(args)
            main(args)
