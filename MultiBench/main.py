import os
import sys
from models import Transformer, Linear, UML
from train import train
from utilis import set_seed
from torch import optim
from datasets.affect.get_data import get_dataloader
import torch
import yaml
from itertools import product
import random
import numpy as np
import argparse
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import wandb
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--modality', type=str, default='x')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--zdim', type=int, default=10)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--n_seeds', type=int, default=1)
parser.add_argument('--ds_name', type=str, default='mosi')
parser.add_argument('--step_k', type=int, default=-1)
parser.add_argument('--augment', action='store_true')
parser.add_argument('--pos_embd', action='store_true')
parser.add_argument('--pos_learnable', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--results_dir', type=str, default='./results')
parser.add_argument('--log_dir', type=str, default='./logs')
parser.add_argument('--alpha_x', type=float, default=1.0)
parser.add_argument('--alpha_y', type=float, default=1.0)
parser.add_argument('--infoNCE_loss', action='store_true')


# change this to your data directory
data_dir = '.'
def main(args):
    print("Command-line arguments:", sys.argv)
    print("Parsed arguments:", args)

    exp_name = f"log_{args.run_name}{args.ds_name}/mod{args.modality}/epochs{args.num_epochs}/zdim{args.zdim}/alpha_x{args.alpha_x}_alpha_y{args.alpha_y}/step_k{args.step_k}/pos_embd_{args.pos_embd}_learnable_{args.pos_learnable}/lr{args.lr}"
    results_dir = os.path.join(args.results_dir, exp_name)
    seeds = [i for i in range(args.n_seeds)]
    outs = {'test/score_x': [], 'test/score_y': [], 'test/score_xy': [], 'val/score_x': [], 'val/score_y': [], 'val/score_xy': []}
    for seed in seeds:
        set_seed(seed)
        log_dir = args.log_dir
        seed_dir = os.path.join(results_dir, f"seed_{seed}")
        print(f"Results will be saved to {seed_dir}")
        if not args.debug:  
            os.makedirs(log_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = log_dir + "/wandb"
            loss_tag = "infoNCE" if args.infoNCE_loss else "MSE"
            wandb.init(entity="unpaired_multimodal", project="unpaired_multimodal", tags=[args.ds_name, args.modality, "self-supervised"], name = os.path.join(exp_name, f"seed_{seed}"), reinit="finish_previous")
            wandb.config.update(args)
            wandb.config.update({'seed': seed})
        if args.ds_name == 'mosi':
            batch_size=32
            indims = [20, 300]
            # Sample two train loaders to shuffle x-only and y-only data within each dataloader
            train_loader, *_ = get_dataloader(f'{data_dir}/data_files/mosi_data.pkl', robust_test=False, batch_size=batch_size, train_shuffle=True)
            train_loader_2, *_ = get_dataloader(f'{data_dir}/data_files/mosi_data.pkl', robust_test=False, batch_size=batch_size, train_shuffle=True)
            eval_train_loader, eval_valid_loader, eval_test_loader = get_dataloader(f'{data_dir}/data_files/mosi_data.pkl', robust_test=False, batch_size=batch_size, train_shuffle=False)

        elif args.ds_name == 'sarcasm':
            batch_size=128
            indims = [371, 300]
            train_loader, *_ = get_dataloader(f'{data_dir}/data_files/sarcasm.pkl', batch_size=batch_size, data_type='sarcasm', vision_norm=True)
            train_loader_2, *_ = get_dataloader(f'{data_dir}/data_files/sarcasm.pkl', batch_size=batch_size, data_type='sarcasm', vision_norm=True)
            eval_train_loader, eval_valid_loader, eval_test_loader = get_dataloader(f'{data_dir}/data_files/sarcasm.pkl', batch_size=batch_size, data_type='sarcasm', train_shuffle=False, vision_norm=True)
            

        elif args.ds_name == 'humor':
            batch_size=128
            indims = [371, 300]
            train_loader, *_ = get_dataloader(f'{data_dir}/data_files/humor.pkl', batch_size=batch_size, data_type='humor')
            train_loader_2, *_ = get_dataloader(f'{data_dir}/data_files/humor.pkl', batch_size=batch_size, data_type='humor')
            eval_train_loader, eval_valid_loader, eval_test_loader = get_dataloader(f'{data_dir}/data_files/humor.pkl', batch_size=batch_size, data_type='humor', train_shuffle=False)
            
        elif args.ds_name == 'mimic':
            from datasets.mimic.get_data import get_dataloader as get_mimic_dataloader
            batch_size = 128
            indims = [5, 12]
            train_loader, *_ = get_mimic_dataloader(7, batch_size=batch_size, imputed_path=f'{data_dir}/data_files/im.pk')
            train_loader_2, *_ = get_mimic_dataloader(7, batch_size=batch_size, imputed_path=f'{data_dir}/data_files/im.pk')
            eval_train_loader, eval_valid_loader, eval_test_loader = get_mimic_dataloader(7, imputed_path=f'{data_dir}/data_files/im.pk', train_shuffle=False)        
            eval_test_loader = eval_valid_loader # as per FACTOR-CL codebase
        
        elif args.ds_name == 'mosei':
            batch_size=32
            indims = [35, 300]
            train_loader, *_ = get_dataloader(f'{data_dir}/data_files/mosei_senti_data.pkl', robust_test=False, batch_size=batch_size, data_type='mosei', train_shuffle=True)
            train_loader_2, *_ = get_dataloader(f'{data_dir}/data_files/mosei_senti_data.pkl', robust_test=False, batch_size=batch_size, data_type='mosei', train_shuffle=True)
            eval_train_loader, eval_valid_loader, eval_test_loader = get_dataloader(f'{data_dir}/data_files/mosei_senti_data.pkl', robust_test=False, batch_size=batch_size, data_type='mosei', train_shuffle=False)
        else:
            raise NotImplementedError('Dataset not implemented yet')

        # Dataset stats
        print("Dataset: ", args.ds_name)
        print("Batch size: ", batch_size)
        print("Train dataset: ", len(train_loader)*batch_size)
        print("Eval train dataset: ", len(eval_train_loader)*batch_size)
        print("Eval test dataset: ", len(eval_test_loader)*batch_size)
        print(f"Modality Info: xdim: {indims[0]}, ydim: {indims[1]}, zdim: {args.zdim}")


        # Initialize model and optimizer
        xproj_in = Linear(indims[0], args.zdim)
        yproj_in = Linear(indims[1], args.zdim)
        shared_encoder = Transformer(args.zdim, args.zdim, nhead=5, num_layers=5, conv1d=True, out_last=False, pos_embd=args.pos_embd, pos_learnable=args.pos_learnable, max_len=128)
        decoders = [Linear(args.zdim, indims[0]), Linear(args.zdim, indims[1])]
        model = UML(xproj_in, yproj_in, shared_encoder, decoders, modality=args.modality, infoNCE_loss=args.infoNCE_loss).cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        capture_embeddings_during_training = True
        results = train(model, args.modality, train_loader, train_loader_2, optimizer, 
                    num_epoch=args.num_epochs, step_k=args.step_k, ds_name=args.ds_name,
                    eval_config={'train': eval_train_loader, 'val': eval_valid_loader, 'test': eval_test_loader, 'freq': 100},
                    alpha_x=args.alpha_x, alpha_y=args.alpha_y, capture_embeddings_during_training=capture_embeddings_during_training,
                    augment=args.augment, debug=args.debug, args=args)
        
        if capture_embeddings_during_training:
            score, embeddings = results
        else:
            score = results

        print('seed: ', seed, ' score: ', score)
        print('=====================================')
        outs['test/score_x'].append(100*score['test/score_x'])
        outs['test/score_y'].append(100*score['test/score_y'])
        outs['test/score_xy'].append(100*score['test/score_xy'])
        outs['val/score_x'].append(100*score['val/score_x'])
        outs['val/score_y'].append(100*score['val/score_y'])
        outs['val/score_xy'].append(100*score['val/score_xy'])

        # save model and log outputs
        os.makedirs(seed_dir, exist_ok=True)
        print(f"Saving model to {os.path.join(seed_dir, 'model.pth')}")
        torch.save(model.state_dict(), os.path.join(seed_dir, "model.pth"))
        torch.save(score, os.path.join(seed_dir, "results.pth"))
        if capture_embeddings_during_training:
            torch.save(embeddings, os.path.join(seed_dir, "embeddings.pth"))

    print(outs)
    
    # Mean across seeds
    outs_mean = {k: np.mean(v) for k, v in outs.items()}
    outs_std = {k: np.std(v) for k, v in outs.items()}
    print("Final scores (mean): ", outs_mean)
    print("Final scores (std): ", outs_std)

    # save model and log outputs
    os.makedirs(results_dir, exist_ok=True)
    torch.save(outs, os.path.join(results_dir, "results.pth"))

if __name__ == "__main__":
    outer_parser = argparse.ArgumentParser(description="MultiBench Experiment")
    outer_parser.add_argument("-c", "--config", type=str, default="config.json", help="Configuration file")
    outer_parser.add_argument("-s", "--slurm", action="store_true", help="Launched with slurm")
    outer_parser.add_argument("-r", "--run", action="store_true", help="Run the experiments")
    outer_parser.add_argument("-d", "--outer_debug", action="store_true", help="Debug mode")
    outer_parser.add_argument("-f", "--flag", action="store_true", help="Run despite existing experiments directory")
    outer_parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing experiments directory")
    outer_args, remaining_args = outer_parser.parse_known_args()

    FLAG = int(outer_args.flag)

    if outer_args.outer_debug:
        print("Running command-line arguments...")
        args = parser.parse_args(remaining_args)
        args.overwrite = outer_args.overwrite
        args.debug = True
        main(args)
        sys.exit(0)
    
    with open(outer_args.config, "r") as f:
        sweep_args = yaml.load(f, Loader=yaml.FullLoader)
    
    keys, values = zip(*sweep_args.items())
    combinations = [dict(zip(keys, v)) for v in product(*[v if isinstance(v, list) else [v] for v in values])]

    print("Total combinations:", len(combinations))
    for i, combo in enumerate(combinations):
        print(f"Combination {i}: {combo}")
    # modification: only run if --run is specified
    if not outer_args.run:
        print("Exiting without running experiments (use -r to run).")
        sys.exit(0)
    if outer_args.slurm:
        # args.debug = True
        job_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "-1"))
        if job_id < 0 or job_id >= len(combinations):
            print("Invalid SLURM_ARRAY_TASK_ID")
            sys.exit(1)
        combination = combinations[job_id]
        print(f"=> Running combination {job_id}: {combination}")
        args = parser.parse_args([], argparse.Namespace(**combination))
        args.overwrite = outer_args.overwrite
        print("=> Parsed arguments:", args)
        main(args)
    else:
        for i, combination in enumerate(combinations):
            print(f"=> Running combination {i}: {combination}")
            args = parser.parse_args([], argparse.Namespace(**combination))
            args.overwrite = outer_args.overwrite
            print("=> Parsed arguments:", args)
            main(args)
