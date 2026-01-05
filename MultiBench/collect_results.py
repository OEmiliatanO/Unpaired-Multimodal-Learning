import torch
import os

def collect_results(
    experiments_dir,
    run_names,
    datasets,
    modality_types,
    zdim,
    num_epochs,
    pos_embd,
    pos_learnable,
    step_k,
    n_seeds,
    alpha_y,
    lrs,
):
    # 1) Gather all raw results by key
    raw = {}
    for dataset in datasets:
        for run_name in run_names:
            for modality_type in modality_types:
                for ne in num_epochs:
                    for zd in zdim:
                        for ay in alpha_y if modality_type in ['y', 'xy'] else [0.0]:
                            for sk in step_k:
                                for pe in pos_embd:
                                    for pl in pos_learnable:
                                        for lr in lrs:
                                            seed_dir = os.path.join(
                                                experiments_dir,
                                                f"log_{run_name}{dataset}/mod{modality_type}/epochs{ne}/zdim{zd}/alpha_y{ay}/step_k{sk}/pos_embd_{pe}_learnable_{pl}/lr{lr}"
                                            )
                                            # f'log_{args.run_name}{args.ds_name}_mod{args.modality}_zdim{args.zdim}_epochs{args.num_epochs}_pos_embd_{args.pos_embd}_learnable_{args.pos_learnable}_step_k{args.step_k}_n_seeds{args.n_seeds}'
                                            # print(f"seek {seed_dir}")
                                            if not os.path.isdir(seed_dir):
                                                continue
                                            seed_results = []
                                            for seed in range(n_seeds):
                                                seed_dir = os.path.join(
                                                    seed_dir, 
                                                    f"seed_{seed}"
                                                )
                                                results_path = os.path.join(seed_dir, "results.pth")
                                                # print(f"seek {results_path}")
                                                if not os.path.isfile(results_path):
                                                    continue
                                                results = torch.load(results_path, weights_only=False)
                                                # print(results)
                                                seed_results.append(results)
                                            results = {}
                                            # average over seeds
                                            for sr in seed_results:
                                                for k, v in sr.items():
                                                    if k not in results:
                                                        results[k] = []
                                                    results[k].append(v)
                                            for k, v in results.items():
                                                results[k] = torch.tensor(v).mean().item()

                                            key = (dataset, modality_type, ay, ne, zd) # aggregate by these params
                                            raw.setdefault(key, []).append(results)

    # 2) Aggregate per key
    summary = {}
    for key, results_list in raw.items():
        aggregated = {}

        if key[1] == 'x':
            results_list = sorted(results_list, key=lambda x: x['val/score_x'], reverse=True)
        elif key[1] == 'y':
            results_list = sorted(results_list, key=lambda x: x['val/score_y'], reverse=True)
        else:
            results_list = sorted(results_list, key=lambda x: x['val/score_xy'], reverse=True)
        
        for results in results_list[:1]:
            for metric, value in results.items():
                if metric not in aggregated:
                    aggregated[metric] = []
                aggregated[metric].append(value)
        summary[key] = {
            metric: {
                'top-1': torch.tensor(values).mean().item(),
                # 'std': torch.tensor(values).std().item()
            }
            for metric, values in aggregated.items()
        }

    return summary

if __name__ == "__main__":
    experiments_dir = "results"
    run_names = ['']
    # ds_name = ['mosi', 'sarcasm', 'humor', 'mimic', 'mosei']
    ds_name = ['sarcasm']
    modality = ['x', 'xy']
    zdim = [10, 40, 150, 300]
    num_epochs = [100]
    pos_embd = [True, False]
    pos_learnable = [True, False]
    step_k = [-1, 30, 50, 70]
    # alpha_y = [0.0, 0.2, 0.5, 0.7, 1.0, 1.5]
    alpha_y = [1.0]
    lrs = [1.0e-2, 1.0e-3, 1.0e-4]
    n_seeds = 3

    summary = collect_results(
        experiments_dir= experiments_dir,
        run_names      = run_names,
        datasets       = ds_name,
        modality_types = modality,
        zdim           = zdim,
        num_epochs     = num_epochs,
        pos_embd       = pos_embd,
        pos_learnable  = pos_learnable,
        step_k         = step_k,
        n_seeds        = n_seeds,
        alpha_y        = alpha_y,
        lrs            = lrs,
    )

    for key, metrics in summary.items():
        print(f"Key: {key}")
        for metric, stats in metrics.items():
            print(f"  {metric}: top-1={stats['top-1']}")
        print()