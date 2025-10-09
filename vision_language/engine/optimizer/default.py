HYPER_DICT = {
    'full_ds_full_model_finetune': {
        'optim': "adamw",
        'lr': [5.0e-5],
        'weight_decay': [0.0],
        'lr_scheduler': "cosine",
        'batch_size': [64],
        'max_iter': [12800],
        'warmup_iter': 50,
        'warmup_type': "linear",
        'warmup_min_lr': 1e-5,
        'dropout': [0.0],
        'learnable_temp': [False],
        'patience': [10]
    },
}