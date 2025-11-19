import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DIM_C = 10
DIM_X = 5
DIM_Y = 5
DIM_OBS = 50
NOISE_STD = 0.09

N_TOTAL = 10000
N_HALF = N_TOTAL // 2
N_VAL = 1000

A_c = torch.randn(DIM_OBS, DIM_C)
A_x = torch.randn(DIM_OBS, DIM_X)
B_c = torch.randn(DIM_OBS, DIM_C)
B_y = torch.randn(DIM_OBS, DIM_Y)

attenuation = torch.full((DIM_C,), 0.05)
attenuation[0] = 1.0 # 10% of 10 is 1 component

def generate_data(num_samples, attenuate_x=True):
    theta_c = torch.randn(num_samples, DIM_C)
    theta_x = torch.randn(num_samples, DIM_X)
    theta_y = torch.randn(num_samples, DIM_Y)

    noise_x = torch.randn(num_samples, DIM_OBS) * NOISE_STD
    noise_y = torch.randn(num_samples, DIM_OBS) * NOISE_STD

    if attenuate_x:
        theta_c_x = theta_c * attenuation
    else:
        theta_c_x = theta_c
        
    # X = Ac * (W * theta_c) + Ax * theta_x + eps_X
    data_x = (theta_c_x @ A_c.T) + (theta_x @ A_x.T) + noise_x

    # Y = Bc * theta_c + By * theta_y + eps_Y
    data_y = (theta_c @ B_c.T) + (theta_y @ B_y.T) + noise_y

    return {'x': data_x, 'y': data_y}

print("Generating data...")
data_x_only_train = generate_data(N_TOTAL, attenuate_x=True)['x']
unpaired_train_data = generate_data(N_HALF, attenuate_x=True)
val_data = generate_data(N_VAL, attenuate_x=False)

torch.save(data_x_only_train, "data/data_x_only_train.pt")
torch.save(unpaired_train_data, "data/unpaired_train_data.pt")
torch.save(val_data, "data/val_data.pt")