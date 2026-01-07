import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from utils import make_reproducible

# configs = {'seed': 0, 'num_samples': 10000, 'dim_c': 100, 'dim_x': 3, 'dim_y': 3,
#            'dim_obs': 50, 'noise_std': 0.1, 'attenuate_x': True, 'attenuation': 0.05}
def generate_data(configs):
    make_reproducible(configs['seed'])
    theta_c = torch.randn(configs['num_samples'], configs['dim_c'])
    theta_x = torch.randn(configs['num_samples'], configs['dim_x'])
    theta_y = torch.randn(configs['num_samples'], configs['dim_y'])

    noise_x = torch.randn(configs['num_samples'], configs['dim_obs']) * configs['noise_std']
    noise_y = torch.randn(configs['num_samples'], configs['dim_obs']) * configs['noise_std']

    A_c = torch.randn(configs['dim_obs'], configs['dim_c'])
    A_x = torch.randn(configs['dim_obs'], configs['dim_x'])
    B_c = torch.randn(configs['dim_obs'], configs['dim_c'])
    B_y = torch.randn(configs['dim_obs'], configs['dim_y'])

    if configs['attenuate_x']:
        attenuation = torch.full((configs['dim_c'],), configs['attenuation'])
        attenuation[:int(configs['dim_c']*0.1)] = 1.0
        theta_c_x = theta_c * attenuation
    else:
        theta_c_x = theta_c
        
    # X = Ac * (W * theta_c) + Ax * theta_x + eps_X
    data_x = (theta_c_x @ A_c.T) + (theta_x @ A_x.T) + noise_x

    # Y = Bc * theta_c + By * theta_y + eps_Y
    data_y = (theta_c @ B_c.T) + (theta_y @ B_y.T) + noise_y

    return {'x': data_x, 'y': data_y}

if __name__ == "__main__":
    configs = {'seed': 0, 'num_samples': 5, 'dim_c': 3, 'dim_x': 5, 'dim_y': 5,
               'dim_obs': 10, 'noise_std': 0.0, 'attenuate_x': True, 'attenuation': 0.05}
    data = generate_data(configs)
    print(data['x'])