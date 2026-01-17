import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from utils import make_reproducible
import torch.distributions as dist

def sample_latent(num_samples, dim, dist_type='gaussian', **kwargs):
    if dist_type == 'gaussian':
        latents = torch.randn(num_samples, dim)
        latents = latents - latents.mean(0)
        return latents
    elif dist_type == 'gmm':
        n_clusters = kwargs.get('n_clusters', 10)
        centroids = torch.randn(n_clusters, dim) * 5.0
        cluster_ids = torch.randint(0, n_clusters, (num_samples,))
        noise = torch.randn(num_samples, dim) * 0.5
        latents = centroids[cluster_ids] + noise
        latents = latents - latents.mean(0)
        return latents
    elif dist_type == 'laplace':
        m = dist.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        return m.sample((num_samples, dim)).squeeze(-1)
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

# configs = {'seed': 0, 'num_samples': 10000, 'dim_c': 100, 'dim_x': 3, 'dim_y': 3,
#            'dim_obs': 50, 'noise_std': 0.1, 'attenuate_x': True, 'attenuation': 0.05}
# new options: {'shared_latent_distribution_type' }
def generate_data(configs):
    make_reproducible(configs['seed'])

    # latent reality (shared)
    theta_c = sample_latent(configs['num_samples'], configs['dim_c'], 
                            dist_type=configs['shared_latent_distribution_type'], n_clusters=10)
    # latent reality (private)
    theta_x = torch.randn(configs['num_samples'], configs['dim_x'])
    theta_y = torch.randn(configs['num_samples'], configs['dim_y'])

    # noise
    noise_x = torch.randn(configs['num_samples'], configs['dim_obs']) * configs['noise_std']
    noise_y = torch.randn(configs['num_samples'], configs['dim_obs']) * configs['noise_std']

    A_c = torch.randn(configs['dim_obs'], configs['dim_c']) # latent reality (shared) -> observated space
    A_x = torch.randn(configs['dim_obs'], configs['dim_x']) # latent reality (private) -> observated space
    B_c = torch.randn(configs['dim_obs'], configs['dim_c']) # latent reality (shared) -> observated space
    B_y = torch.randn(configs['dim_obs'], configs['dim_y']) # latent reality (private) -> observated space

    if configs['attenuate_x']:
        attenuation = torch.full((configs['dim_c'],), configs['attenuation'])
        attenuation[:int(configs['dim_c']*0.1)] = 1.0
        theta_c_x = theta_c * attenuation
    else:
        theta_c_x = theta_c
        
    # X = Ac * (W * theta_c) + Ax * theta_x + eps_X
    data_x = theta_c_x @ A_c.T + theta_x @ A_x.T + noise_x

    # Y = Bc * theta_c + By * theta_y + eps_Y
    data_y = theta_c @ B_c.T + theta_y @ B_y.T + noise_y

    return {'x': data_x, 'y': data_y}

if __name__ == "__main__":
    configs = {'seed': 0, 'num_samples': 5, 'dim_c': 3, 'dim_x': 5, 'dim_y': 5,
               'dim_obs': 10, 'noise_std': 0.0, 'attenuate_x': True, 'attenuation': 0.05}
    data = generate_data(configs)
    print(data['x'])