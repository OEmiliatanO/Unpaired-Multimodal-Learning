import torch
import torch.nn as nn
import numpy as np

class SharedAutoencoder(nn.Module):
    def __init__(self, dim_obs, dim_common, dim_latent):
        super(SharedAutoencoder, self).__init__()
        
        self.in_head_x = nn.Linear(dim_obs, dim_common)
        self.in_head_y = nn.Linear(dim_obs, dim_common)
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(dim_common, dim_latent),
            nn.ReLU(),
            nn.Linear(dim_latent, dim_latent)
        )
        
        self.shared_decoder = nn.Sequential(
            nn.Linear(dim_latent, dim_latent),
            nn.ReLU(),
            nn.Linear(dim_latent, dim_common)
        )
        
        self.out_head_x = nn.Linear(dim_common, dim_obs)
        self.out_head_y = nn.Linear(dim_common, dim_obs)
        
        self.loss_fn = nn.MSELoss()

    def forward(self, x=None, y=None):
        loss_x = torch.tensor(0.0)
        loss_y = torch.tensor(0.0)
        recon_x = None
        recon_y = None

        if x is not None:
            z_x = self.in_head_x(x)
            latent_x = self.shared_encoder(z_x)
            recon_common_x = self.shared_decoder(latent_x)
            recon_x = self.out_head_x(recon_common_x)
            loss_x = self.loss_fn(recon_x, x)

        if y is not None:
            z_y = self.in_head_y(y)
            latent_y = self.shared_encoder(z_y)
            recon_common_y = self.shared_decoder(latent_y)
            recon_y = self.out_head_y(recon_common_y)
            loss_y = self.loss_fn(recon_y, y)

        return loss_x, loss_y, recon_x, recon_y

    def get_embeddings(self, x=None, y=None):
        embedding_x, embedding_y = None, None
        if x is not None:
            z_x = self.in_head_x(x)
            embedding_x = self.shared_encoder(z_x)
        if y is not None:
            z_y = self.in_head_y(y)
            embedding_y = self.shared_encoder(z_y)
        return embedding_x, embedding_y
