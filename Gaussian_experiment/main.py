import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

DIM_C = 10
DIM_X = 5
DIM_Y = 5
DIM_OBS = 50
NOISE_STD = 0.3

N_TOTAL = 10000
N_HALF = 5000
N_VAL = 1000
BATCH_SIZE = 64
NUM_STEPS = 500 
LR = 1e-3
EVAL_EVERY = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


class UnpairedDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.len_x = len(data_x)
        self.len_y = len(data_y)
        self.length = max(self.len_x, self.len_y)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx_x = idx % self.len_x
        idx_y = idx % self.len_y
        
        return {'x': self.data_x[idx_x], 'y': self.data_y[idx_y]}


class SharedAutoencoder(nn.Module):
    def __init__(self, dim_obs=DIM_OBS, dim_common=128, dim_latent=DIM_C):
        super(SharedAutoencoder, self).__init__()
        
        self.in_head_x = nn.Linear(dim_obs, dim_common)
        self.in_head_y = nn.Linear(dim_obs, dim_common)
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(dim_common, 64),
            nn.ReLU(),
            nn.Linear(64, dim_latent)
        )
        
        self.shared_decoder = nn.Sequential(
            nn.Linear(dim_latent, 64),
            nn.ReLU(),
            nn.Linear(64, dim_common)
        )
        
        self.out_head_x = nn.Linear(dim_common, dim_obs)
        self.out_head_y = nn.Linear(dim_common, dim_obs)
        
        self.loss_fn = nn.MSELoss()

    def forward(self, x=None, y=None):
        loss_x = torch.tensor(0.0, device=device)
        loss_y = torch.tensor(0.0, device=device)
        recon_x = None

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

        return loss_x, loss_y, recon_x


def train_model_steps(model, data_loader, optimizer, num_steps, val_data_x, mode='unpaired'):
    model.train()
    data_iter = iter(data_loader)
    val_losses = []
    steps = []

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        optimizer.zero_grad()
        
        if mode == 'unpaired':
            x, y = batch['x'].to(device), batch['y'].to(device)
            loss_x, loss_y, _ = model(x, y)
            loss = loss_x + loss_y
        
        elif mode == 'x_only':
            x = batch[0].to(device)
            loss_x, _, _ = model(x=x)
            loss = loss_x
        
        loss.backward()
        optimizer.step()

        if (step + 1) % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                _, _, recon_val_x = model(x=val_data_x)
                val_loss = model.loss_fn(recon_val_x, val_data_x)
                val_losses.append(val_loss.item())
                steps.append(step + 1)
            model.train()
            
            if (step + 1) % 100 == 0:
                print(f"Step [{step + 1}/{num_steps}], Mode: {mode}, Val Recon Error X: {val_loss.item():.4f}")

    return steps, val_losses



print("Generating data...")
data_x_only_train = generate_data(N_TOTAL, attenuate_x=True)['x']
unpaired_train_data = generate_data(N_HALF, attenuate_x=True)
val_data = generate_data(N_VAL, attenuate_x=False)
val_data_x = val_data['x'].to(device)

dataset_x_only = TensorDataset(data_x_only_train)
loader_x_only = DataLoader(dataset_x_only, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

dataset_unpaired = UnpairedDataset(unpaired_train_data['x'], unpaired_train_data['y'])
loader_unpaired = DataLoader(dataset_unpaired, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


print("\nRunning Experiment 1: X-data only")
model_x_only = SharedAutoencoder().to(device)
optimizer_x_only = optim.Adam(model_x_only.parameters(), lr=LR)
steps_x, losses_x = train_model_steps(
    model_x_only, loader_x_only, optimizer_x_only, NUM_STEPS, val_data_x, mode='x_only'
)

print("\nRunning Experiment 2: Unpaired (X + Y)-data")
model_unpaired = SharedAutoencoder().to(device)
optimizer_unpaired = optim.Adam(model_unpaired.parameters(), lr=LR)
steps_unpaired, losses_unpaired = train_model_steps(
    model_unpaired, loader_unpaired, optimizer_unpaired, NUM_STEPS, val_data_x, mode='unpaired'
)

print("\nPlotting results...")
plt.figure(figsize=(10, 6))
plt.plot(steps_x, losses_x, label="X-data only", color="pink", linewidth=2)
plt.plot(steps_unpaired, losses_unpaired, label="Unpaired (X + Y)-data", color="turquoise", linewidth=2)

plt.title("Gaussian Experiment (Replication of Fig. 39)")
plt.xlabel("Training Steps")
plt.ylabel("Reconstruction Error on X")
plt.legend()
plt.grid(True, linestyle='--')
plt.ylim(bottom=min(min(losses_x), min(losses_unpaired)) * 0.95)
plt.show()