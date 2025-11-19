import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from dataset import UnpairedDataset
from tqdm import tqdm

SEED = 42

DIM_C = 10
DIM_X = 5
DIM_Y = 5
DIM_OBS = 50

BATCH_SIZE = 512
NUM_STEPS = 1000000
LR = 1e-3
EVAL_EVERY = 5

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

    tqdm_range = tqdm(range(num_steps), desc=f"Training ({mode})")
    for step in tqdm_range:
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
                tqdm_range.set_description(f"Training ({mode}), Val Loss: {val_loss.item():.6f}")

    return steps, val_losses

data_x_only_train = torch.load("data/data_x_only_train.pt")
unpaired_train_data = torch.load("data/unpaired_train_data.pt")
val_data = torch.load("data/val_data.pt")
val_data_x = val_data['x'].to(device)

dataset_x_only = TensorDataset(data_x_only_train)
loader_x_only = DataLoader(dataset_x_only, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

dataset_unpaired = UnpairedDataset(unpaired_train_data['x'], unpaired_train_data['y'])
loader_unpaired = DataLoader(dataset_unpaired, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


torch.manual_seed(SEED)
np.random.seed(SEED)
print("\nRunning Experiment 1: X-data only")
model_x_only = SharedAutoencoder().to(device)
optimizer_x_only = optim.Adam(model_x_only.parameters(), lr=LR)
steps_x, losses_x = train_model_steps(
    model_x_only, loader_x_only, optimizer_x_only, NUM_STEPS, val_data_x, mode='x_only'
)

print("Saved model for X-data only experiment.")
torch.save(model_x_only.state_dict(), "model/model_x_only.pth")

torch.manual_seed(SEED)
np.random.seed(SEED)
print("\nRunning Experiment 2: Unpaired (X + Y)-data")
model_unpaired = SharedAutoencoder().to(device)
optimizer_unpaired = optim.Adam(model_unpaired.parameters(), lr=LR)
steps_unpaired, losses_unpaired = train_model_steps(
    model_unpaired, loader_unpaired, optimizer_unpaired, NUM_STEPS, val_data_x, mode='unpaired'
)

print("Saved model for Unpaired (X + Y)-data experiment.")
torch.save(model_unpaired.state_dict(), "model/model_unpaired.pth")

print("\nPlotting results...")
plt.figure(figsize=(10, 6))
plt.plot(steps_x, losses_x, label="X-data only", color="pink", linewidth=2)
plt.plot(steps_unpaired, losses_unpaired, label="Unpaired (X + Y)-data", color="turquoise", linewidth=2)

plt.title(f"Gaussian Experiment batch size = {BATCH_SIZE}")
plt.xlabel("Training Steps")
plt.ylabel("Reconstruction Error on X")
plt.legend()
plt.grid(True, linestyle='--')
plt.ylim(bottom=min(min(losses_x), min(losses_unpaired)) * 0.95)
plt.savefig("result.png")