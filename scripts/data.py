import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vqvae_beta import VQVAE  # <- your new Conv1D+normalized tanh model
from plot.plot import plot, plot_tensor
from dataloader.dataloader import load_jetclass_label_as_tensor

# === Config ===
batch_size = 128
num_epochs = 100
lr = 2e-4
start = 10
end = 12
checkpoint_dir = "checkpoints/checkpoints_vqvae_convnorm"
os.makedirs(checkpoint_dir, exist_ok=True)

# === Device ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# === Load Dataset Once (for scaling too) ===
dataloader = load_jetclass_label_as_tensor("HToBB", start=start, end=end, batch_size=batch_size)

#standard deviation and mean for each feature
x_particles, x_jets, y = next(iter(dataloader))
feature_mean = x_particles.mean(dim=(0, 2))  # shape: (4,)
feature_std = x_particles.std(dim=(0, 2))    # shape: (4,)

print("Feature-wise Mean:", feature_mean)
print("Feature-wise Std:", feature_std)