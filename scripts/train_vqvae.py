import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
import models.vqvaeMLP as vqvae
from plot.plot import plot, plot_tensor
from dataloader.dataloader import load_jetclass_label_as_tensor

# === Config ===
batch_size = 512
num_epochs = 1
lr = 2e-4
start = 10
end = 12
checkpoint_dir = "checkpoints/checkpoints_vqvae_norm"
os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# === Load Dataset Once (for normalization) ===
dataloader = load_jetclass_label_as_tensor(label="HToBB", start=start, end=end, batch_size=batch_size)


model = vqvae.VQVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
recon_loss_fn = nn.MSELoss()

# === Load Checkpoint if Available ===
latest_checkpoint = None
start_epoch = 0

if os.path.exists(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("vqvae_epoch_") and f.endswith(".pth")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"🔁 Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"]
    else:
        print("📭 No valid checkpoint files found in the directory.")
else:
    print(f"📁 Checkpoint directory '{checkpoint_dir}' does not exist.")

# === Plot Original Before Training ===
plot(label="HToBB", start=start, end=end, batch_size=batch_size)

# === Training Loop ===
print("🚀 Starting training...")
for epoch in range(start_epoch, start_epoch + num_epochs):
    model.train()
    total_recon_loss = 0.0
    total_vq_loss = 0.0

    for _, x_jets, _ in dataloader:
        x_jets = x_jets.to(device)
        mean = x_jets.mean(dim=0, keepdim=True)  # Shape: (1, 4)
        std = x_jets.std(dim=0, keepdim=True) + 1e-6
        # === Normalize input ===
        x_norm = (x_jets - mean) / std

        optimizer.zero_grad()
        recon, vq_loss = model(x_norm)

        # === Denormalize output before computing loss ===
        recon_denorm = recon * std + mean
        recon_loss = recon_loss_fn(recon_denorm, x_jets)

        loss = recon_loss + vq_loss
        loss.backward()
        optimizer.step()

        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()

    scheduler.step()
    print(f"[{epoch+1}/{num_epochs}] Recon Loss: {total_recon_loss:.4f} | VQ Loss: {total_vq_loss:.4f}")

    # Save at final epoch
    if epoch == num_epochs - 1:
        torch.save(model.state_dict(), f"{checkpoint_dir}/vqvae_epoch_{epoch+1}.pth")
        print(f"💾 Saved checkpoint for epoch {epoch+1}")


print("🎯 Training complete. Evaluating 10 batches of reconstruction...")

model.eval()
original_batches = []
reconstructed_batches = []

with torch.no_grad():
    for i, (x_particles, x_jets, _) in enumerate(dataloader):
        if i >= 15:
            break
        x_jets = x_jets.to(device)
        x_particles = x_particles.to(device)
        mean = x_jets.mean(dim=0, keepdim=True)  # Shape: (1, 4)
        std = x_jets.std(dim=0, keepdim=True) + 1e-6
        x_norm = (x_jets - mean) / std

        recon, _ = model(x_norm)
        recon_denorm = recon * std + mean

        original_batches.append(x_particles.mean(dim=2))  # Convert to (B, 4)
        reconstructed_batches.append(recon_denorm)

    x_orig_all = torch.cat(original_batches, dim=0).cpu()
    x_recon_all = torch.cat(reconstructed_batches, dim=0).cpu()

    plot_tensor(
        x_orig_all,
        x_recon_all,
        label="Reconstructed",
        filename=f"{checkpoint_dir}/reconstructed_epoch.png"
    )


