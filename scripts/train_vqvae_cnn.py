import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from models.vqvaeCNN import VQVAE

def main():
    # === Configuration ===
    dataroot = "/Users/haoming/Desktop/Make_Authentic_Soccer_Players/data/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 2 if device.type == "cuda" else 0  # safer on macOS CPU
    batch_size = 128
    image_size = 128
    nc = 3
    num_epochs = 10
    lr = 2e-4
    embedding_dim = 64
    num_embeddings = 512
    hidden_channels = 128
    res_hidden_channels = 32
    num_res_layers = 2
    commitment_cost = 0.25
    checkpoint_dir = "checkpoints/checkpoints_vqvae_cnn"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # === Dataset and DataLoader ===
    dataset = dset.ImageFolder(
        root=dataroot,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # === Model ===
    model = VQVAE(
        in_channels=nc,
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        hidden_channels=hidden_channels,
        res_hidden_channels=res_hidden_channels,
        num_res_layers=num_res_layers
    ).to(device)

    # === Load Latest Checkpoint (if any) ===
    latest_checkpoint = None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("vqvae_epoch_") and f.endswith(".pth")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("No checkpoint found. Starting from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    recon_loss_fn = nn.MSELoss()

    # === Training Loop ===
    for epoch in range(num_epochs):
        model.train()
        total_recon_loss = 0.0
        total_vq_loss = 0.0

        for i, data in enumerate(dataloader, 0):
            real_images = data[0].to(device)

            optimizer.zero_grad()
            recon_images, vq_loss = model(real_images)
            recon_loss = recon_loss_fn(recon_images, real_images)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()

        print(f"[{epoch+1}/{num_epochs}] Recon Loss: {total_recon_loss:.4f} | VQ Loss: {total_vq_loss:.4f}")

        # Final epoch save
        if (epoch + 1) == num_epochs:
            # Save model checkpoint
            torch.save(model.state_dict(), f"{checkpoint_dir}/vqvae_epoch_{num_epochs}.pth")

            # Save reconstructed images
            model.eval()
            with torch.no_grad():
                example_images = next(iter(dataloader))[0].to(device)[:64]
                recon_images, _ = model(example_images)
                recon = recon_images.detach().cpu()
            vutils.save_image(recon, f"{checkpoint_dir}/reconstructed_epoch_{num_epochs}.png", normalize=True, nrow=8)

if __name__ == "__main__":
    # Fix multiprocessing on macOS
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()

