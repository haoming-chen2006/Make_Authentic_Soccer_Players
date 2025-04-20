import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import Generator, Discriminator

# === Configuration ===
dataroot = "data"         # Make sure your images are here in subfolders
workers = 2
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 1000
lr = 0.0002
beta1 = 0.5
ngpu = 1

# Device setup
device = torch.device("cuda:0" if (torch.cuda.is_available() and config.ngpu > 0) else "cpu")

# Weight initialization function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Dataset and DataLoader
dataset = dset.ImageFolder(root=config.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(config.image_size),
                               transforms.CenterCrop(config.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               transforms.RandomHorizontalFlip(),
                           ]))
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)

# Model setup
netG = Generator(config.ngpu, config.nz, config.ngf, config.nc).to(device)
netG.apply(weights_init)

netD = Discriminator(config.ngpu, config.ndf, config.nc).to(device)
netD.apply(weights_init)

# Loss and optimizers
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, config.nz, 1, 1, device=device)
real_label = 1.0
fake_label = 0.0

optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

# Create checkpoint directory
os.makedirs("checkpoints", exist_ok=True)

# Training Loop
print("Starting Training Loop...")
for epoch in range(1000):
    netG.train()
    netD.train()
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network
        ###########################
        netD.zero_grad()
        real_images = data[0].to(device)

        # Add noise to real images
        real_images += 0.05 * torch.randn_like(real_images)

        b_size = real_images.size(0)
        real_label_val = 0.9
        fake_label_val = 0.1

        # Train with real images
        output = netD(real_images)
        label = torch.full_like(output, real_label_val, dtype=torch.float)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Train with fake images
        noise = torch.randn(b_size, config.nz, 1, 1, device=device)
        fake = netG(noise)
        output = netD(fake.detach())
        label = torch.full_like(output, fake_label_val, dtype=torch.float)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
        optimizerD.step()

        ############################
        # (2) Update G network twice
        ###########################
        for _ in range(2):
            netG.zero_grad()
            noise2 = torch.randn(b_size, config.nz, 1, 1, device=device)
            fake2 = netG(noise2)
            output = netD(fake2)
            label = torch.full_like(output, real_label_val, dtype=torch.float)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

        D_G_z2 = output.mean().item()

    # === End of epoch ===
    print("[{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f} / {:.4f}".format(
        epoch+1, config.num_epochs, (errD_real + errD_fake), errG, D_x, D_G_z1, D_G_z2
    ))

    # === Save weights and fake images every 10 epochs ===
    if (epoch + 1) % 10 == 0 or epoch == 0:
        torch.save(netG.state_dict(), f"checkpoints/netG_epoch_{epoch+1}.pth")
        torch.save(netD.state_dict(), f"checkpoints/netD_epoch_{epoch+1}.pth")

        with torch.no_grad():
            fake_samples = netG(fixed_noise).detach().cpu()
        vutils.save_image(
            fake_samples,
            f"checkpoints/fake_samples_epoch_{epoch+1}.png",
            normalize=True,
            nrow=8
        )


