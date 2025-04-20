import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_res_layers, num_res_hiddens):
        super().__init__()
        layers = []
        for _ in range(num_res_layers):
            layers.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(num_hiddens, num_res_hiddens, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(num_res_hiddens, num_hiddens, kernel_size=1),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, res_hidden_channels, num_res_layers):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),  # 128 → 64
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=4, stride=2, padding=1),  # 64 → 32
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        )
        self.res_stack = ResidualStack(hidden_channels, num_res_layers, res_hidden_channels)

    def forward(self, x):
        x = self.conv(x)
        return self.res_stack(x)


class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_channels, res_hidden_channels, num_res_layers):
        super().__init__()
        self.conv = nn.Conv2d(embedding_dim, hidden_channels, kernel_size=3, padding=1)
        self.res_stack = ResidualStack(hidden_channels, num_res_layers, res_hidden_channels)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),  # 32 → 64
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels // 2, 3, kernel_size=4, stride=2, padding=1),  # 64 → 128
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.res_stack(x)
        return self.deconv(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        flat_z = z_perm.view(-1, self.embedding_dim)  # (BHW, C)

        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_z, self.embeddings.weight.t())
            + torch.sum(self.embeddings.weight ** 2, dim=1)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(encoding_indices, self.embeddings.weight)
        quantized = quantized.view(z_perm.shape).permute(0, 3, 1, 2).contiguous()

        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = z + (quantized - z).detach()  # Straight-through estimator

        return quantized, loss


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embedding_dim=64,
        num_embeddings=512,
        hidden_channels=128,
        res_hidden_channels=32,
        num_res_layers=2
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels, res_hidden_channels, num_res_layers)
        self.pre_vq = nn.Conv2d(hidden_channels, embedding_dim, kernel_size=1)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_channels, res_hidden_channels, num_res_layers)

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_vq(z)
        quantized, vq_loss = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss
