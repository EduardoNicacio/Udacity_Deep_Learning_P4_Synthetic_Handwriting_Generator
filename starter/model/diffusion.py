"""
diffusion.py
------------
Implements a lightweight Conditional UNet denoiser for MNIST-sized diffusion training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def timestep_embedding(timesteps, dim):
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / half)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.time_emb = nn.Linear(time_dim, out_channels)
        self.label_emb = nn.Embedding(num_classes, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t, y):
        h = F.silu(self.norm1(self.conv1(x)))
        emb = self.time_emb(t)[:, :, None, None] + self.label_emb(y)[:, :, None, None]
        h = h + emb
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)

class ConditionalUNet(nn.Module):
    def __init__(self, num_classes=10, img_channels=1, base_channels=64, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.down1 = ResidualBlock(img_channels, base_channels, time_dim, num_classes)
        self.down2 = ResidualBlock(base_channels, base_channels*2, time_dim, num_classes)
        self.down3 = ResidualBlock(base_channels*2, base_channels*4, time_dim, num_classes)

        self.mid = ResidualBlock(base_channels*4, base_channels*4, time_dim, num_classes)

        self.up3 = ResidualBlock(base_channels * 6, base_channels * 2, time_dim, num_classes)
        self.up2 = ResidualBlock(base_channels * 3, base_channels, time_dim, num_classes)
        self.up1 = ResidualBlock(base_channels + img_channels, base_channels, time_dim, num_classes)

        self.output = nn.Conv2d(base_channels, img_channels, 1)

    def forward(self, x, t, y):
        t_emb = self.time_mlp(timestep_embedding(t, t.shape[0] * 0 + 128))
        h1 = self.down1(x, t_emb, y)
        h2 = self.down2(F.avg_pool2d(h1, 2), t_emb, y)
        h3 = self.down3(F.avg_pool2d(h2, 2), t_emb, y)

        h_mid = self.mid(h3, t_emb, y)

        h = self.up3(torch.cat([F.interpolate(h_mid, scale_factor=2), h2], dim=1), t_emb, y)
        h = self.up2(torch.cat([F.interpolate(h, scale_factor=2), h1], dim=1), t_emb, y)
        h = self.up1(torch.cat([F.interpolate(h, scale_factor=1), x], dim=1), t_emb, y)

        return self.output(h)