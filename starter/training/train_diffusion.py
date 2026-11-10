"""
train_diffusion.py
------------------
Defines training loop and sampling utilities for the Conditional Diffusion Model.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from utils.checkpoint import save_checkpoint
from tqdm import tqdm


def linear_beta_schedule(timesteps):
    beta_start, beta_end = 1e-4, 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


@torch.no_grad()
def sample_images(
    model,
    device,
    num_samples=16,
    num_classes=10,
    img_size=(1, 28, 28),
    timesteps=200,
    class_labels=None,
):
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)

    imgs = torch.randn(num_samples, *img_size).to(device)

    if class_labels is None:
        labels = torch.tensor(
            [i % num_classes for i in range(num_samples)], device=device
        )
    else:
        labels = class_labels.to(device)

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        pred_noise = model(imgs, t_tensor, labels)
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        noise = torch.randn_like(imgs) if t > 0 else torch.zeros_like(imgs)
        imgs = (1 / torch.sqrt(alpha)) * (
            imgs - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise
        ) + torch.sqrt(betas[t]) * noise

    return imgs, labels


def train_diffusion(
    model, dataloader, device, num_classes, timesteps=200, epochs=20, lr=1e-4
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            t = torch.randint(0, timesteps, (imgs.size(0),), device=device).long()
            noise = torch.randn_like(imgs)
            sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alphas_cumprod[t])[
                :, None, None, None
            ]
            noisy_imgs = sqrt_alpha_bar * imgs + sqrt_one_minus_alpha_bar * noise

            pred_noise = model(noisy_imgs, t, labels)
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} completed. Loss: {loss.item():.4f}")

        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch + 1, loss.item(), name="diffusion_unet"
            )

    save_checkpoint(model, optimizer, name="diffusion_unet", epoch="final")
