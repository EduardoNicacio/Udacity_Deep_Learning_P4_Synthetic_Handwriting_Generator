"""
train_diffusion.py
------------------
Defines training loop and sampling utilities for the Conditional Diffusion Model.

Training follows the DDPM objective: sample a random timestep, add the
corresponding amount of Gaussian noise, and regress the noise with the UNet.
Sampling runs the reverse Markov chain from ``t = T - 1`` down to ``0``.
"""

import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from utils.checkpoint import save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """Build a linear noise schedule.

    Args:
        timesteps: Total number of diffusion steps ``T``.

    Returns:
        Tensor of betas with shape ``(T,)``.
    """
    beta_start, beta_end = 1e-4, 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


@torch.no_grad()
def sample_images(
    model: torch.nn.Module,
    device: torch.device | str,
    num_samples: int = 16,
    num_classes: int = 10,
    img_size: tuple[int, int, int] = (1, 28, 28),
    timesteps: int = 200,
    class_labels: torch.Tensor | None = None,
    betas: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate images with the DDPM reverse process.

    Sampling runs in float32 without autocast for numerical stability.

    Args:
        model: Trained conditional UNet noise predictor.
        device: Torch device used for sampling.
        num_samples: Number of images to generate.
        num_classes: Number of conditioning classes.
        img_size: Shape of a single image as ``(C, H, W)``.
        timesteps: Number of reverse steps ``T``.
        class_labels: Optional labels of shape ``(num_samples,)``. Defaults to
            cycling through the classes.
        betas: Optional pre-computed beta schedule of shape ``(T,)``. When
            omitted, a linear schedule is built.

    Returns:
        Tuple of generated images in ``[-1, 1]`` with shape
        ``(num_samples, C, H, W)`` and their labels.
    """
    device = torch.device(device)
    betas = (
        linear_beta_schedule(timesteps).to(device)
        if betas is None
        else betas.to(device)
    )
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)

    imgs = torch.randn(num_samples, *img_size, device=device, dtype=torch.float32)

    if class_labels is None:
        labels = torch.tensor(
            [i % num_classes for i in range(num_samples)], device=device
        )
    else:
        labels = class_labels.to(device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for t in reversed(range(timesteps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            pred_noise = model(imgs, t_tensor, labels)

            alpha = alphas[t]
            alpha_bar = alphas_cumprod[t]
            beta = betas[t]
            noise = torch.randn_like(imgs) if t > 0 else torch.zeros_like(imgs)

            imgs = (1.0 / torch.sqrt(alpha)) * (
                imgs - (beta / torch.sqrt(1.0 - alpha_bar)) * pred_noise
            ) + torch.sqrt(beta) * noise

    if was_training:
        model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return torch.clamp(imgs, -1.0, 1.0), labels


def _save_epoch_checkpoints(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: Path,
) -> None:
    """Save ``diffusion_epoch_{epoch}.pt`` and overwrite ``diffusion_latest.pt``.

    Args:
        model: Trained denoiser.
        optimizer: Optimiser used during training.
        epoch: 1-based epoch number.
        loss: Average training loss for the epoch.
        checkpoint_dir: Directory where checkpoint files are written.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    epoch_path = checkpoint_dir / f"diffusion_epoch_{epoch}.pt"
    torch.save(state, epoch_path)
    shutil.copyfile(epoch_path, checkpoint_dir / "diffusion_latest.pt")
    print(f"Saved checkpoint: {epoch_path} (and diffusion_latest.pt)")


def train_diffusion(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device | str,
    num_classes: int,
    timesteps: int = 200,
    epochs: int = 20,
    lr: float = 1e-4,
    checkpoint_dir: str | Path | None = None,
    start_epoch: int = 0,
) -> dict[str, list[float]]:
    """Train a conditional UNet to predict diffusion noise.

    Args:
        model: Conditional UNet noise predictor.
        dataloader: Iterable yielding ``(images, labels)`` batches.
        device: Torch device used for training.
        num_classes: Number of conditioning classes.
        timesteps: Number of diffusion steps ``T``.
        epochs: Number of epochs to train.
        lr: Adam learning rate.
        checkpoint_dir: Destination for checkpoints. Defaults to
            ``<project_root>/checkpoints``.
        start_epoch: Number of epochs already completed (for resuming).

    Returns:
        Dictionary with the per-epoch ``loss`` history.

    Raises:
        RuntimeError: If a CUDA out-of-memory error persists after halving the
            offending batch and retrying once.
    """
    device = torch.device(device)
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
    amp_enabled = device.type == "cuda"
    scaler = GradScaler(device=device.type, enabled=amp_enabled)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)

    model.train()
    history: dict[str, list[float]] = {"loss": []}

    def train_batch(imgs: torch.Tensor, labels: torch.Tensor) -> float:
        """Run one forward-diffusion / noise-regression step."""
        t = torch.randint(0, timesteps, (imgs.size(0),), device=device).long()
        noise = torch.randn_like(imgs)

        sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alphas_cumprod[t])[
            :, None, None, None
        ]
        noisy_imgs = sqrt_alpha_bar * imgs + sqrt_one_minus_alpha_bar * noise

        with autocast(device_type=device.type, enabled=amp_enabled):
            pred_noise = model(noisy_imgs, t, labels)
            loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return float(loss.detach())

    for epoch in range(start_epoch, start_epoch + epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{start_epoch + epochs}")
        epoch_loss, batches = 0.0, 0

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            try:
                loss = train_batch(imgs, labels)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"Epoch {epoch + 1}: CUDA OOM - retrying with batch size "
                      f"{max(1, imgs.size(0) // 2)}")
                loss_total, halves = 0.0, 0
                try:
                    for chunk in torch.chunk(imgs, 2, dim=0):
                        offset = halves * chunk.size(0)
                        chunk_labels = labels[offset:offset + chunk.size(0)]
                        loss_total += train_batch(chunk, chunk_labels)
                        halves += 1
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA out of memory persists after halving the batch size. "
                        "Reduce batch_size (64 -> 32) and restart training."
                    ) from exc
                loss = loss_total / max(1, halves)

            epoch_loss += loss
            batches += 1
            pbar.set_postfix(loss=f"{loss:.4f}")

        mean_loss = epoch_loss / max(1, batches)
        history["loss"].append(mean_loss)
        print(f"Epoch {epoch + 1} completed. Loss: {mean_loss:.4f}")

        _save_epoch_checkpoints(model, optimizer, epoch + 1, mean_loss, ckpt_dir)
        if amp_enabled:
            torch.cuda.empty_cache()

    save_checkpoint(
        model,
        optimizer,
        loss=history["loss"][-1] if history["loss"] else None,
        name="diffusion_unet",
        path=str(ckpt_dir),
    )
    return history
