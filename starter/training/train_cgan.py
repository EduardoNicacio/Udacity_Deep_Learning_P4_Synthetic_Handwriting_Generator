"""
train_cgan.py
-------------
Reusable training function for cGAN experiments.

The loop trains the Discriminator first (real vs. fake) and then the Generator
(fool the Discriminator) for every batch, using mixed precision on CUDA.
"""

import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from utils.checkpoint import save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def _save_epoch_checkpoints(
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    epoch: int,
    g_loss: float,
    d_loss: float,
    checkpoint_dir: Path,
) -> None:
    """Persist per-epoch checkpoints plus an overwritten ``cgan_latest.pt``.

    Args:
        generator: Trained generator module.
        discriminator: Trained discriminator module.
        optimizer_G: Generator optimiser.
        optimizer_D: Discriminator optimiser.
        epoch: 1-based epoch number.
        g_loss: Average generator loss for the epoch.
        d_loss: Average discriminator loss for the epoch.
        checkpoint_dir: Directory where checkpoint files are written.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        generator,
        optimizer_G,
        epoch,
        g_loss,
        name="cgan_generator",
        path=str(checkpoint_dir),
    )
    save_checkpoint(
        discriminator,
        optimizer_D,
        epoch,
        d_loss,
        name="cgan_discriminator",
        path=str(checkpoint_dir),
    )

    state: dict[str, Any] = {
        "epoch": epoch,
        "generator_state": generator.state_dict(),
        "discriminator_state": discriminator.state_dict(),
        "optimizer_G_state": optimizer_G.state_dict(),
        "optimizer_D_state": optimizer_D.state_dict(),
        "g_loss": g_loss,
        "d_loss": d_loss,
    }
    epoch_path = checkpoint_dir / f"cgan_epoch_{epoch}.pt"
    torch.save(state, epoch_path)
    shutil.copyfile(epoch_path, checkpoint_dir / "cgan_latest.pt")
    print(f"Saved checkpoint: {epoch_path} (and cgan_latest.pt)")


def train_cgan(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str,
    z_dim: int,
    num_classes: int,
    epochs: int = 50,
    checkpoint_dir: str | Path | None = None,
    start_epoch: int = 0,
) -> dict[str, list[float]]:
    """Train a conditional GAN with the BCE minimax objective.

    Args:
        generator: Conditional generator network.
        discriminator: Conditional discriminator network.
        dataloader: Iterable yielding ``(images, labels)`` batches.
        optimizer_G: Optimiser for the generator parameters.
        optimizer_D: Optimiser for the discriminator parameters.
        criterion: Binary cross-entropy loss (``nn.BCELoss``).
        device: Torch device used for training.
        z_dim: Latent noise dimensionality expected by the generator.
        num_classes: Number of conditioning classes.
        epochs: Number of epochs to train.
        checkpoint_dir: Destination for checkpoints. Defaults to
            ``<project_root>/checkpoints``.
        start_epoch: Number of epochs already completed (for resuming).

    Returns:
        Dictionary with per-epoch ``g_loss`` and ``d_loss`` history.

    Raises:
        RuntimeError: If a CUDA out-of-memory error persists after splitting the
            offending batch in half and retrying once.
    """
    device = torch.device(device)
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
    amp_enabled = device.type == "cuda"
    scaler = GradScaler(device=device.type, enabled=amp_enabled)

    generator.train()
    discriminator.train()

    history: dict[str, list[float]] = {"g_loss": [], "d_loss": []}

    def train_batch(imgs: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
        """Run one discriminator update followed by one generator update."""
        batch_size = imgs.size(0)
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # --- Discriminator update -------------------------------------------
        optimizer_D.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            real_pred = discriminator(imgs, labels)

            z = torch.randn(batch_size, z_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            gen_imgs = generator(z, gen_labels)

            fake_pred = discriminator(gen_imgs.detach(), gen_labels)

        # BCE is unsafe under autocast, so the losses are computed in fp32.
        real_loss = criterion(real_pred.float(), valid)
        fake_loss = criterion(fake_pred.float(), fake)
        d_loss = (real_loss + fake_loss) / 2.0

        scaler.scale(d_loss).backward()
        scaler.step(optimizer_D)

        # --- Generator update -----------------------------------------------
        optimizer_G.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            g_pred = discriminator(gen_imgs, gen_labels)

        g_loss = criterion(g_pred.float(), valid)

        scaler.scale(g_loss).backward()
        scaler.step(optimizer_G)
        scaler.update()

        return float(g_loss.detach()), float(d_loss.detach())

    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_g, epoch_d, batches = 0.0, 0.0, 0

        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            try:
                g_loss, d_loss = train_batch(imgs, labels)
            except torch.cuda.OutOfMemoryError:
                # Guard-rail for a 4 GB GPU: retry once with half the batch.
                torch.cuda.empty_cache()
                print(f"[Epoch {epoch + 1}] CUDA OOM - retrying with batch size "
                      f"{max(1, imgs.size(0) // 2)}")
                g_total, d_total, halves = 0.0, 0.0, 0
                try:
                    for chunk in torch.chunk(imgs, 2, dim=0):
                        offset = halves * chunk.size(0)
                        chunk_labels = labels[offset:offset + chunk.size(0)]
                        g_half, d_half = train_batch(chunk, chunk_labels)
                        g_total += g_half
                        d_total += d_half
                        halves += 1
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA out of memory persists after halving the batch size. "
                        "Reduce batch_size (128 -> 64) and restart training."
                    ) from exc
                g_loss, d_loss = g_total / max(1, halves), d_total / max(1, halves)

            epoch_g += g_loss
            epoch_d += d_loss
            batches += 1

        mean_g = epoch_g / max(1, batches)
        mean_d = epoch_d / max(1, batches)
        history["g_loss"].append(mean_g)
        history["d_loss"].append(mean_d)

        print(
            f"[Epoch {epoch + 1}] D loss: {mean_d:.4f} | G loss: {mean_g:.4f}"
        )

        _save_epoch_checkpoints(
            generator,
            discriminator,
            optimizer_G,
            optimizer_D,
            epoch + 1,
            mean_g,
            mean_d,
            ckpt_dir,
        )
        if amp_enabled:
            torch.cuda.empty_cache()

    return history
