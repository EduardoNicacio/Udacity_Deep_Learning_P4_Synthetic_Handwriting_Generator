"""
metrics.py
----------
Evaluation helpers for generative models: Frechet Inception Distance (FID) and
downstream classifier accuracy.

All heavy tensor work is batched so evaluation fits on a 4 GB GPU.
"""

from typing import Iterable

import torch
import torch.nn as nn
from torchmetrics.image.fid import FrechetInceptionDistance


def _prepare_for_inception(
    imgs: torch.Tensor, input_range: tuple[float, float]
) -> torch.Tensor:
    """Convert greyscale images to 3-channel ``[0, 1]`` tensors for Inception.

    Args:
        imgs: Images of shape ``(B, 1, H, W)`` or ``(B, 3, H, W)``.
        input_range: ``(min, max)`` of the incoming pixel values.

    Returns:
        RGB images in ``[0, 1]`` with shape ``(B, 3, H, W)``.
    """
    low, high = input_range
    imgs = (imgs - low) / (high - low)
    imgs = imgs.clamp(0.0, 1.0)
    if imgs.size(1) == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
    return imgs


def fid_score(
    real_imgs: torch.Tensor,
    fake_imgs: torch.Tensor,
    device: torch.device | str = "cuda",
    batch_size: int = 32,
    input_range: tuple[float, float] = (-1.0, 1.0),
) -> float:
    """Compute FID between two image sets using torchmetrics' InceptionV3.

    Args:
        real_imgs: Real images with shape ``(N, 1, H, W)``.
        fake_imgs: Generated images with shape ``(M, 1, H, W)``.
        device: Torch device used for Inception evaluation.
        batch_size: Number of images per forward pass.
        input_range: ``(min, max)`` of the incoming pixel values. Defaults to
            ``(-1, 1)``, matching the GAN/diffusion output range.

    Returns:
        The FID score (lower is better).
    """
    device = torch.device(device)
    fid = FrechetInceptionDistance(normalize=True).to(device)
    fid.reset()

    def batched_update(imgs: torch.Tensor, real: bool) -> None:
        imgs = _prepare_for_inception(imgs, input_range)
        for start in range(0, imgs.size(0), batch_size):
            batch = imgs[start:start + batch_size].to(device, non_blocking=True)
            fid.update(batch, real=real)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    batched_update(real_imgs, real=True)
    batched_update(fake_imgs, real=False)

    score = float(fid.compute().item())
    del fid
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score


@torch.no_grad()
def classifier_accuracy(
    model: nn.Module,
    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str = "cuda",
) -> float:
    """Compute top-1 accuracy of a classifier on a labelled dataloader.

    Args:
        model: Classifier returning logits of shape ``(B, num_classes)``.
        dataloader: Iterable yielding ``(images, labels)`` batches.
        device: Torch device used for evaluation.

    Returns:
        Accuracy in ``[0, 1]``.
    """
    device = torch.device(device)
    was_training = model.training
    model.eval()

    correct, total = 0, 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())

    if was_training:
        model.train()
    return correct / max(1, total)
