"""
visualize.py
------------
Utility functions for displaying batches of images.
"""

import matplotlib.pyplot as plt
import torch

def show_batch(images, labels, n=16):
    """
    Visualizes a batch of MNIST images with labels.
    """
    n = min(n, len(images))
    images = images[:n]
    labels = labels[:n]
    grid_size = int(n ** 0.5)
    side = max(6.0, grid_size * 1.4)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(side, side))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(images[i].squeeze().cpu().numpy(), cmap="gray")
        ax.set_title(f"Label: {labels[i].item()}", fontsize=8 if grid_size <= 6 else 6)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
