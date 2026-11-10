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
    images = images[:n]
    labels = labels[:n]
    grid_size = int(n ** 0.5)
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(6,6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i].squeeze().cpu().numpy(), cmap="gray")
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
