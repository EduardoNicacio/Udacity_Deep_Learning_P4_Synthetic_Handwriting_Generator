"""
cgan.py
-------
Defines Generator and Discriminator architectures for Conditional GAN.

Both networks are conditioned on a class label (digit 0-9). The Generator
concatenates a learned label embedding with the latent noise vector, while the
Discriminator broadcasts the label embedding as an additional image channel.
"""

import math

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Fully-connected conditional generator for 28x28 greyscale digits.

    The latent vector is concatenated with a learned class embedding before
    being upsampled through a stack of linear layers.

    Args:
        z_dim: Dimensionality of the latent noise vector.
        num_classes: Number of conditioning classes (digits).
        img_shape: Shape of the generated image as ``(C, H, W)``.
        embed_dim: Dimensionality of the class embedding.

    Attributes:
        latent_dim: Alias for ``z_dim``.
        label_emb: Embedding table mapping class labels to vectors.
    """

    def __init__(
        self,
        z_dim: int = 100,
        num_classes: int = 10,
        img_shape: tuple[int, int, int] = (1, 28, 28),
        embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.latent_dim = z_dim
        self.img_shape = img_shape
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        out_features = int(math.prod(img_shape))
        self.model = nn.Sequential(
            nn.Linear(z_dim + embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, out_features),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate images conditioned on class labels.

        Args:
            noise: Latent vectors of shape ``(B, z_dim)``.
            labels: Integer class labels of shape ``(B,)``.

        Returns:
            Generated images in ``[-1, 1]`` with shape ``(B, C, H, W)``.
        """
        x = torch.cat((noise, self.label_emb(labels)), dim=1)
        img = self.model(x)
        return img.view(img.size(0), self.img_shape[0], self.img_shape[1], self.img_shape[2])


class Discriminator(nn.Module):
    """Fully-connected conditional discriminator for 28x28 greyscale digits.

    The class label embedding is broadcast to a full spatial channel and
    concatenated with the image, so the network receives ``(B, 2, H, W)``.

    Args:
        num_classes: Number of conditioning classes (digits).
        img_shape: Shape of the input image as ``(C, H, W)``.
        dropout: Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        num_classes: int = 10,
        img_shape: tuple[int, int, int] = (1, 28, 28),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, 1)

        in_features = int((img_shape[0] + 1) * math.prod(img_shape[1:]))
        self.model = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Score images as real (1) or fake (0), conditioned on class labels.

        Args:
            img: Input images of shape ``(B, C, H, W)``.
            labels: Integer class labels of shape ``(B,)``.

        Returns:
            Probabilities of shape ``(B, 1)``.
        """
        label_channel = self.label_emb(labels).view(-1, 1, 1, 1)
        label_channel = label_channel.expand(-1, 1, self.img_shape[1], self.img_shape[2])
        x = torch.cat((img, label_channel), dim=1)
        x = x.view(x.size(0), -1)
        return self.model(x)
