"""MNIST data loading, FID helper, and rate estimation utilities."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


def get_mnist_loaders(batch_size: int = 128, data_dir: str = "./data"):
    """Return train and test DataLoaders for MNIST (28×28, single channel)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
    return train_loader, test_loader


def estimate_rate(z: torch.Tensor) -> float:
    """Estimate I(X;Z) via Gaussian entropy upper bound: 0.5 * sum_i log(1 + var_i).

    Assumes z has shape (N, k) with zero mean across batch.
    """
    var = z.var(dim=0, unbiased=True)  # (k,)
    var = var.clamp(min=1e-8)
    rate = 0.5 * torch.log1p(var).sum() / np.log(2)  # bits per k dimensions
    return float(rate.item())


def compute_mse(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    return float(torch.mean((x - x_hat) ** 2).item())


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())
