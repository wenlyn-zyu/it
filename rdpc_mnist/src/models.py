"""Encoder, Decoder, WGAN-GP Discriminator, and Classifier for RDPC on MNIST."""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Conv encoder: 28×28×1 → latent Z ∈ R^k."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 14×14
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 7×7
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 4×4
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Flatten(),
        )
        conv_out = 128 * 4 * 4
        self.fc = nn.Linear(conv_out, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc(h)


class Decoder(nn.Module):
    """ConvTranspose decoder: Z ∈ R^k → 28×28×1."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=0),  # 7×7→8×8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14×14
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 28×28
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 128, 4, 4)
        return self.net(h)


class Discriminator(nn.Module):
    """WGAN critic: 28×28×1 → scalar score. No sigmoid (Wasserstein)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 14×14
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 7×7
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 4×4
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
        )
        conv_out = 128 * 4 * 4
        self.fc = nn.Linear(conv_out, 1)

    def forward(self, x):
        h = self.net(x)
        return self.fc(h)


class Classifier(nn.Module):
    """CNN classifier: 28×28×1 → 10-class logits."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 14×14
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 7×7
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),  # 1×1
            nn.Flatten(),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        h = self.net(x)
        return self.fc(h)
