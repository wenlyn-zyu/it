"""Model shape and output range tests for RDPC."""

import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import Encoder, Decoder, Discriminator, Classifier


def test_encoder_output_shape():
    model = Encoder(latent_dim=16)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 16)


def test_encoder_different_latent_dims():
    for k in [4, 8, 32, 64]:
        model = Encoder(latent_dim=k)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, k)


def test_decoder_output_shape_and_range():
    encoder = Encoder(latent_dim=16)
    decoder = Decoder(latent_dim=16)
    x = torch.randn(4, 1, 28, 28)
    z = encoder(x)
    x_hat = decoder(z)
    assert x_hat.shape == (4, 1, 28, 28)
    assert x_hat.min() >= 0.0
    assert x_hat.max() <= 1.0


def test_discriminator_output_shape():
    d = Discriminator()
    x = torch.randn(8, 1, 28, 28)
    out = d(x)
    assert out.shape == (8, 1)


def test_classifier_output_shape():
    clf = Classifier(num_classes=10)
    x = torch.randn(8, 1, 28, 28)
    out = clf(x)
    assert out.shape == (8, 10)


def test_classifier_num_classes():
    clf = Classifier(num_classes=5)
    x = torch.randn(4, 1, 28, 28)
    out = clf(x)
    assert out.shape == (4, 5)
