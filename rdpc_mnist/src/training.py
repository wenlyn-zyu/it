"""Multi-task RDPC training loop with WGAN-GP perception constraint."""

import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from .models import Encoder, Decoder, Discriminator, Classifier
from .utils import estimate_rate, compute_mse, compute_accuracy


def gradient_penalty(discriminator, real, fake, device):
    """WGAN-GP gradient penalty on random interpolations."""
    B = real.size(0)
    eps = torch.rand(B, 1, 1, 1, device=device)
    interpolates = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    grads = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    grads = grads.view(B, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def pretrain_classifier(latent_dim: int, epochs: int = 5, lr: float = 1e-3,
                        data_dir: str = "./data", device: str = "cpu") -> Classifier:
    """Train a CNN classifier on real MNIST images. Returns frozen classifier."""
    from .utils import get_mnist_loaders
    train_loader, test_loader = get_mnist_loaders(batch_size=128, data_dir=data_dir)

    model = Classifier().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"  Classifier epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f}, test_acc={acc:.4f}")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def train_rdpc(
    latent_dim: int = 16,
    lambda_d: float = 1.0,
    lambda_p: float = 0.1,
    lambda_c: float = 1.0,
    gp_weight: float = 10.0,
    n_critic: int = 5,
    epochs: int = 30,
    lr_g: float = 1e-3,
    lr_d: float = 1e-4,
    data_dir: str = "./data",
    device: str = "cpu",
    classifier: Classifier = None,
    log_interval: int = 100,
    checkpoint_dir: str = None,
):
    """Train RDPC autoencoder with perception and classification constraints.

    Joint loss: L = λ_d·MSE(x,x̂) + λ_p·(-D(x̂)) + λ_c·CE(s,ŝ)
    where D is the WGAN critic (higher = more realistic).
    """
    from .utils import get_mnist_loaders
    train_loader, test_loader = get_mnist_loaders(batch_size=128, data_dir=data_dir)

    encoder = Encoder(latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    if classifier is None:
        classifier = Classifier().to(device)
    classifier.eval()

    opt_g = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr_g, betas=(0.5, 0.9))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9))

    ce_loss = nn.CrossEntropyLoss()

    metrics_history = []
    step = 0

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        discriminator.train()

        for x_real, labels in train_loader:
            x_real = x_real.to(device)
            labels = labels.to(device)
            B = x_real.size(0)

            # --- Discriminator update (n_critic times) ---
            d_loss_total = 0.0
            for _ in range(n_critic):
                with torch.no_grad():
                    z = encoder(x_real)
                    x_fake = decoder(z)

                d_real = discriminator(x_real)
                d_fake = discriminator(x_fake.detach())
                gp = gradient_penalty(discriminator, x_real, x_fake.detach(), device)
                d_loss = d_fake.mean() - d_real.mean() + gp_weight * gp

                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                d_loss_total += d_loss.item()

            # --- Generator update (encoder + decoder) ---
            z = encoder(x_real)
            x_hat = decoder(z)

            # (1a) Distortion: MSE
            loss_mse = nn.functional.mse_loss(x_hat, x_real)

            # (1b) Perception: push critic score higher
            d_hat = discriminator(x_hat)
            loss_perception = -d_hat.mean()

            # (1c) Classification: cross-entropy on reconstructed image
            logits = classifier(x_hat)
            loss_cls = ce_loss(logits, labels)

            loss_g = lambda_d * loss_mse + lambda_p * loss_perception + lambda_c * loss_cls

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            step += 1

            if step % log_interval == 0:
                rate = estimate_rate(z)
                metrics_history.append({
                    "epoch": epoch,
                    "step": step,
                    "rate_bits": rate,
                    "mse": loss_mse.item(),
                    "perception_loss": loss_perception.item(),
                    "cls_loss": loss_cls.item(),
                    "d_loss": d_loss_total / n_critic,
                })

        # Epoch-end evaluation
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            test_z_list = []
            test_mse = 0.0
            test_acc = 0.0
            n_test = 0
            for x_test, y_test in test_loader:
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                z_test = encoder(x_test)
                x_hat_test = decoder(z_test)
                test_z_list.append(z_test)
                test_mse += nn.functional.mse_loss(x_hat_test, x_test, reduction='sum').item()
                logits_test = classifier(x_hat_test)
                test_acc += (logits_test.argmax(1) == y_test).sum().item()
                n_test += x_test.size(0)
            test_mse /= n_test
            test_acc /= n_test
            test_z = torch.cat(test_z_list, dim=0)
            test_rate = estimate_rate(test_z)

        print(f"Epoch {epoch+1}/{epochs}: R={test_rate:.3f}, MSE={test_mse:.4f}, ClsAcc={test_acc:.4f}")

    if checkpoint_dir:
        ckpt = Path(checkpoint_dir)
        ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(encoder.state_dict(), ckpt / f"encoder_k{latent_dim}_ld{lambda_d}_lp{lambda_p}_lc{lambda_c}.pt")
        torch.save(decoder.state_dict(), ckpt / f"decoder_k{latent_dim}_ld{lambda_d}_lp{lambda_p}_lc{lambda_c}.pt")

    return {
        "latent_dim": latent_dim,
        "lambda_d": lambda_d,
        "lambda_p": lambda_p,
        "lambda_c": lambda_c,
        "test_rate": test_rate,
        "test_mse": test_mse,
        "test_acc": test_acc,
        "metrics_history": metrics_history,
    }
