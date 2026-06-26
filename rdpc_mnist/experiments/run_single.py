"""Train a single RDPC configuration on MNIST."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training import pretrain_classifier, train_rdpc


def main():
    parser = argparse.ArgumentParser(description="Train RDPC on MNIST")
    parser.add_argument("--k", type=int, default=16, help="Latent dimension")
    parser.add_argument("--lambda_d", type=float, default=1.0, help="Distortion weight")
    parser.add_argument("--lambda_p", type=float, default=0.1, help="Perception weight")
    parser.add_argument("--lambda_c", type=float, default=1.0, help="Classification weight")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--data_dir", type=str, default="./data", help="MNIST data directory")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    args = parser.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu"
    print(f"Device: {device}")

    checkpoint_dir = ROOT / "checkpoints"
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print("Pre-training classifier on real MNIST...")
    classifier = pretrain_classifier(args.k, epochs=5, data_dir=args.data_dir, device=device)
    print("Classifier ready.\n")

    print(f"Training RDPC: k={args.k}, λ_d={args.lambda_d}, λ_p={args.lambda_p}, λ_c={args.lambda_c}")
    result = train_rdpc(
        latent_dim=args.k,
        lambda_d=args.lambda_d,
        lambda_p=args.lambda_p,
        lambda_c=args.lambda_c,
        epochs=args.epochs,
        data_dir=args.data_dir,
        device=device,
        classifier=classifier,
        checkpoint_dir=str(checkpoint_dir),
    )

    # Save final metrics to CSV
    import csv
    csv_path = results_dir / "single_result.csv"
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "lambda_d", "lambda_p", "lambda_c", "rate", "mse", "acc"])
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({
            "k": args.k, "lambda_d": args.lambda_d, "lambda_p": args.lambda_p, "lambda_c": args.lambda_c,
            "rate": result["test_rate"], "mse": result["test_mse"], "acc": result["test_acc"],
        })

    print(f"\nFinal: R={result['test_rate']:.3f} bits, MSE={result['test_mse']:.4f}, Acc={result['test_acc']:.4f}")


if __name__ == "__main__":
    main()
