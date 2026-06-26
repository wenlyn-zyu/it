"""Sweep over latent dimensions and λ values to map the R-D-P-C trade-off surface."""

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training import pretrain_classifier, train_rdpc


def main():
    parser = argparse.ArgumentParser(description="RDPC λ-sweep on MNIST")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu"
    print(f"Device: {device}")

    k_values = [4, 8, 16, 32, 64]
    lambda_p_values = [0.01, 0.1, 1.0]
    lambda_c_values = [0.1, 1.0, 10.0]

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "sweep_results.csv"

    print("Pre-training classifier (shared across all configs)...")
    classifier = pretrain_classifier(16, epochs=5, data_dir=args.data_dir, device=device)
    print("Classifier ready.\n")

    results = []
    total = len(k_values) * len(lambda_p_values) * len(lambda_c_values)
    idx = 0

    for k in k_values:
        for lp in lambda_p_values:
            for lc in lambda_c_values:
                idx += 1
                print(f"\n[{idx}/{total}] k={k}, λ_d=1.0, λ_p={lp}, λ_c={lc}")
                result = train_rdpc(
                    latent_dim=k,
                    lambda_d=1.0,
                    lambda_p=lp,
                    lambda_c=lc,
                    epochs=args.epochs,
                    data_dir=args.data_dir,
                    device=device,
                    classifier=classifier,
                    log_interval=500,
                )
                results.append({
                    "k": k, "lambda_d": 1.0, "lambda_p": lp, "lambda_c": lc,
                    "rate": result["test_rate"], "mse": result["test_mse"], "acc": result["test_acc"],
                })

                # Write incrementally
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["k", "lambda_d", "lambda_p", "lambda_c", "rate", "mse", "acc"])
                    writer.writeheader()
                    writer.writerows(results)

    print(f"\nSweep complete. {len(results)} results saved to {csv_path}")


if __name__ == "__main__":
    main()
