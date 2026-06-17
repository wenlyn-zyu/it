"""Plot R-D-P-C trade-off curves and reconstructed sample grids."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def plot_tradeoffs(csv_path: str, output_dir: str):
    """Generate trade-off scatter/multi-curve plots from sweep CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            rows.append({
                "k": int(row["k"]),
                "lambda_p": float(row["lambda_p"]),
                "lambda_c": float(row["lambda_c"]),
                "rate": float(row["rate"]),
                "mse": float(row["mse"]),
                "acc": float(row["acc"]),
            })

    if not rows:
        print("No data found.")
        return

    # --- Plot 1: Rate vs MSE colored by k ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for k in sorted({r["k"] for r in rows}):
        pts = [r for r in rows if r["k"] == k]
        pts.sort(key=lambda r: r["rate"])
        x, y = [p["rate"] for p in pts], [p["mse"] for p in pts]
        axes[0].plot(x, y, marker="o", label=f"k={k}")
    axes[0].set_xlabel("Rate R (bits)")
    axes[0].set_ylabel("MSE (Distortion)")
    axes[0].set_title("Rate-Distortion Trade-off")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: Rate vs Accuracy colored by k ---
    for k in sorted({r["k"] for r in rows}):
        pts = [r for r in rows if r["k"] == k]
        pts.sort(key=lambda r: r["rate"])
        x, y = [p["rate"] for p in pts], [p["acc"] for p in pts]
        axes[1].plot(x, y, marker="s", label=f"k={k}")
    axes[1].set_xlabel("Rate R (bits)")
    axes[1].set_ylabel("Classification Accuracy")
    axes[1].set_title("Rate-Classification Trade-off")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    # --- Plot 3: MSE vs Accuracy (D-C trade-off) colored by k ---
    for k in sorted({r["k"] for r in rows}):
        pts = [r for r in rows if r["k"] == k]
        x, y = [p["mse"] for p in pts], [p["acc"] for p in pts]
        axes[2].scatter(x, y, label=f"k={k}", s=20, alpha=0.7)
    axes[2].set_xlabel("MSE (Distortion)")
    axes[2].set_ylabel("Classification Accuracy")
    axes[2].set_title("Distortion-Classification Trade-off")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "rdpc_tradeoffs.png", dpi=150)
    plt.close(fig)
    print(f"Saved trade-off plot to {output_dir / 'rdpc_tradeoffs.png'}")

    # --- Plot 4: Effect of λ values at fixed k ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    k_fixed = 16
    pts_k = [r for r in rows if r["k"] == k_fixed]

    # λ_p effect (fix λ_c=1.0)
    lc_fixed = 1.0
    for lp in sorted({r["lambda_p"] for r in pts_k}):
        pts = [r for r in pts_k if abs(r["lambda_p"] - lp) < 1e-9 and abs(r["lambda_c"] - lc_fixed) < 1e-9]
        if pts:
            axes[0].bar(str(lp), pts[0]["mse"], label=f"λ_p={lp}")
    axes[0].set_xlabel("λ_p (perception weight)")
    axes[0].set_ylabel("MSE")
    axes[0].set_title(f"Perception Effect (k={k_fixed}, λ_c={lc_fixed})")

    # λ_c effect (fix λ_p=0.1)
    lp_fixed = 0.1
    for lc in sorted({r["lambda_c"] for r in pts_k}):
        pts = [r for r in pts_k if abs(r["lambda_c"] - lc) < 1e-9 and abs(r["lambda_p"] - lp_fixed) < 1e-9]
        if pts:
            axes[1].bar(str(lc), pts[0]["acc"], label=f"λ_c={lc}")
    axes[1].set_xlabel("λ_c (classification weight)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Classification Effect (k={k_fixed}, λ_p={lp_fixed})")

    plt.tight_layout()
    fig.savefig(output_dir / "rdpc_lambda_effects.png", dpi=150)
    plt.close(fig)
    print(f"Saved lambda effects plot to {output_dir / 'rdpc_lambda_effects.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RDPC trade-off results")
    parser.add_argument("--csv", type=str, default=None, help="Path to sweep CSV")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots")
    args = parser.parse_args()

    csv_path = args.csv or str(ROOT / "results" / "sweep_results.csv")
    output_dir = args.output_dir or str(ROOT / "results")
    plot_tradeoffs(csv_path, output_dir)
