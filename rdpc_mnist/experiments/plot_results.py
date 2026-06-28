"""Plot representative R-D-P-C operating points.

The default CSV is the representative table used in the report, not a full
MNIST sweep. The plots intentionally use scatter/operating-point views instead
of connected curves so sparse points are not visually over-interpreted.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _read_rows(csv_path: str):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "k": int(row["k"]),
                "lambda_p": float(row["lambda_p"]),
                "lambda_c": float(row["lambda_c"]),
                "rate": float(row["rate"]),
                "mse": float(row["mse"]),
                "acc": float(row["acc"]),
            })
    return rows


def _set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _finish_axis(ax):
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_tradeoffs(csv_path: str, output_dir: str):
    """Generate publication-style plots from a sparse RDPC result CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(csv_path)
    if not rows:
        print("No data found.")
        return

    _set_style()
    k_values = sorted({r["k"] for r in rows})
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(k_values)))
    color_by_k = {k: colors[i] for i, k in enumerate(k_values)}
    marker_by_lp = {0.01: "o", 0.1: "s", 1.0: "^"}

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), constrained_layout=True)
    panels = [
        ("rate", "mse", "Rate proxy $\\hat{R}$ (bits)", "MSE", "Rate-distortion"),
        ("rate", "acc", "Rate proxy $\\hat{R}$ (bits)", "Accuracy", "Rate-classification"),
        ("mse", "acc", "MSE", "Accuracy", "Distortion-classification"),
    ]

    for ax, (x_key, y_key, x_label, y_label, title) in zip(axes, panels):
        for r in rows:
            ax.scatter(
                r[x_key],
                r[y_key],
                s=58 + 12 * np.log2(r["lambda_c"] + 1.0),
                color=color_by_k[r["k"]],
                marker=marker_by_lp.get(r["lambda_p"], "o"),
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        if y_key == "acc":
            ax.set_ylim(0.9765, 0.9910)
        _finish_axis(ax)

    for r in rows:
        if r["k"] in {4, 16, 32, 64}:
            axes[0].annotate(
                f"$k={r['k']}$",
                (r["rate"], r["mse"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="#333333",
            )

    k_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=color_by_k[k],
            markeredgecolor="white",
            label=f"$k={k}$",
        )
        for k in k_values
    ]
    lp_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker_by_lp[lp],
            linestyle="",
            color="#555555",
            label=f"$\\lambda_p={lp}$",
        )
        for lp in sorted(marker_by_lp)
        if any(abs(r["lambda_p"] - lp) < 1e-9 for r in rows)
    ]
    axes[2].legend(handles=k_handles + lp_handles, loc="lower right", frameon=False)
    fig.suptitle("Representative RDPC Operating Points", y=1.04, fontsize=11)
    fig.savefig(output_dir / "rdpc_tradeoffs.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trade-off plot to {output_dir / 'rdpc_tradeoffs.png'}")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
    k_fixed = 16
    pts_k = [r for r in rows if r["k"] == k_fixed]

    lc_fixed = 1.0
    pts_lp = [r for r in pts_k if abs(r["lambda_c"] - lc_fixed) < 1e-9]
    pts_lp.sort(key=lambda r: r["lambda_p"])
    axes[0].plot(
        [r["lambda_p"] for r in pts_lp],
        [r["mse"] for r in pts_lp],
        color="#2a6fbb",
        marker="o",
        linewidth=1.6,
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("$\\lambda_p$ (perception weight)")
    axes[0].set_ylabel("MSE")
    axes[0].set_title(f"$k={k_fixed}$, $\\lambda_c={lc_fixed}$")
    _finish_axis(axes[0])

    for r in pts_k:
        axes[1].scatter(
            r["mse"],
            r["acc"],
            s=75,
            color=color_by_k[k_fixed],
            marker=marker_by_lp.get(r["lambda_p"], "o"),
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        axes[1].annotate(
            f"$\\lambda_p={r['lambda_p']}$\n$\\lambda_c={r['lambda_c']}$",
            (r["mse"], r["acc"]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=7,
            va="center",
        )
    axes[1].set_xlabel("MSE")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"$k={k_fixed}$ operating points")
    axes[1].set_ylim(0.9765, 0.9910)
    _finish_axis(axes[1])

    fig.suptitle("Sparse Lagrange-Weight Sensitivity", y=1.05, fontsize=11)
    fig.savefig(output_dir / "rdpc_lambda_effects.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved lambda effects plot to {output_dir / 'rdpc_lambda_effects.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RDPC representative results")
    parser.add_argument("--csv", type=str, default=None, help="Path to result CSV")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots")
    args = parser.parse_args()

    csv_path = args.csv or str(ROOT / "results" / "sweep_results.csv")
    output_dir = args.output_dir or str(ROOT / "results")
    plot_tradeoffs(csv_path, output_dir)
