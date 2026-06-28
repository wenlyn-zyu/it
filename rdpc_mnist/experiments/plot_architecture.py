"""Draw the RDPC training/evaluation pipeline used in the report."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


def add_box(ax, xy, width, height, text, color):
    rect = Rectangle(xy, width, height, linewidth=1.4, edgecolor="#222222", facecolor=color)
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=9)
    return rect


def add_arrow(ax, start, end, text=None):
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color="#222222")
    ax.add_patch(arrow)
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12, text, ha="center", fontsize=8)


def main():
    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    add_box(ax, (0.35, 2.0), 1.2, 0.8, "MNIST\nimage $X$", "#f2f2f2")
    add_box(ax, (2.0, 2.0), 1.35, 0.8, "Encoder\n$q_\\phi(z|x)$", "#d7e8fb")
    add_box(ax, (3.85, 2.0), 1.15, 0.8, "Latent\n$Z\\in\\mathbb{R}^k$", "#eef6ff")
    add_box(ax, (5.45, 2.0), 1.35, 0.8, "Decoder\n$p_\\theta(\\hat{x}|z)$", "#d8f0df")
    add_box(ax, (7.25, 2.0), 1.25, 0.8, "Recon.\n$\\hat{X}$", "#f2f2f2")

    add_box(ax, (7.0, 3.55), 1.65, 0.65, "Frozen classifier\nsemantic loss", "#fff2cc")
    add_box(ax, (7.0, 0.65), 1.65, 0.65, "WGAN-GP critic\nperception loss", "#fde0dc")
    add_box(ax, (3.55, 0.65), 1.75, 0.65, "Gaussian entropy\nrate estimate", "#eadcf8")
    add_box(ax, (8.95, 2.0), 0.75, 0.8, "MSE\nloss", "#fce5cd")

    add_arrow(ax, (1.55, 2.4), (2.0, 2.4))
    add_arrow(ax, (3.35, 2.4), (3.85, 2.4))
    add_arrow(ax, (5.0, 2.4), (5.45, 2.4))
    add_arrow(ax, (6.8, 2.4), (7.25, 2.4))
    add_arrow(ax, (8.5, 2.4), (8.95, 2.4))
    add_arrow(ax, (7.9, 2.8), (7.9, 3.55), "$\\lambda_c$")
    add_arrow(ax, (7.9, 2.0), (7.9, 1.3), "$\\lambda_p$")
    add_arrow(ax, (4.42, 2.0), (4.42, 1.3), "$R$")

    ax.text(
        5.05,
        4.72,
        "RDPC objective: $\\lambda_d\\,\\mathrm{MSE}(X,\\hat X) - \\lambda_p\\,D_\\psi(\\hat X) + \\lambda_c\\,\\mathrm{CE}(S,C(\\hat X))$",
        ha="center",
        va="center",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "rdpc_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_dir / 'rdpc_architecture.png'}")


if __name__ == "__main__":
    main()
