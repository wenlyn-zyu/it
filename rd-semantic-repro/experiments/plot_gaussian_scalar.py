import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gaussian_case import gaussian_scalar_sordf


SIGMA_X = 2.0
SIGMA_SX = 1.0
MMSE = 0.25
DS_VALUES = np.linspace(MMSE + 0.02, 1.6, 80)
DA_VALUES = np.linspace(0.05, 2.6, 80)


if __name__ == "__main__":
    output_dir = ROOT / "figures"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "gaussian_scalar_sordf.csv"
    png_path = output_dir / "gaussian_scalar_sordf_contour.png"

    rates = np.zeros((len(DS_VALUES), len(DA_VALUES)))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Ds", "Da", "rate_bits"])
        for i, Ds in enumerate(DS_VALUES):
            for j, Da in enumerate(DA_VALUES):
                rate = gaussian_scalar_sordf(SIGMA_X, SIGMA_SX, MMSE, float(Ds), float(Da))
                rates[i, j] = rate
                writer.writerow([float(Ds), float(Da), rate])

    da_grid, ds_grid = np.meshgrid(DA_VALUES, DS_VALUES)
    plt.figure(figsize=(6.4, 4.8))
    contour = plt.contourf(ds_grid, da_grid, rates, levels=20, cmap="viridis")
    lines = plt.contour(ds_grid, da_grid, rates, levels=8, colors="white", linewidths=0.6, alpha=0.8)
    plt.clabel(lines, inline=True, fontsize=7, fmt="%.2f")
    plt.colorbar(contour, label="Rate [bits]")
    plt.xlabel("Semantic distortion $D_s$")
    plt.ylabel("Appearance distortion $D_a$")
    plt.title("Scalar Gaussian SORDF (Proposition 3)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"Wrote scalar Gaussian data to {csv_path}")
    print(f"Wrote scalar Gaussian contour to {png_path}")