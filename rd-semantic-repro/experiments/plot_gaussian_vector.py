import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gaussian_case import gaussian_vector_parameters, gaussian_vector_sordf


M = 2
SIGMA_S2 = 1.0
SIGMA_Z2 = 1.0
MMSE, ALPHA = gaussian_vector_parameters(M, SIGMA_S2, SIGMA_Z2)
DS_VALUES = np.linspace(MMSE + 0.01, 1.15, 90)
DA_VALUES = np.linspace(0.05, M * SIGMA_S2 + M * SIGMA_Z2 + 0.4, 90)


if __name__ == "__main__":
    output_dir = ROOT / "figures"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "gaussian_vector_sordf_m2.csv"
    png_path = output_dir / "gaussian_vector_sordf_m2_contour.png"

    rates = np.zeros((len(DS_VALUES), len(DA_VALUES)))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Ds", "Da", "rate_bits"])
        for i, Ds in enumerate(DS_VALUES):
            for j, Da in enumerate(DA_VALUES):
                rate = gaussian_vector_sordf(M, SIGMA_S2, SIGMA_Z2, float(Ds), float(Da))
                rates[i, j] = rate
                writer.writerow([float(Ds), float(Da), rate])

    da_grid, ds_grid = np.meshgrid(DA_VALUES, DS_VALUES)
    plt.figure(figsize=(6.6, 4.9))
    contour = plt.contourf(ds_grid, da_grid, rates, levels=24, cmap="magma")
    lines = plt.contour(ds_grid, da_grid, rates, levels=9, colors="white", linewidths=0.55, alpha=0.75)
    plt.clabel(lines, inline=True, fontsize=7, fmt="%.2f")

    ds_boundary = np.linspace(MMSE, 1.0, 200)
    semantic_band = ALPHA * ALPHA * (ds_boundary - MMSE)
    plt.plot(ds_boundary, M * semantic_band, "c--", linewidth=1.2, label=r"$D_a=m\alpha^2(D_s-mmse)$")
    plt.plot(ds_boundary, semantic_band + (M - 1) * SIGMA_Z2, "w--", linewidth=1.2, label=r"$D_a=\alpha^2(D_s-mmse)+(m-1)\sigma_Z^2$")

    plt.colorbar(contour, label="Rate [bits]")
    plt.xlabel("Semantic distortion $D_s$")
    plt.ylabel("Appearance distortion $D_a$")
    plt.title("Vector Gaussian SORDF, m=2 (Proposition 4)")
    plt.ylim(0.0, M * SIGMA_S2 + M * SIGMA_Z2 + 0.4)
    plt.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"Wrote vector Gaussian data to {csv_path}")
    print(f"Wrote vector Gaussian contour to {png_path}")