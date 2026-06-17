import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.binary_case import save_rd_infinite_curve_csv


if __name__ == "__main__":
    output_dir = ROOT / "figures"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "binary_rd_infinite_a1_sigma1.csv"
    png_path = output_dir / "binary_rd_infinite_a1_sigma1.png"

    save_rd_infinite_curve_csv(output_path=csv_path, A=1.0, sigma=1.0, num_points=41)

    ds_values = []
    rate_values = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds_values.append(float(row["Ds"]))
            rate_values.append(float(row["rate_bits"]))

    # Hard (naive) baseline: sweep decision threshold τ, compute (Ds, H(p)) pairs
    # X ~ 0.5*N(+A,σ²) + 0.5*N(-A,σ²), hard decision Ŝ = sign(x - τ)
    A_val, sigma_val = 1.0, 1.0
    tau_vals = np.linspace(0, A_val + 6 * sigma_val, 300)  # symmetric: τ≥0 suffices
    ds_naive, rate_naive = [], []
    for tau in tau_vals:
        p_pos = 0.5 * scipy_norm.sf((tau - A_val) / sigma_val) + \
                0.5 * scipy_norm.sf((tau + A_val) / sigma_val)   # P(Ŝ=+1)
        ds = 0.5 * scipy_norm.cdf((tau - A_val) / sigma_val) + \
             0.5 * scipy_norm.sf((tau + A_val) / sigma_val)      # P(error)
        p_pos = np.clip(p_pos, 1e-12, 1 - 1e-12)
        h = -p_pos * np.log2(p_pos) - (1 - p_pos) * np.log2(1 - p_pos)
        ds_naive.append(ds)
        rate_naive.append(h)

    plt.figure(figsize=(6, 4))
    plt.plot(ds_values, rate_values, linewidth=2, label='Soft (optimal)')

    plt.plot(ds_naive, rate_naive, 'orange', linewidth=2, label='Hard (naive)')
    plt.legend()
    plt.xlabel("Semantic distortion $D_s$")
    plt.ylabel(r"$R(D_s, \infty)$ [bits]")
    plt.title("Binary classification case (A=1, sigma=1)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"Wrote curve data to {csv_path}")
    print(f"Wrote plot to {png_path}")

    