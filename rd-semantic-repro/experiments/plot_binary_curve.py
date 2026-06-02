import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

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

    plt.figure(figsize=(6, 4))
    plt.plot(ds_values, rate_values, linewidth=2)
    plt.xlabel("Semantic distortion $D_s$")
    plt.ylabel(r"$R(D_s, \infty)$ [bits]")
    plt.title("Binary classification case (A=1, sigma=1)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"Wrote curve data to {csv_path}")
    print(f"Wrote plot to {png_path}")