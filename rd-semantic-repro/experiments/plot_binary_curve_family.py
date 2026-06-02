import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.binary_case import save_rd_infinite_curve_csv


SETTINGS = [
    (0.5, 1.0),
    (1.0, 1.0),
    (2.0, 1.0),
]


def load_curve(csv_path: Path):
    ds_values = []
    rate_values = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds_values.append(float(row["Ds"]))
            rate_values.append(float(row["rate_bits"]))
    return ds_values, rate_values


if __name__ == "__main__":
    output_dir = ROOT / "figures"
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(6.5, 4.5))

    for A, sigma in SETTINGS:
        stem = f"binary_rd_infinite_a{A}_sigma{sigma}".replace('.', 'p')
        csv_path = output_dir / f"{stem}.csv"
        save_rd_infinite_curve_csv(output_path=csv_path, A=A, sigma=sigma, num_points=41)
        ds_values, rate_values = load_curve(csv_path)
        plt.plot(ds_values, rate_values, linewidth=2, label=f"A={A}, sigma={sigma}")
        print(f"Wrote curve data to {csv_path}")

    png_path = output_dir / "binary_rd_infinite_curve_family.png"
    plt.xlabel("Semantic distortion $D_s$")
    plt.ylabel(r"$R(D_s, \infty)$ [bits]")
    plt.title("Binary classification case under different class separations")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"Wrote plot to {png_path}")