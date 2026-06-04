import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.binary_case import sample_soft_decision_curve


SETTINGS = [0.50, 0.39, 0.27, 0.16]


if __name__ == "__main__":
    output_dir = ROOT / "figures"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "binary_soft_decision_a1_sigma1.csv"
    png_path = output_dir / "binary_soft_decision_a1_sigma1.png"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Ds", "x", "g_x"])
        plt.figure(figsize=(6, 4))
        for Ds in SETTINGS:
            points = sample_soft_decision_curve(A=1.0, sigma=1.0, Ds=Ds, x_min=-3.0, x_max=3.0, num_points=241)
            for x, g_x in points:
                writer.writerow([Ds, x, g_x])
            plt.plot([p[0] for p in points], [p[1] for p in points], linewidth=2, label=f"Ds={Ds:.2f}")

    plt.xlabel("Observation x")
    plt.ylabel("Soft decision g(x)")
    plt.title("Optimal soft decision function (A=1, sigma=1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"Wrote soft-decision data to {csv_path}")
    print(f"Wrote soft-decision plot to {png_path}")
