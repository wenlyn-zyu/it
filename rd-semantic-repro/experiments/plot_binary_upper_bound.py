import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.binary_case import rd_upper_bound


A = math.sqrt(10.0)
SIGMA = 1.0
DS_VALUES = [0.50, 0.30, 0.10]
DA_VALUES = [0.1 + i * (12.0 - 0.1) / 29 for i in range(30)]


def log_plus(value: float) -> float:
    return max(0.0, math.log2(value))


if __name__ == "__main__":
    output_dir = ROOT / "figures"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "binary_upper_bound_a2sqrt10_sigma1.csv"
    png_path = output_dir / "binary_upper_bound_a2sqrt10_sigma1.png"

    plt.figure(figsize=(6.5, 4.5))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["curve", "Ds", "Da", "rate_bits"])

        for Ds in DS_VALUES:
            rates = []
            for Da in DA_VALUES:
                rate = rd_upper_bound(A=A, sigma=SIGMA, Ds=Ds, Da=Da, num_D_samples=12)
                rates.append(rate)
                writer.writerow(["upper_bound", Ds, Da, rate])
            plt.plot(DA_VALUES, rates, linewidth=2, label=f"Ds={Ds:.2f}")

        naive_rates = [0.5 * log_plus((A * A + SIGMA * SIGMA) / Da) for Da in DA_VALUES]
        ideal_rates = [0.5 * log_plus((SIGMA * SIGMA) / Da) for Da in DA_VALUES]
        for Da, rate in zip(DA_VALUES, naive_rates):
            writer.writerow(["naive_gaussian", "", Da, rate])
        for Da, rate in zip(DA_VALUES, ideal_rates):
            writer.writerow(["known_state", "", Da, rate])

    plt.plot(DA_VALUES, naive_rates, "--", linewidth=1.8, label="naive Gaussian coding")
    plt.plot(DA_VALUES, ideal_rates, ":", linewidth=2.2, label="known-state lower reference")
    plt.xlabel("Appearance distortion $D_a$")
    plt.ylabel("Achievable rate [bits]")
    plt.title(r"Binary upper bound, $A^2/\sigma^2=10$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"Wrote upper-bound data to {csv_path}")
    print(f"Wrote upper-bound plot to {png_path}")