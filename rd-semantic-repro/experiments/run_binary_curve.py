import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.binary_case import sample_rd_infinite_curve


if __name__ == "__main__":
    output_dir = ROOT / "figures"
    output_dir.mkdir(exist_ok=True)
    points = sample_rd_infinite_curve(A=1.0, sigma=1.0, num_points=41)
    output_path = output_dir / "binary_rd_infinite_a1_sigma1.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Ds", "rate_bits"])
        writer.writerows(points)
    print(f"Wrote {len(points)} points to {output_path}")