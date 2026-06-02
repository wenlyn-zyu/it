import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline import save_comparison_csv


if __name__ == "__main__":
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "baseline_comparison.csv"
    png_path = output_dir / "baseline_comparison.png"

    save_comparison_csv(output_path=csv_path, dimensions=[2, 4, 8, 16], random_state=0)

    curves = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            curves.setdefault(method, {"x": [], "y": []})
            curves[method]["x"].append(int(row["requested_n_components"]))
            curves[method]["y"].append(float(row["accuracy"]))

    plt.figure(figsize=(6.5, 4.5))
    for method, values in curves.items():
        plt.plot(values["x"], values["y"], marker="o", linewidth=2, label=method)

    plt.xlabel("Requested bottleneck dimension")
    plt.ylabel("Classification accuracy")
    plt.title("Task-agnostic vs task-oriented bottleneck comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"Wrote comparison data to {csv_path}")
    print(f"Wrote comparison plot to {png_path}")