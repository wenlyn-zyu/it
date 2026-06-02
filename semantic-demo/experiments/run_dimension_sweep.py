import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline import compare_methods_across_dimensions


if __name__ == "__main__":
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    rows = compare_methods_across_dimensions(dimensions=[2, 4, 8, 16], random_state=0)
    output_path = output_dir / "baseline_comparison.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "requested_n_components", "n_components", "accuracy", "input_dim"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")