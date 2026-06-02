import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline import run_pca_logistic_baseline, run_supervised_projection_baseline


if __name__ == "__main__":
    pca_result = run_pca_logistic_baseline(n_components=16, random_state=0)
    supervised_result = run_supervised_projection_baseline(n_components=16, random_state=0)
    print({"baseline": pca_result, "task_oriented": supervised_result})