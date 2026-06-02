import csv
from pathlib import Path

from src.baseline import (
    run_pca_logistic_baseline,
    run_supervised_projection_baseline,
    compare_methods_across_dimensions,
    save_comparison_csv,
)


def test_pca_logistic_baseline_returns_reasonable_accuracy():
    result = run_pca_logistic_baseline(n_components=16, random_state=0)
    assert result["input_dim"] == 64
    assert result["n_components"] == 16
    assert 0.8 <= result["accuracy"] <= 1.0


def test_supervised_projection_baseline_returns_valid_result():
    result = run_supervised_projection_baseline(n_components=16, random_state=0)
    assert result["input_dim"] == 64
    assert result["n_components"] == 9
    assert 0.8 <= result["accuracy"] <= 1.0
    assert result["method"] == "supervised_projection"


def test_compare_methods_across_dimensions_returns_one_row_per_requested_dimension():
    rows = compare_methods_across_dimensions(dimensions=[4, 8], random_state=0)
    assert len(rows) == 4
    methods = {row["method"] for row in rows}
    assert methods == {"pca_logistic", "supervised_projection"}
    requested_dims = {row["requested_n_components"] for row in rows}
    assert requested_dims == {4, 8}


def test_save_comparison_csv_writes_expected_header_and_rows(tmp_path: Path):
    output_path = tmp_path / "comparison.csv"
    save_comparison_csv(output_path=output_path, dimensions=[2, 4], random_state=0)
    rows = list(csv.reader(output_path.open("r", encoding="utf-8")))
    assert rows[0] == ["method", "requested_n_components", "n_components", "accuracy", "input_dim"]
    assert len(rows) == 5