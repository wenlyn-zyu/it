from src.baseline import (
    run_pca_logistic_baseline,
    run_supervised_projection_baseline,
    compare_methods_across_dimensions,
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