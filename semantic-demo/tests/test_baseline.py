from src.baseline import run_pca_logistic_baseline


def test_pca_logistic_baseline_returns_reasonable_accuracy():
    result = run_pca_logistic_baseline(n_components=16, random_state=0)
    assert result["input_dim"] == 64
    assert result["n_components"] == 16
    assert 0.8 <= result["accuracy"] <= 1.0