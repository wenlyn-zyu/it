import csv
import math
from pathlib import Path

import pytest

from src.binary_case import (
    classification_rd_infinite,
    gaussian_q,
    posterior_probability_s0,
    sample_rd_infinite_curve,
    save_rd_infinite_curve_csv,
)


def test_gaussian_q_matches_known_value_at_one():
    assert math.isclose(gaussian_q(1.0), 0.15865525393145707, rel_tol=1e-9)


def test_posterior_is_half_at_zero_for_symmetric_binary_model():
    assert math.isclose(posterior_probability_s0(x=0.0, A=1.0, sigma=1.0), 0.5, abs_tol=1e-9)


def test_posterior_is_large_for_large_positive_observation():
    assert posterior_probability_s0(x=5.0, A=1.0, sigma=1.0) > 0.99


def test_rate_is_one_bit_at_bayes_error_boundary_when_a_equals_sigma():
    Ds = gaussian_q(1.0)
    rate = classification_rd_infinite(A=1.0, sigma=1.0, Ds=Ds)
    assert math.isclose(rate, 1.0, rel_tol=1e-6)


def test_rate_decreases_as_allowed_semantic_distortion_increases():
    lower = classification_rd_infinite(A=1.0, sigma=1.0, Ds=0.2)
    upper = classification_rd_infinite(A=1.0, sigma=1.0, Ds=0.3)
    assert lower >= upper


def test_rate_is_zero_when_semantic_distortion_is_half():
    rate = classification_rd_infinite(A=1.0, sigma=1.0, Ds=0.5)
    assert math.isclose(rate, 0.0, abs_tol=1e-6)


def test_rejects_semantic_distortion_below_bayes_error():
    with pytest.raises(ValueError):
        classification_rd_infinite(A=1.0, sigma=1.0, Ds=0.1)


def test_curve_sampler_includes_endpoints_and_returns_sorted_points():
    points = sample_rd_infinite_curve(A=1.0, sigma=1.0, num_points=5)
    ds_values = [point[0] for point in points]
    rates = [point[1] for point in points]
    assert len(points) == 5
    assert math.isclose(ds_values[0], gaussian_q(1.0), rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(ds_values[-1], 0.5, rel_tol=1e-9, abs_tol=1e-9)
    assert ds_values == sorted(ds_values)
    assert math.isclose(rates[0], 1.0, rel_tol=1e-6)
    assert math.isclose(rates[-1], 0.0, abs_tol=1e-6)


def test_save_curve_csv_writes_header_and_requested_number_of_rows(tmp_path: Path):
    output_path = tmp_path / "curve.csv"
    save_rd_infinite_curve_csv(output_path=output_path, A=1.0, sigma=1.0, num_points=5)
    rows = list(csv.reader(output_path.open("r", encoding="utf-8")))
    assert rows[0] == ["Ds", "rate_bits"]
    assert len(rows) == 6


def test_better_class_separation_reduces_left_endpoint_distortion():
    points_low = sample_rd_infinite_curve(A=0.5, sigma=1.0, num_points=5)
    points_high = sample_rd_infinite_curve(A=2.0, sigma=1.0, num_points=5)
    assert points_high[0][0] < points_low[0][0]

def test_soft_decision_curve_is_symmetric_around_origin():
    from src.binary_case import sample_soft_decision_curve

    points = sample_soft_decision_curve(A=1.0, sigma=1.0, Ds=0.3, x_min=-2.0, x_max=2.0, num_points=5)
    assert len(points) == 5
    left_x, left_g = points[0]
    mid_x, mid_g = points[2]
    right_x, right_g = points[-1]
    assert math.isclose(left_x, -2.0, abs_tol=1e-12)
    assert math.isclose(mid_x, 0.0, abs_tol=1e-12)
    assert math.isclose(right_x, 2.0, abs_tol=1e-12)
    assert math.isclose(mid_g, 0.5, abs_tol=1e-9)
    assert math.isclose(left_g + right_g, 1.0, abs_tol=1e-6)


def test_soft_decision_curve_increases_with_observation():
    from src.binary_case import sample_soft_decision_curve

    points = sample_soft_decision_curve(A=1.0, sigma=1.0, Ds=0.3, x_min=-3.0, x_max=3.0, num_points=31)
    g_values = [point[1] for point in points]
    assert g_values == sorted(g_values)
    assert g_values[0] < 0.5 < g_values[-1]
