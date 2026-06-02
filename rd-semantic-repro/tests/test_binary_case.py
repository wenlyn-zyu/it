import math

import pytest

from src.binary_case import classification_rd_infinite, gaussian_q


def test_gaussian_q_matches_known_value_at_one():
    assert math.isclose(gaussian_q(1.0), 0.15865525393145707, rel_tol=1e-9)


def test_rate_is_one_bit_at_bayes_error_boundary_when_a_equals_sigma():
    Ds = gaussian_q(1.0)
    rate = classification_rd_infinite(A=1.0, sigma=1.0, Ds=Ds)
    assert math.isclose(rate, 1.0, rel_tol=1e-6)


def test_rate_is_zero_when_semantic_distortion_is_half():
    rate = classification_rd_infinite(A=1.0, sigma=1.0, Ds=0.5)
    assert math.isclose(rate, 0.0, abs_tol=1e-6)


def test_rejects_semantic_distortion_below_bayes_error():
    with pytest.raises(ValueError):
        classification_rd_infinite(A=1.0, sigma=1.0, Ds=0.1)