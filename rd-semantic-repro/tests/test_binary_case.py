import math

import pytest

from src.binary_case import classification_rd_infinite


def test_rate_is_zero_when_semantic_distortion_is_half():
    rate = classification_rd_infinite(A=1.0, sigma=1.0, Ds=0.5)
    assert math.isclose(rate, 0.0, abs_tol=1e-6)


def test_rejects_semantic_distortion_below_bayes_error():
    with pytest.raises(ValueError):
        classification_rd_infinite(A=1.0, sigma=1.0, Ds=0.1)