import math

import pytest

from src.gaussian_case import gaussian_scalar_sordf


def test_scalar_gaussian_sordf_matches_proposition_3_appearance_dominates():
    rate = gaussian_scalar_sordf(sigma_x=2.0, sigma_sx=1.0, mmse=0.25, Ds=2.0, Da=0.5)
    assert math.isclose(rate, 1.0, rel_tol=1e-12)


def test_scalar_gaussian_sordf_matches_proposition_3_semantics_dominates():
    rate = gaussian_scalar_sordf(sigma_x=2.0, sigma_sx=1.0, mmse=0.25, Ds=0.5, Da=2.0)
    assert math.isclose(rate, 0.5, rel_tol=1e-12)


def test_scalar_gaussian_sordf_is_zero_when_both_distortions_are_loose():
    rate = gaussian_scalar_sordf(sigma_x=2.0, sigma_sx=1.0, mmse=0.25, Ds=2.0, Da=3.0)
    assert math.isclose(rate, 0.0, abs_tol=1e-12)


def test_scalar_gaussian_sordf_rejects_ds_at_or_below_mmse():
    with pytest.raises(ValueError):
        gaussian_scalar_sordf(sigma_x=2.0, sigma_sx=1.0, mmse=0.25, Ds=0.25, Da=1.0)


def test_sample_scalar_gaussian_surface_returns_grid_points():
    from src.gaussian_case import sample_scalar_gaussian_surface

    points = sample_scalar_gaussian_surface(
        sigma_x=2.0,
        sigma_sx=1.0,
        mmse=0.25,
        ds_values=[0.5, 1.0],
        da_values=[0.5, 2.0],
    )
    assert len(points) == 4
    assert points[0] == (0.5, 0.5, 1.0)
    assert points[-1] == (1.0, 2.0, 0.0)
