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


def test_vector_gaussian_sordf_region_1_tradeoff_band():
    from src.gaussian_case import gaussian_vector_sordf

    rate = gaussian_vector_sordf(m=2, sigma_s2=1.0, sigma_z2=1.0, Ds=4.0 / 9.0, Da=1.0)
    expected = 0.5 * math.log2(6.0) + 0.5 * math.log2(2.0)
    assert math.isclose(rate, expected, rel_tol=1e-12)


def test_vector_gaussian_sordf_region_2_appearance_dominates():
    from src.gaussian_case import gaussian_vector_sordf

    rate = gaussian_vector_sordf(m=2, sigma_s2=1.0, sigma_z2=1.0, Ds=22.0 / 45.0, Da=1.0)
    expected = 0.5 * math.log2(6.0) + 0.5 * math.log2(2.0)
    assert math.isclose(rate, expected, rel_tol=1e-12)


def test_vector_gaussian_sordf_region_3_semantics_dominates():
    from src.gaussian_case import gaussian_vector_sordf

    rate = gaussian_vector_sordf(m=2, sigma_s2=1.0, sigma_z2=1.0, Ds=4.0 / 9.0, Da=2.0)
    expected = 0.5 * math.log2(6.0)
    assert math.isclose(rate, expected, rel_tol=1e-12)


def test_vector_gaussian_sordf_region_4_hybrid_boundary():
    from src.gaussian_case import gaussian_vector_sordf

    rate = gaussian_vector_sordf(m=2, sigma_s2=1.0, sigma_z2=1.0, Ds=2.0 / 3.0, Da=2.5)
    expected = 0.5 * math.log2(2.0)
    assert math.isclose(rate, expected, rel_tol=1e-12)


def test_vector_gaussian_sordf_region_5_zero_rate():
    from src.gaussian_case import gaussian_vector_sordf

    rate = gaussian_vector_sordf(m=2, sigma_s2=1.0, sigma_z2=1.0, Ds=1.0, Da=4.0)
    assert math.isclose(rate, 0.0, abs_tol=1e-12)


def test_sample_vector_gaussian_surface_returns_grid_points():
    from src.gaussian_case import sample_vector_gaussian_surface

    points = sample_vector_gaussian_surface(
        m=2,
        sigma_s2=1.0,
        sigma_z2=1.0,
        ds_values=[4.0 / 9.0, 1.0],
        da_values=[1.0, 4.0],
    )
    assert len(points) == 4
    assert points[0][0] == 4.0 / 9.0
    assert points[0][1] == 1.0
    assert points[-1][0] == 1.0
    assert points[-1][1] == 4.0
    assert math.isclose(points[-1][2], 0.0, abs_tol=1e-12)
