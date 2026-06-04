from math import log2


def log_plus(value: float) -> float:
    return max(0.0, log2(value))


def gaussian_scalar_sordf(sigma_x: float, sigma_sx: float, mmse: float, Ds: float, Da: float) -> float:
    if Ds <= mmse:
        raise ValueError("Ds must be larger than mmse.")
    if Da <= 0.0:
        raise ValueError("Da must be positive.")
    appearance_rate = log_plus(sigma_x / Da)
    semantic_rate = log_plus((sigma_sx * sigma_sx) / (sigma_x * (Ds - mmse)))
    return 0.5 * max(appearance_rate, semantic_rate)


def sample_scalar_gaussian_surface(
    sigma_x: float,
    sigma_sx: float,
    mmse: float,
    ds_values,
    da_values,
):
    points = []
    for Ds in ds_values:
        for Da in da_values:
            points.append((Ds, Da, gaussian_scalar_sordf(sigma_x, sigma_sx, mmse, Ds, Da)))
    return points