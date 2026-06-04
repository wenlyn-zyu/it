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

def gaussian_vector_parameters(m: int, sigma_s2: float, sigma_z2: float):
    mmse = sigma_s2 * sigma_z2 / (m * sigma_s2 + sigma_z2)
    alpha = (m * sigma_s2 + sigma_z2) / ((m * sigma_s2) ** 0.5)
    return mmse, alpha


def gaussian_vector_sordf(m: int, sigma_s2: float, sigma_z2: float, Ds: float, Da: float) -> float:
    if m < 2:
        raise ValueError("m must be at least 2.")
    if Da < 0.0:
        raise ValueError("Da must be nonnegative.")
    mmse, alpha = gaussian_vector_parameters(m, sigma_s2, sigma_z2)
    if Ds < mmse:
        raise ValueError("Ds must be at least mmse.")

    semantic_band = alpha * alpha * (Ds - mmse)
    lambda_1 = m * sigma_s2 + sigma_z2
    lambda_noise = sigma_z2

    if Da >= m * sigma_s2 + m * sigma_z2 and semantic_band >= lambda_1:
        return 0.0
    if semantic_band >= Da / m and Da < m * sigma_z2:
        return 0.5 * log_plus((m * m * sigma_s2 + m * sigma_z2) / Da) + (m - 1) * 0.5 * log_plus((m * sigma_z2) / Da)
    if semantic_band < lambda_1 and Da > semantic_band + (m - 1) * sigma_z2:
        return 0.5 * log_plus(lambda_1 / semantic_band)
    if m * sigma_z2 <= Da < m * sigma_s2 + m * sigma_z2 and semantic_band >= Da - (m - 1) * sigma_z2:
        return 0.5 * log_plus(lambda_1 / (Da - (m - 1) * sigma_z2))
    if m * semantic_band <= Da <= semantic_band + (m - 1) * sigma_z2:
        return 0.5 * log_plus(lambda_1 / semantic_band) + (m - 1) * 0.5 * log_plus(((m - 1) * sigma_z2) / (Da - semantic_band))
    return 0.0

def sample_vector_gaussian_surface(m: int, sigma_s2: float, sigma_z2: float, ds_values, da_values):
    points = []
    for Ds in ds_values:
        for Da in da_values:
            points.append((Ds, Da, gaussian_vector_sordf(m, sigma_s2, sigma_z2, Ds, Da)))
    return points