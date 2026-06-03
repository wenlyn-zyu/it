import csv
from math import erfc, exp, isclose, log, log2, sqrt
from pathlib import Path


def gaussian_q(x: float) -> float:
    return 0.5 * erfc(x / sqrt(2.0))


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * log2(p) - (1.0 - p) * log2(1.0 - p)


def gaussian_pdf(x: float, mean: float, sigma: float) -> float:
    return (1.0 / (sigma * sqrt(2.0 * 3.141592653589793))) * exp(-0.5 * ((x - mean) / sigma) ** 2)


def posterior_probability_s0(x: float, A: float, sigma: float) -> float:
    pos = gaussian_pdf(x, mean=A, sigma=sigma)
    neg = gaussian_pdf(x, mean=-A, sigma=sigma)
    denom = pos + neg
    if denom == 0.0:
        return 0.5
    return pos / denom


def _compute_g(x: float, lam: float, A: float, sigma: float) -> float:
    r = exp(-2.0 * A * x / (sigma * sigma))
    numerator = 1.0 - r
    denominator = 1.0 + r
    if denominator == 0.0:
        return 0.5
    exponent = lam * numerator / denominator
    if exponent > 100.0:
        return 0.0
    if exponent < -100.0:
        return 1.0
    return 1.0 / (1.0 + exp(exponent))


def _dist_constraint(lam: float, A: float, sigma: float, Ds: float) -> float:
    n_pts = 2000
    x_min = -10.0 * sigma
    x_max = 10.0 * sigma
    dx = (x_max - x_min) / n_pts
    integral = 0.0
    for i in range(n_pts):
        x = x_min + (i + 0.5) * dx
        n_plus = gaussian_pdf(x, mean=A, sigma=sigma)
        n_minus = gaussian_pdf(x, mean=-A, sigma=sigma)
        g = _compute_g(x, lam, A, sigma)
        integral += (n_plus - n_minus) * g * dx
    return integral - (1.0 - 2.0 * Ds)


def _find_lambda(A: float, sigma: float, Ds: float) -> float:
    hi = -1e-6
    lo = -50.0
    f_hi = _dist_constraint(hi, A, sigma, Ds)
    f_lo = _dist_constraint(lo, A, sigma, Ds)
    for _ in range(20):
        if f_hi * f_lo < 0.0:
            break
        lo *= 2.0
        f_lo = _dist_constraint(lo, A, sigma, Ds)
        if f_hi * f_lo < 0.0:
            break
        hi /= 2.0
        f_hi = _dist_constraint(hi, A, sigma, Ds)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        f_mid = _dist_constraint(mid, A, sigma, Ds)
        if f_mid == 0.0 or (hi - lo) < 1e-12:
            return mid
        if f_mid * f_lo < 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return (lo + hi) / 2.0


def _integrand_for_rate(x: float, lam: float, A: float, sigma: float) -> float:
    n_plus = gaussian_pdf(x, mean=A, sigma=sigma)
    n_minus = gaussian_pdf(x, mean=-A, sigma=sigma)
    g = _compute_g(x, lam, A, sigma)
    return (n_plus + n_minus) * binary_entropy(g)


def _compute_rate_integral(lam: float, A: float, sigma: float) -> float:
    n_pts = 2000
    x_min = -10.0 * sigma
    x_max = 10.0 * sigma
    dx = (x_max - x_min) / n_pts
    integral = 0.0
    for i in range(n_pts):
        x = x_min + (i + 0.5) * dx
        integral += _integrand_for_rate(x, lam, A, sigma) * dx
    return 1.0 - 0.5 * integral


def classification_rd_infinite(A: float, sigma: float, Ds: float) -> float:
    bayes_error = gaussian_q(A / sigma)
    if Ds < bayes_error - 1e-12:
        raise ValueError("Ds={} is below the minimum achievable classification error Q(A/sigma={}).".format(Ds, bayes_error))
    if Ds >= 0.5 - 1e-12:
        return 0.0
    if isclose(Ds, bayes_error, rel_tol=1e-9, abs_tol=1e-9):
        return 1.0
    lam = _find_lambda(A, sigma, Ds)
    return _compute_rate_integral(lam, A, sigma)

def sample_rd_infinite_curve(A: float, sigma: float, num_points: int):
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")
    left = gaussian_q(A / sigma)
    right = 0.5
    step = (right - left) / (num_points - 1)
    points = []
    for idx in range(num_points):
        ds = left + idx * step
        if idx == num_points - 1:
            ds = right
        rate = classification_rd_infinite(A=A, sigma=sigma, Ds=ds)
        points.append((ds, rate))
    return points


def save_rd_infinite_curve_csv(output_path: Path, A: float, sigma: float, num_points: int) -> None:
    points = sample_rd_infinite_curve(A=A, sigma=sigma, num_points=num_points)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Ds", "rate_bits"])
        writer.writerows(points)
