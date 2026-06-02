import csv
from math import erfc, isclose, log2, sqrt
from pathlib import Path


def gaussian_q(x: float) -> float:
    return 0.5 * erfc(x / sqrt(2.0))


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * log2(p) - (1.0 - p) * log2(1.0 - p)


def classification_rd_infinite(A: float, sigma: float, Ds: float) -> float:
    bayes_error = gaussian_q(A / sigma)
    if Ds < bayes_error:
        raise ValueError("Ds is below the minimum achievable classification error.")
    if isclose(Ds, bayes_error, rel_tol=1e-12, abs_tol=1e-12):
        return 1.0
    if Ds >= 0.5:
        return 0.0
    return 1.0 - binary_entropy(Ds)


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
        points.append((ds, classification_rd_infinite(A=A, sigma=sigma, Ds=ds)))
    return points


def save_rd_infinite_curve_csv(output_path: Path, A: float, sigma: float, num_points: int) -> None:
    points = sample_rd_infinite_curve(A=A, sigma=sigma, num_points=num_points)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Ds", "rate_bits"])
        writer.writerows(points)