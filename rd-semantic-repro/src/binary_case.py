from math import erfc, isclose, log2, sqrt


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