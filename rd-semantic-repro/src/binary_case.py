from math import erfc, sqrt


def gaussian_q(x: float) -> float:
    return 0.5 * erfc(x / sqrt(2.0))


def classification_rd_infinite(A: float, sigma: float, Ds: float) -> float:
    bayes_error = gaussian_q(A / sigma)
    if Ds < bayes_error:
        raise ValueError("Ds is below the minimum achievable classification error.")
    if Ds >= 0.5:
        return 0.0
    raise NotImplementedError("Only the boundary behaviors are implemented so far.")