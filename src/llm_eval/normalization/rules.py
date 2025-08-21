

from math import erf, sqrt


def minmax_normalize(x: float, min_value: float, max_value: float, higher_is_better: bool) -> float:
    if max_value == min_value:
        return 50.0
    ratio = (x - min_value) / (max_value - min_value)
    if not higher_is_better:
        ratio = 1.0 - ratio
    return float(max(0.0, min(100.0, ratio * 100.0)))


def _phi(z: float) -> float:
    # standard normal CDF
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def zscore_cdf_normalize(x: float, mean: float, std: float, higher_is_better: bool) -> float:
    if std <= 0:
        return 50.0
    z = (x - mean) / std
    p = _phi(z)
    if not higher_is_better:
        p = 1.0 - p
    return float(max(0.0, min(100.0, p * 100.0)))


