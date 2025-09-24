
import numpy as np

def step_cdf_left(x: np.ndarray, F: np.ndarray, p_neg_inf: float, p_pos_inf: float, q: float) -> float:
    if q < x[0]:
        return float(p_neg_inf)
    if q >= x[-1]:
        return float(1.0 - p_pos_inf)
    idx = int(np.searchsorted(x, q, side="left")) - 1
    if idx < 0:
        return float(p_neg_inf)
    return float(F[idx])

def step_cdf_right(x: np.ndarray, F: np.ndarray, p_neg_inf: float, p_pos_inf: float, q: float) -> float:
    if q < x[0]:
        return float(p_neg_inf)
    if q >= x[-1]:
        return float(1.0 - p_pos_inf)
    idx = int(np.searchsorted(x, q, side="right")) - 1
    if idx < 0:
        return float(p_neg_inf)
    return float(F[idx])

def step_ccdf_left(x: np.ndarray, S: np.ndarray, p_neg_inf: float, p_pos_inf: float, q: float) -> float:
    if q < x[0]:
        return float(1.0 - p_neg_inf)
    if q >= x[-1]:
        return float(p_pos_inf)
    idx = int(np.searchsorted(x, q, side="left")) - 1
    if idx < 0:
        return float(1.0 - p_neg_inf)
    return float(S[idx])

def step_ccdf_right(x: np.ndarray, S: np.ndarray, p_neg_inf: float, p_pos_inf: float, q: float) -> float:
    if q < x[0]:
        return float(1.0 - p_neg_inf)
    if q >= x[-1]:
        return float(p_pos_inf)
    idx = int(np.searchsorted(x, q, side="right")) - 1
    if idx < 0:
        return float(1.0 - p_neg_inf)
    return float(S[idx])
