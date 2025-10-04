import sys
from pathlib import Path

# Add project root to path so we can import discrete_conv_api
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

TOL_MICRO = 1e-12
TOL_STRESS = 1e-10

@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(12345)

def assert_monotone_nondecreasing(a: np.ndarray, tol: float = 0.0):
    diffs = np.diff(a)
    if np.any(diffs < -tol):
        idx = int(np.where(diffs < -tol)[0][0])
        raise AssertionError(f"Sequence not nondecreasing at idx {idx}: diff={diffs[idx]:.3e}")

def assert_monotone_nonincreasing(a: np.ndarray, tol: float = 0.0):
    diffs = np.diff(a)
    if np.any(diffs > tol):
        idx = int(np.where(diffs > tol)[0][0])
        raise AssertionError(f"Sequence not nonincreasing at idx {idx}: diff={diffs[idx]:.3e}")

def assert_budget_close(p_neg: float, pmf: np.ndarray, p_pos: float, expected: float = 1.0, tol: float = TOL_MICRO):
    total = float(p_neg) + float(pmf.sum()) + float(p_pos)
    if abs(total - expected) > tol:
        raise AssertionError(f"Budget mismatch: got {total:.12f}, expected {expected:.12f}, |err|={abs(total-expected):.3e}")

def pmf_to_cdf(x: np.ndarray, pmf: np.ndarray, p_neg: float, p_pos: float) -> np.ndarray:
    F = p_neg + np.cumsum(pmf, dtype=np.float64)
    F = np.minimum(F, 1.0 - p_pos)
    return F

def pmf_to_ccdf(x: np.ndarray, pmf: np.ndarray, p_neg: float, p_pos: float) -> np.ndarray:
    S = 1.0 - p_neg - np.cumsum(pmf, dtype=np.float64)
    S = np.maximum(S, p_pos)
    return S

def make_strict_grid(n: int, lo: float = -5.0, hi: float = 5.0) -> np.ndarray:
    x = np.linspace(lo, hi, n, dtype=np.float64)
    # enforce strict increase if n>1
    if n > 1:
        x[1:] += np.linspace(1e-12, 1e-9, n-1)
    return x

def normalize_nonneg(a: np.ndarray) -> np.ndarray:
    s = float(a.sum())
    return (a / s) if s > 0 else a

def assert_domination(lower: np.ndarray, upper: np.ndarray, tol: float = 0.0):
    gaps = lower - upper
    ok = np.all(gaps <= tol)
    count = int(np.sum(gaps > tol))
    max_gap = float(np.max(gaps)) if count else 0.0
    idx = int(np.argmax(gaps)) if count else -1
    if not ok:
        raise AssertionError(f"Domination violated {count} times; max overshoot {max_gap:.3e} at index {idx}")
    return ok, count, max_gap, idx
