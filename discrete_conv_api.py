
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import numpy as np

Kind = Literal["pmf", "cdf", "ccdf"]
Mode = Literal["DOMINATES", "IS_DOMINATED"]

from implementation import utils as _impl_utils
from implementation import steps as _impl_steps
from implementation import kernels as _impl_kernels
from implementation import grids as _impl_grids
from implementation import selfconv as _impl_selfconv

@dataclass
class DiscreteDist:
    x: np.ndarray
    kind: Kind
    vals: np.ndarray
    p_neg_inf: float = 0.0
    p_pos_inf: float = 0.0
    name: Optional[str] = None
    debug_check: bool = False
    tol: float = 1e-12
    def __post_init__(self):
        self.x = np.ascontiguousarray(self.x, dtype=np.float64)
        self.vals = np.ascontiguousarray(self.vals, dtype=np.float64)
        if self.x.ndim != 1 or self.vals.ndim != 1 or self.x.shape != self.vals.shape:
            raise ValueError("x and vals must be 1-D arrays of equal length")
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("x must be strictly increasing")
        if self.p_neg_inf < -self.tol or self.p_pos_inf < -self.tol:
            raise ValueError("p_neg_inf and p_pos_inf must be nonnegative")
        if self.debug_check:
            if self.kind == "pmf":
                if np.any(self.vals < -self.tol):
                    raise ValueError("PMF must be nonnegative")
            elif self.kind == "cdf":
                if np.any(np.diff(self.vals) < -1e-12):
                    raise ValueError("CDF must be nondecreasing")
            elif self.kind == "ccdf":
                if np.any(np.diff(self.vals) > 1e-12):
                    raise ValueError("CCDF must be nonincreasing")

def cdf_to_pmf(x: np.ndarray, F: np.ndarray, p_neg_inf: float, p_pos_inf: float, *, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, float, float]:
    x = np.ascontiguousarray(x, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.float64)
    pmf = np.diff(np.concatenate(([float(p_neg_inf)], F))).astype(np.float64, copy=False)
    pmf[pmf < 0.0] = 0.0
    _impl_utils.budget_correction_last_bin(pmf, float(p_neg_inf), float(p_pos_inf), expected_total=1.0, tol=tol)
    return x, pmf, float(p_neg_inf), float(p_pos_inf)

def ccdf_to_pmf(x: np.ndarray, S: np.ndarray, p_neg_inf: float, p_pos_inf: float, *, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, float, float]:
    x = np.ascontiguousarray(x, dtype=np.float64)
    S = np.ascontiguousarray(S, dtype=np.float64)
    pmf = np.empty_like(S)
    if S.size == 0:
        pmf = S.copy()
    else:
        pmf[0] = max(0.0, 1.0 - float(p_neg_inf) - float(S[0]))
        if S.size > 1:
            pmf[1:] = S[:-1] - S[1:]
            pmf[1:][pmf[1:] < 0.0] = 0.0
    _impl_utils.budget_correction_last_bin(pmf, float(p_neg_inf), float(p_pos_inf), expected_total=1.0, tol=tol)
    return x, pmf, float(p_neg_inf), float(p_pos_inf)

def step_cdf_left(x, F, p_neg_inf, p_pos_inf, q):  return _impl_steps.step_cdf_left(x, F, p_neg_inf, p_pos_inf, q)
def step_cdf_right(x, F, p_neg_inf, p_pos_inf, q): return _impl_steps.step_cdf_right(x, F, p_neg_inf, p_pos_inf, q)
def step_ccdf_left(x, S, p_neg_inf, p_pos_inf, q): return _impl_steps.step_ccdf_left(x, S, p_neg_inf, p_pos_inf, q)
def step_ccdf_right(x, S, p_neg_inf, p_pos_inf, q):return _impl_steps.step_ccdf_right(x, S, p_neg_inf, p_pos_inf, q)

def convolve_pmf_pmf_to_pmf(X: DiscreteDist, Y: DiscreteDist, t: Optional[np.ndarray] = None, mode: Mode = 'DOMINATES') -> DiscreteDist:
    if X.kind != 'pmf' or Y.kind != 'pmf':
        raise ValueError('convolve_pmf_pmf_to_pmf expects PMF inputs')
    if t is None:
        t = _impl_grids.build_grid_trim_log_from_dists(X, Y)
    t = np.ascontiguousarray(t, dtype=np.float64)
    pmf_out, pneg, ppos = _impl_kernels.convolve_pmf_pmf_to_pmf_core(X.x, X.vals, X.p_neg_inf, X.p_pos_inf, Y.x, Y.vals, Y.p_neg_inf, Y.p_pos_inf, t, mode)
    return DiscreteDist(x=t, kind='pmf', vals=pmf_out, p_neg_inf=pneg, p_pos_inf=ppos, name='pmf⊕pmf')

def convolve_pmf_cdf_to_cdf(X: DiscreteDist, Y: DiscreteDist, t: Optional[np.ndarray] = None, mode: Mode = 'DOMINATES') -> DiscreteDist:
    if X.kind != 'pmf' or Y.kind != 'cdf':
        raise ValueError('convolve_pmf_cdf_to_cdf expects X:PMF, Y:CDF')
    if t is None:
        t = _impl_grids.build_grid_trim_log_from_dists(X, Y)
    t = np.ascontiguousarray(t, dtype=np.float64)
    F, pneg, ppos = _impl_kernels.convolve_pmf_cdf_to_cdf_core(X.x, X.vals, X.p_neg_inf, X.p_pos_inf, Y.x, Y.vals, Y.p_neg_inf, Y.p_pos_inf, t, mode)
    return DiscreteDist(x=t, kind='cdf', vals=F, p_neg_inf=pneg, p_pos_inf=ppos, name='pmf⊕cdf')

def convolve_pmf_ccdf_to_ccdf(X: DiscreteDist, Y: DiscreteDist, t: Optional[np.ndarray] = None, mode: Mode = 'DOMINATES') -> DiscreteDist:
    if X.kind != 'pmf' or Y.kind != 'ccdf':
        raise ValueError('convolve_pmf_ccdf_to_ccdf expects X:PMF, Y:CCDF')
    if t is None:
        t = _impl_grids.build_grid_trim_log_from_dists(X, Y)
    t = np.ascontiguousarray(t, dtype=np.float64)
    S, pneg, ppos = _impl_kernels.convolve_pmf_ccdf_to_ccdf_core(X.x, X.vals, X.p_neg_inf, X.p_pos_inf, Y.x, Y.vals, Y.p_neg_inf, Y.p_pos_inf, t, mode)
    return DiscreteDist(x=t, kind='ccdf', vals=S, p_neg_inf=pneg, p_pos_inf=ppos, name='pmf⊕ccdf')

def self_convolve_envelope(base: DiscreteDist, T: int, t: Optional[np.ndarray] = None, mode: Mode = 'DOMINATES', backend: Literal["cdf","ccdf"] = 'cdf') -> DiscreteDist:
    if base.kind != 'pmf':
        raise ValueError('self_convolve_envelope currently expects base as PMF')
    if t is None:
        t = _impl_grids.build_grid_self_convolution(base.x, int(T))
    t = np.ascontiguousarray(t, dtype=np.float64)
    env, pneg, ppos = _impl_selfconv.self_convolve_envelope_core(base.x, base.vals, base.p_neg_inf, base.p_pos_inf, int(T), t, mode, backend)
    kind = 'cdf' if backend == 'cdf' else 'ccdf'
    return DiscreteDist(x=t, kind=kind, vals=env, p_neg_inf=pneg, p_pos_inf=ppos, name=f'self⊕^{T}')
