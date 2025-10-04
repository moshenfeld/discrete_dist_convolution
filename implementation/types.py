"""
Common types and enums for the discrete convolution library.

This module contains all shared types, enums, and the main DiscreteDist class
to avoid circular imports and provide a single source of truth.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
import numpy as np

# Enums
class Mode(Enum):
    """Tie-breaking mode for discretization."""
    DOMINATES = "DOMINATES"  # Upper bound (exact hits round up)
    IS_DOMINATED = "IS_DOMINATED"  # Lower bound (exact hits round down)

class Spacing(Enum):
    """Grid spacing strategy."""
    LINEAR = "linear"  # Linear spacing (linspace)
    GEOMETRIC = "geometric"  # Geometric spacing (geomspace)

class DistKind(Enum):
    """Distribution kind."""
    PMF = "pmf"  # Probability mass function
    CDF = "cdf"  # Cumulative distribution function
    CCDF = "ccdf"  # Complementary cumulative distribution function

# Main data class
@dataclass
class DiscreteDist:
    x: np.ndarray
    kind: DistKind
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
            if self.kind == DistKind.PMF:
                if np.any(self.vals < -self.tol):
                    raise ValueError("PMF must be nonnegative")
            elif self.kind == DistKind.CDF:
                if np.any(np.diff(self.vals) < -1e-12):
                    raise ValueError("CDF must be nondecreasing")
            elif self.kind == DistKind.CCDF:
                if np.any(np.diff(self.vals) > 1e-12):
                    raise ValueError("CCDF must be nonincreasing")
