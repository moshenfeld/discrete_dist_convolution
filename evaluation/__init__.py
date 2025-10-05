"""
Evaluation utilities for discrete convolution methods.

This package contains utilities for evaluating and comparing different
convolution methods through various statistical measures.
"""

from .moments import compute_alpha_moment, compute_log_moment, compute_moment_sequence, compute_log_moment_sequence

__all__ = [
    'compute_alpha_moment',
    'compute_log_moment', 
    'compute_moment_sequence',
    'compute_log_moment_sequence'
]
