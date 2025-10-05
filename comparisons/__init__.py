"""
Comparison implementations for discrete convolution methods.

This package contains alternative convolution implementations for comparison
with the main implementation:

1. FFT-based convolution: Uses Fast Fourier Transform for linear grids
2. Monte Carlo convolution: Uses sampling-based approximation
3. Analytic convolution: Uses exact formulas for Gaussian and lognormal distributions
"""

from .fft_convolution import (
    fft_self_convolve_pmf
)

from .monte_carlo_convolution import (
    monte_carlo_self_convolve_pmf
)

from .analytic_convolution import (
    analytic_convolve_gaussian,
    analytic_convolve_lognormal,
    analytic_convolve_pmf_pmf_gaussian,
    analytic_convolve_with_continuous_gaussian,
    analytic_convolve_with_continuous_lognormal
)

__all__ = [
    # FFT methods
    'fft_self_convolve_pmf',
    
    # Monte Carlo methods
    'monte_carlo_self_convolve_pmf',
    
    # Analytic methods
    'analytic_convolve_gaussian',
    'analytic_convolve_lognormal',
    'analytic_convolve_pmf_pmf_gaussian',
    'analytic_convolve_with_continuous_gaussian',
    'analytic_convolve_with_continuous_lognormal'
]
