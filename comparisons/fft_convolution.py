"""
FFT-based convolution implementation for discrete distributions.

This module provides an FFT-based convolution method that works specifically
for linear grids. It leverages the convolution theorem: convolution in the
time domain equals multiplication in the frequency domain.
"""

import numpy as np
from scipy import fft, stats
from typing import Optional
import sys
import os

# Add parent directory to path to import implementation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation.types import DiscreteDist, DistKind, Mode, Spacing
from implementation.grids import discretize_continuous_to_pmf


def fft_convolve_pmf_pmf(X: DiscreteDist, Y: DiscreteDist, mode: Mode, spacing: Spacing) -> DiscreteDist:
    """
    FFT-based convolution of two PMF distributions.
    
    This method uses the convolution theorem: FFT(X * Y) = FFT(X) * FFT(Y)
    where * denotes convolution. The result is then inverse FFT'd to get
    the convolution in the time domain.
    
    Parameters:
    -----------
    X, Y : DiscreteDist
        Input distributions (must be PMF kind)
    mode : Mode
        Tie-breaking mode (used for consistency with other methods)
    spacing : Spacing
        Grid spacing strategy (must be LINEAR for FFT method)
    
    Returns:
    --------
    DiscreteDist
        Convolution result as PMF
        
    Notes:
    ------
    - Only works with LINEAR spacing grids
    - Assumes both distributions have the same grid size
    - Uses zero-padding to avoid circular convolution effects
    """
    if X.kind != DistKind.PMF or Y.kind != DistKind.PMF:
        raise ValueError(f'fft_convolve_pmf_pmf expects PMF inputs, got {X.kind}, {Y.kind}')
    
    if spacing != Spacing.LINEAR:
        raise ValueError(f'FFT convolution only supports LINEAR spacing, got {spacing}')
    
    # Check if grids are linearly spaced
    if not _is_linear_grid(X.x) or not _is_linear_grid(Y.x):
        raise ValueError('FFT convolution requires linearly spaced grids')
    
    # Ensure both distributions have the same grid size
    if X.x.size != Y.x.size:
        raise ValueError(f'Both distributions must have same grid size: {X.x.size} vs {Y.y.size}')
    
    # Get grid parameters
    dx = X.x[1] - X.x[0]  # Grid spacing
    dy = Y.x[1] - Y.x[0]  # Grid spacing
    
    if abs(dx - dy) > 1e-10:
        raise ValueError(f'Grid spacings must be equal: {dx} vs {dy}')
    
    # Zero-pad to avoid circular convolution
    n_pad = X.x.size
    total_size = 2 * n_pad - 1
    
    # Pad PMFs with zeros
    X_padded = np.zeros(total_size)
    Y_padded = np.zeros(total_size)
    
    X_padded[:n_pad] = X.vals
    Y_padded[:n_pad] = Y.vals
    
    # Compute FFT convolution
    X_fft = fft.fft(X_padded)
    Y_fft = fft.fft(Y_padded)
    
    # Multiply in frequency domain
    Z_fft = X_fft * Y_fft
    
    # Inverse FFT to get convolution result
    Z_padded = np.real(fft.ifft(Z_fft))
    
    # Extract the valid portion (first n_pad elements)
    Z_vals = Z_padded[:n_pad]
    
    # Create output grid
    # The convolution shifts the support by the sum of the starting points
    z_start = X.x[0] + Y.x[0]
    z_end = X.x[-1] + Y.x[-1]
    z_grid = np.linspace(z_start, z_end, n_pad)
    
    # Handle infinity masses
    p_neg_inf = X.p_neg_inf + Y.p_neg_inf
    p_pos_inf = X.p_pos_inf + Y.p_pos_inf
    
    return DiscreteDist(
        x=z_grid,
        kind=DistKind.PMF,
        vals=Z_vals,
        p_neg_inf=p_neg_inf,
        p_pos_inf=p_pos_inf,
        name=f"FFT_conv({X.name or 'X'}, {Y.name or 'Y'})"
    )


def fft_self_convolve_pmf(base: DiscreteDist, T: int, mode: Mode, spacing: Spacing) -> DiscreteDist:
    """
    FFT-based self-convolution of a PMF distribution T times.
    
    Uses the fact that T-fold self-convolution can be computed efficiently
    using FFT by raising the FFT of the base distribution to the T-th power.
    
    Parameters:
    -----------
    base : DiscreteDist
        Base distribution (must be PMF)
    T : int
        Number of times to convolve (must be >= 1)
    mode : Mode
        Tie-breaking mode (used for consistency)
    spacing : Spacing
        Grid spacing strategy (must be LINEAR)
    
    Returns:
    --------
    DiscreteDist
        T-fold self-convolution result
    """
    if base.kind != DistKind.PMF:
        raise ValueError(f'fft_self_convolve_pmf expects PMF input, got {base.kind}')
    
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    
    if spacing != Spacing.LINEAR:
        raise ValueError(f'FFT convolution only supports LINEAR spacing, got {spacing}')
    
    if T == 1:
        return base
    
    # Check if grid is linearly spaced
    if not _is_linear_grid(base.x):
        raise ValueError('FFT convolution requires linearly spaced grids')
    
    # Get grid parameters
    dx = base.x[1] - base.x[0]  # Grid spacing
    n = base.x.size
    
    # For T-fold convolution, we need to maintain the same resolution
    # This means keeping the same grid spacing but expanding the support
    
    # Calculate the new support range
    z_start = T * base.x[0]
    z_end = T * base.x[-1]
    
    # Calculate how many points we need to maintain the same resolution
    n_output = int(np.ceil((z_end - z_start) / dx)) + 1
    
    # Use a power-of-2 size for efficient FFT
    fft_size = 2 ** int(np.ceil(np.log2(n_output + n)))
    
    # Pad the input PMF
    base_padded = np.zeros(fft_size)
    base_padded[:n] = base.vals
    
    # Compute FFT
    base_fft = fft.fft(base_padded)
    
    # Raise to T-th power (T-fold convolution in frequency domain)
    result_fft = base_fft ** T
    
    # Inverse FFT
    result_padded = np.real(fft.ifft(result_fft))
    
    # Extract the valid portion (first n_output elements)
    result_vals = result_padded[:n_output]
    
    # Create output grid with the same spacing as input
    z_grid = np.linspace(z_start, z_end, n_output)
    
    # Normalize the result to ensure it's a proper PMF
    total_mass = np.sum(result_vals) + T * base.p_neg_inf + T * base.p_pos_inf
    if total_mass > 0:
        result_vals = result_vals / total_mass * (1.0 - T * base.p_neg_inf - T * base.p_pos_inf)
    
    # Ensure non-negative
    result_vals = np.maximum(result_vals, 0.0)
    
    # Handle infinity masses (they scale with T)
    p_neg_inf = T * base.p_neg_inf
    p_pos_inf = T * base.p_pos_inf
    
    return DiscreteDist(
        x=z_grid,
        kind=DistKind.PMF,
        vals=result_vals,
        p_neg_inf=p_neg_inf,
        p_pos_inf=p_pos_inf,
        name=f"FFT_selfconv_{T}({base.name or 'base'})"
    )


def fft_self_convolve_continuous(dist: stats.rv_continuous, T: int, mode: Mode, spacing: Spacing, 
                                n_bins: int, beta: float) -> DiscreteDist:
    """
    FFT-based self-convolution of a continuous distribution T times.
    
    This function discretizes the continuous distribution internally using linear spacing,
    then applies FFT-based convolution.
    
    Parameters:
    -----------
    dist : scipy.stats.rv_continuous
        Continuous distribution object
    T : int
        Number of times to convolve (must be >= 1)
    mode : Mode
        Tie-breaking mode (used for consistency)
    spacing : Spacing
        Grid spacing strategy (must be LINEAR for FFT)
    n_bins : int
        Number of bins for discretization
    beta : float
        Tail probability to trim during discretization
        
    Returns:
    --------
    DiscreteDist
        T-fold self-convolution result
    """
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    
    if spacing != Spacing.LINEAR:
        raise ValueError(f'FFT convolution only supports LINEAR spacing, got {spacing}')
    
    if T == 1:
        # For T=1, just discretize and return
        x, pmf, p_neg_inf, p_pos_inf = discretize_continuous_to_pmf(
            dist, n_bins, beta, mode, spacing)
        return DiscreteDist(
            x=x, kind=DistKind.PMF, vals=pmf,
            p_neg_inf=p_neg_inf, p_pos_inf=p_pos_inf,
            name=f"FFT_discretized_{dist.__class__.__name__}"
        )
    
    # Discretize the continuous distribution
    x, pmf, p_neg_inf, p_pos_inf = discretize_continuous_to_pmf(
        dist, n_bins, beta, mode, spacing)
    
    # Create DiscreteDist object
    base = DiscreteDist(
        x=x, kind=DistKind.PMF, vals=pmf,
        p_neg_inf=p_neg_inf, p_pos_inf=p_pos_inf,
        name=f"FFT_base_{dist.__class__.__name__}"
    )
    
    # Apply FFT self-convolution
    return fft_self_convolve_pmf(base, T, mode, spacing)


def _is_linear_grid(x: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if a grid is linearly spaced.
    
    Parameters:
    -----------
    x : np.ndarray
        Grid points
    tol : float
        Tolerance for linearity check
        
    Returns:
    --------
    bool
        True if grid is linearly spaced
    """
    if x.size < 2:
        return True
    
    dx = np.diff(x)
    return np.allclose(dx, dx[0], rtol=tol, atol=tol)


def fft_convolve_with_adaptive_grid(X: DiscreteDist, Y: DiscreteDist, mode: Mode, spacing: Spacing) -> DiscreteDist:
    """
    FFT convolution with adaptive grid generation for non-linear inputs.
    
    This method interpolates non-linear grids onto linear grids before
    applying FFT convolution. This allows FFT to work with any input grid.
    
    Parameters:
    -----------
    X, Y : DiscreteDist
        Input distributions (must be PMF)
    mode : Mode
        Tie-breaking mode
    spacing : Spacing
        Grid spacing strategy (must be LINEAR)
    
    Returns:
    --------
    DiscreteDist
        Convolution result
    """
    if spacing != Spacing.LINEAR:
        raise ValueError(f'FFT convolution only supports LINEAR spacing, got {spacing}')
    
    # Create linear grids for interpolation
    n_points = max(X.x.size, Y.x.size)
    
    # Determine support bounds
    x_min, x_max = X.x[0], X.x[-1]
    y_min, y_max = Y.x[0], Y.x[-1]
    
    # Create linear grids
    x_linear = np.linspace(x_min, x_max, n_points)
    y_linear = np.linspace(y_min, y_max, n_points)
    
    # Interpolate PMFs onto linear grids
    X_interp = np.interp(x_linear, X.x, X.vals)
    Y_interp = np.interp(y_linear, Y.x, Y.vals)
    
    # Normalize interpolated PMFs
    X_interp = X_interp / np.sum(X_interp) * np.sum(X.vals)
    Y_interp = Y_interp / np.sum(Y_interp) * np.sum(Y.vals)
    
    # Create temporary DiscreteDist objects with linear grids
    X_linear = DiscreteDist(x=x_linear, kind=DistKind.PMF, vals=X_interp, 
                           p_neg_inf=X.p_neg_inf, p_pos_inf=X.p_pos_inf)
    Y_linear = DiscreteDist(x=y_linear, kind=DistKind.PMF, vals=Y_interp,
                           p_neg_inf=Y.p_neg_inf, p_pos_inf=Y.p_pos_inf)
    
    # Apply FFT convolution
    return fft_convolve_pmf_pmf(X_linear, Y_linear, mode, spacing)
