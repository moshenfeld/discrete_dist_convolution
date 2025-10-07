"""
Enhanced comparison methods for discrete distribution convolution v2.

Updates in v2:
- Consistent use of DiscreteDist API
- No tuple unpacking for discretization results
"""

import numpy as np
from scipy import stats, fft
from typing import Optional, Union, Tuple

from implementation import (
    DiscreteDist, DistKind, Mode, Spacing, SummationMethod,
    discretize_continuous_to_pmf
)

# =============================================================================
# FFT-BASED CONVOLUTION METHODS
# =============================================================================

def _is_linear_grid(x: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if a grid is linearly spaced."""
    if x.size < 2:
        return True
    dx = np.diff(x)
    return np.allclose(dx, dx[0], rtol=tol, atol=tol)

def fft_self_convolve_continuous(dist: stats.rv_continuous, T: int, mode: Mode, spacing: Spacing, 
                                n_bins: int, beta: float) -> DiscreteDist:
    """
    FFT-based self-convolution with special handling for lognormal distributions.
    
    For lognormal distributions, works in log-space to handle geometric nature properly.
    For other distributions, uses standard linear-space FFT.
    """
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    
    # Check if this is a lognormal distribution
    dist_name = getattr(dist, 'dist', dist).__class__.__name__ if hasattr(dist, 'dist') else dist.__class__.__name__
    is_lognormal = 'lognorm' in dist_name.lower()
    
    if is_lognormal and spacing == Spacing.GEOMETRIC:
        # Special handling for lognormal: use Fenton-Wilkinson approximation
        if isinstance(dist, stats.lognorm):
            sigma = dist.kwds.get('s', dist.args[0] if dist.args else 1.0)
            scale = dist.kwds.get('scale', 1.0)
            mu = np.log(scale)
        else:
            mean = dist.mean()
            var = dist.var()
            sigma_sq = np.log(1 + var / mean**2)
            mu = np.log(mean) - sigma_sq / 2
            sigma = np.sqrt(sigma_sq)
        
        # For sum of T lognormals, use Fenton-Wilkinson approximation
        mean_sum = T * np.exp(mu + sigma**2 / 2)
        var_sum = T * np.exp(2*mu + sigma**2) * (np.exp(sigma**2) - 1)
        
        # Convert back to lognormal parameters
        cv_sq = var_sum / mean_sum**2
        sigma_sum_sq = np.log(1 + cv_sq)
        mu_sum = np.log(mean_sum) - sigma_sum_sq / 2
        
        # Create approximated lognormal
        approx_dist = stats.lognorm(s=np.sqrt(sigma_sum_sq), scale=np.exp(mu_sum))
        
        # Discretize with geometric spacing
        return discretize_continuous_to_pmf(
            approx_dist, n_bins, beta, mode, Spacing.GEOMETRIC,
            name=f"FFT_approx_selfconv_{T}({dist.__class__.__name__})"
        )
    
    # For non-lognormal or linear spacing: use standard FFT
    if spacing != Spacing.LINEAR:
        import warnings
        warnings.warn("Standard FFT requires LINEAR spacing; using linear spacing")
        spacing = Spacing.LINEAR
    
    if T == 1:
        return discretize_continuous_to_pmf(
            dist, n_bins, beta, mode, Spacing.LINEAR,
            name=f"FFT_discretized_{dist.__class__.__name__}"
        )
    
    # Discretize with linear spacing
    base_dist = discretize_continuous_to_pmf(
        dist, n_bins, beta, mode, Spacing.LINEAR)
    
    x = base_dist.x
    pmf = base_dist.vals
    p_neg_inf = base_dist.p_neg_inf
    p_pos_inf = base_dist.p_pos_inf
    
    # Check if grid is linearly spaced
    if not _is_linear_grid(x):
        raise ValueError('FFT convolution requires linearly spaced grids')
    
    dx = x[1] - x[0]
    n = x.size
    
    # Calculate the new support range for T-fold convolution
    z_start = T * x[0]
    z_end = T * x[-1]
    
    # Calculate required output size
    n_output = int(np.ceil((z_end - z_start) / dx)) + 1
    
    # Use power-of-2 size for efficient FFT
    fft_size = 2 ** int(np.ceil(np.log2(n_output + n)))
    
    # Pad the input PMF
    base_padded = np.zeros(fft_size)
    base_padded[:n] = pmf
    
    # Compute FFT
    base_fft = fft.fft(base_padded)
    
    # Raise to T-th power (T-fold convolution in frequency domain)
    result_fft = base_fft ** T
    
    # Inverse FFT
    result_padded = np.real(fft.ifft(result_fft))
    
    # Extract the valid portion
    result_vals = result_padded[:n_output]
    
    # Create output grid with the same spacing as input
    z_grid = np.linspace(z_start, z_end, n_output)
    
    # Normalize the result
    total_mass = np.sum(result_vals) + T * p_neg_inf + T * p_pos_inf
    if total_mass > 0:
        result_vals = result_vals / total_mass * (1.0 - T * p_neg_inf - T * p_pos_inf)
    
    result_vals = np.maximum(result_vals, 0.0)
    
    # Handle infinity masses
    p_neg_inf_out = T * p_neg_inf
    p_pos_inf_out = T * p_pos_inf
    
    return DiscreteDist(
        x=z_grid,
        kind=DistKind.PMF,
        vals=result_vals,
        p_neg_inf=p_neg_inf_out,
        p_pos_inf=p_pos_inf_out,
        name=f"FFT_selfconv_{T}({dist.__class__.__name__})"
    )

# =============================================================================
# CLEAN FFT CONVOLUTION METHOD
# =============================================================================

def clean_fft_self_convolve_continuous(dist: stats.rv_continuous, T: int, mode: Mode, 
                                      n_bins: int, beta: float) -> DiscreteDist:
    """
    Clean FFT-based self-convolution. ONLY does FFT with linear spacing.
    
    FFT requires linear spacing by definition - no exceptions, no tricks.
    """
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    
    if T == 1:
        return discretize_continuous_to_pmf(
            dist, n_bins, beta, mode, Spacing.LINEAR,
            name=f"FFT_discretized_{dist.__class__.__name__}"
        )
    
    # Discretize with linear spacing (FFT requirement)
    base_dist = discretize_continuous_to_pmf(
        dist, n_bins, beta, mode, Spacing.LINEAR)
    
    x = base_dist.x
    pmf = base_dist.vals
    p_neg_inf = base_dist.p_neg_inf
    p_pos_inf = base_dist.p_pos_inf
    
    # Check if grid is linearly spaced
    if not _is_linear_grid(x):
        raise ValueError('FFT convolution requires linearly spaced grids')
    
    dx = x[1] - x[0]
    n = x.size
    
    # Calculate the new support range for T-fold convolution
    z_start = T * x[0]
    z_end = T * x[-1]
    
    # Calculate required output size
    n_output = int(np.ceil((z_end - z_start) / dx)) + 1
    
    # Use power-of-2 size for efficient FFT
    fft_size = 2 ** int(np.ceil(np.log2(n_output + n)))
    
    # Pad the input PMF
    base_padded = np.zeros(fft_size)
    base_padded[:n] = pmf
    
    # Compute FFT
    base_fft = fft.fft(base_padded)
    
    # Raise to T-th power (T-fold convolution in frequency domain)
    result_fft = base_fft ** T
    
    # Inverse FFT
    result_padded = np.real(fft.ifft(result_fft))
    
    # Extract the valid portion
    result_vals = result_padded[:n_output]
    
    # Create output grid
    z_grid = np.linspace(z_start, z_end, n_output)
    
    # Normalize
    result_vals = result_vals / np.sum(result_vals)
    
    # Handle infinity masses
    p_neg_inf_out = p_neg_inf * T  # Approximate
    p_pos_inf_out = p_pos_inf * T  # Approximate
    
    return DiscreteDist(
        x=z_grid,
        kind=DistKind.PMF,
        vals=result_vals,
        p_neg_inf=p_neg_inf_out,
        p_pos_inf=p_pos_inf_out,
        name=f"Clean_FFT_selfconv_{T}({dist.__class__.__name__})"
    )

# =============================================================================
# MONTE CARLO SAMPLING CONVOLUTION METHODS
# =============================================================================

def monte_carlo_self_convolve_pmf(base_dist: stats.rv_continuous, T: int, mode: Mode, spacing: Spacing,
                                 n_samples: int, n_bins: int, block_size: int = 100000,
                                 use_importance_sampling: bool = False) -> DiscreteDist:
    """
    Monte Carlo self-convolution using standard random sampling.
    
    Note: Importance sampling disabled by default as it introduces bias.
    Use standard Monte Carlo sampling for unbiased estimates.
    """
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    
    if n_samples <= 0:
        raise ValueError(f'n_samples must be positive, got {n_samples}')
    
    if block_size <= 0:
        raise ValueError(f'block_size must be positive, got {block_size}')
    
    # Always use standard sampling
    z_samples = []
    
    # Process samples in blocks
    n_blocks = (n_samples + block_size - 1) // block_size
    
    for block_idx in range(n_blocks):
        current_block_size = min(block_size, n_samples - block_idx * block_size)
        
        # Standard iid sampling: sample T values and sum
        block_samples = base_dist.rvs(size=(current_block_size, T))
        Z_block = np.sum(block_samples, axis=1)
        z_samples.append(Z_block)
    
    # Combine all blocks
    Z_all = np.concatenate(z_samples)
    
    # Verify we have exactly n_samples
    actual_samples = len(Z_all)
    assert actual_samples == n_samples, f"Sample count mismatch: {actual_samples} != {n_samples}"
    
    # Create histogram
    z_min = Z_all.min()
    z_max = Z_all.max()
    
    if z_max <= z_min:
        z_max = z_min + 1e-10
    
    # Create bins based on spacing
    if spacing == Spacing.GEOMETRIC:
        if z_min > 0:
            bins = np.geomspace(z_min, z_max, n_bins + 1)
        elif z_max < 0:
            bins = -np.geomspace(-z_max, -z_min, n_bins + 1)[::-1]
        else:
            bins = np.linspace(z_min, z_max, n_bins + 1)
    else:  # LINEAR
        bins = np.linspace(z_min, z_max, n_bins + 1)
    
    # Compute histogram
    hist, bins_used = np.histogram(Z_all, bins=bins, density=False)
    
    # Normalize by actual sample count
    pmf = hist.astype(np.float64) / actual_samples
    
    # Create grid points (bin centers)
    x_grid = (bins_used[:-1] + bins_used[1:]) / 2
    
    # Verify grid is strictly increasing
    if not np.all(np.diff(x_grid) > 0):
        raise ValueError("Grid is not strictly increasing after histogram binning")
    
    # Handle infinity masses (set to zero for MC)
    p_neg_inf = 0.0
    p_pos_inf = 0.0
    
    base_name = base_dist.dist.name if hasattr(base_dist, 'dist') else 'dist'
    
    return DiscreteDist(
        x=x_grid,
        kind=DistKind.PMF,
        vals=pmf,
        p_neg_inf=p_neg_inf,
        p_pos_inf=p_pos_inf,
        name=f"MC_selfconv_{T}({base_name})"
    )

# =============================================================================
# ANALYTIC CONVOLUTION METHODS
# =============================================================================

def analytic_convolve_gaussian(base_dist: stats.norm, T: int, mode: Mode, spacing: Spacing,
                              n_points: int = 1000, beta: float = 1e-6) -> DiscreteDist:
    """Analytic convolution of T identical Gaussian distributions."""
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    
    if T == 1:
        return discretize_continuous_to_pmf(
            base_dist, n_points, beta, mode, spacing,
            name=f"Analytic_Gaussian({base_dist.mean():.3f}, {base_dist.std():.3f})"
        )
    
    # For Gaussian: sum of T identical Gaussians ~ N(T*μ, T*σ²)
    new_mean = T * base_dist.mean()
    new_std = np.sqrt(T) * base_dist.std()
    
    convolved_dist = stats.norm(loc=new_mean, scale=new_std)
    
    return discretize_continuous_to_pmf(
        convolved_dist, n_points, beta, mode, spacing,
        name=f"Analytic_Gaussian_T{T}({new_mean:.3f}, {new_std:.3f})"
    )

def analytic_convolve_lognormal(base_dist: stats.lognorm, T: int, mode: Mode, spacing: Spacing,
                               n_points: int = 1000, beta: float = 1e-6) -> DiscreteDist:
    """Analytic approximation for convolution of T identical lognormal distributions."""
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    
    if T == 1:
        return discretize_continuous_to_pmf(
            base_dist, n_points, beta, mode, spacing,
            name=f"Analytic_LogNormal({base_dist.mean():.3f}, {base_dist.std():.3f})"
        )
    
    # Method of moments approximation
    base_mean = base_dist.mean()
    base_var = base_dist.var()
    
    approx_mean = T * base_mean
    approx_var = T * base_var
    
    # Convert back to lognormal parameters
    cv_squared = approx_var / (approx_mean ** 2)
    sigma_squared = np.log(1 + cv_squared)
    mu = np.log(approx_mean) - sigma_squared / 2
    
    approx_dist = stats.lognorm(s=np.sqrt(sigma_squared), scale=np.exp(mu))
    
    return discretize_continuous_to_pmf(
        approx_dist, n_points, beta, mode, spacing,
        name=f"Analytic_LogNormal_T{T}({approx_mean:.3f}, {np.sqrt(approx_var):.3f})"
    )