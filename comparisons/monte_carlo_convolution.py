"""
Monte Carlo sampling convolution implementation for discrete distributions.

This module provides a Monte Carlo-based convolution method that works by
sampling from the underlying distributions and computing the sum of T elements.
The resulting samples are then used to construct an empirical distribution.
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path to import implementation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation.types import DiscreteDist, DistKind, Mode, Spacing


def monte_carlo_convolve_pmf_pmf(X: DiscreteDist, Y: DiscreteDist, mode: Mode, spacing: Spacing, 
                                n_samples: int = 100000, n_bins: Optional[int] = None) -> DiscreteDist:
    """
    Monte Carlo convolution of two PMF distributions.
    
    This method samples from both distributions and computes the sum of samples
    to approximate the convolution. The resulting samples are histogrammed to
    create an empirical PMF.
    
    Parameters:
    -----------
    X, Y : DiscreteDist
        Input distributions (must be PMF kind)
    mode : Mode
        Tie-breaking mode (used for consistency with other methods)
    spacing : Spacing
        Grid spacing strategy (used to determine output grid)
    n_samples : int
        Number of Monte Carlo samples to generate
    n_bins : int, optional
        Number of bins for histogramming. If None, uses input grid size
        
    Returns:
    --------
    DiscreteDist
        Convolution result as PMF
    """
    if X.kind != DistKind.PMF or Y.kind != DistKind.PMF:
        raise ValueError(f'monte_carlo_convolve_pmf_pmf expects PMF inputs, got {X.kind}, {Y.kind}')
    
    if n_samples <= 0:
        raise ValueError(f'n_samples must be positive, got {n_samples}')
    
    # Determine number of bins
    if n_bins is None:
        n_bins = max(X.x.size, Y.x.size)
    
    # Sample from X and Y distributions
    X_samples = _sample_from_pmf(X, n_samples)
    Y_samples = _sample_from_pmf(Y, n_samples)
    
    # Compute convolution samples (sum)
    Z_samples = X_samples + Y_samples
    
    # Create histogram bins
    z_min = Z_samples.min()
    z_max = Z_samples.max()
    
    # Ensure we have a valid range
    if z_max <= z_min:
        z_max = z_min + 1e-10
    
    if spacing == Spacing.GEOMETRIC:
        if z_min > 0:
            bins = np.geomspace(z_min, z_max, n_bins + 1)
        elif z_max < 0:
            bins = -np.geomspace(-z_max, -z_min, n_bins + 1)[::-1]
        else:
            # Fall back to linear if range contains 0
            bins = np.linspace(z_min, z_max, n_bins + 1)
    else:  # LINEAR
        bins = np.linspace(z_min, z_max, n_bins + 1)
    
    # Ensure bins are strictly increasing and unique
    bins = np.sort(np.unique(bins))
    
    # Compute histogram
    hist, bin_edges = np.histogram(Z_samples, bins=bins, density=False)
    
    # Convert to PMF (normalize by number of samples)
    pmf = hist.astype(np.float64) / n_samples
    
    # Create grid points (bin centers)
    x_grid = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Ensure grid is strictly increasing and unique
    x_grid = np.sort(np.unique(x_grid))
    
    # Ensure PMF matches grid size
    if len(pmf) != len(x_grid):
        # If sizes don't match, create a new PMF with correct size
        pmf = np.zeros(len(x_grid))
        for i in range(len(bin_edges) - 1):
            if i < len(hist):
                pmf[i] = hist[i] / n_samples
    
    # Handle infinity masses (approximate from tail samples)
    p_neg_inf = np.sum(Z_samples < z_min) / n_samples
    p_pos_inf = np.sum(Z_samples > z_max) / n_samples
    
    return DiscreteDist(
        x=x_grid,
        kind=DistKind.PMF,
        vals=pmf,
        p_neg_inf=p_neg_inf,
        p_pos_inf=p_pos_inf,
        name=f"MC_conv({X.name or 'X'}, {Y.name or 'Y'})"
    )


def monte_carlo_self_convolve_pmf(base: DiscreteDist, T: int, mode: Mode, spacing: Spacing,
                                 n_samples: int = 100000, n_bins: Optional[int] = None) -> DiscreteDist:
    """
    Monte Carlo self-convolution of a PMF distribution T times.
    
    This method samples T elements from the base distribution and computes
    their sum to approximate the T-fold self-convolution.
    
    Parameters:
    -----------
    base : DiscreteDist
        Base distribution (must be PMF)
    T : int
        Number of times to convolve (must be >= 1)
    mode : Mode
        Tie-breaking mode (used for consistency)
    spacing : Spacing
        Grid spacing strategy (used to determine output grid)
    n_samples : int
        Number of Monte Carlo samples to generate
    n_bins : int, optional
        Number of bins for histogramming. If None, uses input grid size
        
    Returns:
    --------
    DiscreteDist
        T-fold self-convolution result
    """
    if base.kind != DistKind.PMF:
        raise ValueError(f'monte_carlo_self_convolve_pmf expects PMF input, got {base.kind}')
    
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    
    if n_samples <= 0:
        raise ValueError(f'n_samples must be positive, got {n_samples}')
    
    # Determine number of bins
    if n_bins is None:
        n_bins = base.x.size
    
    # Sample T elements from base distribution for each Monte Carlo sample
    # Shape: (n_samples, T)
    samples_matrix = _sample_from_pmf(base, n_samples * T).reshape(n_samples, T)
    
    # Compute sum of T elements for each Monte Carlo sample
    Z_samples = np.sum(samples_matrix, axis=1)
    
    # Create histogram bins
    z_min = Z_samples.min()
    z_max = Z_samples.max()
    
    # Ensure we have a valid range
    if z_max <= z_min:
        z_max = z_min + 1e-10
    
    if spacing == Spacing.GEOMETRIC:
        if z_min > 0:
            bins = np.geomspace(z_min, z_max, n_bins + 1)
        elif z_max < 0:
            bins = -np.geomspace(-z_max, -z_min, n_bins + 1)[::-1]
        else:
            # Fall back to linear if range contains 0
            bins = np.linspace(z_min, z_max, n_bins + 1)
    else:  # LINEAR
        bins = np.linspace(z_min, z_max, n_bins + 1)
    
    # Ensure bins are strictly increasing and unique
    bins = np.sort(np.unique(bins))
    
    # Compute histogram
    hist, bin_edges = np.histogram(Z_samples, bins=bins, density=False)
    
    # Convert to PMF (normalize by number of samples)
    pmf = hist.astype(np.float64) / n_samples
    
    # Create grid points (bin centers)
    x_grid = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Ensure grid is strictly increasing and unique
    x_grid = np.sort(np.unique(x_grid))
    
    # Ensure PMF matches grid size
    if len(pmf) != len(x_grid):
        # If sizes don't match, create a new PMF with correct size
        pmf = np.zeros(len(x_grid))
        for i in range(len(bin_edges) - 1):
            if i < len(hist):
                pmf[i] = hist[i] / n_samples
    
    # Handle infinity masses (approximate from tail samples)
    p_neg_inf = np.sum(Z_samples < z_min) / n_samples
    p_pos_inf = np.sum(Z_samples > z_max) / n_samples
    
    return DiscreteDist(
        x=x_grid,
        kind=DistKind.PMF,
        vals=pmf,
        p_neg_inf=p_neg_inf,
        p_pos_inf=p_pos_inf,
        name=f"MC_selfconv_{T}({base.name or 'base'})"
    )


def _sample_from_pmf(dist: DiscreteDist, n_samples: int) -> np.ndarray:
    """
    Sample from a discrete PMF distribution.
    
    Parameters:
    -----------
    dist : DiscreteDist
        Discrete distribution (must be PMF)
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    np.ndarray
        Array of samples
    """
    if dist.kind != DistKind.PMF:
        raise ValueError(f'_sample_from_pmf expects PMF input, got {dist.kind}')
    
    # Create cumulative distribution for sampling
    cumsum = np.cumsum(dist.vals)
    
    # Normalize to ensure it sums to 1 (accounting for infinity masses)
    total_mass = cumsum[-1] + dist.p_neg_inf + dist.p_pos_inf
    
    # Generate uniform random numbers
    u = np.random.random(n_samples)
    
    # Sample using inverse CDF method
    samples = np.zeros(n_samples)
    
    for i, u_val in enumerate(u):
        if u_val < dist.p_neg_inf / total_mass:
            # Sample from negative infinity mass
            samples[i] = -np.inf
        elif u_val < (dist.p_neg_inf + cumsum[-1]) / total_mass:
            # Sample from finite part
            idx = np.searchsorted(cumsum, u_val * total_mass - dist.p_neg_inf)
            samples[i] = dist.x[idx]
        else:
            # Sample from positive infinity mass
            samples[i] = np.inf
    
    return samples


def monte_carlo_convolve_with_continuous(X: DiscreteDist, continuous_dist: stats.rv_continuous, 
                                       mode: Mode, spacing: Spacing, n_samples: int = 100000,
                                       n_bins: Optional[int] = None) -> DiscreteDist:
    """
    Monte Carlo convolution of a discrete PMF with a continuous distribution.
    
    This method samples from both the discrete PMF and the continuous distribution
    to approximate their convolution.
    
    Parameters:
    -----------
    X : DiscreteDist
        Discrete distribution (must be PMF)
    continuous_dist : scipy.stats.rv_continuous
        Continuous distribution object
    mode : Mode
        Tie-breaking mode
    spacing : Spacing
        Grid spacing strategy
    n_samples : int
        Number of Monte Carlo samples
    n_bins : int, optional
        Number of bins for histogramming
        
    Returns:
    --------
    DiscreteDist
        Convolution result
    """
    if X.kind != DistKind.PMF:
        raise ValueError(f'monte_carlo_convolve_with_continuous expects PMF input, got {X.kind}')
    
    # Determine number of bins
    if n_bins is None:
        n_bins = X.x.size
    
    # Sample from discrete distribution
    X_samples = _sample_from_pmf(X, n_samples)
    
    # Sample from continuous distribution
    Y_samples = continuous_dist.rvs(size=n_samples)
    
    # Compute convolution samples
    Z_samples = X_samples + Y_samples
    
    # Filter out infinite samples for histogramming
    finite_mask = np.isfinite(Z_samples)
    Z_finite = Z_samples[finite_mask]
    
    if len(Z_finite) == 0:
        raise ValueError("All convolution samples are infinite")
    
    # Create histogram bins
    z_min = Z_finite.min()
    z_max = Z_finite.max()
    
    # Ensure we have a valid range
    if z_max <= z_min:
        z_max = z_min + 1e-10
    
    if spacing == Spacing.GEOMETRIC:
        if z_min > 0:
            bins = np.geomspace(z_min, z_max, n_bins + 1)
        elif z_max < 0:
            bins = -np.geomspace(-z_max, -z_min, n_bins + 1)[::-1]
        else:
            bins = np.linspace(z_min, z_max, n_bins + 1)
    else:  # LINEAR
        bins = np.linspace(z_min, z_max, n_bins + 1)
    
    # Ensure bins are strictly increasing
    bins = np.sort(bins)
    
    # Compute histogram
    hist, bin_edges = np.histogram(Z_finite, bins=bins, density=False)
    
    # Convert to PMF
    pmf = hist.astype(np.float64) / n_samples
    
    # Create grid points
    x_grid = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Ensure grid is strictly increasing
    x_grid = np.sort(x_grid)
    
    # Handle infinity masses
    p_neg_inf = np.sum(Z_samples == -np.inf) / n_samples
    p_pos_inf = np.sum(Z_samples == np.inf) / n_samples
    
    return DiscreteDist(
        x=x_grid,
        kind=DistKind.PMF,
        vals=pmf,
        p_neg_inf=p_neg_inf,
        p_pos_inf=p_pos_inf,
        name=f"MC_conv({X.name or 'X'}, continuous)"
    )