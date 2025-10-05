"""
Monte Carlo sampling convolution implementation for continuous distributions.

This module provides a Monte Carlo-based convolution method that works by
sampling from continuous distributions and computing the sum of T elements.
The resulting samples are then used to construct an empirical distribution.
"""

import numpy as np
from scipy import stats
from typing import Tuple
import sys
import os

# Add parent directory to path to import implementation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation.types import DiscreteDist, DistKind, Mode, Spacing


def monte_carlo_self_convolve_pmf(base_dist: stats.rv_continuous, T: int, mode: Mode, spacing: Spacing,
                                 n_samples: int, n_bins: int, block_size: int = 100000) -> DiscreteDist:
    """
    Monte Carlo self-convolution of a continuous distribution T times.
    
    This method samples T elements from the base continuous distribution and computes
    their sum to approximate the T-fold self-convolution. Uses block-based processing
    to handle large sample sizes efficiently.
    
    Memory usage: ~block_size * T * 8 bytes (for double precision)
    - Default block_size=100k: ~0.8MB for T=100, ~8MB for T=1000
    - For high-memory systems, use larger block_size (e.g., 500k-1M)
    - For memory-constrained systems, use smaller block_size (e.g., 10k-50k)
    
    Parameters:
    -----------
    base_dist : scipy.stats.rv_continuous
        Base continuous distribution
    T : int
        Number of times to convolve (must be >= 1)
    mode : Mode
        Tie-breaking mode (used for consistency)
    spacing : Spacing
        Grid spacing strategy (used to determine output grid)
    n_samples : int
        Number of Monte Carlo samples to generate
    n_bins : int
        Number of bins for histogramming
    block_size : int
        Size of blocks for memory-efficient sampling (default: 100000)
        
    Returns:
    --------
    DiscreteDist
        T-fold self-convolution result
    """
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    
    if n_samples <= 0:
        raise ValueError(f'n_samples must be positive, got {n_samples}')
    
    if block_size <= 0:
        raise ValueError(f'block_size must be positive, got {block_size}')
    
    # Initialize histogram accumulator
    hist_accumulator = None
    bins = None
    z_min = float('inf')
    z_max = float('-inf')
    
    # Process samples in blocks to avoid memory issues
    n_blocks = (n_samples + block_size - 1) // block_size
    
    for block_idx in range(n_blocks):
        # Calculate block boundaries
        start_idx = block_idx * block_size
        end_idx = min(start_idx + block_size, n_samples)
        current_block_size = end_idx - start_idx
        
        # Sample T elements from base continuous distribution for current block
        # Shape: (current_block_size, T) - vectorized sampling
        samples_matrix = _sample_from_continuous(base_dist, current_block_size * T).reshape(current_block_size, T)
        
        # Compute sum of T elements for each Monte Carlo sample in this block
        Z_block = np.sum(samples_matrix, axis=1)
        
        # Update global min/max
        z_min = min(z_min, Z_block.min())
        z_max = max(z_max, Z_block.max())
        
        # Create histogram bins on first iteration
        if bins is None:
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
        
        # Compute histogram for this block
        hist_block, _ = np.histogram(Z_block, bins=bins, density=False)
        
        # Accumulate histogram
        if hist_accumulator is None:
            hist_accumulator = hist_block.astype(np.float64)
        else:
            hist_accumulator += hist_block.astype(np.float64)
    
    # Convert accumulated histogram to PMF (normalize by total number of samples)
    pmf = hist_accumulator / n_samples
    
    # Create grid points (bin centers)
    x_grid = (bins[:-1] + bins[1:]) / 2
    
    # Ensure grid is strictly increasing and unique
    x_grid = np.sort(np.unique(x_grid))
    
    # Ensure PMF matches grid size
    if len(pmf) != len(x_grid):
        # If sizes don't match, create a new PMF with correct size
        pmf = np.zeros(len(x_grid))
        for i in range(len(bins) - 1):
            if i < len(hist_accumulator):
                pmf[i] = hist_accumulator[i] / n_samples
    
    # Handle infinity masses (approximate from tail samples)
    p_neg_inf = 0.0  # Will be updated after final sampling if needed
    p_pos_inf = 0.0  # Will be updated after final sampling if needed
    
    # Determine name for continuous distribution
    base_name = f"continuous_{base_dist.dist.name if hasattr(base_dist, 'dist') else 'dist'}"
    
    return DiscreteDist(
        x=x_grid,
        kind=DistKind.PMF,
        vals=pmf,
        p_neg_inf=p_neg_inf,
        p_pos_inf=p_pos_inf,
        name=f"MC_selfconv_{T}({base_name})"
    )




def _sample_from_continuous(dist: stats.rv_continuous, n_samples: int) -> np.ndarray:
    """
    Sample from a continuous distribution using vectorized operations.
    
    Parameters:
    -----------
    dist : scipy.stats.rv_continuous
        Continuous distribution object
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    np.ndarray
        Array of samples
    """
    return dist.rvs(size=n_samples)

