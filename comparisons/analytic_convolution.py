"""
Analytic approximation convolution implementation for Gaussian and lognormal distributions.

This module provides analytic convolution methods that work exactly for Gaussian
distributions and approximately for lognormal distributions. For Gaussian distributions,
the convolution of T identical Gaussians is exactly Gaussian with mean T*μ and
variance T*σ². For lognormal distributions, this is an approximation.
"""

import numpy as np
from scipy import stats
from typing import Optional, Union, Tuple
import sys
import os

# Add parent directory to path to import implementation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from implementation.types import DiscreteDist, DistKind, Mode, Spacing


def analytic_convolve_gaussian(base_dist: stats.norm, T: int, mode: Mode, spacing: Spacing,
                              n_points: int = 1000, beta: float = 1e-6) -> DiscreteDist:
    """
    Analytic convolution of T identical Gaussian distributions.
    
    For Gaussian distributions, the convolution of T identical Gaussians is exactly
    Gaussian with mean T*μ and variance T*σ². This method creates the exact result
    by constructing a new Gaussian distribution with the appropriate parameters.
    
    Parameters:
    -----------
    base_dist : scipy.stats.norm
        Base Gaussian distribution
    T : int
        Number of times to convolve (must be >= 1)
    mode : Mode
        Tie-breaking mode (used for consistency)
    spacing : Spacing
        Grid spacing strategy (used to determine output grid)
    n_points : int
        Number of points for discretization
    beta : float
        Tail probability to trim
        
    Returns:
    --------
    DiscreteDist
        T-fold convolution result as PMF
    """
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    
    if T == 1:
        # Single convolution - discretize the base distribution
        return _discretize_gaussian(base_dist, n_points, beta, mode, spacing)
    
    # Compute parameters for T-fold convolution
    # For Gaussian: sum of T identical Gaussians ~ N(T*μ, T*σ²)
    new_mean = T * base_dist.mean()
    new_std = np.sqrt(T) * base_dist.std()
    
    # Create new Gaussian distribution
    convolved_dist = stats.norm(loc=new_mean, scale=new_std)
    
    # Discretize the convolved distribution
    return _discretize_gaussian(convolved_dist, n_points, beta, mode, spacing)


def analytic_convolve_lognormal(base_dist: stats.lognorm, T: int, mode: Mode, spacing: Spacing,
                               n_points: int = 1000, beta: float = 1e-6) -> DiscreteDist:
    """
    Analytic approximation for convolution of T identical lognormal distributions.
    
    For lognormal distributions, the convolution is not exactly lognormal, but we
    approximate it as lognormal with parameters computed from the sum of means and
    variances. This is an approximation that becomes more accurate as T increases
    due to the central limit theorem.
    
    Parameters:
    -----------
    base_dist : scipy.stats.lognorm
        Base lognormal distribution
    T : int
        Number of times to convolve (must be >= 1)
    mode : Mode
        Tie-breaking mode (used for consistency)
    spacing : Spacing
        Grid spacing strategy (used to determine output grid)
    n_points : int
        Number of points for discretization
    beta : float
        Tail probability to trim
        
    Returns:
    --------
    DiscreteDist
        T-fold convolution result as PMF (approximate)
    """
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    
    if T == 1:
        # Single convolution - discretize the base distribution
        return _discretize_lognormal(base_dist, n_points, beta, mode, spacing)
    
    # For lognormal approximation:
    # If X ~ LogNormal(μ, σ²), then sum of T identical lognormals
    # is approximately LogNormal(μ_sum, σ²_sum) where:
    # μ_sum = T * μ + log(T) * σ²/2  (approximation)
    # σ²_sum = T * σ²  (approximation)
    
    # Get base parameters
    base_mean = base_dist.mean()
    base_var = base_dist.var()
    
    # Compute approximate parameters for T-fold convolution
    # Using method of moments approximation
    approx_mean = T * base_mean
    approx_var = T * base_var
    
    # Convert back to lognormal parameters
    # For LogNormal(μ, σ²): mean = exp(μ + σ²/2), var = exp(2μ + σ²)(exp(σ²) - 1)
    # Solving: σ² = log(1 + var/mean²), μ = log(mean) - σ²/2
    cv_squared = approx_var / (approx_mean ** 2)
    sigma_squared = np.log(1 + cv_squared)
    mu = np.log(approx_mean) - sigma_squared / 2
    
    # Create approximate lognormal distribution
    # scipy.stats.lognorm uses s=σ, scale=exp(μ)
    approx_dist = stats.lognorm(s=np.sqrt(sigma_squared), scale=np.exp(mu))
    
    # Discretize the approximated distribution
    return _discretize_lognormal(approx_dist, n_points, beta, mode, spacing)


def analytic_convolve_pmf_pmf_gaussian(X: DiscreteDist, Y: DiscreteDist, mode: Mode, spacing: Spacing,
                                     n_points: int = 1000, beta: float = 1e-6) -> DiscreteDist:
    """
    Analytic convolution of two Gaussian PMFs.
    
    This method fits Gaussian distributions to the input PMFs and then uses
    the analytic convolution formula for Gaussians.
    
    Parameters:
    -----------
    X, Y : DiscreteDist
        Input distributions (must be PMF)
    mode : Mode
        Tie-breaking mode
    spacing : Spacing
        Grid spacing strategy
    n_points : int
        Number of points for discretization
    beta : float
        Tail probability to trim
        
    Returns:
    --------
    DiscreteDist
        Convolution result
    """
    if X.kind != DistKind.PMF or Y.kind != DistKind.PMF:
        raise ValueError(f'analytic_convolve_pmf_pmf_gaussian expects PMF inputs, got {X.kind}, {Y.kind}')
    
    # Fit Gaussian distributions to the PMFs
    X_gaussian = _fit_gaussian_to_pmf(X)
    Y_gaussian = _fit_gaussian_to_pmf(Y)
    
    # Compute convolution parameters
    # Sum of two independent Gaussians: N(μ₁, σ₁²) + N(μ₂, σ₂²) = N(μ₁+μ₂, σ₁²+σ₂²)
    conv_mean = X_gaussian.mean() + Y_gaussian.mean()
    conv_var = X_gaussian.var() + Y_gaussian.var()
    conv_std = np.sqrt(conv_var)
    
    # Create convolved Gaussian
    conv_gaussian = stats.norm(loc=conv_mean, scale=conv_std)
    
    # Discretize
    return _discretize_gaussian(conv_gaussian, n_points, beta, mode, spacing)


def _discretize_gaussian(dist: stats.norm, n_points: int, beta: float, mode: Mode, spacing: Spacing) -> DiscreteDist:
    """
    Discretize a Gaussian distribution onto a grid.
    
    Parameters:
    -----------
    dist : scipy.stats.norm
        Gaussian distribution
    n_points : int
        Number of grid points
    beta : float
        Tail probability to trim
    mode : Mode
        Tie-breaking mode
    spacing : Spacing
        Grid spacing strategy
        
    Returns:
    --------
    DiscreteDist
        Discretized distribution
    """
    # Determine range via quantiles
    q_min = dist.ppf(beta / 2)
    q_max = dist.ppf(1 - beta / 2)
    
    # Create grid based on spacing
    if spacing == Spacing.GEOMETRIC:
        if q_min > 0:
            x = np.geomspace(q_min, q_max, n_points)
        elif q_max < 0:
            x = -np.geomspace(-q_max, -q_min, n_points)[::-1]
        else:
            # Fall back to linear if range contains 0
            x = np.linspace(q_min, q_max, n_points)
    else:  # LINEAR
        x = np.linspace(q_min, q_max, n_points)
    
    # Compute PMF using CDF differences
    F = dist.cdf(x)
    
    if mode == Mode.DOMINATES:
        pmf = np.zeros(n_points)
        pmf[:-1] = np.diff(F)
        pmf[-1] = 0.0
        
        p_neg_inf = F[0]
        p_pos_inf = 1.0 - F[-1]
    else:  # IS_DOMINATED
        pmf = np.zeros(n_points)
        pmf[0] = F[0]
        pmf[1:] = np.diff(F)
        
        p_neg_inf = 0.0
        p_pos_inf = 1.0 - F[-1]
    
    # Ensure non-negative and normalize
    pmf = np.maximum(pmf, 0.0)
    total = pmf.sum() + p_neg_inf + p_pos_inf
    if total > 0:
        pmf = pmf / total * (1.0 - p_neg_inf - p_pos_inf)
    
    return DiscreteDist(
        x=x,
        kind=DistKind.PMF,
        vals=pmf,
        p_neg_inf=p_neg_inf,
        p_pos_inf=p_pos_inf,
        name=f"Analytic_Gaussian({dist.mean():.3f}, {dist.std():.3f})"
    )


def _discretize_lognormal(dist: stats.lognorm, n_points: int, beta: float, mode: Mode, spacing: Spacing) -> DiscreteDist:
    """
    Discretize a lognormal distribution onto a grid.
    
    Parameters:
    -----------
    dist : scipy.stats.lognorm
        Lognormal distribution
    n_points : int
        Number of grid points
    beta : float
        Tail probability to trim
    mode : Mode
        Tie-breaking mode
    spacing : Spacing
        Grid spacing strategy
        
    Returns:
    --------
    DiscreteDist
        Discretized distribution
    """
    # Determine range via quantiles
    q_min = dist.ppf(beta / 2)
    q_max = dist.ppf(1 - beta / 2)
    
    # Create grid based on spacing
    if spacing == Spacing.GEOMETRIC:
        if q_min > 0:
            x = np.geomspace(q_min, q_max, n_points)
        else:
            # Fall back to linear if range contains 0 or negative values
            x = np.linspace(q_min, q_max, n_points)
    else:  # LINEAR
        x = np.linspace(q_min, q_max, n_points)
    
    # Compute PMF using CDF differences
    F = dist.cdf(x)
    
    if mode == Mode.DOMINATES:
        pmf = np.zeros(n_points)
        pmf[:-1] = np.diff(F)
        pmf[-1] = 0.0
        
        p_neg_inf = F[0]
        p_pos_inf = 1.0 - F[-1]
    else:  # IS_DOMINATED
        pmf = np.zeros(n_points)
        pmf[0] = F[0]
        pmf[1:] = np.diff(F)
        
        p_neg_inf = 0.0
        p_pos_inf = 1.0 - F[-1]
    
    # Ensure non-negative and normalize
    pmf = np.maximum(pmf, 0.0)
    total = pmf.sum() + p_neg_inf + p_pos_inf
    if total > 0:
        pmf = pmf / total * (1.0 - p_neg_inf - p_pos_inf)
    
    return DiscreteDist(
        x=x,
        kind=DistKind.PMF,
        vals=pmf,
        p_neg_inf=p_neg_inf,
        p_pos_inf=p_pos_inf,
        name=f"Analytic_LogNormal({dist.mean():.3f}, {dist.std():.3f})"
    )


def _fit_gaussian_to_pmf(dist: DiscreteDist) -> stats.norm:
    """
    Fit a Gaussian distribution to a discrete PMF using method of moments.
    
    Parameters:
    -----------
    dist : DiscreteDist
        Discrete distribution (must be PMF)
        
    Returns:
    --------
    scipy.stats.norm
        Fitted Gaussian distribution
    """
    if dist.kind != DistKind.PMF:
        raise ValueError(f'_fit_gaussian_to_pmf expects PMF input, got {dist.kind}')
    
    # Compute moments
    mean = np.sum(dist.x * dist.vals)
    var = np.sum((dist.x - mean) ** 2 * dist.vals)
    
    # Account for infinity masses
    if dist.p_neg_inf > 0 or dist.p_pos_inf > 0:
        # Adjust variance to account for infinity masses
        # This is a heuristic - in practice, infinity masses should be small
        var = var / (1.0 - dist.p_neg_inf - dist.p_pos_inf)
    
    return stats.norm(loc=mean, scale=np.sqrt(var))


def analytic_convolve_with_continuous_gaussian(continuous_dist: stats.norm, T: int, mode: Mode, spacing: Spacing,
                                             n_points: int = 1000, beta: float = 1e-6) -> DiscreteDist:
    """
    Analytic convolution of T identical continuous Gaussian distributions.
    
    This is a convenience function that directly works with continuous Gaussian
    distributions without requiring discretization first.
    
    Parameters:
    -----------
    continuous_dist : scipy.stats.norm
        Continuous Gaussian distribution
    T : int
        Number of times to convolve
    mode : Mode
        Tie-breaking mode
    spacing : Spacing
        Grid spacing strategy
    n_points : int
        Number of points for discretization
    beta : float
        Tail probability to trim
        
    Returns:
    --------
    DiscreteDist
        T-fold convolution result
    """
    return analytic_convolve_gaussian(continuous_dist, T, mode, spacing, n_points, beta)


def analytic_convolve_with_continuous_lognormal(continuous_dist: stats.lognorm, T: int, mode: Mode, spacing: Spacing,
                                              n_points: int = 1000, beta: float = 1e-6) -> DiscreteDist:
    """
    Analytic approximation for convolution of T identical continuous lognormal distributions.
    
    This is a convenience function that directly works with continuous lognormal
    distributions without requiring discretization first.
    
    Parameters:
    -----------
    continuous_dist : scipy.stats.lognorm
        Continuous lognormal distribution
    T : int
        Number of times to convolve
    mode : Mode
        Tie-breaking mode
    spacing : Spacing
        Grid spacing strategy
    n_points : int
        Number of points for discretization
    beta : float
        Tail probability to trim
        
    Returns:
    --------
    DiscreteDist
        T-fold convolution result (approximate)
    """
    return analytic_convolve_lognormal(continuous_dist, T, mode, spacing, n_points, beta)
