"""
Moment computation utilities for discrete distributions.

This module provides functions to compute statistical moments of discrete
distributions, particularly useful for evaluating convolution methods.
"""

import numpy as np
from typing import Union
from math import comb
from discrete_conv_api import DiscreteDist


def compute_log_moment(dist: DiscreteDist, alpha: float) -> float:
    """
    Compute the alpha-th moment of log(X) for a discrete distribution.
    
    This function computes E[(log(X))^α] = Σ (log(x))^α * p(x) where the sum
    is over all finite positive x values.
    
    Parameters:
    -----------
    dist : DiscreteDist
        Discrete distribution (must be PMF kind)
    alpha : float
        Power for the moment computation (must be >= 0)
        
    Returns:
    --------
    float
        The alpha-th moment E[(log(X))^α]
        
    Notes:
    ------
    - Only considers finite positive x values (log(x) must be finite)
    - Ignores p_neg_inf and p_pos_inf
    - For alpha = 0, returns 1.0 (sum of probabilities)
    - For alpha = 1, returns E[log(X)]
    - For alpha = 2, returns E[(log(X))²]
    """
    if alpha < 0:
        raise ValueError(f"Alpha must be >= 0, got {alpha}")
    
    if dist.kind.name != 'PMF':
        raise ValueError(f"Distribution must be PMF kind, got {dist.kind}")
    
    # Handle special cases
    if alpha == 0:
        return 1.0
    
    # Only consider finite positive x values
    finite_positive_mask = (dist.x > 0) & np.isfinite(dist.x)
    if not np.any(finite_positive_mask):
        return np.nan
    
    x_finite = dist.x[finite_positive_mask]
    vals_finite = dist.vals[finite_positive_mask]
    
    # Compute log(x) for finite positive values
    log_x = np.log(x_finite)
    
    # Handle potential overflow/underflow
    if not np.all(np.isfinite(log_x)):
        return np.nan
    
    if alpha == 1:
        return np.sum(log_x * vals_finite)
    
    # General case: E[(log(X))^α] = Σ (log(x))^α * p(x)
    log_x_power = np.power(log_x, alpha)
    
    # Handle potential overflow/underflow
    if not np.all(np.isfinite(log_x_power)):
        return np.nan
    
    # Compute the moment
    moment = np.sum(log_x_power * vals_finite)
    
    return float(moment)


def compute_log_moment_sequence(dist: DiscreteDist, alpha_values: np.ndarray) -> np.ndarray:
    """
    Compute a sequence of log moments for different alpha values.
    
    Parameters:
    -----------
    dist : DiscreteDist
        Discrete distribution (must be PMF kind)
    alpha_values : np.ndarray
        Array of alpha values to compute log moments for
        
    Returns:
    --------
    np.ndarray
        Array of log moments corresponding to alpha_values
    """
    moments = np.zeros_like(alpha_values, dtype=float)
    
    for i, alpha in enumerate(alpha_values):
        moments[i] = compute_log_moment(dist, alpha)
    
    return moments


def compute_alpha_moment(dist: DiscreteDist, alpha: float) -> float:
    """
    Compute the alpha-th moment of a discrete distribution.
    
    The alpha-th moment is defined as E[X^α] = Σ x^α * p(x) where the sum
    is over all finite x values, discarding infinity probabilities.
    
    Parameters:
    -----------
    dist : DiscreteDist
        Discrete distribution (must be PMF kind)
    alpha : float
        Power for the moment computation (must be >= 0)
        
    Returns:
    --------
    float
        The alpha-th moment E[X^α]
        
    Notes:
    ------
    - Only considers finite x values, ignoring p_neg_inf and p_pos_inf
    - For alpha = 0, returns 1.0 (sum of probabilities)
    - For alpha = 1, returns the expectation E[X]
    - For alpha = 2, returns E[X²] (second moment)
    """
    if alpha < 0:
        raise ValueError(f"Alpha must be >= 0, got {alpha}")
    
    if dist.kind.name != 'PMF':
        raise ValueError(f"Distribution must be PMF kind, got {dist.kind}")
    
    # Handle special cases
    if alpha == 0:
        return 1.0
    
    if alpha == 1:
        return np.sum(dist.x * dist.vals)
    
    # General case: E[X^α] = Σ x^α * p(x)
    # Only consider finite x values
    finite_mask = np.isfinite(dist.x)
    if not np.any(finite_mask):
        return np.nan
    
    x_finite = dist.x[finite_mask]
    vals_finite = dist.vals[finite_mask]
    
    # Compute x^α for finite values
    x_power = np.power(x_finite, alpha)
    
    # Handle potential overflow/underflow
    if not np.all(np.isfinite(x_power)):
        # If some powers are infinite, we can't compute the moment reliably
        return np.nan
    
    # Compute the moment
    moment = np.sum(x_power * vals_finite)
    
    return float(moment)


def compute_moment_sequence(dist: DiscreteDist, alpha_values: np.ndarray) -> np.ndarray:
    """
    Compute a sequence of moments for different alpha values.
    
    Parameters:
    -----------
    dist : DiscreteDist
        Discrete distribution (must be PMF kind)
    alpha_values : np.ndarray
        Array of alpha values to compute moments for
        
    Returns:
    --------
    np.ndarray
        Array of moments corresponding to alpha_values
    """
    moments = np.zeros_like(alpha_values, dtype=float)
    
    for i, alpha in enumerate(alpha_values):
        moments[i] = compute_alpha_moment(dist, alpha)
    
    return moments


def compute_central_moment(dist: DiscreteDist, alpha: float) -> float:
    """
    Compute the alpha-th central moment of a discrete distribution.
    
    The alpha-th central moment is defined as E[(X - μ)^α] where μ = E[X].
    
    Parameters:
    -----------
    dist : DiscreteDist
        Discrete distribution (must be PMF kind)
    alpha : float
        Power for the central moment computation (must be >= 0)
        
    Returns:
    --------
    float
        The alpha-th central moment E[(X - μ)^α]
    """
    if alpha < 0:
        raise ValueError(f"Alpha must be >= 0, got {alpha}")
    
    if alpha == 0:
        return 1.0
    
    if alpha == 1:
        return 0.0  # E[X - μ] = E[X] - μ = 0
    
    # Compute the mean
    mean = compute_alpha_moment(dist, 1.0)
    
    if not np.isfinite(mean):
        return np.nan
    
    # Compute E[(X - μ)^α]
    finite_mask = np.isfinite(dist.x)
    if not np.any(finite_mask):
        return np.nan
    
    x_finite = dist.x[finite_mask]
    vals_finite = dist.vals[finite_mask]
    
    # Compute (x - μ)^α for finite values
    x_centered_power = np.power(x_finite - mean, alpha)
    
    # Handle potential overflow/underflow
    if not np.all(np.isfinite(x_centered_power)):
        return np.nan
    
    # Compute the central moment
    central_moment = np.sum(x_centered_power * vals_finite)
    
    return float(central_moment)


def compute_variance(dist: DiscreteDist) -> float:
    """
    Compute the variance of a discrete distribution.
    
    Variance = E[X²] - (E[X])² = second central moment.
    
    Parameters:
    -----------
    dist : DiscreteDist
        Discrete distribution (must be PMF kind)
        
    Returns:
    --------
    float
        The variance Var[X]
    """
    return compute_central_moment(dist, 2.0)


def compute_skewness(dist: DiscreteDist) -> float:
    """
    Compute the skewness of a discrete distribution.
    
    Skewness = E[(X - μ)³] / σ³ where σ is the standard deviation.
    
    Parameters:
    -----------
    dist : DiscreteDist
        Discrete distribution (must be PMF kind)
        
    Returns:
    --------
    float
        The skewness
    """
    variance = compute_variance(dist)
    if variance <= 0 or not np.isfinite(variance):
        return np.nan
    
    std_dev = np.sqrt(variance)
    third_central_moment = compute_central_moment(dist, 3.0)
    
    if not np.isfinite(third_central_moment):
        return np.nan
    
    return third_central_moment / (std_dev ** 3)


def compute_kurtosis(dist: DiscreteDist) -> float:
    """
    Compute the kurtosis of a discrete distribution.
    
    Kurtosis = E[(X - μ)⁴] / σ⁴ where σ is the standard deviation.
    
    Parameters:
    -----------
    dist : DiscreteDist
        Discrete distribution (must be PMF kind)
        
    Returns:
    --------
    float
        The kurtosis
    """
    variance = compute_variance(dist)
    if variance <= 0 or not np.isfinite(variance):
        return np.nan
    
    std_dev = np.sqrt(variance)
    fourth_central_moment = compute_central_moment(dist, 4.0)
    
    if not np.isfinite(fourth_central_moment):
        return np.nan
    
    return fourth_central_moment / (std_dev ** 4)


def moments_of_t_convolution(rv, t, alphas):
    """
    Compute E[(X1+...+Xt)^alpha] for iid X~rv (SciPy continuous RV), integer t>=1,
    for each alpha in `alphas` (array-like of nonnegative integers).

    Parameters
    ----------
    rv : scipy.stats.rv_continuous frozen distribution
        e.g., rv = scipy.stats.norm(loc=..., scale=...).freeze(...) is NOT required;
        you typically get a "frozen RV" by calling .rvs with parameters or .freeze.
        Must support rv.moment(k) for integer k >= 0.
    t : int
        Number of iid summands (t-fold convolution), t >= 1.
    alphas : array-like
        Nonnegative integer exponents. (If any alpha is non-integer or negative, raises ValueError.)

    Returns
    -------
    out : np.ndarray
        Array of the same shape as `alphas`, where out[i] = E[(sum_{j=1}^t X_j) ** alphas[i]].

    Notes
    -----
    Uses the partial Bell polynomial identity:
        E[S_t^n] = sum_{k=1}^n t^{↓k} * B_{n,k}(μ1,..., μ_{n-k+1}),
    with μ_r = E[X^r]. Here t^{↓k} = t*(t-1)*...*(t-k+1) (falling factorial).
    """
    # ---- input checks ----
    alphas = np.asarray(alphas)
    if np.any(alphas < 0):
        raise ValueError("All alpha must be nonnegative.")
    # Require integers:
    if not np.all(np.equal(alphas, np.floor(alphas))):
        raise ValueError("This function requires integer alphas. "
                         "If you need non-integer moments, we can add a numeric fallback.")
    if not isinstance(t, (int, np.integer)) or t < 1:
        raise ValueError("t must be an integer >= 1.")

    alphas = alphas.astype(int)
    max_n = int(alphas.max(initial=0))
    if max_n == 0:
        # E[S_t^0] = 1
        return np.ones_like(alphas, dtype=float)

    # ---- collect raw moments μ_r up to r = max_n ----
    # SciPy's frozen RVs typically provide .moment(n) for integer n >= 0
    mu = np.empty(max_n + 1, dtype=float)
    mu[0] = 1.0
    for r in range(1, max_n + 1):
        val = rv.moment(r)
        mu[r] = float(val)

    # ---- partial Bell polynomials B_{n,k}(x1, x2, ..., x_{n-k+1}) ----
    # Recurrence: B_{0,0}=1; B_{n,0}=0 for n>=1; B_{n,k}=0 if k>n
    # B_{n,k} = sum_{j=1}^{n-k+1} C(n-1, j-1) * x_j * B_{n-j, k-1}
    # Here x_j = μ_j (raw moments).
    # We'll compute all needed B_{n,k} up to n=max_n.
    B = [[0.0]*(max_n+1) for _ in range(max_n+1)]
    B[0][0] = 1.0
    x = mu  # alias

    for n in range(1, max_n + 1):
        B[n][0] = 0.0
        for k in range(1, n + 1):
            s = 0.0
            # j runs to n-k+1 to ensure the remaining (n-j,k-1) is valid
            jmax = n - k + 1
            for j in range(1, jmax + 1):
                s += comb(n - 1, j - 1) * x[j] * B[n - j][k - 1]
            B[n][k] = s

    # ---- falling factorial t^{↓k} ----
    def falling(t_int, k_int):
        prod = 1
        for m in range(k_int):
            prod *= (t_int - m)
        return prod

    # ---- assemble moments for each n ----
    # E[S_t^n] = sum_{k=1}^n t^{↓k} * B_{n,k}(μ1..)
    E_sum_pow = np.zeros(max_n + 1, dtype=float)
    E_sum_pow[0] = 1.0
    for n in range(1, max_n + 1):
        total = 0.0
        for k in range(1, n + 1):
            total += falling(t, k) * B[n][k]
        E_sum_pow[n] = total

    # Map back to requested alphas
    return E_sum_pow[alphas]
