# Discrete Distribution Convolution: Implementation Details

## Overview

This document describes the implementation of numerical convolution for probability distributions using discrete upper and lower bounds on arbitrary grids. The implementation provides rigorous bounds on cumulative distribution functions (CDFs) through careful discretization and convolution operations.

## Table of Contents

1. [Mathematical Foundation](#mathematical-foundation)
2. [Core Data Structures](#core-data-structures)
3. [Discretization Algorithm](#discretization-algorithm)
4. [Convolution Algorithm](#convolution-algorithm)
5. [Grid Strategies](#grid-strategies)
6. [Numerical Precision](#numerical-precision)
7. [Mode System](#mode-system)
8. [Alternative Methods](#alternative-methods)

---

## Mathematical Foundation

### Problem Statement

Given a continuous probability distribution with CDF F(x), we want to compute rigorous upper and lower bounds on the CDF of the sum of T independent copies:

Z = X₁ + X₂ + ... + Xₜ

where each Xᵢ ~ F independently.

### Discretization Approach

We discretize the continuous distribution onto a grid of points {x₁, x₂, ..., xₙ} to create discrete distributions that bound the continuous distribution:

- **Upper Bound (DOMINATES mode)**: For each grid point xᵢ, we assign all probability mass from the interval [xᵢ, xᵢ₊₁) to point xᵢ₊₁. This creates a **CCDF (survival function) that dominates** the true CCDF: S_upper(x) ≥ S_true(x), which means the **CDF is dominated**: F_upper(x) ≤ F_true(x). This gives *conservative* (lower) probabilities in the CDF.

- **Lower Bound (IS_DOMINATED mode)**: For each grid point xᵢ, we assign all probability mass from the interval [xᵢ, xᵢ₊₁) to point xᵢ. This creates a **CCDF that is dominated** by the true CCDF: S_lower(x) ≤ S_true(x), which means the **CDF dominates**: F_lower(x) ≥ F_true(x). This gives *aggressive* (higher) probabilities in the CDF.

**Key inequality relationships:**
- **For CCDF:** S_upper(x) ≥ S_true(x) ≥ S_lower(x)
- **For CDF:** F_lower(x) ≥ F_true(x) ≥ F_upper(x)
- **For expectations (by first-order stochastic dominance):** E[upper] ≥ E[true] ≥ E[lower]

Note: The expectation ordering may seem counter-intuitive given the CDF ordering, but it follows from stochastic dominance: higher CCDF means more probability mass on larger values, leading to higher expectation.

### Key Properties

1. **Domination Preservation**: When convolving upper bounds, the result is an upper bound on the convolution (conservative CCDF). Similarly for lower bounds (aggressive CCDF).

2. **Mass Conservation**: Total probability mass (including infinity masses) must equal 1.

3. **Grid Monotonicity**: Grid points must be strictly increasing: x₁ < x₂ < ... < xₙ.

---

## Core Data Structures

### DiscreteDist

The fundamental data structure representing a discrete distribution:

```python
@dataclass
class DiscreteDist:
    x: np.ndarray          # Grid points (strictly increasing)
    kind: DistKind         # PMF, CDF, or CCDF
    vals: np.ndarray       # Probability values
    p_neg_inf: float       # Probability mass at -∞
    p_pos_inf: float       # Probability mass at +∞
    name: Optional[str]    # Optional name
    debug_check: bool      # Enable validation checks
```

**Invariants:**
- `x` and `vals` have the same length
- `x` is strictly increasing
- `p_neg_inf, p_pos_inf >= 0`
- For PMF: `sum(vals) + p_neg_inf + p_pos_inf = 1`
- For CDF: `vals` is non-decreasing
- For CCDF: `vals` is non-increasing

### Enums

```python
class Mode(Enum):
    DOMINATES = "DOMINATES"        # Upper bound (CDF dominates true CDF)
    IS_DOMINATED = "IS_DOMINATED"  # Lower bound (CDF dominated by true CDF)

class Spacing(Enum):
    LINEAR = "linear"       # Uniform spacing: xᵢ₊₁ - xᵢ = const
    GEOMETRIC = "geometric" # Logarithmic spacing: xᵢ₊₁/xᵢ = const

class GridStrategy(Enum):
    FIXED_POINTS = "fixed_points"  # Maintain constant number of grid points
    FIXED_WIDTH = "fixed_width"    # Maintain constant bin width

class SummationMethod(Enum):
    STANDARD = "standard"  # Standard floating-point summation
    KAHAN = "kahan"       # Kahan compensated summation
    SORTED = "sorted"     # Sum smallest-to-largest by absolute value
```

---

## Discretization Algorithm

### Overview

The `discretize_continuous_to_pmf` function converts a continuous distribution into a discrete PMF on a grid while maintaining rigorous bounds.

### Algorithm Steps

#### 1. Quantile-Based Grid Generation

```python
x_min = dist.ppf(beta / 2)      # Left quantile
x_max = dist.ppf(1 - beta / 2)  # Right quantile
```

We trim probability mass β from the tails (β/2 from each side) to ensure finite support.

#### 2. Grid Point Selection

**Linear Spacing:**
```python
x = np.linspace(x_min, x_max, n_grid)
```

**Geometric Spacing** (for positive distributions):
```python
x = np.geomspace(x_min, x_max, n_grid)
```

#### 3. Bin Edge Construction

For each grid point xᵢ, we create bin edges:
```python
edges[0] = x[0] - (x[1] - x[0]) / 2        # Left edge
edges[i] = (x[i-1] + x[i]) / 2             # Midpoint between consecutive points
edges[n] = x[n-1] + (x[n-1] - x[n-2]) / 2  # Right edge
```

#### 4. Stable Mass Computation

For each bin [a, b], we compute P(a < X ≤ b) using numerically stable methods:

**Middle Region** (standard precision adequate):
```python
mass = F(b) - F(a)
```

**Right Tail** (when S(a), S(b) < tail_switch):
```python
mass = exp(log S(a)) × (-expm1(log S(b) - log S(a)))
```
where S(x) = 1 - F(x) is the survival function.

**Left Tail** (when F(a), F(b) < tail_switch):
```python
mass = exp(log F(b)) × (-expm1(log F(a) - log F(b)))
```

This uses the identity: `exp(a) - exp(b) = exp(a) × (1 - exp(b-a)) = exp(a) × (-expm1(b-a))`

#### 5. Infinity Mass Assignment

Based on the mode:

**DOMINATES (upper bound):**
- `p_neg_inf = 0` (all left-tail mass goes to finite grid)
- `p_pos_inf = F(edges[0]) + (1 - F(edges[n]))` (capture all tail mass)

**IS_DOMINATED (lower bound):**
- `p_neg_inf = F(edges[0]) + (1 - F(edges[n]))` (capture all tail mass)
- `p_pos_inf = 0` (all right-tail mass goes to finite grid)

---

## Convolution Algorithm

### PMF × PMF Convolution

The core operation is convolving two discrete PMFs: Z = X + Y.

#### Kernel Algorithm

For each combination of points (xᵢ, yⱼ):

1. Compute sum: `z = xᵢ + yⱼ`
2. Compute mass: `mass = P(X = xᵢ) × P(Y = yⱼ)`
3. Find target bin in output grid based on mode:
   - **DOMINATES**: Find smallest k where `z_k ≥ z`
   - **IS_DOMINATED**: Find largest k where `z_k ≤ z`
4. Add mass to target bin (with Kahan summation if enabled)

#### Standard Kernel

```python
@njit
def _pmf_pmf_kernel_standard(x1, p1, x2, p2, x_out, mode_val):
    pmf_out = np.zeros(x_out.size)
    for i in range(x1.size):
        for j in range(x2.size):
            z = x1[i] + x2[j]
            mass = p1[i] * p2[j]
            # Find target bin and add mass
            ...
    return pmf_out, pneg_extra, ppos_extra
```

#### Kahan Kernel

The Kahan version maintains compensation terms to reduce floating-point rounding errors:

```python
@njit
def _pmf_pmf_kernel_kahan(x1, p1, x2, p2, x_out, mode_val):
    pmf_out = np.zeros(x_out.size)
    compensations = np.zeros(x_out.size)
    for i in range(x1.size):
        for j in range(x2.size):
            z = x1[i] + x2[j]
            mass = p1[i] * p2[j]
            # Kahan summation:
            y = mass - compensations[k]
            t = pmf_out[k] + y
            compensations[k] = (t - pmf_out[k]) - y
            pmf_out[k] = t
    return pmf_out, pneg_extra, ppos_extra
```

### Self-Convolution via Exponentiation by Squaring

To compute X^(⊕T) = X ⊕ X ⊕ ... ⊕ X (T times), we use binary exponentiation:

```python
def self_convolve_pmf_core(base, T, ...):
    base_dist = base
    acc_dist = None
    while T > 0:
        if T & 1:  # If T is odd
            acc_dist = base_dist if acc_dist is None else convolve(acc_dist, base_dist)
        T >>= 1
        if T > 0:
            base_dist = convolve(base_dist, base_dist)  # Square
    return acc_dist
```

This reduces the number of convolutions from O(T) to O(log T).

---

## Grid Strategies

### Fixed Points Strategy (FIXED_POINTS)

**Original behavior**: Maintains a constant number of grid points across convolutions.

**Algorithm:**
```python
out_size = max(x1.size, x2.size)
x_out = sample_from_range(x_min, x_max, out_size, spacing)
```

**Advantages:**
- Predictable memory usage
- Consistent grid resolution

**Disadvantages:**
- Bin width grows with each convolution
- May lose detail in regions of interest

### Fixed Width Strategy (FIXED_WIDTH)

**New in v2**: Maintains approximately constant bin width across convolutions.

**Algorithm:**

1. **Compute characteristic bin width:**
   ```python
   width1 = median(diff(x1))
   width2 = median(diff(x2))
   target_width = min(width1, width2)
   ```

2. **Determine output size:**
   
   **Linear spacing:**
   ```python
   out_size = ceil((x_max - x_min) / target_width) + 1
   ```
   
   **Geometric spacing:**
   ```python
   ratio = 1 + target_width  # target_width is relative width
   out_size = ceil(log(x_max / x_min) / log(ratio)) + 1
   ```

3. **Create output grid:**
   ```python
   x_out = sample_from_range(x_min, x_max, out_size, spacing)
   ```

**Advantages:**
- Maintains resolution across convolutions
- Better accuracy for large T

**Disadvantages:**
- Number of grid points grows with T
- Higher memory usage

---

## Numerical Precision

### Floating-Point Error Accumulation

Standard floating-point arithmetic accumulates errors during:
1. Repeated additions in convolution kernel
2. Mass conservation checks via summation
3. Multiple convolutions (T convolutions → ~log₂(T) levels)

### Kahan Summation

Compensated summation reduces rounding errors:

```python
def kahan_sum(arr):
    s = 0.0
    c = 0.0  # Running compensation
    for x in arr:
        y = x - c           # Compensated value
        t = s + y           # New sum
        c = (t - s) - y     # New compensation
        s = t
    return s
```

**Error reduction:** From O(nε) to O(ε) where n is array size and ε is machine epsilon.

### Sorted Summation

Summing from smallest to largest absolute value reduces catastrophic cancellation:

```python
def sorted_sum(arr):
    sorted_vals = np.sort(np.abs(arr))
    signs = np.sign(arr[np.argsort(np.abs(arr))])
    return np.sum(sorted_vals * signs)
```

### Tail Computation

For extreme tail probabilities (< 10⁻¹²), we use log-space arithmetic:

**Standard (loses precision):**
```python
mass = F(b) - F(a)  # Catastrophic cancellation when both ≈ 0
```

**Stable (maintains precision):**
```python
mass = exp(log F(b)) × (-expm1(log F(a) - log F(b)))
```

---

## Mode System

### DOMINATES Mode (Upper Bound)

**Discretization:**
- Assigns interval [xᵢ, xᵢ₊₁) mass to right endpoint xᵢ₊₁
- Results in CDF that is ≥ true CDF everywhere
- `p_neg_inf = 0` (no mass at -∞)
- `p_pos_inf > 0` (captures right tail)

**Convolution:**
- For sum z = xᵢ + yⱼ, assigns mass to smallest grid point ≥ z
- Preserves upper bound property

**Use case:** When we need P(Z ≤ z) ≤ F_upper(z) (conservative quantiles)

### IS_DOMINATED Mode (Lower Bound)

**Discretization:**
- Assigns interval [xᵢ, xᵢ₊₁) mass to left endpoint xᵢ
- Results in CDF that is ≤ true CDF everywhere
- `p_neg_inf > 0` (captures left tail)
- `p_pos_inf = 0` (no mass at +∞)

**Convolution:**
- For sum z = xᵢ + yⱼ, assigns mass to largest grid point ≤ z
- Preserves lower bound property

**Use case:** When we need P(Z ≤ z) ≥ F_lower(z) (aggressive quantiles)

### Mode Constraints

**DOMINATES requirements:**
- Input distributions must have `p_neg_inf = 0`
- Output will have `p_neg_inf = 0`

**IS_DOMINATED requirements:**
- Input distributions must have `p_pos_inf = 0`
- Output will have `p_pos_inf = 0`

**Violation consequences:** Raises `ValueError`

---

## Alternative Methods

### FFT-Based Convolution

**Principle:** Uses Fast Fourier Transform for efficient convolution.

**Algorithm:**
```python
def fft_self_convolve(dist, T):
    pmf_fft = fft(pmf, n=fft_size)
    result_fft = pmf_fft ** T  # T-fold convolution
    result = ifft(result_fft)
```

**Requirements:**
- Linear spacing (uniform grid)
- Power-of-2 size for efficiency

**Special handling for lognormal:**
- Uses Fenton-Wilkinson approximation
- Approximates sum of lognormals as single lognormal via moment matching

**Advantages:**
- O(n log n) complexity
- Very fast for large T

**Disadvantages:**
- Requires linear spacing
- No rigorous bounds (approximate only)
- Lognormal sum approximation has error

### Monte Carlo Sampling

**Algorithm:**
```python
def monte_carlo_convolve(dist, T, n_samples):
    samples = dist.rvs(size=(n_samples, T))
    Z = np.sum(samples, axis=1)
    hist, bins = np.histogram(Z, bins=n_bins)
    pmf = hist / n_samples
```

**Advantages:**
- Works with any distribution
- Asymptotically exact as n_samples → ∞
- No accumulation of discretization error

**Disadvantages:**
- Stochastic (different results each run)
- Requires many samples for accuracy
- No rigorous bounds

**Importance sampling (deprecated):**
- Earlier versions attempted stratified sampling
- Removed due to bias issues

### Analytic Methods

**Gaussian (exact):**
```python
X₁ + X₂ + ... + Xₜ ~ N(T×μ, T×σ²)
```

**Lognormal (approximate):**
```python
# Fenton-Wilkinson approximation
mean_sum = T × E[X]
var_sum = T × Var[X]
# Fit lognormal to match moments
```

**Advantages:**
- Exact for Gaussian
- Very fast (no numerical computation)

**Disadvantages:**
- Limited to specific distributions
- Lognormal approximation has error
- No general solution

---

## Implementation Notes

### Grid Support Bounds

When convolving X and Y, we determine output grid range using probability mass thresholds:

```python
# Find quantiles that trim β/2 mass from each tail
x_min = quantile(X, β/2)
x_max = quantile(X, 1 - β/2)
y_min = quantile(Y, β/2)
y_max = quantile(Y, 1 - β/2)

# Output range
z_min = x_min + y_min
z_max = x_max + y_max
```

### Mass Conservation Checks

Before and after each convolution:
```python
total_mass = sum(pmf) + p_neg_inf + p_pos_inf
assert abs(total_mass - 1.0) < tolerance
```

**Summation methods:**
- `STANDARD`: Fast but accumulates error
- `KAHAN`: Better accuracy, slight overhead
- `SORTED`: Best for poorly conditioned sums

### Performance Characteristics

**Time complexity:**
- Discretization: O(n)
- Single convolution: O(n₁ × n₂)
- Self-convolution T times: O(n² log T) via binary exponentiation
- FFT: O(n log n)
- Monte Carlo: O(T × n_samples + n_bins)

**Space complexity:**
- Main method: O(n) per distribution
- FFT: O(fft_size) ≈ O(n)
- Monte Carlo: O(n_samples) during sampling, O(n_bins) for result

### Numerical Stability

**Critical operations requiring care:**
1. Tail probability computation (use log-space)
2. Repeated summation (use Kahan or sorted)
3. Mass conservation verification (use sorted sum)
4. CDF evaluation in extremes (use survival function)

**Error sources:**
1. Discretization error: O(1/n) for smooth distributions
2. Convolution kernel error: O(ε × n²) standard, O(ε) with Kahan
3. Accumulated error over T convolutions: O(log T) growth

---

## Usage Examples

### Basic Discretization and Convolution

```python
from scipy import stats

# Create continuous distribution
dist = stats.norm(0, 1)

# Discretize to upper bound
upper = discretize_continuous_to_pmf(
    dist, n_grid=1000, beta=1e-12,
    mode=Mode.DOMINATES, spacing=Spacing.LINEAR
)

# Discretize to lower bound
lower = discretize_continuous_to_pmf(
    dist, n_grid=1000, beta=1e-12,
    mode=Mode.IS_DOMINATED, spacing=Spacing.LINEAR
)

# Self-convolve 100 times
result_upper = self_convolve_pmf(upper, T=100, mode=Mode.DOMINATES)
result_lower = self_convolve_pmf(lower, T=100, mode=Mode.IS_DOMINATED)

# Now we have rigorous bounds on the 100-fold convolution CDF
```

### Using Fixed Width Strategy

```python
# Maintain constant bin width across convolutions
result = self_convolve_pmf(
    base, T=1000,
    mode=Mode.DOMINATES,
    spacing=Spacing.LINEAR,
    grid_strategy=GridStrategy.FIXED_WIDTH
)

# Number of points will grow, but resolution remains constant
print(f"Grid size: {len(result.x)}")
print(f"Median bin width: {np.median(np.diff(result.x)):.6f}")
```

### Enhanced Numerical Precision

```python
# Use Kahan summation and sorted mass checks
result = self_convolve_pmf(
    base, T=1000,
    mode=Mode.DOMINATES,
    use_kahan=True,
    sum_method=SummationMethod.SORTED
)
```

### Geometric Spacing for Log-Scale Distributions

```python
# For lognormal or exponential distributions
dist = stats.lognorm(s=1, scale=1)

upper = discretize_continuous_to_pmf(
    dist, n_grid=1000, beta=1e-12,
    mode=Mode.DOMINATES,
    spacing=Spacing.GEOMETRIC  # Logarithmically-spaced grid
)
```

---

## References

1. **Kahan Summation:** Kahan, W. (1965). "Further remarks on reducing truncation errors". Communications of the ACM 8 (1): 40.

2. **Exponentiation by Squaring:** Knuth, D. E. (1997). The Art of Computer Programming, Volume 2: Seminumerical Algorithms (3rd ed.).

3. **Fenton-Wilkinson Approximation:** Fenton, L. (1960). "The Sum of Log-Normal Probability Distributions in Scatter Transmission Systems". IRE Transactions on Communications Systems.

4. **FFT Convolution:** Cooley, J. W.; Tukey, J. W. (1965). "An algorithm for the machine calculation of complex Fourier series". Mathematics of Computation 19 (90): 297–301.

---

## Appendix: Common Pitfalls

### 1. Mode Mismatch

**Wrong:**
```python
# Trying to convolve DOMINATES with IS_DOMINATED
result = convolve_pmf_pmf_to_pmf(upper, lower, mode=Mode.DOMINATES)
# Error: IS_DOMINATED input has p_pos_inf > 0 but DOMINATES requires 0
```

**Right:**
```python
# Keep modes consistent
result_upper = convolve_pmf_pmf_to_pmf(upper1, upper2, mode=Mode.DOMINATES)
result_lower = convolve_pmf_pmf_to_pmf(lower1, lower2, mode=Mode.IS_DOMINATED)
```

### 2. Geometric Spacing with Negative Values

**Wrong:**
```python
dist = stats.norm(0, 1)  # Has negative values
result = discretize_continuous_to_pmf(
    dist, 1000, 1e-12, Mode.DOMINATES,
    spacing=Spacing.GEOMETRIC  # Error!
)
```

**Right:**
```python
# Use LINEAR for distributions with negative support
result = discretize_continuous_to_pmf(
    dist, 1000, 1e-12, Mode.DOMINATES,
    spacing=Spacing.LINEAR
)
```

### 3. Insufficient Grid Resolution

**Wrong:**
```python
# Too few points for large T
base = discretize_continuous_to_pmf(dist, n_grid=50, ...)
result = self_convolve_pmf(base, T=1000, ...)  # Mass leaks to infinity
```

**Right:**
```python
# Use adequate resolution and/or FIXED_WIDTH strategy
base = discretize_continuous_to_pmf(dist, n_grid=5000, ...)
result = self_convolve_pmf(
    base, T=1000,
    grid_strategy=GridStrategy.FIXED_WIDTH
)
```

### 4. Ignoring Mass Conservation Errors

**Wrong:**
```python
try:
    result = self_convolve_pmf(base, T=100, ...)
except ValueError as e:
    pass  # Ignoring mass conservation errors
```

**Right:**
```python
# Investigate and fix the root cause
try:
    result = self_convolve_pmf(base, T=100, ...)
except ValueError as e:
    print(f"Mass conservation error: {e}")
    # Increase n_grid, adjust beta, or use FIXED_WIDTH strategy
```

---

## Version History

**v1:**
- Initial implementation
- Fixed points strategy only
- Basic Kahan summation

**v2:**
- Consistent use of DiscreteDist throughout
- Added FIXED_WIDTH grid strategy
- Enhanced property testing
- Improved documentation
- Better API design (discretize_continuous_to_pmf returns DiscreteDist)