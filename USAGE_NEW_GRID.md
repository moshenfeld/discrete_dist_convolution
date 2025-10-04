# API Guide: Automatic Grid Generation

## Overview

The convolution API has been redesigned for simplicity and power:
- **Grid generation happens automatically** inside kernels based on support bounds
- **Functions return `DiscreteDist` objects** directly (no tuples)
- **Spacing strategies** (`LINEAR` or `GEOMETRIC`) control grid generation
- **Grids evolve naturally** during self-convolution (no manual computation needed)

## Grid Generation Strategy

Given two distributions with grids `xX` and `xY`, the output grid is computed as:

1. **Output size**: `z_size = max(len(xX), len(xY))`
2. **Support bounds**: `z_min = xX[0] + xY[0]`, `z_max = xX[-1] + xY[-1]`
3. **Spacing**: Linear (`np.linspace`) or Geometric (`np.geomspace`) between bounds

This ensures:
- Predictable output grid size
- Full coverage of convolution support
- Consistent spacing strategy across operations

## Basic Usage

### Pairwise Convolution

```python
from discrete_conv_api import (
    discretize_continuous_to_pmf,
    convolve_pmf_pmf_to_pmf,
    DiscreteDist,
    GridMode,
    Spacing
)
from scipy import stats

# Discretize continuous distributions
N_BINS = 1000
BETA = 1e-6

dist_x = stats.norm(0, 1)
x, pmf_x, pneg_x, ppos_x = discretize_continuous_to_pmf(
    dist_x, N_BINS, BETA, 
    mode=GridMode.DOMINATES, 
    spacing=Spacing.LINEAR
)
X = DiscreteDist(x=x, kind='pmf', vals=pmf_x, 
                 p_neg_inf=pneg_x, p_pos_inf=ppos_x)

dist_y = stats.norm(1, 0.8)
y, pmf_y, pneg_y, ppos_y = discretize_continuous_to_pmf(
    dist_y, N_BINS, BETA,
    mode=GridMode.DOMINATES,
    spacing=Spacing.LINEAR
)
Y = DiscreteDist(x=y, kind='pmf', vals=pmf_y,
                 p_neg_inf=pneg_y, p_pos_inf=ppos_y)

# Convolve: Z = X + Y (grid generated automatically)
Z = convolve_pmf_pmf_to_pmf(X, Y, mode='DOMINATES', spacing=Spacing.LINEAR)

# Z is a DiscreteDist - ready to use!
print(f"Result: {len(Z.x)} bins")
print(f"E[Z] = {(Z.x * Z.vals).sum():.4f}")
print(f"Sum(PMF) = {Z.vals.sum():.6f}")
```

### Self-Convolution

```python
from discrete_conv_api import self_convolve_pmf

# Create base distribution
dist_base = stats.norm(0, 1)
x_base, pmf_base, pneg_base, ppos_base = discretize_continuous_to_pmf(
    dist_base, N_BINS, BETA,
    mode=GridMode.DOMINATES,
    spacing=Spacing.LINEAR
)
base = DiscreteDist(x=x_base, kind='pmf', vals=pmf_base,
                    p_neg_inf=pneg_base, p_pos_inf=ppos_base)

# Self-convolve: Z = base + base + ... + base (T times)
T = 10
Z = self_convolve_pmf(base, T, mode='DOMINATES', spacing=Spacing.LINEAR)

# That's it! Grids computed automatically at each step
print(f"Result: {Z.name}")
print(f"E[Z] = {(Z.x * Z.vals).sum():.4f} (theory: {T * 0.0:.1f})")
```

## Spacing Strategies

### `Spacing.LINEAR`

- Uses `np.linspace(z_min, z_max, z_size)`
- Uniform bin spacing
- **Best for**: Distributions that cross zero or have symmetric support
- **Examples**: Normal, Uniform

```python
# Linear spacing for distributions crossing zero
Z = convolve_pmf_pmf_to_pmf(X, Y, mode='DOMINATES', spacing=Spacing.LINEAR)
```

### `Spacing.GEOMETRIC`

- Uses `np.geomspace(z_min, z_max, z_size)`  
- Logarithmic bin spacing (constant ratio between consecutive points)
- **Best for**: Strictly positive or strictly negative distributions
- **Examples**: Exponential, Lognormal
- **Constraint**: Cannot be used if `[z_min, z_max]` contains 0

```python
# Geometric spacing for positive distributions
Z = convolve_pmf_pmf_to_pmf(X, Y, mode='DOMINATES', spacing=Spacing.GEOMETRIC)
```

## Chaining Operations

Since functions return `DiscreteDist` objects, operations chain naturally:

```python
# Compute (X + Y) + X
Z1 = convolve_pmf_pmf_to_pmf(X, Y, mode='DOMINATES', spacing=Spacing.LINEAR)
Z2 = convolve_pmf_pmf_to_pmf(Z1, X, mode='DOMINATES', spacing=Spacing.LINEAR)

print(f"E[(X+Y)+X] = {(Z2.x * Z2.vals).sum():.4f}")
```

## Mode: Tie-Breaking

The `mode` parameter controls how exact grid hits are handled:

### `mode='DOMINATES'` (Upper Bound)
- Exact hits round **up** to next grid point
- Uses `searchsorted(..., side='right')`
- Produces **upper bound** on true distribution

### `mode='IS_DOMINATED'` (Lower Bound)  
- Exact hits round **down** to previous grid point
- Uses `searchsorted(..., side='left') - 1`
- Produces **lower bound** on true distribution

```python
# Compute both bounds
Z_upper = convolve_pmf_pmf_to_pmf(X, Y, mode='DOMINATES', spacing=Spacing.LINEAR)
Z_lower = convolve_pmf_pmf_to_pmf(X, Y, mode='IS_DOMINATED', spacing=Spacing.LINEAR)

# Compute bias (discretization error)
E_upper = (Z_upper.x * Z_upper.vals).sum()
E_lower = (Z_lower.x * Z_lower.vals).sum()
bias = E_upper - E_lower
print(f"Bias: {bias:.6f}")
```

## Advanced: Explicit Grid (Legacy)

For backwards compatibility, you can still provide an explicit grid:

```python
import numpy as np

# Compute grid manually
z_min = X.x[0] + Y.x[0]
z_max = X.x[-1] + Y.x[-1]
t = np.linspace(z_min, z_max, 1000)

# Use explicit grid (spacing parameter ignored)
Z = convolve_pmf_pmf_to_pmf(X, Y, mode='DOMINATES', t=t)
```

**Note**: This is **not recommended** for new code. The automatic grid generation is simpler and more robust.

## Discretization from Continuous Distributions

The `discretize_continuous_to_pmf` function converts continuous distributions to discrete PMFs:

```python
from discrete_conv_api import discretize_continuous_to_pmf, GridMode, Spacing
from scipy import stats

dist = stats.norm(0, 1)

# Parameters
n_grid = 1000  # Number of grid points
beta = 1e-6    # Tail probability to trim (quantiles: [beta/2, 1-beta/2])
mode = GridMode.DOMINATES  # or GridMode.IS_DOMINATED
spacing = Spacing.LINEAR    # or Spacing.GEOMETRIC

# Discretize
x, pmf, p_neg_inf, p_pos_inf = discretize_continuous_to_pmf(
    dist, n_grid, beta, mode, spacing
)

# Create DiscreteDist
X = DiscreteDist(x=x, kind='pmf', vals=pmf,
                 p_neg_inf=p_neg_inf, p_pos_inf=p_pos_inf,
                 name='X')
```

### Grid Range Selection

- **Quantile-based**: `[dist.ppf(beta/2), dist.ppf(1-beta/2)]`
- **Trims tails**: Total `beta` probability trimmed, split evenly at both ends
- **Spacing**: Linear or geometric within this range

### Mode Effects

- **`GridMode.DOMINATES`**: 
  - Uses CDF for discretization
  - `p_neg_inf = CDF(x[0])`
  - `p_pos_inf = 1 - CDF(x[-1])`

- **`GridMode.IS_DOMINATED`**:
  - Uses CCDF for discretization  
  - `p_neg_inf = 0` (lower bound has no mass at -∞)
  - `p_pos_inf = CCDF(x[-1])`

## Performance Considerations

### Self-Convolution Efficiency

Binary exponentiation provides massive speedup:
- **T=10**: 7 convolutions (vs 9 naive)
- **T=100**: 13 convolutions (vs 99 naive)
- **T=1000**: 17 convolutions (vs 999 naive)

### Grid Size

- **Typical**: 1,000 - 5,000 bins provides good accuracy/speed balance
- **High accuracy**: 10,000+ bins (but slower)
- **Memory**: O(n²) per convolution for n-bin grids

### Spacing Choice

- **Linear**: More uniform coverage, better for symmetric distributions
- **Geometric**: Better resolution near origin, ideal for heavy-tailed distributions

## Complete Example

```python
from discrete_conv_api import (
    discretize_continuous_to_pmf,
    self_convolve_pmf,
    DiscreteDist,
    GridMode,
    Spacing
)
from scipy import stats
import numpy as np

# Setup
N_BINS = 5000
BETA = 1e-6
T = 100

# Create base distribution
dist_base = stats.expon(scale=1.0)
x, pmf, pneg, ppos = discretize_continuous_to_pmf(
    dist_base, N_BINS, BETA,
    mode=GridMode.DOMINATES,
    spacing=Spacing.GEOMETRIC  # Exponential is positive
)
base = DiscreteDist(x=x, kind='pmf', vals=pmf,
                    p_neg_inf=pneg, p_pos_inf=ppos,
                    name='Exp(1)')

print(f"Base: {len(base.x)} bins, range=[{base.x[0]:.4f}, {base.x[-1]:.4f}]")
print(f"E[base] = {(base.x * base.vals).sum():.4f} (theory: 1.0)")

# Self-convolve
Z = self_convolve_pmf(base, T, mode='DOMINATES', spacing=Spacing.GEOMETRIC)

print(f"\nResult: {Z.name}")
print(f"  Bins: {len(Z.x)}")
print(f"  Range: [{Z.x[0]:.4f}, {Z.x[-1]:.4f}]")
print(f"  E[Z] = {(Z.x * Z.vals).sum():.4f} (theory: {T * 1.0:.1f})")
print(f"  Sum(PMF) = {Z.vals.sum():.6f}")
```

## Summary

### Key Improvements

1. **No manual grid computation** - happens automatically
2. **Clean API** - functions return `DiscreteDist` objects
3. **Easy chaining** - compose operations naturally
4. **Flexible spacing** - choose linear or geometric
5. **Efficient** - binary exponentiation for self-convolution

### Migration from Old API

**Old:**
```python
# Manual grid computation
z_min = T * base.x[0]
z_max = T * base.x[-1]
t = np.linspace(z_min, z_max, N_BINS)

# Returns tuple
pmf_out, pneg, ppos = self_convolve_pmf_core(base, T, t, mode)
Z = DiscreteDist(x=t, kind='pmf', vals=pmf_out, p_neg_inf=pneg, p_pos_inf=ppos)
```

**New:**
```python
# Automatic - just specify spacing
Z = self_convolve_pmf(base, T, mode='DOMINATES', spacing=Spacing.LINEAR)
```

Much simpler! 🎉
