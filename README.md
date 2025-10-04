# Discrete Distribution Convolution

High-performance discrete distribution convolution with automatic grid generation, proper tie-breaking, and infinity mass handling. Implements PMF convolution and self-convolution using Numba-accelerated kernels.

## Features

- **Automatic Grid Generation**: Grids computed from support bounds with linear or geometric spacing
- **Flexible API**: Returns `DiscreteDist` objects for easy chaining
- **Binary Exponentiation**: Efficient self-convolution (O(log T) operations for T-fold convolution)
- **Proper Tie-Breaking**: DOMINATES/IS_DOMINATED modes for bound computation
- **Numba-Optimized**: JIT-compiled kernels for high performance

## Quickstart

```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -U pip

# Install library (editable) + dev tools
pip install -e ".[dev,viz,test]"

# Run tests
pytest -q

# Run demonstrations
python experiments/demo_5k_bins.py
python experiments/demo_spacing_comparison.py
```

## Quick Example

```python
from discrete_conv_api import (
    discretize_continuous_to_pmf,
    convolve_pmf_pmf_to_pmf,
    self_convolve_pmf,
    DiscreteDist,
    GridMode,
    Spacing
)
from scipy import stats

# Discretize continuous distributions
dist_x = stats.norm(0, 1)
x, pmf_x, pneg_x, ppos_x = discretize_continuous_to_pmf(
    dist_x, n_grid=1000, beta=1e-6, 
    mode=GridMode.DOMINATES, spacing=Spacing.LINEAR
)
X = DiscreteDist(x=x, kind='pmf', vals=pmf_x, 
                 p_neg_inf=pneg_x, p_pos_inf=ppos_x)

# Pairwise convolution (automatic grid)
Y = DiscreteDist(...)  # another distribution
Z = convolve_pmf_pmf_to_pmf(X, Y, mode='DOMINATES', spacing=Spacing.LINEAR)

# Self-convolution: X + X + ... + X (10 times)
Z_self = self_convolve_pmf(X, T=10, mode='DOMINATES', spacing=Spacing.LINEAR)

# Results are DiscreteDist objects
print(f"Result: {len(Z_self.x)} bins, E[Z] = {(Z_self.x * Z_self.vals).sum():.4f}")
```

## API Overview

- **Entry Points**: `discrete_conv_api.py`
  - `convolve_pmf_pmf_to_pmf(X, Y, mode, spacing)` - Pairwise PMF convolution
  - `self_convolve_pmf(base, T, mode, spacing)` - T-fold self-convolution  
  - `discretize_continuous_to_pmf(dist, n_grid, beta, mode, spacing)` - Discretization
  - All functions return `DiscreteDist` objects
  
- **Grid Generation**: `implementation/grids.py`
  - `build_grid_from_support_bounds(xX, xY, spacing)` - Support-based grid generation
  - Automatic sizing: output grid size = max(len(xX), len(xY))
  - Linear or geometric spacing options

- **Core Kernels**: `implementation/kernels.py`
  - `convolve_pmf_pmf_to_pmf_core(X, Y, mode, spacing)` - Main convolution kernel
  - Grid generation happens internally
  - Numba-optimized double loop with tie-breaking

- **Self-Convolution**: `implementation/selfconv.py`
  - Binary exponentiation algorithm
  - Grids evolve naturally at each step
  - O(log T) convolutions instead of O(T)

## Implementation Details

- **Grid Strategy**: Support bounds (z_min = x_min + y_min, z_max = x_max + y_max)
- **Tie-Breaking**: 
  - DOMINATES: exact hits round up (`searchsorted(..., 'right')`)
  - IS_DOMINATED: exact hits round down (`searchsorted(..., 'left') - 1`)
- **Infinity Handling**: Proper ledger for mass at ±∞
- **Performance**: O(mn) per convolution, O(log T * n²) for self-convolution

## Documentation

- `README.md` - Grid generation and API guide
- `docs/IMPLEMENTATION_GUIDE_NUMBA.md` - Implementation details and algorithms
- `docs/DERIVATION.tex` - Mathematical derivations
- `STATUS.md` - Current implementation status

## Tests

```bash
pytest -v                    # Run all tests
pytest tests/test_pmf_pmf_kernel.py  # Test core kernel
pytest tests/test_selfconv_core.py   # Test self-convolution
```

- 22 tests passing, 4 xpassing
- Full coverage of PMF×PMF kernel and self-convolution
- Property-based tests for budget conservation

## Coding Style

- Python 3.10+
- Type hints throughout
- Ruff + MyPy configured in `pyproject.toml`
- Prefer `float64` contiguous arrays
