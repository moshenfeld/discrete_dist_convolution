# Discrete Distribution Convolution - Version 2

## Summary of Changes

Version 2 includes the following major improvements requested:

### 1. ✅ Consistent Use of DiscreteDist Objects

**Before (v1):**
```python
x, pmf, p_neg_inf, p_pos_inf = discretize_continuous_to_pmf(dist, n_bins, beta, mode, spacing)
base = DiscreteDist(x=x, kind=DistKind.PMF, vals=pmf, p_neg_inf=p_neg_inf, p_pos_inf=p_pos_inf)
```

**After (v2):**
```python
base = discretize_continuous_to_pmf(dist, n_bins, beta, mode, spacing)
# Returns DiscreteDist directly - no tuple unpacking needed
```

All functions now use `DiscreteDist` objects consistently for inputs and outputs. No more manual tuple unpacking!

### 2. ✅ Fixed Bin Width Grid Strategy

**New feature:** Alternative grid generation strategy that maintains constant bin width across convolutions instead of constant number of points.

```python
from implementation_enhanced_v2 import GridStrategy

# Original behavior (maintains number of points)
result = self_convolve_pmf(
    base, T=1000,
    grid_strategy=GridStrategy.FIXED_POINTS  # Default
)

# New behavior (maintains bin width)
result = self_convolve_pmf(
    base, T=1000,
    grid_strategy=GridStrategy.FIXED_WIDTH  # Better for large T
)
```

**Key differences:**
- **FIXED_POINTS**: Grid size stays ~constant, bin width grows with each convolution
- **FIXED_WIDTH**: Bin width stays ~constant, grid size grows proportionally to the support width

**When to use FIXED_WIDTH:**
- Large T values (T > 100)
- When accuracy is more important than memory
- When the distribution support expands significantly

### 3. ✅ CDF Domination Tests

New comprehensive tests ensure that the upper bound CDF always dominates (is ≥) the lower bound CDF:

```python
# From tests_enhanced_v2.py
class TestCDFDomination:
    def test_cdf_domination_gaussian(self):
        """Test CDF domination for Gaussian distribution."""
        # Creates upper and lower bounds
        # Verifies F_upper(x) >= F_lower(x) for all x
        
    def test_cdf_domination_after_convolution(self):
        """Test that CDF domination is preserved after convolution."""
        # Verifies property is maintained through T convolutions
        
    def test_expected_value_bounds(self):
        """Test that E[upper] >= E[lower] for convolutions."""
        # Checks expectation ordering
```

### 4. ✅ Comprehensive Property Tests

New test classes verify all assumed properties:

```python
class TestDistributionProperties:
    test_pmf_nonnegativity()         # PMF values >= 0
    test_cdf_monotonicity()          # CDF is non-decreasing
    test_cdf_bounds()                # 0 <= CDF <= 1
    test_grid_monotonicity()         # Grid strictly increasing
    test_infinity_mass_non_negative() # p_neg_inf, p_pos_inf >= 0
    test_mode_constraints()          # Mode-specific constraints
```

Additional test categories:
- Grid strategy tests (FIXED_POINTS vs FIXED_WIDTH)
- Conversion tests (PMF ↔ CDF ↔ CCDF)
- Mass conservation tests
- Edge cases and error conditions

### 5. ✅ Implementation Documentation

Comprehensive documentation file `IMPLEMENTATION_DETAILS.md` covers:
- Mathematical foundation and theory
- Algorithm descriptions with pseudocode
- Grid strategy details
- Numerical precision techniques
- Mode system explanation
- Usage examples and common pitfalls
- Version history

## File Structure

```
project/
├── implementation_enhanced_v2.py      # Core implementation with v2 improvements
├── comparisons_enhanced_v2.py         # Alternative methods (FFT, MC, Analytic)
├── tests_enhanced_v2.py               # Comprehensive test suite
├── exp_enhanced_v2.py                 # Experiments and visualization
├── IMPLEMENTATION_DETAILS.md          # Complete documentation
└── README_V2.md                       # This file
```

## Quick Start

### Basic Usage

```python
from scipy import stats
from implementation_enhanced_v2 import (
    discretize_continuous_to_pmf, self_convolve_pmf,
    Mode, Spacing, GridStrategy
)

# 1. Create continuous distribution
dist = stats.norm(0, 1)

# 2. Discretize to upper and lower bounds
upper = discretize_continuous_to_pmf(
    dist, n_grid=1000, beta=1e-12,
    mode=Mode.DOMINATES,
    spacing=Spacing.LINEAR,
    name='Gaussian-upper'
)

lower = discretize_continuous_to_pmf(
    dist, n_grid=1000, beta=1e-12,
    mode=Mode.IS_DOMINATED,
    spacing=Spacing.LINEAR,
    name='Gaussian-lower'
)

# 3. Self-convolve T times
T = 100
result_upper = self_convolve_pmf(upper, T=T, mode=Mode.DOMINATES)
result_lower = self_convolve_pmf(lower, T=T, mode=Mode.IS_DOMINATED)

# 4. Access results
print(f"Grid size: {len(result_upper.x)}")
print(f"Expected value (upper): {np.sum(result_upper.x * result_upper.vals):.3f}")
print(f"Expected value (lower): {np.sum(result_lower.x * result_lower.vals):.3f}")
```

### Using Fixed Width Strategy

```python
# For large T, use FIXED_WIDTH to maintain resolution
result = self_convolve_pmf(
    base,
    T=1000,
    mode=Mode.DOMINATES,
    spacing=Spacing.LINEAR,
    grid_strategy=GridStrategy.FIXED_WIDTH,  # NEW!
    use_kahan=True,
    sum_method=SummationMethod.SORTED
)

print(f"Grid size grew from {len(base.x)} to {len(result.x)}")
print(f"But median bin width stayed ~constant")
```

### Working with CDFs

```python
from implementation_enhanced_v2 import pmf_to_cdf

# Convert PMF to CDF
cdf_upper = pmf_to_cdf(result_upper)
cdf_lower = pmf_to_cdf(result_lower)

# CDF values are in cdf.vals
# Can verify: cdf_upper.vals[i] >= cdf_lower.vals[i] for all i
```

## Running Tests

```bash
# Run all tests
python tests_enhanced_v2.py

# Run specific test class
pytest tests_enhanced_v2.py::TestCDFDomination -v

# Run with verbose output
pytest tests_enhanced_v2.py -v -s
```

## Running Experiments

```bash
# Run full experiment suite
python exp_enhanced_v2.py

# This will:
# 1. Run diagnostics on error accumulation
# 2. Compare methods for Gaussian distributions
# 3. Compare methods for LogNormal distributions
# 4. Generate comparison plots
# 5. Save results to data/ directory
```

## API Changes from V1 to V2

### Function Signatures

**`discretize_continuous_to_pmf`** - Now returns `DiscreteDist`:
```python
# V1
x, pmf, p_neg_inf, p_pos_inf = discretize_continuous_to_pmf(...)

# V2
dist = discretize_continuous_to_pmf(...)  # Returns DiscreteDist directly
```

**`build_grid_from_support_bounds`** - New parameters:
```python
# V2 adds:
grid_strategy: GridStrategy = GridStrategy.FIXED_POINTS
target_bin_width: Optional[float] = None
```

**`self_convolve_pmf`** - New parameters:
```python
# V2 adds:
grid_strategy: GridStrategy = GridStrategy.FIXED_POINTS
target_bin_width: Optional[float] = None
```

**New conversion functions:**
```python
pmf_to_cdf(dist: DiscreteDist) -> DiscreteDist
cdf_to_pmf(dist: DiscreteDist) -> DiscreteDist
ccdf_to_pmf(dist: DiscreteDist) -> DiscreteDist
```

### Backward Compatibility

V2 is **mostly backward compatible** with V1, with these exceptions:

1. ❌ `discretize_continuous_to_pmf` returns `DiscreteDist` instead of tuple
2. ✅ All other functions maintain same interface (new parameters are optional)

**Migration guide:**
```python
# Old V1 code
x, pmf, pneg, ppos = discretize_continuous_to_pmf(dist, n_bins, beta, mode, spacing)
base = DiscreteDist(x=x, kind=DistKind.PMF, vals=pmf, p_neg_inf=pneg, p_pos_inf=ppos)

# New V2 code (simpler!)
base = discretize_continuous_to_pmf(dist, n_bins, beta, mode, spacing)
```

## Performance Characteristics

### FIXED_POINTS Strategy
- **Time:** O(n² log T) for T convolutions
- **Space:** O(n) - constant grid size
- **Best for:** Small to medium T (< 100), memory-constrained scenarios

### FIXED_WIDTH Strategy
- **Time:** O(n²T log T) - grows with T due to larger grids
- **Space:** O(nT) - grid size proportional to support width
- **Best for:** Large T (> 100), accuracy-critical scenarios

### Numerical Precision Options

**Standard:**
- Fast but accumulates floating-point errors
- Error: O(n²ε) per convolution

**Kahan:**
- ~10-20% slower
- Error: O(ε) per convolution (much better!)
- Recommended for T > 10

**Sorted summation:**
- ~20-30% slower for mass checks
- Best accuracy for verification
- Recommended for mass conservation checks

## Common Patterns

### Pattern 1: Rigorous Bounds

```python
# Create tight bounds around true distribution
upper = discretize_continuous_to_pmf(dist, 5000, 1e-15, Mode.DOMINATES, Spacing.LINEAR)
lower = discretize_continuous_to_pmf(dist, 5000, 1e-15, Mode.IS_DOMINATED, Spacing.LINEAR)

# Convolve with high precision
result_u = self_convolve_pmf(upper, T=100, use_kahan=True, sum_method=SummationMethod.SORTED)
result_l = self_convolve_pmf(lower, T=100, use_kahan=True, sum_method=SummationMethod.SORTED)

# Now result_l.vals ≤ true_pmf ≤ result_u.vals (pointwise)
```

### Pattern 2: Large T with Fixed Width

```python
# For very large T, maintain resolution
base = discretize_continuous_to_pmf(dist, 2000, 1e-12, Mode.DOMINATES, Spacing.LINEAR)

result = self_convolve_pmf(
    base, T=10000,
    grid_strategy=GridStrategy.FIXED_WIDTH,
    use_kahan=True
)

# Grid grows but resolution maintained
```

### Pattern 3: Lognormal with Geometric Spacing

```python
# Lognormal distribution benefits from geometric spacing
dist = stats.lognorm(s=1, scale=1)

upper = discretize_continuous_to_pmf(
    dist, 5000, 1e-12,
    mode=Mode.DOMINATES,
    spacing=Spacing.GEOMETRIC  # Better for log-scale
)

result = self_convolve_pmf(upper, T=100, spacing=Spacing.GEOMETRIC)
```

### Pattern 4: Checking Domination

```python
# Verify CDF domination property
cdf_upper = pmf_to_cdf(result_upper)
cdf_lower = pmf_to_cdf(result_lower)

# Check at every point
for i in range(len(cdf_upper.x)):
    x = cdf_upper.x[i]
    # Find corresponding point in lower bound
    idx_lower = np.searchsorted(cdf_lower.x, x)
    if idx_lower < len(cdf_lower.x):
        assert cdf_upper.vals[i] >= cdf_lower.vals[idx_lower] - 1e-9
```

## Troubleshooting

### Issue: Mass Conservation Error

```
ValueError: MASS CONSERVATION ERROR: Error=1.23e-11
```

**Solution:**
- Increase `n_grid` (more resolution)
- Use `use_kahan=True` (better precision)
- Use `sum_method=SummationMethod.SORTED` (accurate verification)
- Use `GridStrategy.FIXED_WIDTH` for large T

### Issue: Grid Not Strictly Increasing

```
ValueError: x must be strictly increasing
```

**Solution:**
- Check for duplicate points in input
- Increase numerical tolerance in grid generation
- Verify spacing strategy is appropriate (LINEAR vs GEOMETRIC)

### Issue: Invalid Support Bounds

```
ValueError: Invalid support bounds: x_out_min >= x_out_max
```

**Solution:**
- Increase `beta` to trim more tail mass
- Check that distributions have finite support in the trimmed region
- For degenerate cases (all mass at one point), handle separately

### Issue: Geometric Spacing with Negative Values

```
ValueError: Cannot use geometric spacing when range contains negative values
```

**Solution:**
- Use `Spacing.LINEAR` for distributions with negative support
- For positive distributions only, use `Spacing.GEOMETRIC`

## Testing Checklist

When adding new features:

- [ ] Add unit tests to `tests_enhanced_v2.py`
- [ ] Verify mass conservation with `check_mass_conservation`
- [ ] Test CDF domination property if applicable
- [ ] Check edge cases (T=1, small n_grid, extreme beta)
- [ ] Verify backward compatibility
- [ ] Update documentation in `IMPLEMENTATION_DETAILS.md`
- [ ] Add usage example to this README

## References

- See `IMPLEMENTATION_DETAILS.md` for complete mathematical background
- Original implementation in `implementation_enhanced.py` (v1)
- Test suite demonstrates all features and edge cases

## Future Improvements (Potential V3)

Ideas for future versions:

1. **Adaptive grid refinement**: Automatically add points in regions of high probability mass
2. **Parallel convolution**: Use multiprocessing for large convolutions
3. **Sparse representation**: More efficient storage for distributions with localized mass
4. **GPU acceleration**: CUDA/OpenCL for very large problems
5. **Interval arithmetic**: Fully rigorous bounds accounting for all floating-point errors
6. **Non-uniform spacing**: Custom spacing functions beyond linear/geometric

## Contact

For questions, issues, or contributions, please refer to the project repository or contact the maintainers.

---

**Version 2.0** - Enhanced numerical stability, consistent API, and comprehensive testing