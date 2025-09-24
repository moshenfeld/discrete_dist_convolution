# Implementation Plan for Convolution Bounds Kernels

## Overview
The project requires implementing Numba-optimized kernels for discrete convolution bounds with proper handling of ±∞ masses, tie-breaking conventions, and envelope semantics.

## Phase 1: Core PMF×PMF Kernel
**File**: `implementation/kernels.py::convolve_pmf_pmf_to_pmf_core`

### Requirements:
- Compute finite-finite convolution: `∑_i ∑_j pX[i] * pY[j] * δ(t_k - (x_i + y_j))`
- Implement tie-breaking for rounding:
  - **DOMINATES**: Use `searchsorted(t, z, 'right')` - exact hits go up
  - **IS_DOMINATED**: Use `searchsorted(t, z, 'left') - 1` - exact hits go down
- Handle edge cases:
  - If `z == t[-1]` in DOMINATES mode → mass to `+∞`
  - If `z == t[0]` in IS_DOMINATED mode → mass to `−∞`
- Add infinity ledger contributions (already implemented in `ledger.py`)

### Implementation Steps:
1. Double loop over `(i,j)` pairs computing `z = xX[i] + xY[j]`
2. Apply searchsorted with proper side parameter
3. Accumulate mass with boundary handling
4. Use compensated summation for numerical stability
5. Apply `@njit(cache=True)` decorator

## Phase 2: PMF×CDF Stieltjes Kernel
**File**: `implementation/kernels.py::convolve_pmf_cdf_to_cdf_core`

### Requirements:
- Compute Stieltjes convolution for CDF envelope
- **DOMINATES** (lower): Use right limit at `t_k`
- **IS_DOMINATED** (upper): Use left limit at `t_{k+1}`
- Apply boundary clamps: `F_Y(q<y_0) = p_neg_Y`, `F_Y(q≥y_{K-1}) = 1 - p_pos_Y`
- Feasible interval: `[p_neg_X, p_neg_X + m_X*(1 - p_pos_Y)]`
- Apply monotone projection via `running_max_inplace`

### Implementation Steps:
1. Initialize output CDF array
2. For each `t_k`, compute: `F_Z(t_k) = ∑_i pX[i] * F_Y(t_k - x_i)`
3. Use appropriate step evaluator (already in `steps.py`)
4. Clip to feasible interval
5. Apply monotone projection

## Phase 3: PMF×CCDF Stieltjes Kernel
**File**: `implementation/kernels.py::convolve_pmf_ccdf_to_ccdf_core`

### Requirements:
- Similar to PMF×CDF but for CCDF
- **DOMINATES** (lower): left limit behavior
- **IS_DOMINATED** (upper): right limit behavior  
- Feasible interval: `[p_pos_X, 1]`
- Apply reverse monotone projection via `running_min_reverse_inplace`

### Implementation Steps:
1. Initialize output CCDF array
2. For each `t_k`, compute: `S_Z(t_k) = ∑_i pX[i] * S_Y(t_k - x_i)`
3. Use appropriate step evaluator
4. Clip to feasible interval
5. Apply reverse monotone projection

## Phase 4: Self-Convolution Core
**File**: `implementation/selfconv.py::self_convolve_envelope_core`

### Requirements:
- Implement exponentiation-by-squaring strategy
- Maintain `(cur_pmf, acc_env)` pair
- Update `cur_pmf` via PMF×PMF kernel
- Update `acc_env` via PMF×CDF or PMF×CCDF kernel
- Handle identity δ₀ placement at `i0 = max{j | t[j] ≤ 0}` (or 0 if none)

### Implementation Steps:
1. Initialize identity PMF on output grid `t`
2. Set `cur_pmf = base_pmf`, `acc = identity`
3. While `T > 0`:
   - If `T` is odd: `acc = convolve(acc, cur_pmf)`
   - `cur_pmf = convolve(cur_pmf, cur_pmf)`
   - `T = T // 2`
4. Convert final PMF to requested envelope type

## Phase 5: Optimization & Testing

### Numba Optimizations:
- Use `@njit(cache=True, parallel=False)` initially
- Consider tiling for cache efficiency
- Use compensated (Kahan) summation for large grids
- Profile with `line_profiler` for bottlenecks

### Testing Strategy:
1. Remove `xfail` markers from kernel tests
2. Add property-based tests:
   - Budget conservation
   - Monotonicity of envelopes
   - Domination property (lower ≤ upper)
3. Stress tests with large grids (K > 10000)
4. Comparison with Monte Carlo for validation

## Phase 6: Final Integration

### Documentation:
- Add docstrings with complexity analysis
- Include numerical stability notes
- Document any deviations from spec

### Performance Targets:
- PMF×PMF: O(mn) where m,n are input sizes
- Stieltjes: O(mn) with step evaluator calls
- Self-convolution: O(log T) iterations of base kernels

## Deliverables Checklist:
- [x] `convolve_pmf_pmf_to_pmf_core` with proper rounding ✅ **COMPLETED**
- [ ] `convolve_pmf_cdf_to_cdf_core` with Stieltjes integration
- [ ] `convolve_pmf_ccdf_to_ccdf_core` with reverse monotonicity
- [ ] `self_convolve_envelope_core` with exp-by-squaring
- [x] All tests passing (remove xfail markers) ✅ **COMPLETED for PMF×PMF**
- [ ] Performance benchmarks documented
- [ ] Numerical stability validated