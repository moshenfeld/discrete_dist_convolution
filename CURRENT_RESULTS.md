# Current Valid Results

**Date**: October 4, 2025

## ✅ **Working Implementation: PMF×PMF Convolution**

### What's Implemented

The **PMF×PMF convolution kernel** is fully implemented and validated. This computes the discrete convolution of two probability mass functions on fixed grids with proper:

- **Tie-breaking rules** (DOMINATES vs IS_DOMINATED modes)
- **Infinity mass accounting** (±∞ mass handling)
- **Budget conservation** (total probability = 1.0)
- **Numba optimization** (fast JIT-compiled code)

### Demonstration Results

Run the demo with:
```bash
python experiments/demo_pmf_pmf.py
```

#### Example 1: Gaussian + Gaussian
- **Input**: Two N(0,1) distributions on 50-point grids
- **Output**: Properly convolved distribution (approximates N(0,√2))
- **Mass conservation**: 1.0000 (perfect)
- **Plot**: `plots/demo_gaussian_gaussian.png`

Key observations:
- DOMINATES mode: Rounds exact hits upward
- IS_DOMINATED mode: Rounds exact hits downward
- Result is wider than inputs (correct for sum of independent variables)

#### Example 2: Uniform + Uniform
- **Input**: Two uniform distributions
- **Output**: Triangular-shaped distribution (theoretically correct)
- **Mass conservation**: 1.0000 (perfect)
- **Plot**: `plots/demo_uniform_uniform.png`

#### Example 3: Gaussian + Uniform
- **Input**: Gaussian and uniform on different grids
- **Output**: Smoothed distribution
- **Mass conservation**: 1.0000 (perfect)
- **Plot**: `plots/demo_gaussian_uniform.png`

### Test Results

All tests for PMF×PMF functionality pass:
```bash
pytest tests/test_pmf_pmf_kernel.py -v
```

Results:
- ✅ test_pmf_pmf_simple_no_infinity
- ✅ test_pmf_pmf_edge_routing
- ✅ test_pmf_pmf_with_infinity_masses
- ✅ test_pmf_pmf_budget_conservation

### API Usage

```python
from discrete_conv_api import DiscreteDist, convolve_pmf_pmf_to_pmf
import numpy as np

# Create two distributions
X = DiscreteDist(
    x=np.array([0.0, 1.0, 2.0]),
    vals=np.array([0.5, 0.3, 0.2]),
    kind="pmf",
    p_neg_inf=0.0,
    p_pos_inf=0.0
)

Y = DiscreteDist(
    x=np.array([0.0, 1.0]),
    vals=np.array([0.6, 0.4]),
    kind="pmf",
    p_neg_inf=0.0,
    p_pos_inf=0.0
)

# Convolve (will auto-generate output grid)
Z = convolve_pmf_pmf_to_pmf(X, Y, mode='DOMINATES')

print(f"Output grid: {Z.x}")
print(f"Output PMF: {Z.vals}")
print(f"Total mass: {Z.vals.sum() + Z.p_neg_inf + Z.p_pos_inf}")
```

---

## ⚠️ **Not Implemented: CDF/CCDF Kernels**

The following functions are **stub implementations only** and return invalid results:

### ❌ convolve_pmf_cdf_to_cdf_core
- Returns zeros for CDF
- Cannot compute CDF envelopes
- **Do not use** - results are meaningless

### ❌ convolve_pmf_ccdf_to_ccdf_core
- Returns zeros for CCDF
- Cannot compute CCDF envelopes
- **Do not use** - results are meaningless

### ❌ self_convolve_envelope_core
- Returns trivial bounds (0 or 1 everywhere)
- Cannot perform iterated convolutions
- **Do not use** - results are meaningless

### Why Previous Experiments Failed

The experiment script `experiments/run_comparisons.py` was calling these unimplemented functions, which is why all the plots showed:
- CDF lower = 0 everywhere (all mass at +∞)
- CCDF upper = 1 everywhere (all mass at +∞)
- Gaps showing invalid bounds

These plots have been **removed** as they represent failures, not results.

---

## 📊 **Summary**

### What You Can Do Now:
✅ Convolve two PMF distributions  
✅ Handle infinity masses  
✅ Use both DOMINATES and IS_DOMINATED modes  
✅ Verify budget conservation  
✅ Run demonstrations and tests  

### What You Cannot Do Yet:
❌ Compute CDF envelopes  
❌ Compute CCDF envelopes  
❌ Perform self-convolution (X+X+...+X)  
❌ Generate bounds for iterated sums  

### To Get Full Functionality:
Implement the missing kernels following the specifications in:
- `docs/IMPLEMENTATION_PLAN.md` - Step-by-step implementation guide
- `docs/IMPLEMENTATION_GUIDE_NUMBA.md` - Detailed Numba specifications
- `docs/DERIVATION.tex` - Mathematical foundations

---

**For questions or to report issues**: See `STATUS.md` for detailed technical information.

