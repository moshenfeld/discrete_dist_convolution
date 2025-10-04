# Results Summary (5,000-bin standard)

**Date**: October 4, 2025  
**Standard Resolution**: 5,000 bins  
**Implementation Status**: PMF kernels complete

---

## ✅ What's Working

### 1. PMF×PMF Pairwise Convolution
**Function**: `convolve_pmf_pmf_to_pmf(X, Y, mode='DOMINATES')`

Computes the distribution of X + Y where X and Y are independent random variables.

**Performance (5k × 5k bins)**:
- Time: ~1.2 seconds
- Throughput: 21 M operations/sec
- Mass conservation: Perfect (1.0000000000)

**Features**:
- Two tie-breaking modes: DOMINATES and IS_DOMINATED
- Handles infinity masses at ±∞
- Numba JIT-compiled for speed

### 2. PMF Self-Convolution
**Function**: `self_convolve_pmf(base, T, mode='DOMINATES')`

Computes the distribution of X + X + ... + X (T times) using exponentiation-by-squaring.

**Performance (5k bins)**:
| T | Time | Convolutions | Efficiency |
|---|------|--------------|------------|
| 10 | 4.0s | 5 | 44% fewer operations |
| 100 | 7.2s | 9 | 91% fewer operations |
| 1000 | 12.0s | 15 | 98.5% fewer operations |

**Algorithm**: Binary exponentiation (O(log T) instead of O(T))

---

## 📊 Current Results

### Generated Plots (in `plots/`)

1. **`pairwise_convolution_5k.png`**
   - Shows N(0,1) + N(1,0.8)
   - Demonstrates proper convolution
   - Perfect mass conservation

2. **`self_convolution_5k.png`**
   - Shows base N(0,1) and results for T=10, 100, 1000
   - Distributions properly widen as √T
   - All match theoretical expectations

---

## 🔧 Usage

### Quick Start
```bash
# Run standard 5k-bin demonstrations
python experiments/demo_5k_bins.py
```

### API Examples

**Pairwise Convolution**:
```python
from discrete_conv_api import DiscreteDist, convolve_pmf_pmf_to_pmf
import numpy as np

# Create distributions (5000 bins recommended)
x = np.linspace(-5, 5, 5000)
pmf1 = ... # your PMF values
X = DiscreteDist(x=x, kind='pmf', vals=pmf1, p_neg_inf=0.0, p_pos_inf=0.0)

y = np.linspace(-3, 7, 5000)
pmf2 = ... # your PMF values
Y = DiscreteDist(x=y, kind='pmf', vals=pmf2, p_neg_inf=0.0, p_pos_inf=0.0)

# Compute X + Y
Z = convolve_pmf_pmf_to_pmf(X, Y, mode='DOMINATES')
```

**Self-Convolution**:
```python
from discrete_conv_api import self_convolve_pmf

# Compute X + X + ... + X (1000 times)
# Takes ~12 seconds with 5000 bins
Z = self_convolve_pmf(base, T=1000, mode='DOMINATES')
```

---

## 🎯 Performance Guidelines

### Recommended Grid Sizes

| Bins | Pairwise | T=10 | T=100 | T=1000 | Use Case |
|------|----------|------|-------|--------|----------|
| 1,000 | 0.02s | 0.1s | 0.2s | 0.4s | Quick prototyping |
| **5,000** | **1.2s** | **4.0s** | **7.2s** | **12.0s** | **Production (recommended)** |
| 10,000 | 6.5s | 18.2s | 33s | 55s | Ultra-high precision |

**Recommendation**: Use 5,000 bins for the best balance of accuracy and speed.

### Computational Complexity

**Pairwise**: O(m × n) where m, n are grid sizes  
**Self-convolution**: O(log T × n²) where T is number of iterations, n is grid size

---

## ⚠️ Not Implemented

The following features are **stub implementations only** and return invalid results:

❌ `convolve_pmf_cdf_to_cdf` - CDF envelope convolution  
❌ `convolve_pmf_ccdf_to_ccdf` - CCDF envelope convolution  
❌ `self_convolve_envelope` - Self-convolution with envelope backend  

These require implementation of Stieltjes integration kernels (see `docs/IMPLEMENTATION_PLAN.md`).

---

## 📁 Project Structure

```
discrete_dist_convolution/
├── implementation/
│   ├── kernels.py          # PMF×PMF kernel (✅ complete)
│   ├── selfconv.py         # Self-convolution (✅ complete)
│   ├── grids.py            # Grid builders
│   ├── ledger.py           # Infinity mass accounting
│   ├── steps.py            # Step evaluators
│   └── utils.py            # Utilities
├── experiments/
│   ├── demo_5k_bins.py     # Main demonstration (5k bins)
│   ├── test_self_conv_progressive.py  # Performance testing
│   └── visualize_self_conv_results.py # Visualization tools
├── plots/
│   ├── pairwise_convolution_5k.png
│   └── self_convolution_5k.png
├── tests/                  # 22 passing + 4 xpass
└── docs/                   # Implementation guides

```

---

## ✅ Validation

All implementations validated with:
- ✅ Perfect mass conservation (10 decimal places)
- ✅ Proper distribution shapes
- ✅ Expected theoretical properties (e.g., N(0,1) ⊕ N(0,1) ≈ N(0,√2))
- ✅ Binary exponentiation efficiency verified
- ✅ 22 passing unit tests

---

## 🚀 Next Steps

To complete the system:
1. Implement PMF×CDF Stieltjes kernel (~500 lines, 2-3 days)
2. Implement PMF×CCDF Stieltjes kernel (similar structure)
3. Integrate with self-convolution envelope backend

See `docs/IMPLEMENTATION_PLAN.md` for detailed specifications.

