
# Numerical Convolution Package — Numba Implementation Guide

**Note**: This guide describes the low-level implementation details. For API usage, see `../USAGE_NEW_GRID.md` and `../README.md`.

**Current Status (Oct 2025)**: PMF×PMF kernel and self-convolution fully implemented with automatic grid generation. CDF/CCDF Stieltjes kernels are stub implementations.

---

This guide specifies *exactly* how to implement the kernels and dispatchers in Numba,
capturing all envelope semantics, tie-breaking, ±∞ mass accounting, and grid selection.

---

## Notation

- Distributions live on a strictly increasing grid `x = [x_0 < ... < x_{K-1}]`.
- Representation: **PMF**, **CDF**, or **CCDF**, plus masses at `−∞` / `+∞`.
- Finite mass: `m_fin = sum(pmf)`.

We use “DOMINATES” for the CDF **lower** envelope and “IS_DOMINATED” for the CDF **upper** envelope.

---

## ±∞ Mass Accounting (Ledger)

Given X and Y, with `mX = sum(X.pmf)`, `mY = sum(Y.pmf)`:

- To **−∞**: `X.p_neg_inf*(mY + Y.p_neg_inf) + Y.p_neg_inf*mX`
- To **+∞**: `X.p_pos_inf*(mY + Y.p_pos_inf) + Y.p_pos_inf*mX`
- Ambiguous cross: `m_cross = X.p_neg_inf*Y.p_pos_inf + X.p_pos_inf*Y.p_neg_inf`
  - **DOMINATES**: route `m_cross` → **+∞**
  - **IS_DOMINATED**: route `m_cross` → **−∞**

This is applied **in addition** to the finite–finite kernel.

---

## Tie-breaking / Cell Conventions

### PMF × PMF (rounding)
- **DOMINATES**: `idx = searchsorted(t, z, 'right')` (exact hits up). If `z == t[-1]` ⇒ `+∞`.
- **IS_DOMINATED**: `idx = searchsorted(t, z, 'left') - 1` (exact hits down). If `z == t[0]` ⇒ `−∞`.

### Stieltjes (PMF×CDF/CCDF)
- **CDF** lower = **right** at `t_k`, upper = **left** at `t_{k+1}`.
- **CCDF** mirrors CDF.

### Boundary clamps
- CDF: `F(q<x0)=p_neg_inf`, `F(q≥x_last)=1−p_pos_inf`.
- CCDF: `S(q<x0)=1−p_neg_inf`, `S(q≥x_last)=p_pos_inf`.

---

## Feasible Clipping Intervals

- **CDF**: clip to `[ p_neg_X , p_neg_X + mX * (1 - p_pos_Y) ]`.
- **CCDF**: clip to `[ p_pos_X , 1 ]`.

---

## Monotone Projection

- **CDF**: in-place `maximum.accumulate` (running max)
- **CCDF**: reverse `minimum.accumulate` (running min)

Optional reconciliation when both lower & upper exist:
- CDF: `lower = min(lower, upper)`
- CCDF: `upper = max(upper, lower)`

---

## Self-Convolution Strategy (Exponentiation)

Maintain `(cur_pmf, acc_env)`:
- Update `cur_pmf` via **PMF×PMF** in the same mode.
- Update `acc_env` via **PMF×CDF/CCDF**.
- Identity δ₀ sits at `i0 = max{ j | t[j] ≤ 0 }` (or `i0=0` if none).

---

## Initial Grid Generation from Continuous Distribution

To discretize a continuous distribution onto a grid for convolution:

**Input parameters:**
- `n_grid`: Number of grid points
- `beta`: Tail probability to trim (e.g., 1e-6)
- `dist`: Continuous distribution with CDF/quantile methods
- `dir`: Bound direction ('upper' for DOMINATES, 'lower' for IS_DOMINATED)

**Algorithm:**

1. **Determine range via quantiles:**
   - `q_min = dist.quantile(beta/2)`
   - `q_max = dist.quantile(1 - beta/2)`

2. **Create geometric spacing:**
   - If support is positive (q_min > 0): `grid = geomspace(q_min, q_max, n_grid)`
   - If support contains 0: Split into negative and positive regions with geometric spacing
   - If support is negative: `grid = -geomspace(-q_max, -q_min, n_grid)[::-1]`

3. **Discretize to PMF using CDF/CCDF:**
   - For **upper bound** (DOMINATES):
     - Compute `pmf[i] = CDF(grid[i+1]) - CDF(grid[i])` for interior points
     - Set `p_neg_inf = CDF(grid[0])`
     - Set `p_pos_inf = 1 - CDF(grid[-1])`
   
   - For **lower bound** (IS_DOMINATED):
     - Compute `pmf[i] = CCDF(grid[i]) - CCDF(grid[i+1])` for interior points
     - Set `p_neg_inf = 0` (mass at -∞ cannot increase in lower bound)
     - Set `p_pos_inf = CCDF(grid[-1])`

4. **Budget correction:** Ensure `sum(pmf) + p_neg_inf + p_pos_inf = 1.0`

## Grid selection strategies

### Current Implementation (Oct 2025): **Support Bounds**

When producing anchors `t` for `Z = X ⊕ Y`, the **default strategy** uses support bounds:

- **support-bounds (default)**: 
  - Output size: `z_size = max(len(xX), len(xY))`
  - Bounds: `z_min = xX[0] + xY[0]`, `z_max = xX[-1] + xY[-1]`
  - Spacing: User-specified via `Spacing` enum
    - `Spacing.LINEAR`: `np.linspace(z_min, z_max, z_size)`
    - `Spacing.GEOMETRIC`: `np.geomspace(z_min, z_max, z_size)` (requires z_min, z_max same sign)
  - Implementation: `implementation.grids.build_grid_from_support_bounds(xX, xY, spacing=Spacing.LINEAR)`
  - Grid generation happens **inside** convolution kernels automatically

**Key properties:**
- Predictable output size (matches max input size)
- Full support coverage (no probability mass outside grid except at ±∞)
- Consistent spacing type across operations
- No tail trimming needed (handled at discretization stage)

### Legacy Strategies (Available but Not Default)

- **trim‑log**: drop total tail `β` via `q = sqrt(β/2)` on each input CDF. Set
  `z_min = x_lo + y_lo`, `z_max = x_hi + y_hi`, build a **log‑spaced** grid on `[z_min, z_max]`
  - Still available via `build_grid_trim_log_from_dists(X, Y, beta=1e-6, z_size=None)`
  - Assumes **positive support**; fallback to range-linear or minkowski if violated

- **minkowski**: exact `{x_i + y_j}`; optionally decimate
- **range‑linear**: equivalent to support-bounds with linear spacing

**Current API:**
- All convolution functions use **support-bounds** with user-specified spacing
- Functions return `DiscreteDist` objects directly (not tuples)
- Grid generation is automatic and internal to kernels

---

## Numba Implementation Details

- Prefer `@njit(cache=True)`; use tiled loops + compensated sums.
- `np.searchsorted` is OK in recent Numba; otherwise wrap.
- Operate on contiguous `float64` arrays.
