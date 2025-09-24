
# Numerical Convolution Package — Numba Implementation Guide

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

## Grid selection strategies (default = **trim‑log**)

When producing anchors `t` for `Z = X ⊕ Y`:

- **trim‑log (default)**: drop total tail `β` via `q = sqrt(β/2)` on each input CDF. Set
  `z_min = x_lo + y_lo`, `z_max = x_hi + y_hi`, build a **log‑spaced** grid on `[z_min, z_max]`
  with `z_size` points. Assumes **positive support**; if violated/degenerate, fallback to
  `range-linear` (default) or `minkowski`.
  - `implementation.grids.build_grid_trim_log_from_dists(X, Y, beta=1e-6, z_size=None)`
  - `..._from_cdfs(xX, FX, xY, FY, ...)`

- **minkowski**: exact `{x_i + y_j}`; optionally decimate.
- **range‑linear**: `linspace` over `[xX[0]+xY[0], xX[-1]+xY[-1]]` (or `[T*min(x), T*max(x)]`).

**API defaults (`t=None`)**:
- Pairwise: uses **trim‑log** (from `DiscreteDist` inputs).
- Self‑convolution: uses **range‑linear** with a heuristic size.

---

## Numba Implementation Details

- Prefer `@njit(cache=True)`; use tiled loops + compensated sums.
- `np.searchsorted` is OK in recent Numba; otherwise wrap.
- Operate on contiguous `float64` arrays.
