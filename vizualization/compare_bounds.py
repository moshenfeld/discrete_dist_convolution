
from typing import Dict, Tuple, Optional, Literal
import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from discrete_conv_api import DiscreteDist
from implementation.utils import union_grid
from implementation.steps import step_cdf_right, step_cdf_left, step_ccdf_right, step_ccdf_left

def _sample_cdf_on_grid(dist: DiscreteDist, grid: np.ndarray, side: Literal["left","right"] = "right") -> np.ndarray:
    if dist.kind != "cdf":
        raise ValueError("Expected a CDF DiscreteDist")
    F = np.empty_like(grid, dtype=np.float64)
    step = step_cdf_right if side == "right" else step_cdf_left
    for k, q in enumerate(grid):
        F[k] = step(dist.x, dist.vals, dist.p_neg_inf, dist.p_pos_inf, float(q))
    return F

def _sample_ccdf_on_grid(dist: DiscreteDist, grid: np.ndarray, side: Literal["left","right"] = "right") -> np.ndarray:
    if dist.kind != "ccdf":
        raise ValueError("Expected a CCDF DiscreteDist")
    S = np.empty_like(grid, dtype=np.float64)
    step = step_ccdf_right if side == "right" else step_ccdf_left
    for k, q in enumerate(grid):
        S[k] = step(dist.x, dist.vals, dist.p_neg_inf, dist.p_pos_inf, float(q))
    return S

def _align_maps_to_union_grid(dists: Dict[str, DiscreteDist], rep: Literal["cdf","ccdf"], side: Literal["left","right"] = "right"):
    if not dists:
        return np.empty((0,), dtype=np.float64), {}
    ug = union_grid([d.x for d in dists.values()])
    out = {}
    for name, d in dists.items():
        out[name] = _sample_cdf_on_grid(d, ug, side) if rep == "cdf" else _sample_ccdf_on_grid(d, ug, side)
    return ug, out

def _pairwise_max_gap(values: Dict[str, np.ndarray]) -> Dict[Tuple[str,str], float]:
    names = list(values.keys())
    gaps = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = values[names[i]]; b = values[names[j]]
            gaps[(names[i], names[j])] = float(np.max(np.abs(a - b))) if a.size and b.size else float("nan")
    return gaps

def _gap_metrics(upper: np.ndarray, lower: np.ndarray, x: np.ndarray):
    g = upper - lower
    linf = float(np.max(g)) if g.size else float("nan")
    l1 = float(np.trapz(g, x)) if g.size >= 2 else float("nan")
    return linf, l1

def plot_cdf_lowers(lower_map: Dict[str, DiscreteDist], *, side: str = "right"):
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    x, vals = _align_maps_to_union_grid(lower_map, rep="cdf", side=side)  # type: ignore
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for name, v in vals.items():
        ax.plot(x, v, label=name)
    ax.set_title("CDF Lower Bounds")
    ax.set_xlabel("x"); ax.set_ylabel("F_lower(x)"); ax.legend(); ax.grid(True, alpha=0.3)
    ax._pairwise_linf = _pairwise_max_gap(vals)
    return fig, ax, x, vals

def plot_cdf_uppers(upper_map: Dict[str, DiscreteDist], *, side: str = "left"):
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    x, vals = _align_maps_to_union_grid(upper_map, rep="cdf", side=side)  # type: ignore
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for name, v in vals.items():
        ax.plot(x, v, label=name)
    ax.set_title("CDF Upper Bounds")
    ax.set_xlabel("x"); ax.set_ylabel("F_upper(x)"); ax.legend(); ax.grid(True, alpha=0.3)
    ax._pairwise_linf = _pairwise_max_gap(vals)
    return fig, ax, x, vals

def plot_cdf_gaps(lower_map: Dict[str, DiscreteDist], upper_map: Dict[str, DiscreteDist], *, side_lower: str = "right", side_upper: str = "left"):
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    all_grids = [d.x for d in lower_map.values()] + [d.x for d in upper_map.values()]
    ug = union_grid(all_grids)
    gaps = {}; metrics = {}
    for name in sorted(set(lower_map.keys()).intersection(upper_map.keys())):
        L = _sample_cdf_on_grid(lower_map[name], ug, side=side_lower)
        U = _sample_cdf_on_grid(upper_map[name], ug, side=side_upper)
        G = U - L; G[G < 0.0] = 0.0
        gaps[name] = G; metrics[name] = _gap_metrics(U, L, ug)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for name, g in gaps.items():
        linf, l1 = metrics[name]
        ax.plot(ug, g, label=f"{name} (L∞={linf:.2e}, L1≈{l1:.2e})")
    ax.set_title("CDF Upper–Lower Gaps")
    ax.set_xlabel("x"); ax.set_ylabel("F_upper(x) − F_lower(x)"); ax.legend(); ax.grid(True, alpha=0.3)
    return fig, ax, ug, gaps, metrics

def plot_ccdf_lowers(lower_map: Dict[str, DiscreteDist], *, side: str = "left"):
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    x, vals = _align_maps_to_union_grid(lower_map, rep="ccdf", side=side)  # type: ignore
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for name, v in vals.items():
        ax.plot(x, v, label=name)
    ax.set_title("CCDF Lower Bounds")
    ax.set_xlabel("x"); ax.set_ylabel("S_lower(x)"); ax.legend(); ax.grid(True, alpha=0.3)
    ax._pairwise_linf = _pairwise_max_gap(vals)
    return fig, ax, x, vals

def plot_ccdf_uppers(upper_map: Dict[str, DiscreteDist], *, side: str = "right"):
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    x, vals = _align_maps_to_union_grid(upper_map, rep="ccdf", side=side)  # type: ignore
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for name, v in vals.items():
        ax.plot(x, v, label=name)
    ax.set_title("CCDF Upper Bounds")
    ax.set_xlabel("x"); ax.set_ylabel("S_upper(x)"); ax.legend(); ax.grid(True, alpha=0.3)
    ax._pairwise_linf = _pairwise_max_gap(vals)
    return fig, ax, x, vals

def plot_ccdf_gaps(lower_map: Dict[str, DiscreteDist], upper_map: Dict[str, DiscreteDist], *, side_lower: str = "left", side_upper: str = "right"):
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    all_grids = [d.x for d in lower_map.values()] + [d.x for d in upper_map.values()]
    ug = union_grid(all_grids)
    gaps = {}; metrics = {}
    for name in sorted(set(lower_map.keys()).intersection(upper_map.keys())):
        L = _sample_ccdf_on_grid(lower_map[name], ug, side=side_lower)
        U = _sample_ccdf_on_grid(upper_map[name], ug, side=side_upper)
        G = U - L; G[G < 0.0] = 0.0
        gaps[name] = G; metrics[name] = _gap_metrics(U, L, ug)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for name, g in gaps.items():
        linf, l1 = metrics[name]
        ax.plot(ug, g, label=f"{name} (L∞={linf:.2e}, L1≈{l1:.2e})")
    ax.set_title("CCDF Upper–Lower Gaps")
    ax.set_xlabel("x"); ax.set_ylabel("S_upper(x) − S_lower(x)"); ax.legend(); ax.grid(True, alpha=0.3)
    return fig, ax, ug, gaps, metrics
