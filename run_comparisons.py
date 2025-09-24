"""
Run bound comparisons over a variety of settings:
- Number of iterations T
- Starting distributions
- Grid resolutions

This script orchestrates computations (lower/upper envelopes) using the API and
then calls the visualization utilities to produce figures. Designed to be robust
even before all kernels are implemented—methods that raise NotImplementedError
are skipped with a note.
"""

from typing import Dict, Callable, Literal, Tuple, List, Optional
import os
import json
import numpy as np

from discrete_conv_api import DiscreteDist, self_convolve_envelope, cdf_to_pmf
from implementation.utils import union_grid
from viz.compare_bounds import (
    plot_cdf_lowers, plot_cdf_uppers, plot_cdf_gaps,
    plot_ccdf_lowers, plot_ccdf_uppers, plot_ccdf_gaps
)

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ---------------------------------------------------------------------------
# Base distribution generators
# ---------------------------------------------------------------------------

def make_gaussian_pmf(n: int, lo: float, hi: float, mu: float = 0.0, sigma: float = 1.0) -> DiscreteDist:
    x = np.linspace(lo, hi, n, dtype=np.float64)
    # tiny jitter to enforce strict increase
    if n > 1:
        x[1:] += np.linspace(1e-12, 1e-9, n-1)
    z = (x - mu) / sigma
    pdf = np.exp(-0.5 * z * z)
    pdf /= pdf.sum()
    return DiscreteDist(x=x, kind="pmf", vals=pdf, p_neg_inf=0.0, p_pos_inf=0.0, name="gauss")

def make_lognormal_pmf(n: int, lo: float, hi: float, mu: float = 0.0, sigma: float = 1.0) -> DiscreteDist:
    x = np.linspace(max(lo, 1e-9), hi, n, dtype=np.float64)
    if n > 1:
        x[1:] += np.linspace(1e-12, 1e-9, n-1)
    pdf = (1.0 / (x * sigma * np.sqrt(2*np.pi))) * np.exp(- (np.log(x) - mu)**2 / (2*sigma**2))
    pdf /= pdf.sum()
    return DiscreteDist(x=x, kind="pmf", vals=pdf, p_neg_inf=0.0, p_pos_inf=0.0, name="lognorm")

def make_bimodal_pmf(n: int, lo: float, hi: float) -> DiscreteDist:
    x = np.linspace(lo, hi, n, dtype=np.float64)
    if n > 1:
        x[1:] += np.linspace(1e-12, 1e-9, n-1)
    c1 = np.exp(-0.5*((x+1.5)/0.5)**2)
    c2 = np.exp(-0.5*((x-1.5)/0.4)**2)
    pdf = 0.6*c1 + 0.4*c2
    pdf /= pdf.sum()
    return DiscreteDist(x=x, kind="pmf", vals=pdf, p_neg_inf=0.0, p_pos_inf=0.0, name="bimodal")

# ---------------------------------------------------------------------------
# Methods registry
# ---------------------------------------------------------------------------

def method_selfconv_cdf_lower(base_pmf: DiscreteDist, T: int, t: np.ndarray) -> DiscreteDist:
    return self_convolve_envelope(base_pmf, T=T, t=t, mode="DOMINATES", backend="cdf")

def method_selfconv_cdf_upper(base_pmf: DiscreteDist, T: int, t: np.ndarray) -> DiscreteDist:
    return self_convolve_envelope(base_pmf, T=T, t=t, mode="IS_DOMINATED", backend="cdf")

def method_selfconv_ccdf_lower(base_pmf: DiscreteDist, T: int, t: np.ndarray) -> DiscreteDist:
    return self_convolve_envelope(base_pmf, T=T, t=t, mode="DOMINATES", backend="ccdf")

def method_selfconv_ccdf_upper(base_pmf: DiscreteDist, T: int, t: np.ndarray) -> DiscreteDist:
    return self_convolve_envelope(base_pmf, T=T, t=t, mode="IS_DOMINATED", backend="ccdf")

METHODS_CDF_LOWER: Dict[str, Callable[[DiscreteDist,int,np.ndarray], DiscreteDist]] = {
    "pmf×cdf (lower)": method_selfconv_cdf_lower,
}
METHODS_CDF_UPPER: Dict[str, Callable[[DiscreteDist,int,np.ndarray], DiscreteDist]] = {
    "pmf×cdf (upper)": method_selfconv_cdf_upper,
}
METHODS_CCDF_LOWER: Dict[str, Callable[[DiscreteDist,int,np.ndarray], DiscreteDist]] = {
    "pmf×ccdf (lower)": method_selfconv_ccdf_lower,
}
METHODS_CCDF_UPPER: Dict[str, Callable[[DiscreteDist,int,np.ndarray], DiscreteDist]] = {
    "pmf×ccdf (upper)": method_selfconv_ccdf_upper,
}

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_one_setting(
    outdir: str,
    base_gen: Callable[..., DiscreteDist],
    base_kwargs: dict,
    T: int,
    n_out: int,
    x_lo: float,
    x_hi: float,
    do_cdf: bool = True,
    do_ccdf: bool = False
):
    os.makedirs(outdir, exist_ok=True)
    base = base_gen(**base_kwargs)  # PMF
    t = np.linspace(x_lo, x_hi, n_out, dtype=np.float64)
    if n_out > 1:
        t[1:] += np.linspace(1e-12, 1e-9, n_out-1)

    results = {"cdf": {"lower": {}, "upper": {}}, "ccdf": {"lower": {}, "upper": {}}}
    notes = []

    if do_cdf:
        # Lower
        for name, fn in METHODS_CDF_LOWER.items():
            try:
                dist = fn(base, T, t)
                results["cdf"]["lower"][name] = dist
            except NotImplementedError as e:
                notes.append(f"CDF lower method '{name}' not implemented: {e}")
            except Exception as e:
                notes.append(f"CDF lower method '{name}' failed: {e}")
        # Upper
        for name, fn in METHODS_CDF_UPPER.items():
            try:
                dist = fn(base, T, t)
                results["cdf"]["upper"][name] = dist
            except NotImplementedError as e:
                notes.append(f"CDF upper method '{name}' not implemented: {e}")
            except Exception as e:
                notes.append(f"CDF upper method '{name}' failed: {e}")

    if do_ccdf:
        for name, fn in METHODS_CCDF_LOWER.items():
            try:
                dist = fn(base, T, t)
                results["ccdf"]["lower"][name] = dist
            except NotImplementedError as e:
                notes.append(f"CCDF lower method '{name}' not implemented: {e}")
            except Exception as e:
                notes.append(f"CCDF lower method '{name}' failed: {e}")
        for name, fn in METHODS_CCDF_UPPER.items():
            try:
                dist = fn(base, T, t)
                results["ccdf"]["upper"][name] = dist
            except NotImplementedError as e:
                notes.append(f"CCDF upper method '{name}' not implemented: {e}")
            except Exception as e:
                notes.append(f"CCDF upper method '{name}' failed: {e}")

    # Save plots
    if plt is not None:
        if do_cdf:
            if results["cdf"]["lower"]:
                fig, ax, x, vals = plot_cdf_lowers(results["cdf"]["lower"])
                fig.savefig(os.path.join(outdir, f"cdf_lowers_T{T}_n{n_out}.png"), dpi=160, bbox_inches="tight")
                plt.close(fig)
            if results["cdf"]["upper"]:
                fig, ax, x, vals = plot_cdf_uppers(results["cdf"]["upper"])
                fig.savefig(os.path.join(outdir, f"cdf_uppers_T{T}_n{n_out}.png"), dpi=160, bbox_inches="tight")
                plt.close(fig)
            if results["cdf"]["lower"] and results["cdf"]["upper"]:
                fig, ax, x, gaps, metrics = plot_cdf_gaps(results["cdf"]["lower"], results["cdf"]["upper"])
                fig.savefig(os.path.join(outdir, f"cdf_gaps_T{T}_n{n_out}.png"), dpi=160, bbox_inches="tight")
                plt.close(fig)
        if do_ccdf:
            if results["ccdf"]["lower"]:
                fig, ax, x, vals = plot_ccdf_lowers(results["ccdf"]["lower"])
                fig.savefig(os.path.join(outdir, f"ccdf_lowers_T{T}_n{n_out}.png"), dpi=160, bbox_inches="tight")
                plt.close(fig)
            if results["ccdf"]["upper"]:
                fig, ax, x, vals = plot_ccdf_uppers(results["ccdf"]["upper"])
                fig.savefig(os.path.join(outdir, f"ccdf_uppers_T{T}_n{n_out}.png"), dpi=160, bbox_inches="tight")
                plt.close(fig)
            if results["ccdf"]["lower"] and results["ccdf"]["upper"]:
                fig, ax, x, gaps, metrics = plot_ccdf_gaps(results["ccdf"]["lower"], results["ccdf"]["upper"])
                fig.savefig(os.path.join(outdir, f"ccdf_gaps_T{T}_n{n_out}.png"), dpi=160, bbox_inches="tight")
                plt.close(fig)

    # Persist lightweight report (method presence + notes)
    report = {
        "T": T,
        "n_out": n_out,
        "base": base.name,
        "notes": notes,
        "cdf_methods_lower": list(results["cdf"]["lower"].keys()),
        "cdf_methods_upper": list(results["cdf"]["upper"].keys()),
        "ccdf_methods_lower": list(results["ccdf"]["lower"].keys()),
        "ccdf_methods_upper": list(results["ccdf"]["upper"].keys()),
    }
    with open(os.path.join(outdir, f"report_T{T}_n{n_out}.json"), "w") as f:
        json.dump(report, f, indent=2)

    return results, notes

def run_all(out_root: str):
    os.makedirs(out_root, exist_ok=True)
    settings = [
        # (generator, kwargs, T, n_out, lo, hi)
        (make_gaussian_pmf, {"n": 129, "lo": -6.0, "hi": 6.0}, 1, 257, -8.0, 8.0),
        (make_gaussian_pmf, {"n": 129, "lo": -6.0, "hi": 6.0}, 4, 257, -8.0, 8.0),
        (make_bimodal_pmf,  {"n": 161, "lo": -4.0, "hi": 4.0}, 8, 257, -8.0, 8.0),
        (make_lognormal_pmf,{"n": 161, "lo": 1e-6, "hi": 8.0, "mu": 0.0, "sigma": 0.6}, 4, 257, 0.0, 12.0),
    ]
    for (gen, kwargs, T, n_out, lo, hi) in settings:
        sub = os.path.join(out_root, f"{gen.__name__}_T{T}_n{n_out}")
        run_one_setting(sub, gen, kwargs, T, n_out, lo, hi, do_cdf=True, do_ccdf=True)

if __name__ == "__main__":
    outdir = os.environ.get("BOUNDS_RUN_OUTDIR", "./_bounds_runs")
    run_all(outdir)
