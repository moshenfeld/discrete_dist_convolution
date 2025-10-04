# Convolution Bounds

Discrete convolution bounds on fixed grids: PMF rounding and Stieltjes envelopes (CDF/CCDF) with Numba.

## Quickstart

```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -U pip

# Install library (editable) + dev tools
pip install -e ".[dev,viz,test]"

# Run tests
pytest -q

# Optional: run comparison experiments (figures saved under ./plots)
python experiments/run_comparisons.py
```

## Implementor's Map

- **API entry points**: `discrete_conv_api.py`
  - Pairwise: `convolve_pmf_pmf_to_pmf`, `convolve_pmf_cdf_to_cdf`, `convolve_pmf_ccdf_to_ccdf`
  - Self conv: `self_convolve_envelope`
  - All accept `t=None` and will choose an output grid automatically.
- **Inner kernels** (to implement in Numba): `implementation/kernels.py`
- **Envelope/self-conv loop** pseudocode: `implementation/selfconv.py`
- **Step evaluators & utils** (implemented): `implementation/steps.py`, `implementation/utils.py`
- **Grid builders** (default **trim-log**): `implementation/grids.py`
- **Infinity mass ledger**: `implementation/ledger.py`
- **Visualization**: `viz/compare_bounds.py`
- **Experiments runner**: `experiments/run_comparisons.py`
- **Docs**: `IMPLEMENTATION_GUIDE_NUMBA.md` (implementation details), `DERIVATION.tex` (math notes)

## Tests

- Unit tests under `tests/` cover helpers and conversion roundtrips and assert contracts for kernels.
- Some tests are marked `xfail` until the heavy kernels are implemented.

## Coding style

- Python: Ruff + MyPy settings in `pyproject.toml` (py310).
- Prefer `float64` contiguous arrays. See `DiscreteDist.__post_init__` checks.
