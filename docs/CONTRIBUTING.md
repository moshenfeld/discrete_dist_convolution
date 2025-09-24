# Contributing

- Use Python 3.10+.
- Keep arrays `float64`, contiguous; enforce strict-increasing grids.
- Follow tie-breaking, feasible clipping, and ±∞ mass accounting exactly as documented in `IMPLEMENTATION_GUIDE_NUMBA.md`.
- Run `pytest -q` locally before pushing.
- Larger features should include tests and doc updates.
