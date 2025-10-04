import numpy as np
import pytest
from discrete_conv_api import DiscreteDist, DistKind, convolve_pmf_cdf_to_cdf, convolve_pmf_ccdf_to_ccdf

@pytest.mark.xfail(reason="Envelope kernels not implemented yet")
def test_cdf_envelope_monotone_and_feasible():
    # Build X as PMF, Y as CDF, test monotonicity and feasible interval once implemented
    pass

@pytest.mark.xfail(reason="Envelope kernels not implemented yet")
def test_ccdf_envelope_monotone_and_feasible():
    pass
