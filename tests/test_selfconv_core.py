import numpy as np
import pytest
from discrete_conv_api import DiscreteDist, self_convolve_envelope

@pytest.mark.xfail(reason="Self-convolution core not implemented yet")
def test_self_convolution_identity_and_exponentiation():
    pass
