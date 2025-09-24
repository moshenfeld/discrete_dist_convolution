"""
Example: Trim-log grid from DiscreteDist objects.
"""
import numpy as np
from discrete_conv_api import DiscreteDist
from implementation.grids import build_grid_trim_log_from_dists

# X: PMF on positive grid
x = np.linspace(0.1, 5.0, 101)
pmfX = np.exp(-0.5*((x-2.0)/0.6)**2); pmfX /= pmfX.sum()
X = DiscreteDist(x=x, kind="pmf", vals=pmfX)

# Y: CDF on positive grid (monotone increasing)
y = np.linspace(0.2, 6.0, 121)
cdfY = np.linspace(0.0, 1.0, 121) * 0.98  # pretend finite mass 0.98
Y = DiscreteDist(x=y, kind="cdf", vals=cdfY, p_neg_inf=0.01, p_pos_inf=0.01)

t = build_grid_trim_log_from_dists(X, Y, beta=1e-4, z_size=200)
print("trim-log grid size:", t.size, "min:", t.min(), "max:", t.max())
