"""Reproducible OACP-VB comparison-arm implementation.

OACP-VB is the disclosed velocity-bound adaptation documented in this
subpackage's README. It is not the paper's full Bézier/consensus-ADMM planner.
"""

from .oacp_vb import *  # noqa: F401,F403
from .oacp_vb import __all__ as __all__  # noqa: F401
