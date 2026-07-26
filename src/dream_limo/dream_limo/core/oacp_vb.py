"""Compatibility imports for the relocated OACP-VB numerical core.

The canonical implementation is :mod:`dream_limo.OACP.oacp_vb`. New code
should import from that module. This shim preserves existing user scripts and
recorded reproduction commands.
"""

from dream_limo.OACP.oacp_vb import *  # noqa: F401,F403
from dream_limo.OACP.oacp_vb import __all__ as __all__  # noqa: F401
