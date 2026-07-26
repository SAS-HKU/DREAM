"""Compatibility wrapper for the relocated OACP-VB ROS assessor.

The canonical implementation is :mod:`dream_limo.OACP.assessor_node`.
"""

from dream_limo.OACP.assessor_node import *  # noqa: F401,F403
from dream_limo.OACP.assessor_node import main


if __name__ == "__main__":
    main()
