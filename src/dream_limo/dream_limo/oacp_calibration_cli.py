"""Compatibility wrapper for the relocated OACP-VB calibration command.

The canonical implementation is :mod:`dream_limo.OACP.calibration_cli`.
"""

from dream_limo.OACP.calibration_cli import *  # noqa: F401,F403
from dream_limo.OACP.calibration_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
