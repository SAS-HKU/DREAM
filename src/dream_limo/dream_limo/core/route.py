"""Shared station-anchored route geometry for decision and MPC."""

from __future__ import annotations

import numpy as np


def anchored_lane_change_y(
    x_values,
    *,
    source_y: float,
    target_y: float,
    start_x: float,
    end_x: float,
):
    """Return a smooth lane transition that cannot cut through the truck."""
    if end_x <= start_x:
        raise ValueError("lane-change end must be after its start")
    x = np.asarray(x_values, dtype=np.float64)
    phase = np.clip((x - start_x) / (end_x - start_x), 0.0, 1.0)
    blend = phase * phase * (3.0 - 2.0 * phase)
    return float(source_y) + blend * (float(target_y) - float(source_y))
