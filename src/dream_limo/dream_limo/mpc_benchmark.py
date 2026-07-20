"""Reproducible, motion-free MPC deadline check for the onboard computer.

This module imports only the ROS-independent DREAM core.  It never creates a
ROS node or publisher, so running it cannot reach ``/cmd_vel``.  The scenarios
cover standstill, an occluded static truck, and a newly revealed merger for
both the balanced DREAM arm and the matched pure-MPC arm.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import casadi
import cvxpy as cp
import numpy as np
import osqp
import scipy

from .core.mpc import RiskAwareMPC
from .core.risk_field import DREAMRiskField
from .core.types import EgoState, Vehicle
from .limo_scale import default_deployment_config, get_preset


Scenario = Tuple[EgoState, int, Sequence[Vehicle], float]


def _scenarios() -> Dict[str, Scenario]:
    return {
        "standstill_clear": (
            EgoState(0.35, 0.45, 0.0, 0.0, lane_index=0),
            0,
            (),
            0.0,
        ),
        "occluded_truck_lane_keep": (
            EgoState(1.20, 0.45, 0.0, 0.15, lane_index=0),
            0,
            (
                Vehicle(
                    "truck",
                    2.40,
                    0.0,
                    heading=0.0,
                    vehicle_class="truck",
                    length=1.20,
                    width=0.24,
                ),
            ),
            3.0,
        ),
        "revealed_merger_merge": (
            EgoState(2.80, 0.45, 0.0, 0.15, lane_index=0),
            1,
            (
                Vehicle(
                    "merger",
                    4.20,
                    -0.10,
                    vx=0.18,
                    vy=0.05,
                    length=0.22,
                    width=0.22,
                ),
            ),
            2.0,
        ),
    }


def run_benchmark(iterations: int = 20, target_speed: float = 0.15) -> dict:
    """Run all hardware-style MPC cases and return a JSON-safe report."""

    if iterations < 2:
        raise ValueError(
            "iterations must be at least 2 (one cold and one warm solve)"
        )

    config = default_deployment_config()
    config = replace(
        config,
        mpc=replace(config.mpc, target_speed=target_speed),
    )
    report = {
        "platform": {
            "system": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "python": sys.version.split()[0],
        },
        "dependencies": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "cvxpy": cp.__version__,
            "casadi": casadi.__version__,
            "osqp": osqp.__version__,
            "cvxpy_solvers": cp.installed_solvers(),
        },
        "configuration": {
            "horizon": config.mpc.horizon,
            "dt_seconds": config.mpc.dt,
            "control_period_seconds": config.pde.control_dt,
            "target_speed_mps": config.mpc.target_speed,
            "osqp_time_limit_seconds": config.mpc.solver_timeout,
            "map_bounds_enforced": True,
            "maximum_allowed_cbf_slack": config.mpc.maximum_allowed_cbf_slack,
            "iterations_per_case": iterations,
        },
        "cases": {},
    }

    for preset_name in ("balanced", "pure_mpc"):
        preset = get_preset(preset_name)
        for scenario_name, scenario in _scenarios().items():
            ego, target_lane, vehicles, uniform_risk = scenario
            field = DREAMRiskField(config)
            field.R.fill(uniform_risk)
            mpc = RiskAwareMPC(config, enforce_map_bounds=True)
            results = [
                mpc.solve(ego, target_lane, vehicles, field, preset)
                for _ in range(iterations)
            ]
            milliseconds = np.asarray(
                [result.solve_seconds * 1000.0 for result in results],
                dtype=np.float64,
            )
            case_name = f"{preset_name}:{scenario_name}"
            maximum_slack = float(
                max(result.maximum_slack for result in results)
            )
            report["cases"][case_name] = {
                "count": iterations,
                "cold_ms": float(milliseconds[0]),
                "warm_median_ms": float(np.median(milliseconds[1:])),
                "p95_ms": float(np.quantile(milliseconds, 0.95)),
                "p99_ms": float(np.quantile(milliseconds, 0.99)),
                "maximum_ms": float(np.max(milliseconds)),
                "fallbacks": int(
                    sum(result.used_fallback for result in results)
                ),
                "statuses": sorted({result.status for result in results}),
                "maximum_cbf_slack": (
                    maximum_slack if np.isfinite(maximum_slack) else None
                ),
                "all_finite": bool(
                    all(
                        np.all(np.isfinite(result.states))
                        and np.all(np.isfinite(result.controls))
                        for result in results
                    )
                ),
            }
    return report


def verify_report(report: dict, maximum_p99_ms: float = 150.0) -> list[str]:
    """Return failures against the 5 Hz physical-planning acceptance gate."""

    if not np.isfinite(maximum_p99_ms) or maximum_p99_ms <= 0.0:
        raise ValueError("maximum_p99_ms must be positive and finite")
    failures = []
    if "OSQP" not in report["dependencies"]["cvxpy_solvers"]:
        failures.append("CVXPY does not expose the OSQP solver")

    control_deadline_ms = (
        1000.0 * report["configuration"]["control_period_seconds"]
    )
    allowed_slack = report["configuration"]["maximum_allowed_cbf_slack"]
    for case_name, case in report["cases"].items():
        if case["fallbacks"]:
            failures.append(
                f"{case_name}: {case['fallbacks']} fallback solve(s)"
            )
        if not case["all_finite"]:
            failures.append(f"{case_name}: non-finite state/control output")
        if any(
            status not in {"optimal", "optimal_inaccurate"}
            for status in case["statuses"]
        ):
            failures.append(
                f"{case_name}: rejected solver status {case['statuses']}"
            )
        slack = case["maximum_cbf_slack"]
        if slack is None or not np.isfinite(slack):
            failures.append(f"{case_name}: non-finite CBF slack")
        elif slack > allowed_slack:
            failures.append(
                f"{case_name}: CBF slack {slack:.6g} "
                f"exceeds {allowed_slack:.6g}"
            )
        if case["p99_ms"] > maximum_p99_ms:
            failures.append(
                f"{case_name}: p99 {case['p99_ms']:.1f} ms exceeds "
                f"{maximum_p99_ms:.1f} ms"
            )
        if case["maximum_ms"] >= control_deadline_ms:
            failures.append(
                f"{case_name}: maximum {case['maximum_ms']:.1f} ms misses "
                f"the {control_deadline_ms:.1f} ms control deadline"
            )
    return failures


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--target-speed",
        type=float,
        default=0.15,
        help="MPC cruise speed; defaults to the first-motion hardware cap",
    )
    parser.add_argument(
        "--maximum-p99-ms",
        type=float,
        default=150.0,
        help=(
            "acceptance ceiling that leaves 50 ms of a 5 Hz cycle "
            "for non-MPC work"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    report = run_benchmark(arguments.iterations, arguments.target_speed)
    failures = verify_report(report, arguments.maximum_p99_ms)
    report["verification"] = {
        "passed": not failures,
        "maximum_p99_ms": arguments.maximum_p99_ms,
        "failures": failures,
        "motion_output_possible": False,
    }
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    print(rendered, end="")
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
