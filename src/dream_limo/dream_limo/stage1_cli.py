"""Installed entry point for the mandatory Stage 1 replay."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path

from .core.replay import run_stage1


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    results = run_stage1(args.output)
    metrics = {name: asdict(item.metrics) for name, item in results.items()}
    print(json.dumps(_json_safe(metrics), indent=2, allow_nan=False))
