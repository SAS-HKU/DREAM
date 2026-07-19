#!/usr/bin/env python3
"""Run the mandatory offline DREAM scaling/replay gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from dream_limo.core.replay import run_stage1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    results = run_stage1(args.output)
    print(
        json.dumps(
            {name: asdict(result.metrics) for name, result in results.items()},
            indent=2,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()
