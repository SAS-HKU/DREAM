#!/usr/bin/env python3
"""Freeze and optionally execute matched CARLA converging-overtake scenes.

The safe default only freezes the bank.  Pass ``--execute`` to run the four
matched arms for each scene.  The same resolved manifest path and file digest
are supplied to DREAM/IDEAM and true-threat/empty-shadow arms; only the two
explicit command-line factors vary.  Every attempted arm receives a durable
JSONL ledger entry, including launch errors and non-zero exits.

This orchestration layer is Python 3.7 compatible and does not import CARLA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from evaluation.carla_converging_scene import (
        GENERATOR_VERSION,
        construction_hash,
        load_template,
        resolve_scene,
    )
except ImportError:  # Direct invocation from evaluation/ under Python 3.7.
    from carla_converging_scene import (  # type: ignore
        GENERATOR_VERSION,
        construction_hash,
        load_template,
        resolve_scene,
    )


BANK_SCHEMA = "carla_converging_scene_bank_v1"
LEDGER_SCHEMA = "carla_converging_scene_run_ledger_v1"
ARM_ORDER_VERSION = "matched_four_arm_randomization_v1"
ARMS = (
    ("DREAM", "true_threat"),
    ("DREAM", "empty_shadow"),
    ("IDEAM", "true_threat"),
    ("IDEAM", "empty_shadow"),
)


class BankError(ValueError):
    """Raised when a bank cannot be frozen or its integrity is violated."""


def _canonical_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _pretty_bytes(value):
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def randomized_arm_order(scene_seed, randomization_seed=0):
    """Return a deterministic within-scene permutation of all four arms."""

    material = "{}:{}:{}".format(
        ARM_ORDER_VERSION, int(randomization_seed), int(scene_seed)
    ).encode("ascii")
    local_seed = int(hashlib.sha256(material).hexdigest()[:16], 16)
    rng = random.Random(local_seed)
    arms = [
        {"controller": controller, "condition": condition}
        for controller, condition in ARMS
    ]
    rng.shuffle(arms)
    return arms


def _write_frozen(path, content):
    """Create a frozen file, or verify that an existing file is identical."""

    path = Path(path)
    if path.exists():
        existing = path.read_bytes()
        if existing != content:
            raise BankError(
                "refusing to overwrite non-identical frozen file: {}".format(path)
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _normalize_seeds(seeds):
    normalized = []
    seen = set()
    for seed in seeds:
        if isinstance(seed, bool):
            raise BankError("scene seeds must be integers")
        integer = int(seed)
        if integer in seen:
            raise BankError("duplicate scene seed: {}".format(integer))
        normalized.append(integer)
        seen.add(integer)
    if not normalized:
        raise BankError("at least one scene seed is required")
    return normalized


def freeze_bank(template_path, bank_dir, seeds, randomization_seed=0):
    """Resolve and freeze a deterministic bank, returning its index payload."""

    template_path = Path(template_path).expanduser().resolve()
    bank_dir = Path(bank_dir).expanduser().resolve()
    seeds = _normalize_seeds(seeds)
    template_bytes = template_path.read_bytes()
    template_digest = _sha256_bytes(template_bytes)
    template = load_template(template_path)

    bank_identity = {
        "schema_version": BANK_SCHEMA,
        "generator_version": GENERATOR_VERSION,
        "arm_order_version": ARM_ORDER_VERSION,
        "template_sha256": template_digest,
        "scene_seeds": seeds,
        "randomization_seed": int(randomization_seed),
    }
    bank_id = "converging_bank_{}".format(
        _sha256_bytes(_canonical_bytes(bank_identity))[:16]
    )

    scenes = []
    for ordinal, seed in enumerate(seeds):
        manifest = resolve_scene(template, seed)
        manifest_digest = construction_hash(manifest)
        recorded_digest = manifest["scene_construction"]["construction_hash_sha256"]
        if manifest_digest != recorded_digest:
            raise BankError("construction hash mismatch for scene seed {}".format(seed))
        manifest_relpath = Path("manifests") / "scene_{:04d}_seed{}.json".format(
            ordinal + 1, seed
        )
        manifest_path = bank_dir / manifest_relpath
        manifest_bytes = _pretty_bytes(manifest)
        _write_frozen(manifest_path, manifest_bytes)
        order = randomized_arm_order(seed, randomization_seed)
        scenes.append(
            {
                "ordinal": ordinal + 1,
                "seed": seed,
                "scene_id": manifest["scenario_id"],
                "construction_hash_sha256": manifest_digest,
                "manifest_path": manifest_relpath.as_posix(),
                "manifest_file_sha256": _sha256_bytes(manifest_bytes),
                "arm_order": order,
                "arm_count": len(order),
            }
        )

    index = {
        "schema_version": BANK_SCHEMA,
        "bank_id": bank_id,
        "generator_version": GENERATOR_VERSION,
        "arm_order_version": ARM_ORDER_VERSION,
        "template_source": str(template_path),
        "template_sha256": template_digest,
        "randomization_seed": int(randomization_seed),
        "scene_count": len(scenes),
        "planned_run_count": 4 * len(scenes),
        "matched_block_factors": {
            "controller": ["DREAM", "IDEAM"],
            "condition": ["true_threat", "empty_shadow"],
        },
        "manifest_policy": "one_byte_identical_frozen_manifest_per_four_arm_scene",
        "scenes": scenes,
    }
    _write_frozen(bank_dir / "bank_index.json", _pretty_bytes(index))
    return index


def load_bank_index(path):
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    if index.get("schema_version") != BANK_SCHEMA:
        raise BankError("unsupported bank index schema")
    if int(index.get("planned_run_count", -1)) != 4 * int(index.get("scene_count", -1)):
        raise BankError("bank index does not describe complete four-arm blocks")
    return index


def _ledger_records(path):
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except ValueError as error:
                raise BankError(
                    "invalid ledger JSON at line {}: {}".format(line_number, error)
                )
    return records


def _append_ledger(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _attempt_number(existing, scene_id, controller, condition):
    return 1 + sum(
        1
        for record in existing
        if record.get("scene_id") == scene_id
        and record.get("controller") == controller
        and record.get("condition") == condition
    )


def _quote_command(command):
    if os.name == "nt":
        return subprocess.list2cmdline([str(item) for item in command])
    return " ".join(shlex.quote(str(item)) for item in command)


def execute_bank(
    index_path,
    trial_python,
    trial_script,
    output_root,
    base_carla_port=2057,
    base_planner_port=8765,
    launch_server=False,
    pace_realtime=True,
    quality_level="Low",
    planner_python=None,
    carla_executable=None,
    extra_trial_args=None,
    stop_on_failure=False,
    retry_successful=False,
):
    """Execute a frozen bank and return a summary of this invocation."""

    index_path = Path(index_path).expanduser().resolve()
    bank_dir = index_path.parent
    index = load_bank_index(index_path)
    trial_script = Path(trial_script).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ledger_path = bank_dir / "run_ledger.jsonl"
    existing = _ledger_records(ledger_path)
    extra_trial_args = list(extra_trial_args or [])
    invocation = {
        "bank_id": index["bank_id"],
        "attempted": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped_prior_success": 0,
        "ledger_path": str(ledger_path),
    }
    global_arm_index = 0

    for scene in index["scenes"]:
        manifest_path = (bank_dir / scene["manifest_path"]).resolve()
        if not manifest_path.is_file():
            raise BankError("frozen manifest is missing: {}".format(manifest_path))
        manifest_bytes = manifest_path.read_bytes()
        file_digest = _sha256_bytes(manifest_bytes)
        if file_digest != scene["manifest_file_sha256"]:
            raise BankError("frozen manifest file digest changed: {}".format(manifest_path))
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if construction_hash(manifest) != scene["construction_hash_sha256"]:
            raise BankError("construction digest changed: {}".format(manifest_path))

        normalized_arms = {
            (arm["controller"], arm["condition"]) for arm in scene["arm_order"]
        }
        if normalized_arms != set(ARMS) or len(scene["arm_order"]) != 4:
            raise BankError("scene {} is not a complete four-arm block".format(scene["scene_id"]))

        for order_index, arm in enumerate(scene["arm_order"], start=1):
            controller = arm["controller"]
            condition = arm["condition"]
            prior_success = any(
                record.get("scene_id") == scene["scene_id"]
                and record.get("controller") == controller
                and record.get("condition") == condition
                and record.get("status") == "success"
                for record in existing
            )
            if prior_success and not retry_successful:
                invocation["skipped_prior_success"] += 1
                global_arm_index += 1
                continue

            attempt = _attempt_number(
                existing, scene["scene_id"], controller, condition
            )
            stem = "scene_{:04d}_order{}_{}_{}_attempt{}".format(
                int(scene["ordinal"]),
                order_index,
                controller.lower(),
                condition,
                attempt,
            )
            log_dir = bank_dir / "logs" / scene["scene_id"]
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = log_dir / (stem + ".stdout.log")
            stderr_path = log_dir / (stem + ".stderr.log")
            # A user-managed CARLA server is intentionally reused across the
            # sequential bank.  Only self-launched per-arm servers require a
            # distinct port because each trial owns that server process.
            carla_port = (
                int(base_carla_port) + global_arm_index
                if launch_server
                else int(base_carla_port)
            )
            planner_port = int(base_planner_port) + global_arm_index
            command = [
                str(trial_python),
                str(trial_script),
                "--condition",
                condition,
                "--controller",
                controller,
                "--seed",
                str(scene["seed"]),
                "--manifest",
                str(manifest_path),
                "--output-root",
                str(output_root / scene["scene_id"]),
                "--carla-port",
                str(carla_port),
                "--planner-port",
                str(planner_port),
                "--quality-level",
                str(quality_level),
            ]
            if launch_server:
                command.append("--launch-server")
            if pace_realtime:
                command.append("--pace-realtime")
            if planner_python:
                command.extend(["--planner-python", str(planner_python)])
            if carla_executable:
                command.extend(["--carla-executable", str(carla_executable)])
            command.extend(str(item) for item in extra_trial_args)

            started_at = _utc_now()
            started_clock = time.time()
            status = "launch_error"
            return_code = None
            error_text = None
            with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout_handle, stderr_path.open(
                "w", encoding="utf-8", newline="\n"
            ) as stderr_handle:
                try:
                    completed = subprocess.run(
                        command,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        check=False,
                    )
                    return_code = int(completed.returncode)
                    status = "success" if return_code == 0 else "failed"
                except Exception as error:  # Preserve the exact failed arm in the ledger.
                    error_text = "{}: {}".format(type(error).__name__, error)
                    stderr_handle.write(error_text + "\n")
            record = {
                "schema_version": LEDGER_SCHEMA,
                "bank_id": index["bank_id"],
                "scene_ordinal": int(scene["ordinal"]),
                "scene_id": scene["scene_id"],
                "scene_seed": int(scene["seed"]),
                "construction_hash_sha256": scene["construction_hash_sha256"],
                "manifest_path": str(manifest_path),
                "manifest_file_sha256": file_digest,
                "arm_order_index": order_index,
                "controller": controller,
                "condition": condition,
                "attempt": attempt,
                "status": status,
                "return_code": return_code,
                "error": error_text,
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now(),
                "wall_duration_s": round(time.time() - started_clock, 6),
                "carla_port": carla_port,
                "planner_port": planner_port,
                "command": command,
                "command_text": _quote_command(command),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
            _append_ledger(ledger_path, record)
            existing.append(record)
            invocation["attempted"] += 1
            if status == "success":
                invocation["succeeded"] += 1
            else:
                invocation["failed"] += 1
                if stop_on_failure:
                    return invocation
            global_arm_index += 1
    return invocation


def _parse_seed_list(raw_values, start_seed, scene_count):
    if raw_values:
        seeds = []
        for raw in raw_values:
            for token in str(raw).split(","):
                if token.strip():
                    seeds.append(int(token.strip()))
        return seeds
    return [int(start_seed) + offset for offset in range(int(scene_count))]


def _parse_args(argv=None):
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        default=str(Path(__file__).with_name("carla_converging_overtake_manifest.json")),
    )
    parser.add_argument(
        "--bank-dir",
        default=str(repo_root / "outputs" / "carla_converging_bank"),
    )
    parser.add_argument(
        "--seeds",
        action="append",
        help="comma-separated scene seeds; repeatable (overrides start/count)",
    )
    parser.add_argument("--start-seed", type=int, default=1001)
    parser.add_argument("--scene-count", type=int, default=12)
    parser.add_argument("--randomization-seed", type=int, default=20260716)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run CARLA after freezing; otherwise only materialize the bank",
    )
    parser.add_argument("--trial-python", default=sys.executable)
    parser.add_argument(
        "--trial-script", default=str(Path(__file__).with_name("carla_overtaking_trial.py"))
    )
    parser.add_argument(
        "--output-root",
        default=str(repo_root / "outputs" / "carla_converging_runs"),
    )
    parser.add_argument("--base-carla-port", type=int, default=2200)
    parser.add_argument("--base-planner-port", type=int, default=8900)
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument("--quality-level", choices=("Low", "Epic"), default="Low")
    parser.add_argument("--planner-python", default=None)
    parser.add_argument("--carla-executable", default=None)
    parser.add_argument("--extra-trial-arg", action="append", default=[])
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--retry-successful", action="store_true")
    pacing = parser.add_mutually_exclusive_group()
    pacing.add_argument("--pace-realtime", dest="pace_realtime", action="store_true")
    pacing.add_argument("--no-pace-realtime", dest="pace_realtime", action="store_false")
    parser.set_defaults(pace_realtime=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    seeds = _parse_seed_list(args.seeds, args.start_seed, args.scene_count)
    index = freeze_bank(
        args.template,
        args.bank_dir,
        seeds,
        randomization_seed=args.randomization_seed,
    )
    result = {
        "bank_index": str(Path(args.bank_dir).expanduser().resolve() / "bank_index.json"),
        "bank_id": index["bank_id"],
        "scene_count": index["scene_count"],
        "planned_run_count": index["planned_run_count"],
        "executed": bool(args.execute),
    }
    exit_code = 0
    if args.execute:
        invocation = execute_bank(
            Path(args.bank_dir) / "bank_index.json",
            trial_python=args.trial_python,
            trial_script=args.trial_script,
            output_root=args.output_root,
            base_carla_port=args.base_carla_port,
            base_planner_port=args.base_planner_port,
            launch_server=args.launch_server,
            pace_realtime=args.pace_realtime,
            quality_level=args.quality_level,
            planner_python=args.planner_python,
            carla_executable=args.carla_executable,
            extra_trial_args=args.extra_trial_arg,
            stop_on_failure=args.stop_on_failure,
            retry_successful=args.retry_successful,
        )
        result["invocation"] = invocation
        if invocation["failed"]:
            exit_code = 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "ARM_ORDER_VERSION",
    "BANK_SCHEMA",
    "BankError",
    "execute_bank",
    "freeze_bank",
    "load_bank_index",
    "randomized_arm_order",
]
