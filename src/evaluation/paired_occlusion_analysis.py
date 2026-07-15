"""Auditable paired analysis for occlusion-ablation episode records.

The simulator is intentionally not imported here.  This module consumes one
JSON object per episode--variant run, pairs variants at the *scenario* level,
and writes analysis-ready JSON and Markdown.  Keeping the analysis independent
of the runner makes the statistical contract stable while the scenario and
controller implementations evolve.

Expected minimal JSONL schema (additional fields are retained but ignored)::

    {
      "scenario_id": "threat-001",
      "stratum": "true_occluded_threat",
      "variant": "full_dream",
      "validity": {
        "sim_completed": true,
        "valid_reveal": true,
        "fallback_used": false
      },
      "safety": {
        "collision_incident": false,
        "near_collision_incident": false,
        "collision_or_near_incident": false,
        "min_clearance_m": 1.42,
        "post_reveal_min_clearance_m": 1.61,
        "min_ttc_s": 4.8,
        "min_ttc_censored": false,
        "ttc_horizon_s": 10.0
      },
      "field": {"risk_mass_target_maneuver_tube": 0.31},
      "tradeoff": {"progress_m": 80.0}
    }

For censored TTC, request ``safety.restricted_min_ttc_s`` or
``safety.post_reveal_restricted_min_ttc_s``.  These are explicitly defined as
``min(observed TTC, declared horizon)``.  An infinite TTC is therefore mapped
only to the recorded evaluation horizon, never to an arbitrary 60 s cap.  Raw
TTC fields with an infinite/censored value are excluded from continuous tests
and their censoring is reported.

The unit of inference is one paired scenario in one stratum--never a simulation
step.  When requested before analysis, results are also reported within the
predeclared construction-severity bands.  By default every retained scenario
must contain one valid record for the reference and every requested comparator.
Incomplete, duplicate, and invalid runs are counted in the output rather than
silently dropped.  Fallback runs are kept for an intention-to-treat safety
analysis by default, but are flagged; use ``--exclude-fallback`` only for a
declared per-protocol sensitivity analysis.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "paired_occlusion_analysis_v2"


# These defaults intentionally focus on outcome families expected from the
# revised benchmark.  Metrics absent from a file are listed as skipped rather
# than being fabricated as zero-valued outcomes.
DEFAULT_EVENT_METRICS = (
    "safety.collision_incident",
    "safety.near_collision_incident",
    "safety.collision_or_near_incident",
    "safety.ttc_critical_incident",
    "safety_by_actor.latent_target_lane_vehicle.collision_incident",
    "safety_by_actor.latent_target_lane_vehicle.near_collision_incident",
    "safety_by_actor.latent_target_lane_vehicle.collision_or_near_incident",
    "safety_by_actor.latent_target_lane_vehicle.ttc_critical_incident",
    "tradeoff.false_veto_incident",
    "tradeoff.unnecessary_braking_incident",
)

DEFAULT_CONTINUOUS_METRICS = (
    "safety.min_clearance_m",
    "safety.post_reveal_min_clearance_m",
    "safety.restricted_min_ttc_s",
    "safety.post_reveal_restricted_min_ttc_s",
    "safety_by_actor.latent_target_lane_vehicle.min_clearance_m",
    "safety_by_actor.latent_target_lane_vehicle.post_reveal_min_clearance_m",
    "safety_by_actor.latent_target_lane_vehicle.restricted_min_ttc_s",
    "safety_by_actor.latent_target_lane_vehicle.post_reveal_restricted_min_ttc_s",
    "field.reference_playback.risk_mass_target_maneuver_tube",
    "field.reference_playback.risk_contrast_target_vs_control",
    "field.risk_mass_target_maneuver_tube",
    "field.risk_contrast_target_vs_control",
    "field.anticipation_lead_time_s",
    "tradeoff.progress_m",
    "tradeoff.time_loss_s",
    "tradeoff.braking_duration_s",
    "tradeoff.peak_deceleration_mps2",
    "tradeoff.mean_abs_jerk_mps3",
)


_MISSING = object()
_TRUE = {"1", "true", "t", "yes", "y", "on"}
_FALSE = {"0", "false", "f", "no", "n", "off"}


@dataclass(frozen=True)
class MetricValue:
    """One extracted metric value plus its censoring/provenance state."""

    value: float | None
    censored: bool = False
    horizon_s: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class RecordQuality:
    """Eligibility information derived from an episode record."""

    globally_valid: bool
    invalid_reason: str | None
    fallback_used: bool
    reveal_valid: bool | None


class AnalysisInputError(ValueError):
    """Raised when a JSONL input cannot support an auditable paired analysis."""


def load_episode_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL episode records, failing loudly on malformed input."""

    input_path = Path(path)
    if not input_path.is_file():
        raise AnalysisInputError(f"Episode JSONL file does not exist: {input_path}")

    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisInputError(
                    f"Malformed JSON on line {line_number} of {input_path}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise AnalysisInputError(
                    f"Line {line_number} of {input_path} is not a JSON object"
                )
            records.append(record)

    if not records:
        raise AnalysisInputError(f"No episode records found in {input_path}")
    return records


def _nested(record: Mapping[str, Any], dotted_path: str, default: Any = _MISSING) -> Any:
    current: Any = record
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in _TRUE:
            return True
        if normalised in _FALSE:
            return False
    return None


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_float_allow_inf(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_boolean(record: Mapping[str, Any], paths: Iterable[str]) -> bool | None:
    for path in paths:
        value = _nested(record, path)
        parsed = _as_bool(value)
        if parsed is not None:
            return parsed
    return None


def _record_quality(record: Mapping[str, Any]) -> RecordQuality:
    """Classify a record without making an outcome-dependent exclusion."""

    # ``sim_completed``/``episode_valid`` are scenario-execution checks.  A
    # decision/controller outcome such as a collision is deliberately *not*
    # treated as invalid here.
    invalid_paths = (
        "validity.sim_completed",
        "validity.run_valid",
        "validity.episode_valid",
        "validity.scenario_valid",
    )
    for path in invalid_paths:
        value = _nested(record, path)
        parsed = _as_bool(value)
        if parsed is False:
            return RecordQuality(False, path, False, None)

    fallback_paths = (
        "validity.fallback_used",
        "validity.solver_fallback_used",
        "validity.used_fallback",
        "solver.fallback_used",
        "solver.used_fallback",
    )
    fallback_used = any(
        _as_bool(_nested(record, path)) is True for path in fallback_paths
    )
    fallback_steps = _as_finite_float(_nested(record, "solver.n_fallback_steps"))
    if fallback_steps is not None and fallback_steps > 0:
        fallback_used = True

    reveal_valid = _first_boolean(
        record,
        (
            "validity.valid_reveal",
            "validity.reveal_valid",
            "validity.geometric_reveal_valid",
            "safety.post_reveal_available",
        ),
    )
    return RecordQuality(True, None, fallback_used, reveal_valid)


def _is_post_reveal_metric(metric: str) -> bool:
    return "post_reveal" in metric


_RESTRICTED_TTC_METRICS: dict[str, tuple[str, str, str]] = {
    "safety.restricted_min_ttc_s": (
        "safety.min_ttc_s",
        "safety.min_ttc_censored",
        "safety.ttc_horizon_s",
    ),
    "safety.post_reveal_restricted_min_ttc_s": (
        "safety.post_reveal_min_ttc_s",
        "safety.post_reveal_min_ttc_censored",
        "safety.ttc_horizon_s",
    ),
    "safety_by_actor.latent_target_lane_vehicle.restricted_min_ttc_s": (
        "safety_by_actor.latent_target_lane_vehicle.min_ttc_s",
        "safety_by_actor.latent_target_lane_vehicle.min_ttc_censored",
        "safety_by_actor.latent_target_lane_vehicle.ttc_horizon_s",
    ),
    "safety_by_actor.latent_target_lane_vehicle.post_reveal_restricted_min_ttc_s": (
        "safety_by_actor.latent_target_lane_vehicle.post_reveal_min_ttc_s",
        "safety_by_actor.latent_target_lane_vehicle.post_reveal_min_ttc_censored",
        "safety_by_actor.latent_target_lane_vehicle.ttc_horizon_s",
    ),
}


def _extract_continuous(record: Mapping[str, Any], metric: str) -> MetricValue:
    """Read a finite continuous value, handling declared TTC censoring safely."""

    if metric in _RESTRICTED_TTC_METRICS:
        direct = _as_finite_float(_nested(record, metric))
        if direct is not None:
            return MetricValue(direct)

        value_path, censored_path, horizon_path = _RESTRICTED_TTC_METRICS[metric]
        raw = _as_float_allow_inf(_nested(record, value_path))
        censored = _as_bool(_nested(record, censored_path))
        horizon = _as_finite_float(_nested(record, horizon_path))
        if horizon is None or horizon <= 0.0:
            return MetricValue(None, reason="missing_or_invalid_ttc_horizon")
        if censored is True or (raw is not None and math.isinf(raw)):
            return MetricValue(horizon, censored=True, horizon_s=horizon)
        if raw is None or not math.isfinite(raw):
            return MetricValue(None, reason="missing_ttc_value")
        # The restricted estimand is defined at the declared horizon even if a
        # producer accidentally supplies a value beyond it.
        return MetricValue(min(raw, horizon), censored=False, horizon_s=horizon)

    raw = _as_float_allow_inf(_nested(record, metric))
    if raw is None:
        return MetricValue(None, reason="missing_or_non_numeric")
    if not math.isfinite(raw):
        return MetricValue(None, censored=math.isinf(raw), reason="non_finite")
    return MetricValue(raw)


def _extract_event(record: Mapping[str, Any], metric: str) -> tuple[bool | None, str | None]:
    value = _nested(record, metric)
    parsed = _as_bool(value)
    if parsed is not None:
        return parsed, None
    return None, "missing_or_non_binary"


def _stable_seed(base_seed: int, *parts: str) -> int:
    payload = "|".join((str(base_seed), *parts)).encode("utf-8")
    # A stable derived seed makes a metric's CI independent of iteration order.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _binomial_two_sided_exact(successes: int, trials: int) -> float:
    """Two-sided exact binomial p value under p=0.5, without SciPy."""

    if trials <= 0:
        return 1.0
    lower_tail = sum(math.comb(trials, k) for k in range(0, min(successes, trials - successes) + 1))
    return min(1.0, 2.0 * lower_tail / (2 ** trials))


def _mcnemar_exact(reference: np.ndarray, comparator: np.ndarray) -> dict[str, Any]:
    """Exact paired McNemar test for binary event indicators."""

    ref = reference.astype(bool)
    comp = comparator.astype(bool)
    reference_only = int(np.count_nonzero(ref & ~comp))
    comparator_only = int(np.count_nonzero(~ref & comp))
    discordant = reference_only + comparator_only
    p_value = _binomial_two_sided_exact(reference_only, discordant)
    return {
        "test": "exact_mcnemar",
        "reference_only_events": reference_only,
        "comparator_only_events": comparator_only,
        "n_discordant": discordant,
        "p_value": p_value,
    }


def _paired_randomization_test(
    differences: np.ndarray,
    *,
    seed: int,
    monte_carlo_samples: int,
    exact_max_pairs: int = 20,
) -> dict[str, Any]:
    """Two-sided paired sign-flip randomization test of median difference.

    Treatment labels are exchangeable *within each frozen construction pair*.
    For small banks, every sign assignment is enumerated exactly.  Larger
    banks use a reproducible Monte-Carlo sign-flip distribution and report its
    simulation size, rather than presenting a pseudo-exact p value.
    """

    if differences.size == 0:
        raise ValueError("Cannot run a paired randomization test on an empty sample")
    if monte_carlo_samples <= 0:
        raise ValueError("randomization_samples must be positive")

    observed = abs(float(np.median(differences)))
    n_pairs = int(differences.size)
    tolerance = 1e-12
    if n_pairs <= exact_max_pairs:
        total_assignments = 1 << n_pairs
        extreme = 0
        bit_positions = np.arange(n_pairs, dtype=np.uint64)
        for start in range(0, total_assignments, 1_024):
            stop = min(total_assignments, start + 1_024)
            masks = np.arange(start, stop, dtype=np.uint64)[:, None]
            signs = np.where(
                ((masks >> bit_positions) & np.uint64(1)) == np.uint64(1),
                1.0,
                -1.0,
            )
            statistics = np.abs(np.median(signs * differences[None, :], axis=1))
            extreme += int(np.count_nonzero(statistics >= observed - tolerance))
        return {
            "test": "two_sided_exact_paired_sign_flip_randomization",
            "statistic": "absolute_paired_median_difference",
            "observed_statistic": observed,
            "n_pairs": n_pairs,
            "n_sign_assignments": total_assignments,
            "p_value": float(extreme / total_assignments),
        }

    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, monte_carlo_samples, 1_024):
        batch_size = min(1_024, monte_carlo_samples - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(batch_size, n_pairs))
        statistics = np.abs(np.median(signs * differences[None, :], axis=1))
        extreme += int(np.count_nonzero(statistics >= observed - tolerance))
    return {
        "test": "two_sided_monte_carlo_paired_sign_flip_randomization",
        "statistic": "absolute_paired_median_difference",
        "observed_statistic": observed,
        "n_pairs": n_pairs,
        "n_sign_assignments": int(monte_carlo_samples),
        # Add-one correction retains a valid finite-simulation p value.
        "p_value": float((extreme + 1) / (monte_carlo_samples + 1)),
    }


def _is_predeclared_primary(entry: Mapping[str, Any], *, kind: str) -> bool:
    """Identify only the two protocol-declared pooled primary endpoints."""

    if entry.get("severity") != "all":
        return False
    if kind == "event":
        return (
            entry.get("stratum") == "true_occluded_threat"
            and entry.get("reference_variant") == "coupling_full"
            and entry.get("comparator_variant") == "coupling_no_veto"
            and entry.get("metric") == "safety.collision_incident"
        )
    return (
        entry.get("reference_variant") == "field_full"
        and entry.get("comparator_variant") == "field_no_occ_source"
        and entry.get("metric")
        == "field.reference_playback.risk_mass_target_maneuver_tube"
    )


def _bootstrap_median_ci(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Deterministic percentile bootstrap CI for a paired median difference."""

    if differences.size == 0:
        raise ValueError("Cannot bootstrap an empty paired sample")
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    rng = np.random.default_rng(seed)
    n = differences.size
    estimates = np.empty(samples, dtype=float)
    # Chunking prevents a large benchmark from materialising a huge matrix.
    for start in range(0, samples, 512):
        stop = min(samples, start + 512)
        indices = rng.integers(0, n, size=(stop - start, n))
        estimates[start:stop] = np.median(differences[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    """Holm step-down adjustment, preserving the input order."""

    adjusted: list[float | None] = [None] * len(p_values)
    valid = [(index, float(p)) for index, p in enumerate(p_values) if p is not None]
    valid.sort(key=lambda item: item[1])
    total = len(valid)
    previous = 0.0
    for rank, (index, p_value) in enumerate(valid):
        value = min(1.0, max(previous, (total - rank) * p_value))
        adjusted[index] = value
        previous = value
    return adjusted


def _format_count(count: int, denominator: int) -> str:
    return f"{count}/{denominator}" if denominator else "N/A"


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def _format_p(value: Any) -> str:
    if value is None:
        return "N/A"
    number = float(value)
    if number < 0.001:
        return "<0.001"
    return f"{number:.3f}"


def _json_safe(value: Any) -> Any:
    """Convert NumPy/non-finite values to strict JSON-compatible primitives."""

    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_is_available(
    records: Sequence[Mapping[str, Any]], metric: str, kind: str
) -> bool:
    if kind == "event":
        return any(_extract_event(record, metric)[0] is not None for record in records)
    return any(_extract_continuous(record, metric).value is not None for record in records)


def _validate_identity(record: Mapping[str, Any], index: int) -> tuple[str, str, str]:
    scenario_id = record.get("scenario_id")
    variant = record.get("variant")
    if scenario_id is None or str(scenario_id).strip() == "":
        raise AnalysisInputError(f"Record {index} is missing a non-empty scenario_id")
    if variant is None or str(variant).strip() == "":
        raise AnalysisInputError(f"Record {index} is missing a non-empty variant")
    stratum = record.get("stratum", "unspecified")
    if stratum is None or str(stratum).strip() == "":
        stratum = "unspecified"
    return str(scenario_id), str(stratum), str(variant)


def _scenario_severity(record: Mapping[str, Any]) -> str:
    """Return a design label without deriving it from a safety outcome.

    Version-3 benchmark records carry this at ``scenario_design.severity``.
    The top-level fallback keeps the analysis compatible with an explicitly
    labelled external episode file.  Missing labels remain visible as
    ``unspecified`` rather than being inferred from observed clearance/TTC.
    """

    raw = _nested(record, "scenario_design.severity")
    if raw is _MISSING:
        raw = record.get("severity", "unspecified")
    if raw is None or str(raw).strip() == "":
        return "unspecified"
    return str(raw).strip()


def _pair_severity(
    pair_key: tuple[str, str],
    by_variant: Mapping[str, Mapping[str, Any]],
) -> str:
    """Verify that all arms of a paired construction share one design band."""

    labels = sorted({_scenario_severity(record) for record in by_variant.values()})
    if len(labels) != 1:
        scenario_id, stratum = pair_key
        raise AnalysisInputError(
            "Paired variants have inconsistent scenario-design severity labels "
            f"for ({scenario_id}, {stratum}): {', '.join(labels)}"
        )
    return labels[0]


def _build_complete_population(
    records: Sequence[Mapping[str, Any]],
    *,
    required_variants: Sequence[str],
    exclude_fallback: bool,
) -> tuple[dict[tuple[str, str], dict[str, Mapping[str, Any]]], dict[str, Any]]:
    """Return exact complete cases and an explicit exclusion audit."""

    grouped: dict[tuple[str, str], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_variants: set[str] = set()
    for index, record in enumerate(records):
        scenario_id, stratum, variant = _validate_identity(record, index)
        grouped[(scenario_id, stratum)][variant].append(record)
        all_variants.add(variant)

    missing_reference = [variant for variant in required_variants if variant not in all_variants]
    if missing_reference:
        raise AnalysisInputError(
            "Required variants are absent from the JSONL input: " + ", ".join(missing_reference)
        )

    complete: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    exclusions: Counter[str] = Counter()
    fallback_by_variant: Counter[str] = Counter()
    valid_records = 0

    for key in sorted(grouped):
        per_variant = grouped[key]
        missing = [variant for variant in required_variants if variant not in per_variant]
        if missing:
            exclusions["missing_required_variant"] += 1
            continue
        duplicate = [variant for variant in required_variants if len(per_variant[variant]) != 1]
        if duplicate:
            exclusions["duplicate_variant_record"] += 1
            continue
        selected = {variant: per_variant[variant][0] for variant in required_variants}
        qualities = {variant: _record_quality(record) for variant, record in selected.items()}
        invalid = [variant for variant, quality in qualities.items() if not quality.globally_valid]
        if invalid:
            exclusions["invalid_run"] += 1
            continue
        fallbacks = [variant for variant, quality in qualities.items() if quality.fallback_used]
        if fallbacks and exclude_fallback:
            exclusions["fallback_used"] += 1
            continue
        for variant in fallbacks:
            fallback_by_variant[variant] += 1
        complete[key] = selected
        valid_records += len(selected)

    by_stratum: Counter[str] = Counter(stratum for _, stratum in complete)
    by_severity: Counter[str] = Counter(
        _pair_severity(pair_key, by_variant)
        for pair_key, by_variant in complete.items()
    )
    audit = {
        "n_input_records": len(records),
        "n_unique_scenario_strata": len(grouped),
        "n_complete_scenario_pairs": len(complete),
        "n_complete_records": valid_records,
        "complete_pairs_by_stratum": dict(sorted(by_stratum.items())),
        "complete_pairs_by_severity": dict(sorted(by_severity.items())),
        "excluded_scenario_strata": dict(sorted(exclusions.items())),
        "fallback_runs_retained_by_variant": dict(sorted(fallback_by_variant.items())),
        "exclude_fallback": bool(exclude_fallback),
        "required_variants": list(required_variants),
    }
    return complete, audit


def _eligible_pair_values(
    complete: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
    *,
    stratum: str,
    severity: str | None,
    reference_variant: str,
    comparator_variant: str,
    metric: str,
    kind: str,
) -> tuple[list[Any], Counter[str]]:
    """Extract metric-level complete pairs and count every exclusion reason."""

    pairs: list[Any] = []
    exclusions: Counter[str] = Counter()
    requires_reveal = _is_post_reveal_metric(metric)
    for (scenario_id, pair_stratum), by_variant in sorted(complete.items()):
        if pair_stratum != stratum:
            continue
        if severity is not None and _pair_severity(
            (scenario_id, pair_stratum), by_variant
        ) != severity:
            continue
        reference = by_variant[reference_variant]
        comparator = by_variant[comparator_variant]
        if requires_reveal:
            ref_reveal = _record_quality(reference).reveal_valid
            comp_reveal = _record_quality(comparator).reveal_valid
            # Post-reveal estimates must not silently include a run whose
            # reveal was absent or unverified.  All-run safety estimates are
            # still available through non-post-reveal metrics.
            if ref_reveal is not True or comp_reveal is not True:
                exclusions["invalid_or_missing_reveal"] += 1
                continue

        if kind == "event":
            ref_value, ref_reason = _extract_event(reference, metric)
            comp_value, comp_reason = _extract_event(comparator, metric)
            if ref_value is None or comp_value is None:
                exclusions[ref_reason or comp_reason or "missing_metric"] += 1
                continue
            pairs.append((scenario_id, bool(ref_value), bool(comp_value)))
            continue

        ref_value = _extract_continuous(reference, metric)
        comp_value = _extract_continuous(comparator, metric)
        if ref_value.value is None or comp_value.value is None:
            reason = ref_value.reason or comp_value.reason or "missing_metric"
            exclusions[reason] += 1
            continue
        if metric in _RESTRICTED_TTC_METRICS:
            h_ref, h_comp = ref_value.horizon_s, comp_value.horizon_s
            if h_ref is not None and h_comp is not None and not math.isclose(
                h_ref, h_comp, rel_tol=0.0, abs_tol=1e-9
            ):
                exclusions["inconsistent_ttc_horizon"] += 1
                continue
        pairs.append((scenario_id, ref_value, comp_value))
    return pairs, exclusions


def analyze_episode_records(
    records: Sequence[Mapping[str, Any]],
    *,
    reference_variant: str,
    comparator_variants: Sequence[str] | None = None,
    event_metrics: Sequence[str] | None = None,
    continuous_metrics: Sequence[str] | None = None,
    bootstrap_samples: int = 10_000,
    randomization_samples: int = 100_000,
    seed: int = 20_260_713,
    exclude_fallback: bool = False,
    stratify_by_severity: bool = False,
) -> dict[str, Any]:
    """Run the full paired analysis on already-loaded episode records.

    ``comparator_variants=None`` selects every observed variant except the
    reference.  Complete cases are enforced over the reference plus *all*
    selected comparators; this keeps an ablation family's denominators aligned.
    """

    if not records:
        raise AnalysisInputError("No episode records were supplied")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if randomization_samples <= 0:
        raise ValueError("randomization_samples must be positive")

    observed_variants = sorted({str(record.get("variant")) for record in records if record.get("variant") is not None})
    if reference_variant not in observed_variants:
        raise AnalysisInputError(f"Reference variant not present: {reference_variant}")
    if comparator_variants is None:
        comparators = [variant for variant in observed_variants if variant != reference_variant]
    else:
        comparators = []
        for variant in comparator_variants:
            if variant == reference_variant:
                continue
            if variant not in comparators:
                comparators.append(str(variant))
    if not comparators:
        raise AnalysisInputError("At least one comparator variant is required")

    selected_events = list(event_metrics or DEFAULT_EVENT_METRICS)
    selected_continuous = list(continuous_metrics or DEFAULT_CONTINUOUS_METRICS)
    available_events = [
        metric for metric in selected_events if _metric_is_available(records, metric, "event")
    ]
    available_continuous = [
        metric for metric in selected_continuous
        if _metric_is_available(records, metric, "continuous")
    ]
    skipped_events = [metric for metric in selected_events if metric not in available_events]
    skipped_continuous = [metric for metric in selected_continuous if metric not in available_continuous]

    required_variants = [reference_variant, *comparators]
    complete, population = _build_complete_population(
        records,
        required_variants=required_variants,
        exclude_fallback=exclude_fallback,
    )
    strata = sorted({stratum for _, stratum in complete})
    severity_by_pair = {
        pair_key: _pair_severity(pair_key, by_variant)
        for pair_key, by_variant in complete.items()
    }

    event_results: list[dict[str, Any]] = []
    continuous_results: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not complete:
        warnings.append("No complete paired scenario records remain after eligibility checks.")
    if population["fallback_runs_retained_by_variant"] and not exclude_fallback:
        warnings.append(
            "Fallback runs were retained for intention-to-treat safety analysis; "
            "inspect fallback counts before making a controller-performance claim."
        )
    if population["excluded_scenario_strata"]:
        warnings.append("Some scenario strata were excluded from the complete-pair population; see population audit.")

    # The pooled result remains the prespecified benchmark-wide estimate.  The
    # optional severity rows are a transparent design-stratified companion,
    # never a post-hoc safety/outcome classification.
    analysis_groups: list[tuple[str, str | None]] = []
    for stratum in strata:
        analysis_groups.append((stratum, None))
        if stratify_by_severity:
            analysis_groups.extend(
                (stratum, severity)
                for severity in sorted({
                    label
                    for (scenario_id, pair_stratum), label in severity_by_pair.items()
                    if pair_stratum == stratum
                })
            )

    for stratum, severity in analysis_groups:
        severity_label = severity if severity is not None else "all"
        for metric in available_events:
            for comparator in comparators:
                pairs, exclusions = _eligible_pair_values(
                    complete,
                    stratum=stratum,
                    severity=severity,
                    reference_variant=reference_variant,
                    comparator_variant=comparator,
                    metric=metric,
                    kind="event",
                )
                if not pairs:
                    continue
                reference_values = np.asarray([pair[1] for pair in pairs], dtype=bool)
                comparator_values = np.asarray([pair[2] for pair in pairs], dtype=bool)
                test = _mcnemar_exact(reference_values, comparator_values)
                n_pairs = int(reference_values.size)
                reference_events = int(np.count_nonzero(reference_values))
                comparator_events = int(np.count_nonzero(comparator_values))
                event_results.append({
                    "stratum": stratum,
                    "severity": severity_label,
                    "metric": metric,
                    "reference_variant": reference_variant,
                    "comparator_variant": comparator,
                    "n_pairs": n_pairs,
                    "reference_events": reference_events,
                    "comparator_events": comparator_events,
                    "reference_event_rate": float(reference_events / n_pairs),
                    "comparator_event_rate": float(comparator_events / n_pairs),
                    # Positive means the comparator had more events than full DREAM.
                    "paired_risk_difference_comparator_minus_reference": float(
                        np.mean(comparator_values.astype(float) - reference_values.astype(float))
                    ),
                    "metric_exclusions": dict(sorted(exclusions.items())),
                    **test,
                })

        for metric in available_continuous:
            for comparator in comparators:
                pairs, exclusions = _eligible_pair_values(
                    complete,
                    stratum=stratum,
                    severity=severity,
                    reference_variant=reference_variant,
                    comparator_variant=comparator,
                    metric=metric,
                    kind="continuous",
                )
                if not pairs:
                    continue
                reference_values = np.asarray([pair[1].value for pair in pairs], dtype=float)
                comparator_values = np.asarray([pair[2].value for pair in pairs], dtype=float)
                differences = comparator_values - reference_values
                ci_low, ci_high = _bootstrap_median_ci(
                    differences,
                    samples=bootstrap_samples,
                    seed=_stable_seed(
                        seed,
                        "continuous",
                        stratum,
                        severity_label,
                        metric,
                        comparator,
                    ),
                )
                test = _paired_randomization_test(
                    differences,
                    seed=_stable_seed(
                        seed,
                        "randomization",
                        stratum,
                        severity_label,
                        metric,
                        comparator,
                    ),
                    monte_carlo_samples=randomization_samples,
                )
                ref_censored = sum(1 for _, ref, _ in pairs if ref.censored)
                comp_censored = sum(1 for _, _, comp in pairs if comp.censored)
                horizons = sorted({
                    value.horizon_s
                    for _, reference, comparator_value in pairs
                    for value in (reference, comparator_value)
                    if value.horizon_s is not None
                })
                continuous_results.append({
                    "stratum": stratum,
                    "severity": severity_label,
                    "metric": metric,
                    "reference_variant": reference_variant,
                    "comparator_variant": comparator,
                    "n_pairs": int(differences.size),
                    "reference_median": float(np.median(reference_values)),
                    "comparator_median": float(np.median(comparator_values)),
                    # Positive means the comparator is numerically larger.  The
                    # reader must use the metric's stated direction, not infer it.
                    "paired_median_difference_comparator_minus_reference": float(np.median(differences)),
                    "bootstrap_percentile_ci_95": [ci_low, ci_high],
                    "reference_censored_count": int(ref_censored),
                    "comparator_censored_count": int(comp_censored),
                    "restriction_horizons_s": horizons,
                    "metric_exclusions": dict(sorted(exclusions.items())),
                    **test,
                })

    all_tests: list[tuple[str, dict[str, Any]]] = [
        *( ("event", entry) for entry in event_results ),
        *( ("continuous", entry) for entry in continuous_results ),
    ]
    secondary_indices: list[int] = []
    for index, (kind, entry) in enumerate(all_tests):
        is_primary = _is_predeclared_primary(entry, kind=kind)
        entry["inference_role"] = "primary" if is_primary else "secondary"
        if not is_primary:
            secondary_indices.append(index)
    secondary_adjusted = _holm_adjust([
        all_tests[index][1].get("p_value") for index in secondary_indices
    ])
    holm_family_size = sum(value is not None for value in secondary_adjusted)
    for kind, entry in all_tests:
        entry["p_value_holm"] = None
        entry["holm_family_size"] = holm_family_size
    for index, p_holm in zip(secondary_indices, secondary_adjusted):
        all_tests[index][1]["p_value_holm"] = p_holm

    if not event_results:
        warnings.append("No declared event metric had usable complete paired records.")
    if not continuous_results:
        warnings.append("No declared continuous metric had usable complete paired records.")

    return _json_safe({
        "schema_version": SCHEMA_VERSION,
        "analysis_unit": "paired scenario_id within stratum",
        "complete_pair_policy": (
            "A scenario-stratum is retained only when every selected variant "
            "has exactly one globally valid episode record."
        ),
        "reference_variant": reference_variant,
        "comparator_variants": comparators,
        "metrics_requested": {
            "event": selected_events,
            "continuous": selected_continuous,
        },
        "bootstrap": {
            "method": "paired percentile bootstrap of median difference",
            "samples": int(bootstrap_samples),
            "seed": int(seed),
            "confidence_level": 0.95,
        },
        "randomization": {
            "method": "within-pair sign-flip randomization test of the median difference",
            "monte_carlo_samples_when_not_exact": int(randomization_samples),
            "exact_enumeration_max_pairs": 20,
        },
        "stratification": {
            "severity_requested": bool(stratify_by_severity),
            "severity_source": "scenario_design.severity (or explicit top-level severity)",
            "pooled_rows_included": True,
            "available_severity_levels": sorted(set(severity_by_pair.values())),
        },
        "testing": {
            "event": "exact McNemar test",
            "continuous": "two-sided paired sign-flip randomization test",
            "multiple_comparison_adjustment": (
                "Holm across emitted secondary pairwise tests; protocol-declared "
                "pooled primary endpoints retain their raw p values"
            ),
            "holm_family_size": holm_family_size,
        },
        "population": population,
        "skipped_metrics": {
            "event": skipped_events,
            "continuous": skipped_continuous,
        },
        "event_results": event_results,
        "continuous_results": continuous_results,
        "warnings": warnings,
    })


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render a compact, manuscript-audit-friendly Markdown summary."""

    population = result.get("population", {})
    lines = [
        "# Paired occlusion ablation analysis",
        "",
        f"- Analysis unit: {result.get('analysis_unit', 'paired scenario')}",
        f"- Reference variant: `{result.get('reference_variant', 'N/A')}`",
        f"- Complete paired scenarios: {population.get('n_complete_scenario_pairs', 0)}",
        f"- Holm family: {result.get('testing', {}).get('holm_family_size', 0)} pairwise tests",
        (
            "- Severity rows: "
            f"{'pooled plus predeclared bands' if result.get('stratification', {}).get('severity_requested') else 'pooled only'}"
        ),
        "",
        "## Event outcomes",
        "",
        "| Stratum | Outcome | Comparator | Reference n/N | Comparator n/N | Δ risk (comp-ref) | McNemar p / Holm | Pairs |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    event_results = result.get("event_results", [])
    if event_results:
        for entry in event_results:
            lines.append(
                "| {stratum} | `{metric}` | `{comparator}` | {ref} | {comp} | {effect} | {p}/{p_holm} | {n} |".format(
                    stratum=f"{entry['stratum']} / {entry.get('severity', 'all')}",
                    metric=entry["metric"],
                    comparator=entry["comparator_variant"],
                    ref=_format_count(entry["reference_events"], entry["n_pairs"]),
                    comp=_format_count(entry["comparator_events"], entry["n_pairs"]),
                    effect=_format_float(entry["paired_risk_difference_comparator_minus_reference"]),
                    p=_format_p(entry.get("p_value")),
                    p_holm=_format_p(entry.get("p_value_holm")),
                    n=entry["n_pairs"],
                )
            )
    else:
        lines.append("| — | No usable event outcomes | — | — | — | — | — | — |")

    lines.extend([
        "",
        "## Continuous outcomes",
        "",
        "| Stratum | Outcome | Comparator | Median Δ (comp-ref), 95% bootstrap CI | Sign p / Holm | Pairs | TTC censoring (ref/comp) |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    continuous_results = result.get("continuous_results", [])
    if continuous_results:
        for entry in continuous_results:
            ci = entry.get("bootstrap_percentile_ci_95", [None, None])
            lines.append(
                "| {stratum} | `{metric}` | `{comparator}` | {effect} [{low}, {high}] | {p}/{p_holm} | {n} | {rc}/{cc} |".format(
                    stratum=f"{entry['stratum']} / {entry.get('severity', 'all')}",
                    metric=entry["metric"],
                    comparator=entry["comparator_variant"],
                    effect=_format_float(entry["paired_median_difference_comparator_minus_reference"]),
                    low=_format_float(ci[0]),
                    high=_format_float(ci[1]),
                    p=_format_p(entry.get("p_value")),
                    p_holm=_format_p(entry.get("p_value_holm")),
                    n=entry["n_pairs"],
                    rc=entry.get("reference_censored_count", 0),
                    cc=entry.get("comparator_censored_count", 0),
                )
            )
    else:
        lines.append("| — | No usable continuous outcomes | — | — | — | — | — |")

    exclusions = population.get("excluded_scenario_strata", {})
    fallback = population.get("fallback_runs_retained_by_variant", {})
    if exclusions or fallback or result.get("warnings"):
        lines.extend(["", "## Audit notes", ""])
        if exclusions:
            notes = ", ".join(f"{key}: {value}" for key, value in sorted(exclusions.items()))
            lines.append(f"- Excluded scenario strata: {notes}.")
        if fallback:
            notes = ", ".join(f"{key}: {value}" for key, value in sorted(fallback.items()))
            lines.append(f"- Retained fallback runs by variant: {notes}.")
        for warning in result.get("warnings", []):
            lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)


def write_analysis_outputs(
    result: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    """Write strict JSON and the compact Markdown table to caller-chosen paths."""

    output_json = Path(json_path)
    output_markdown = Path(markdown_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(result), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    output_markdown.write_text(render_markdown(result), encoding="utf-8", newline="\n")
    return output_json, output_markdown


def analyze_jsonl(
    episode_path: str | Path | Sequence[str | Path],
    *,
    reference_variant: str,
    comparator_variants: Sequence[str] | None = None,
    event_metrics: Sequence[str] | None = None,
    continuous_metrics: Sequence[str] | None = None,
    bootstrap_samples: int = 10_000,
    randomization_samples: int = 100_000,
    seed: int = 20_260_713,
    exclude_fallback: bool = False,
    stratify_by_severity: bool = False,
) -> dict[str, Any]:
    """Convenience wrapper for one or more JSONL episode files.

    Multiple files are merged only in memory.  The complete-pair audit rejects
    duplicate scenario/stratum/variant records, so a resumed experiment can be
    analysed without editing or concatenating its original episode artifacts.
    """

    if isinstance(episode_path, (str, Path)):
        input_paths = [Path(episode_path)]
    else:
        input_paths = [Path(path) for path in episode_path]
    if not input_paths:
        raise AnalysisInputError("At least one episode JSONL path is required")

    records: list[dict[str, Any]] = []
    for input_path in input_paths:
        records.extend(load_episode_jsonl(input_path))
    result = analyze_episode_records(
        records,
        reference_variant=reference_variant,
        comparator_variants=comparator_variants,
        event_metrics=event_metrics,
        continuous_metrics=continuous_metrics,
        bootstrap_samples=bootstrap_samples,
        randomization_samples=randomization_samples,
        seed=seed,
        exclude_fallback=exclude_fallback,
        stratify_by_severity=stratify_by_severity,
    )
    input_manifest = [
        {"episode_jsonl": str(input_path), "sha256": _sha256_file(input_path)}
        for input_path in input_paths
    ]
    result["input"] = (
        input_manifest[0]
        if len(input_manifest) == 1
        else {"episode_jsonls": input_manifest}
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze paired JSONL occlusion-ablation episode records."
    )
    parser.add_argument(
        "--episodes",
        action="append",
        required=True,
        help="Episode JSONL input path; repeat for an audited read-only merge.",
    )
    parser.add_argument("--reference-variant", required=True)
    parser.add_argument(
        "--comparator-variant",
        action="append",
        default=None,
        help="Comparator variant; repeat as needed. Defaults to all non-reference variants.",
    )
    parser.add_argument(
        "--event-metric",
        action="append",
        default=None,
        help="Dotted boolean event path; repeat to override default outcomes.",
    )
    parser.add_argument(
        "--continuous-metric",
        action="append",
        default=None,
        help="Dotted numeric path; repeat to override default outcomes.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--randomization-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20_260_713)
    parser.add_argument(
        "--exclude-fallback",
        action="store_true",
        help="Exclude a complete pair if any selected arm used a fallback (sensitivity analysis only).",
    )
    parser.add_argument(
        "--stratify-by-severity",
        action="store_true",
        help=(
            "Also emit predeclared design-severity rows from "
            "scenario_design.severity, while retaining pooled rows."
        ),
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = analyze_jsonl(
        args.episodes,
        reference_variant=args.reference_variant,
        comparator_variants=args.comparator_variant,
        event_metrics=args.event_metric,
        continuous_metrics=args.continuous_metric,
        bootstrap_samples=args.bootstrap_samples,
        randomization_samples=args.randomization_samples,
        seed=args.seed,
        exclude_fallback=args.exclude_fallback,
        stratify_by_severity=args.stratify_by_severity,
    )
    output_json, output_markdown = write_analysis_outputs(
        result,
        json_path=args.output_json,
        markdown_path=args.output_markdown,
    )
    print(f"Paired analysis JSON: {output_json}")
    print(f"Paired analysis Markdown: {output_markdown}")
    print(
        "Complete paired scenarios: "
        f"{result['population']['n_complete_scenario_pairs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
