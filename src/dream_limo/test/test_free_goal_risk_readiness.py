import pytest

from dream_limo.free_goal_authorizer_node import (
    evaluate_required_risk_assessment,
)


NOW = 100.0
TIMEOUT = 0.5


def _evaluate(
    *,
    required_provider="oacp_vb",
    payload=None,
    receipt_stamp=99.8,
    shared_minimum_speed=0.0,
    shared_target_speed=0.15,
):
    return evaluate_required_risk_assessment(
        required_provider,
        payload,
        receipt_stamp,
        now=NOW,
        timeout=TIMEOUT,
        shared_minimum_speed=shared_minimum_speed,
        shared_target_speed=shared_target_speed,
    )


def _ready_payload(**overrides):
    payload = {
        "provider": "oacp_vb",
        "assessment_ready": True,
        "pre_goal_bound_valid": True,
        "v_occ_min": 0.08,
        "v_occ_max": 0.15,
        "pre_goal_velocity_bound": 0.08,
    }
    payload.update(overrides)
    return payload


def test_default_empty_provider_preserves_existing_goal_authorization():
    readiness = _evaluate(
        required_provider="",
        payload=None,
        receipt_stamp=None,
    )

    assert not readiness.required
    assert readiness.ready
    assert readiness.reason == "RISK_ASSESSMENT_NOT_REQUIRED"
    assert readiness.provider is None
    assert readiness.age is None


@pytest.mark.parametrize(
    ("payload", "receipt_stamp"),
    (
        (None, None),
        (_ready_payload(), None),
        ("not-a-mapping", 99.9),
    ),
)
def test_missing_or_malformed_required_status_fails_closed(payload, receipt_stamp):
    readiness = _evaluate(payload=payload, receipt_stamp=receipt_stamp)

    assert readiness.required
    assert not readiness.ready
    assert readiness.reason == "OACP_ASSESSMENT_UNAVAILABLE"


@pytest.mark.parametrize("receipt_stamp", [99.5, 100.1, float("nan")])
def test_stale_future_or_nonfinite_required_status_fails_closed(receipt_stamp):
    readiness = _evaluate(
        payload=_ready_payload(),
        receipt_stamp=receipt_stamp,
    )

    assert not readiness.ready
    assert readiness.reason == "OACP_ASSESSMENT_STALE"


def test_wrong_provider_fails_closed():
    readiness = _evaluate(
        payload=_ready_payload(provider="dream"),
    )

    assert not readiness.ready
    assert readiness.reason == "OACP_ASSESSMENT_PROVIDER_MISMATCH"
    assert readiness.provider == "dream"


@pytest.mark.parametrize(
    ("payload", "reason"),
    (
        (
            _ready_payload(assessment_ready=False),
            "OACP_ASSESSMENT_NOT_READY",
        ),
        (
            _ready_payload(assessment_ready=1),
            "OACP_ASSESSMENT_NOT_READY",
        ),
        (
            _ready_payload(pre_goal_bound_valid=False),
            "OACP_PRE_GOAL_BOUND_INVALID",
        ),
        (
            _ready_payload(pre_goal_bound_valid=1),
            "OACP_PRE_GOAL_BOUND_INVALID",
        ),
    ),
)
def test_explicit_boolean_readiness_and_bound_are_required(payload, reason):
    readiness = _evaluate(payload=payload)

    assert not readiness.ready
    assert readiness.reason == reason


def test_fresh_matching_explicit_status_allows_candidate_goal_phase():
    readiness = _evaluate(payload=_ready_payload())

    assert readiness.required
    assert readiness.ready
    assert readiness.reason == "OACP_ASSESSMENT_READY"
    assert readiness.provider == "oacp_vb"
    assert readiness.age == pytest.approx(0.2)


@pytest.mark.parametrize(
    "overrides",
    (
        {"pre_goal_velocity_bound": float("nan")},
        {"pre_goal_velocity_bound": 0.20},
        {"v_occ_min": 0.16},
        {"v_occ_max": "invalid"},
    ),
)
def test_pre_goal_bound_must_be_finite_and_inside_provider_limits(overrides):
    readiness = _evaluate(payload=_ready_payload(**overrides))
    assert not readiness.ready
    assert readiness.reason == "OACP_PRE_GOAL_BOUND_INVALID"


def test_pre_goal_bound_maximum_must_equal_shared_target_speed():
    readiness = _evaluate(
        payload=_ready_payload(v_occ_max=0.16),
        shared_target_speed=0.15,
    )
    assert not readiness.ready
    assert readiness.reason == "OACP_PRE_GOAL_BOUND_INVALID"


@pytest.mark.parametrize(
    ("shared_minimum_speed", "shared_target_speed"),
    (
        (float("nan"), 0.15),
        (-0.01, 0.15),
        (0.16, 0.15),
    ),
)
def test_invalid_shared_speed_contract_fails_closed(
    shared_minimum_speed, shared_target_speed
):
    readiness = _evaluate(
        payload=_ready_payload(),
        shared_minimum_speed=shared_minimum_speed,
        shared_target_speed=shared_target_speed,
    )
    assert not readiness.ready
    assert readiness.reason == "OACP_ASSESSMENT_CONFIG_INVALID"
