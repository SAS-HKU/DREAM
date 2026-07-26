import pytest

from dream_limo.OACP.calibration_cli import (
    extract_calibration_samples,
    summarize_calibration,
)


def _status(*, count, risk, stamp, revision=2, active=True, pvs_length=1.0):
    return {
        "provider": "oacp_vb",
        "calibration_logging_only": True,
        "calibration_run_active": active,
        "ready": True,
        "exact_bound_valid": True,
        "calibration_goal_revision": revision,
        "calibration_goal_receipt_stamp": 100.0,
        "calibration_sample_count": count,
        "risk_total": risk,
        "pvs_length": pvs_length,
        "exploration_velocity_bound": 0.10,
        "fallback_velocity_bound": 0.12,
        "stamp": stamp,
    }


def test_calibration_extraction_deduplicates_status_and_filters_interval():
    records = [
        (101.0, _status(count=1, risk=1.0, stamp=101.0)),
        (101.1, _status(count=1, risk=9.0, stamp=101.1)),
        (102.0, _status(count=2, risk=2.0, stamp=102.0)),
        (103.0, _status(count=3, risk=3.0, stamp=103.0)),
        (104.0, _status(count=4, risk=4.0, stamp=104.0, active=False)),
    ]
    samples = extract_calibration_samples(
        records,
        start_offset=1.5,
        end_offset=3.5,
    )
    assert [sample.sample_count for sample in samples] == [2, 3]
    assert [sample.risk_total for sample in samples] == [2.0, 3.0]


def test_calibration_summary_uses_linear_p70_and_four_thirds_fallback():
    records = [
        (101.0 + index, _status(count=index + 1, risk=float(index), stamp=101.0 + index))
        for index in range(5)
    ]
    summary = summarize_calibration(extract_calibration_samples(records))
    assert summary["sample_count"] == 5
    assert summary["risk_p70_linear"] == pytest.approx(2.8)
    assert summary["suggested_c_th_max_fallback"] == pytest.approx(
        2.8 * 4.0 / 3.0
    )
    assert "not calibrated" in summary["approval"]


def test_calibration_extraction_rejects_mixed_goals_without_selection():
    records = [
        (101.0, _status(count=1, risk=1.0, stamp=101.0, revision=1)),
        (102.0, _status(count=1, risk=2.0, stamp=102.0, revision=2)),
    ]
    with pytest.raises(ValueError, match="multiple calibration goals"):
        extract_calibration_samples(records)


def test_calibration_summary_requires_occluded_samples():
    with pytest.raises(ValueError, match="no valid"):
        summarize_calibration([])


def test_calibration_summary_rejects_a_riskless_interval():
    records = [
        (101.0, _status(count=1, risk=0.0, stamp=101.0)),
        (102.0, _status(count=2, risk=0.0, stamp=102.0)),
    ]
    with pytest.raises(ValueError, match="no positive risk"):
        summarize_calibration(extract_calibration_samples(records))
