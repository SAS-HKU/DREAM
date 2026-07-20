from dream_limo.mpc_benchmark import verify_report


def report(case_updates=None):
    case = {
        "fallbacks": 0,
        "all_finite": True,
        "statuses": ["optimal"],
        "maximum_cbf_slack": 0.0,
        "p99_ms": 80.0,
        "maximum_ms": 100.0,
    }
    case.update(case_updates or {})
    return {
        "dependencies": {"cvxpy_solvers": ["OSQP"]},
        "configuration": {
            "control_period_seconds": 0.2,
            "maximum_allowed_cbf_slack": 0.05,
        },
        "cases": {"balanced:test": case},
    }


def test_benchmark_acceptance_requires_finite_timely_nonfallback_osqp_solution():
    assert verify_report(report()) == []
    assert verify_report(report({"fallbacks": 1}))
    assert verify_report(report({"all_finite": False}))
    assert verify_report(report({"maximum_cbf_slack": 0.051}))
    assert verify_report(report({"p99_ms": 151.0}))
    assert verify_report(report({"maximum_ms": 200.0}))


def test_benchmark_rejects_missing_osqp():
    payload = report()
    payload["dependencies"]["cvxpy_solvers"] = ["SCS"]
    assert "CVXPY does not expose the OSQP solver" in verify_report(payload)
