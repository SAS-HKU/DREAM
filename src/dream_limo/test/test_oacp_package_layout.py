from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OACP = ROOT / "dream_limo" / "OACP"


def test_oacp_reproducibility_artifacts_have_one_canonical_index():
    expected = {
        "README.md",
        "DESIGN_INPUT.md",
        "__init__.py",
        "oacp_vb.py",
        "assessor_node.py",
        "calibration_cli.py",
    }
    assert expected.issubset({path.name for path in OACP.iterdir()})
    assert (
        OACP
        / "results"
        / "oacp_vb_contingency_nuc12_2026-07-25.json"
    ).is_file()
    assert not (ROOT / "OACP_VB.md").exists()
    assert not (
        ROOT
        / "benchmark_results"
        / "oacp_vb_contingency_nuc12_2026-07-25.json"
    ).exists()


def test_shared_dream_runtime_is_not_relabelled_as_oacp():
    assert (ROOT / "dream_limo" / "core" / "mpc.py").is_file()
    assert (ROOT / "dream_limo" / "free_planner_node.py").is_file()
    assert not (OACP / "mpc.py").exists()
    assert not (OACP / "free_planner_node.py").exists()


def test_old_python_paths_are_explicit_compatibility_wrappers():
    wrappers = [
        (
            ROOT / "dream_limo" / "core" / "oacp_vb.py",
            "dream_limo.OACP.oacp_vb",
        ),
        (
            ROOT / "dream_limo" / "oacp_vb_node.py",
            "dream_limo.OACP.assessor_node",
        ),
        (
            ROOT / "dream_limo" / "oacp_calibration_cli.py",
            "dream_limo.OACP.calibration_cli",
        ),
    ]
    for path, canonical_module in wrappers:
        source = path.read_text(encoding="utf-8")
        assert "Compatibility" in source
        assert canonical_module in source


def test_setup_installs_oacp_package_docs_results_and_entry_points():
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert (ROOT / "resource" / "dream_limo").is_file()
    assert '"dream_limo.OACP"' in setup
    assert 'glob("dream_limo/OACP/*.md")' in setup
    assert 'glob("dream_limo/OACP/results/*.json")' in setup
    assert "dream_limo.OACP.assessor_node:main" in setup
    assert "dream_limo.OACP.calibration_cli:main" in setup
