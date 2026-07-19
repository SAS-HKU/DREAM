from pathlib import Path


def test_dream_has_no_sfg_python_imports_or_real_cmd_vel_publisher():
    root = Path(__file__).resolve().parents[1]
    python_sources = list((root / "dream_limo").glob("**/*.py"))
    combined = "\n".join(path.read_text() for path in python_sources)
    assert "from sfg_nav" not in combined
    assert "import sfg_nav" not in combined
    assert 'create_publisher(Twist, "/cmd_vel"' not in combined
    supervisor = (root / "dream_limo" / "safety_supervisor_node.py").read_text()
    assert 'OUTPUT_TOPIC = "/cmd_vel_test"' in supervisor
