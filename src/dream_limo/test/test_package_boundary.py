from pathlib import Path


def test_dream_has_no_sfg_python_imports_and_one_reviewed_cmd_vel_publisher():
    root = Path(__file__).resolve().parents[1]
    python_sources = list((root / "dream_limo").glob("**/*.py"))
    combined = "\n".join(path.read_text() for path in python_sources)
    assert "from sfg_nav" not in combined
    assert "import sfg_nav" not in combined
    real_output_nodes = [
        path.name
        for path in python_sources
        if 'OUTPUT_TOPIC = "/cmd_vel"' in path.read_text()
    ]
    assert real_output_nodes == ["hardware_command_gate_node.py"]
    supervisor = (root / "dream_limo" / "safety_supervisor_node.py").read_text()
    assert 'OUTPUT_TOPIC = "/cmd_vel_test"' in supervisor
    gate = (root / "dream_limo" / "hardware_command_gate_node.py").read_text()
    assert 'self.declare_parameter("hardware_output_enabled", False)' in gate
    assert 'self.declare_parameter("staging_pose_verified", False)' in gate
    assert 'self.declare_parameter("platform_watchdog_verified", False)' in gate
    assert 'self.declare_parameter("operator_kill_verified", False)' in gate
    assert 'OUTPUT_TOPIC = "/cmd_vel"' in gate
