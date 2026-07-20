from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_uses_verified_orbbec_rgb_only_bringup():
    readme = (ROOT / "README.md").read_text()

    assert "ros2 launch orbbec_camera dabai.launch.py" in readme
    assert "enable_point_cloud:=false" in readme
    assert "enable_colored_point_cloud:=false" in readme
    assert "enable_depth:=false" in readme
    assert "enable_ir:=false" in readme
    assert "enable_color:=true" in readme

    # The installed legacy launch silently ignores these arguments and starts
    # both point-cloud components, so it must never appear as an executable
    # command in the operator procedure.
    assert "ros2 launch astra_camera dabai.launch.py" not in readme
