from glob import glob
from setuptools import find_packages, setup


package_name = "dream_limo"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (
            "share/" + package_name,
            [
                "package.xml",
                "README.md",
                "UPSTREAM_DREAM.md",
                "THIRD_PARTY_NOTICES.md",
                "requirements.txt",
            ],
        ),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        ("share/" + package_name + "/patches", glob("patches/*.patch")),
        (
            "share/" + package_name + "/benchmark_results",
            glob("benchmark_results/*.json"),
        ),
        ("share/" + package_name + "/scripts", ["scripts/run_stage1_replay.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="PeterWANGHK",
    maintainer_email="117706125+PeterWANGHK@users.noreply.github.com",
    description="ROS 2 Humble deployment of DREAM occlusion-aware planning for AgileX LIMO.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "dream_state_estimator = dream_limo.state_estimator_node:main",
            "dream_world_model = dream_limo.world_model_node:main",
            "dream_drift_field = dream_limo.drift_field_node:main",
            "dream_planner = dream_limo.planner_node:main",
            "dream_command_adapter = dream_limo.command_adapter_node:main",
            "dream_safety_supervisor = dream_limo.safety_supervisor_node:main",
            "dream_fake_world = dream_limo.fake_world_node:main",
            "dream_metrics = dream_limo.metrics_node:main",
            "dream_preflight = dream_limo.preflight_node:main",
            "dream_visualization = dream_limo.visualization_node:main",
            "dream_smoke_monitor = dream_limo.smoke_monitor_node:main",
            "dream_camera_evidence = dream_limo.camera_evidence_node:main",
            "dream_vehicle_tracker = dream_limo.vehicle_tracker_node:main",
            "dream_collision_monitor = dream_limo.collision_monitor_node:main",
            "dream_hardware_command_gate = dream_limo.hardware_command_gate_node:main",
            "dream_hardware_deadman = dream_limo.hardware_deadman_node:main",
            "dream_merger_odometry_adapter = dream_limo.merger_odometry_adapter_node:main",
            "dream_stage1_replay = dream_limo.stage1_cli:main",
            "dream_mpc_benchmark = dream_limo.mpc_benchmark:main",
        ],
    },
)
