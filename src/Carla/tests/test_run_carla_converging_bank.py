import json
import sys
from pathlib import Path

from run_carla_converging_bank import (
    ARMS,
    execute_bank,
    freeze_bank,
    randomized_arm_order,
)


TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "carla_converging_overtake_manifest.json"
)


def test_freeze_bank_materializes_complete_matched_blocks(tmp_path):
    bank_dir = tmp_path / "bank"
    index = freeze_bank(TEMPLATE, bank_dir, [101, 102], randomization_seed=77)

    assert index["scene_count"] == 2
    assert index["planned_run_count"] == 8
    assert (bank_dir / "bank_index.json").is_file()
    for scene in index["scenes"]:
        manifest_path = bank_dir / scene["manifest_path"]
        assert manifest_path.is_file()
        assert len(scene["arm_order"]) == 4
        assert {
            (item["controller"], item["condition"])
            for item in scene["arm_order"]
        } == set(ARMS)
        # Every arm references this one frozen path; condition is not embedded
        # in the physical manifest.
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "condition" not in manifest
        assert manifest["scene_construction"]["construction_hash_sha256"] == scene[
            "construction_hash_sha256"
        ]

    repeated = freeze_bank(TEMPLATE, bank_dir, [101, 102], randomization_seed=77)
    assert repeated == index


def test_randomized_order_is_deterministic_and_complete():
    first = randomized_arm_order(101, 44)
    assert first == randomized_arm_order(101, 44)
    assert {(item["controller"], item["condition"]) for item in first} == set(ARMS)
    orders = {
        tuple((item["controller"], item["condition"]) for item in randomized_arm_order(seed, 44))
        for seed in range(101, 111)
    }
    assert len(orders) > 1


def test_execute_bank_logs_each_failure_and_preserves_manifest_digest(tmp_path):
    bank_dir = tmp_path / "bank"
    freeze_bank(TEMPLATE, bank_dir, [111], randomization_seed=9)
    fake_trial = tmp_path / "fake_trial.py"
    fake_trial.write_text(
        """
import sys

def value(flag):
    return sys.argv[sys.argv.index(flag) + 1]

controller = value('--controller')
condition = value('--condition')
print(controller, condition, value('--manifest'))
if controller == 'DREAM' and condition == 'empty_shadow':
    raise SystemExit(7)
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = execute_bank(
        bank_dir / "bank_index.json",
        trial_python=sys.executable,
        trial_script=fake_trial,
        output_root=tmp_path / "runs",
        pace_realtime=False,
    )

    assert result["attempted"] == 4
    assert result["succeeded"] == 3
    assert result["failed"] == 1
    ledger_path = Path(result["ledger_path"])
    records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert {record["status"] for record in records} == {"success", "failed"}
    assert next(record for record in records if record["status"] == "failed")[
        "return_code"
    ] == 7
    assert len({record["manifest_path"] for record in records}) == 1
    assert len({record["manifest_file_sha256"] for record in records}) == 1
    assert len({record["construction_hash_sha256"] for record in records}) == 1
    # An externally managed CARLA server remains on one endpoint across the
    # sequential four-arm block.  Planner services still receive distinct
    # ports because each trial launches its own external planner process.
    assert {record["carla_port"] for record in records} == {2057}
    assert len({record["planner_port"] for record in records}) == 4
    assert all("--launch-server" not in record["command"] for record in records)
