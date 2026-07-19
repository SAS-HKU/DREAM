from dream_limo.visualization_node import world_visibility_label


def test_sfg_visibility_label_does_not_claim_ground_truth_reveal():
    source = "perception_only_no_merger_ground_truth"
    assert world_visibility_label(
        merger_visible=False, visibility_source=source, dynamic_track_count=0
    ) == "SHADOW / NO TRACK"
    assert world_visibility_label(
        merger_visible=False, visibility_source=source, dynamic_track_count=2
    ) == "TRACK OBSERVED (2)"


def test_odom_gate_visibility_label_uses_ground_truth_gate():
    assert world_visibility_label(
        merger_visible=False,
        visibility_source="merger_odom_gate",
        dynamic_track_count=0,
    ) == "OCCLUDED"
    assert world_visibility_label(
        merger_visible=True,
        visibility_source="merger_odom_gate",
        dynamic_track_count=1,
    ) == "VISIBLE"
