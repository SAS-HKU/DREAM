from dream_limo.camera_evidence_node import DreamCameraEvidenceNode


def test_camera_visibility_labels_do_not_invent_hidden_ground_truth():
    label = DreamCameraEvidenceNode._visibility_label
    assert label(merger_visible=True, track_count=0, shadow_cells=20) == (
        "REVEALED / ODOM GATE"
    )
    assert label(merger_visible=False, track_count=2, shadow_cells=20) == (
        "TRACK OBSERVED (2)"
    )
    assert label(merger_visible=False, track_count=0, shadow_cells=20) == (
        "SHADOW PRESENT / NO TRACK"
    )
    assert label(merger_visible=False, track_count=0, shadow_cells=0) == (
        "NO TRACK / NO SHADOW"
    )
