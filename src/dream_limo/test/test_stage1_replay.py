from dream_limo.core.replay import run_stage1


def test_stage1_replay_gate():
    results = run_stage1()
    dream = results["balanced"].metrics
    baseline = results["pure_mpc"].metrics
    assert dream.veto_activations > 0
    assert dream.hidden_track_leaks == 0
    assert baseline.hidden_track_leaks == 0
    assert baseline.conflict_zone_overlap_samples > 0
    assert dream.conflict_zone_overlap_samples < baseline.conflict_zone_overlap_samples
    assert dream.predicted_conflict_arrival_margin_at_reveal > (
        baseline.predicted_conflict_arrival_margin_at_reveal
    )
    assert dream.minimum_clearance > baseline.minimum_clearance > 0.0
