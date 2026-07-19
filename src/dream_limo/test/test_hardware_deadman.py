from dream_limo.core.hardware_deadman import evaluate_deadman_buttons


def decide(buttons):
    return evaluate_deadman_buttons(
        buttons, hold_button=4, confirm_button=5, stop_button=1
    )


def test_deadman_requires_two_button_chord():
    assert decide([0, 0, 0, 0, 1, 1]).armed
    partial = decide([0, 0, 0, 0, 1, 0])
    assert not partial.armed
    assert partial.reason == "DEADMAN_CHORD_INCOMPLETE"
    assert not decide([0, 0, 0, 0, 0, 0]).armed


def test_stop_button_overrides_arm_and_latches_request():
    result = decide([0, 1, 0, 0, 1, 1])
    assert result.valid
    assert result.external_stop
    assert not result.armed


def test_invalid_or_short_button_maps_fail_closed():
    assert not decide([0, 0]).valid
    duplicate = evaluate_deadman_buttons(
        [0, 0], hold_button=1, confirm_button=1, stop_button=0
    )
    assert not duplicate.valid
    assert not duplicate.armed
