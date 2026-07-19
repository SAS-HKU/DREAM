"""Pure two-button held-to-run logic for the physical LIMO gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DeadmanDecision:
    """One fail-closed interpretation of a joystick button sample."""

    armed: bool
    external_stop: bool
    valid: bool
    reason: str


def evaluate_deadman_buttons(
    buttons: Sequence[int],
    *,
    hold_button: int,
    confirm_button: int,
    stop_button: int,
) -> DeadmanDecision:
    """Require a two-button chord and reserve a distinct latched-stop button.

    Button indices are deliberately configuration, not guessed controller
    semantics.  A missing/short Joy message fails closed without asserting the
    latched external stop; pressing the configured stop button asserts it.
    """

    indices = (int(hold_button), int(confirm_button), int(stop_button))
    if any(index < 0 for index in indices) or len(set(indices)) != len(indices):
        return DeadmanDecision(False, False, False, "INVALID_BUTTON_MAP")
    if not isinstance(buttons, Sequence) or max(indices) >= len(buttons):
        return DeadmanDecision(False, False, False, "JOY_BUTTONS_MISSING")
    try:
        pressed = tuple(bool(int(buttons[index])) for index in indices)
    except (TypeError, ValueError, OverflowError):
        return DeadmanDecision(False, False, False, "INVALID_JOY_BUTTON_VALUE")
    hold, confirm, stop = pressed
    if stop:
        return DeadmanDecision(False, True, True, "EXTERNAL_STOP_REQUESTED")
    if hold and confirm:
        return DeadmanDecision(True, False, True, "ok")
    if hold or confirm:
        return DeadmanDecision(False, False, True, "DEADMAN_CHORD_INCOMPLETE")
    return DeadmanDecision(False, False, True, "DEADMAN_RELEASED")
