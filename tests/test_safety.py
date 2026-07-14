import dataclasses

import pytest

from aliengo.config import SafetyConfig
from aliengo.robot.interface import RobotState
from aliengo.safety.validator import Decision, SafetySession, validate
from aliengo.skills.definitions import Posture

LIMITS = SafetyConfig(max_cumulative_distance_m=6.0, max_tool_calls_per_command=8)


def standing_state() -> RobotState:
    return RobotState(posture=Posture.STANDING)


def test_valid_move_allowed():
    decision = validate(
        "move_forward", {"distance": 2}, standing_state(), SafetySession(), LIMITS
    )
    assert decision.decision is Decision.ALLOW
    assert decision.params == {"distance": 2.0}


def test_unknown_skill_blocked():
    decision = validate("fly", {}, standing_state(), SafetySession(), LIMITS)
    assert decision.decision is Decision.BLOCK
    assert "not an available skill" in decision.reason


def test_wrong_type_blocked():
    decision = validate(
        "move_forward", {"distance": "far"}, standing_state(), SafetySession(), LIMITS
    )
    assert decision.decision is Decision.BLOCK


def test_out_of_bounds_blocked_with_readable_reason():
    decision = validate(
        "move_forward", {"distance": 100}, standing_state(), SafetySession(), LIMITS
    )
    assert decision.decision is Decision.BLOCK
    assert "distance" in decision.reason


def test_posture_precondition_blocked():
    decision = validate(
        "move_forward", {"distance": 1}, RobotState(), SafetySession(), LIMITS
    )
    assert decision.decision is Decision.BLOCK
    assert "stand_up" in decision.reason


def test_cumulative_distance_limit():
    session = SafetySession(distance_moved_m=5.0)
    decision = validate(
        "move_forward", {"distance": 2}, standing_state(), session, LIMITS
    )
    assert decision.decision is Decision.BLOCK
    assert "6.0" in decision.reason


def test_tool_call_limit():
    session = SafetySession(tool_calls_made=8)
    decision = validate(
        "move_forward", {"distance": 1}, standing_state(), session, LIMITS
    )
    assert decision.decision is Decision.BLOCK


def test_stop_exempt_from_tool_call_limit():
    session = SafetySession(tool_calls_made=99)
    decision = validate("stop", {}, standing_state(), session, LIMITS)
    assert decision.decision is Decision.ALLOW


def test_estop_blocks_movement():
    session = SafetySession(estop_active=True)
    decision = validate(
        "move_forward", {"distance": 1}, standing_state(), session, LIMITS
    )
    assert decision.decision is Decision.BLOCK
    assert "mergency" in decision.reason


@pytest.mark.parametrize("skill", ["stop", "sit_down"])
def test_recovery_skills_allowed_under_estop(skill):
    session = SafetySession(estop_active=True)
    decision = validate(skill, {}, standing_state(), session, LIMITS)
    assert decision.decision is Decision.ALLOW


def test_follow_person_requires_confirmation():
    decision = validate("follow_person", {}, standing_state(), SafetySession(), LIMITS)
    assert decision.decision is Decision.CONFIRM
    assert decision.params == {}


def test_validator_is_pure():
    session = SafetySession(tool_calls_made=3, distance_moved_m=2.0)
    before = dataclasses.replace(session)
    state = standing_state()
    validate("move_forward", {"distance": 1}, state, session, LIMITS)
    assert session == before
    assert state == standing_state()
