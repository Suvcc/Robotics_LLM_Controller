import json

import pytest

from aliengo.actionlog import ActionLog
from aliengo.robot.mock import MockRobotController
from aliengo.skills.definitions import Posture


@pytest.fixture
def robot():
    return MockRobotController()


@pytest.fixture
def standing(robot):
    robot.stand_up()
    return robot


def test_initial_state(robot):
    state = robot.get_state()
    assert state.posture is Posture.SITTING
    assert state.battery_pct == 100.0
    assert (state.x, state.y, state.heading_deg) == (0.0, 0.0, 0.0)


def test_stand_up_is_idempotent(robot):
    assert robot.stand_up().success
    assert robot.get_state().posture is Posture.STANDING
    battery = robot.get_state().battery_pct
    again = robot.stand_up()
    assert again.success  # no-op, not a failure
    assert "lready standing" in again.data["note"]
    assert robot.get_state().battery_pct == battery  # no-op costs nothing


def test_sit_down_is_idempotent(robot):
    result = robot.sit_down()
    assert result.success
    assert "lready sitting" in result.data["note"]
    assert robot.get_state().posture is Posture.SITTING


def test_move_while_sitting_fails(robot):
    result = robot.move_forward(distance=1.0)
    assert not result.success
    assert "sitting" in result.error


def test_pose_math(standing):
    standing.move_forward(distance=2.0)
    state = standing.get_state()
    assert state.x == pytest.approx(2.0)
    assert state.y == pytest.approx(0.0)

    standing.turn_left(angle=90)
    standing.move_forward(distance=1.0)
    state = standing.get_state()
    assert state.x == pytest.approx(2.0)
    assert state.y == pytest.approx(1.0)
    assert state.heading_deg == 90

    standing.move_backward(distance=1.0)
    assert standing.get_state().y == pytest.approx(0.0)

    standing.turn_right(angle=90)
    assert standing.get_state().heading_deg == 0


def test_battery_drains(standing):
    before = standing.get_state().battery_pct
    standing.move_forward(distance=2.0)
    assert standing.get_state().battery_pct < before


def test_low_battery_blocks_movement(standing):
    standing.state.battery_pct = 2.0
    result = standing.move_forward(distance=1.0)
    assert not result.success
    assert "attery" in result.error


def test_injected_failure_is_one_shot(standing):
    standing.inject_failure("move_forward", "motor stall")
    first = standing.move_forward(distance=1.0)
    assert not first.success
    assert first.error == "motor stall"
    assert standing.move_forward(distance=1.0).success


def test_find_object(robot):
    found = robot.find_object(label="bottle")
    assert found.success
    assert found.data["found"] is True
    assert found.data["position"] == "left"

    missing = robot.find_object(label="elephant")
    assert missing.success  # a scan that finds nothing did not fail
    assert missing.data["found"] is False


def test_detect_objects(robot):
    result = robot.detect_objects()
    assert result.success
    labels = [o["label"] for o in result.data["objects"]]
    assert "bottle" in labels and "person" in labels


def test_reset(standing):
    standing.move_forward(distance=2.0)
    standing.inject_failure("stop", "x")
    standing.set_visible_objects([])
    standing.reset()
    state = standing.get_state()
    assert state.posture is Posture.SITTING
    assert (state.x, state.y) == (0.0, 0.0)
    assert state.battery_pct == 100.0
    assert standing.visible_objects  # defaults restored
    assert standing.stop().success  # injected failure cleared


def test_follow_person_and_stop(standing):
    result = standing.follow_person()
    assert result.success
    assert standing.get_state().following is True
    standing.stop()
    state = standing.get_state()
    assert state.following is False and state.moving is False


def test_follow_object(standing):
    result = standing.follow_object("bottle")
    assert result.success
    assert result.data["target"]["label"] == "bottle"
    assert standing.get_state().following is True


def test_follow_object_not_visible(standing):
    result = standing.follow_object("elephant")
    assert not result.success
    assert "elephant" in result.error
    assert standing.get_state().following is False


def test_emergency_stop_clears_following(standing):
    standing.follow_object("bottle")
    standing.emergency_stop()
    state = standing.get_state()
    assert state.following is False and state.moving is False


def test_follow_person_needs_visible_person(standing):
    standing.set_visible_objects([])
    result = standing.follow_person()
    assert not result.success
    assert "person" in result.error.lower()


def test_actions_logged_as_jsonl(tmp_path):
    log_path = tmp_path / "actions.jsonl"
    robot = MockRobotController(log=ActionLog(log_path))
    robot.stand_up()
    robot.move_forward(distance=1.0)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[1])
    assert entry["skill"] == "move_forward"
    assert entry["result"]["success"] is True
    assert entry["state"]["posture"] == "standing"
