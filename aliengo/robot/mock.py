"""Stateful fake robot.

Honest even when bypassed: preconditions and battery are enforced here too,
not only in the safety layer, so a bug above this layer cannot produce an
impossible state trajectory.
"""
import math
from copy import deepcopy

from ..actionlog import ActionLog
from ..skills.definitions import Posture
from .interface import RobotState
from .result import SkillResult

LOW_BATTERY_PCT = 5.0

# Battery cost per action; moves and turns scale with magnitude.
COST_STAND = 1.0
COST_SIT = 0.5
COST_PER_METER = 1.5
COST_PER_DEGREE = 0.01
COST_SCAN = 0.5
COST_FOLLOW = 3.0

DEFAULT_VISIBLE_OBJECTS = [
    {"label": "bottle", "confidence": 0.94, "position": "left"},
    {"label": "person", "confidence": 0.91, "position": "center"},
]


class MockRobotController:
    def __init__(self, log: ActionLog | None = None):
        self.state = RobotState()
        self.log = log or ActionLog(None)
        self.visible_objects = deepcopy(DEFAULT_VISIBLE_OBJECTS)
        self._injected_failures: dict[str, str] = {}

    # -- test/demo hooks ---------------------------------------------------

    def inject_failure(self, skill_name: str, error: str) -> None:
        """Make the next call to `skill_name` fail with `error` (one-shot)."""
        self._injected_failures[skill_name] = error

    def set_visible_objects(self, objects: list[dict]) -> None:
        self.visible_objects = objects

    def reset(self) -> None:
        self.state = RobotState()
        self.visible_objects = deepcopy(DEFAULT_VISIBLE_OBJECTS)
        self._injected_failures.clear()

    # -- skills --------------------------------------------------------------

    def stand_up(self) -> SkillResult:
        # Idempotent: already standing means nothing to do, not a failure.
        if self.state.posture is Posture.STANDING:
            return self._finish(
                "stand_up", {},
                data={"note": "Already standing; no action needed. Continue with the task."},
            )
        err = self._precheck("stand_up", None, COST_STAND)
        if err:
            return self._finish("stand_up", {}, error=err)
        self.state.posture = Posture.STANDING
        self._drain(COST_STAND)
        return self._finish("stand_up", {})

    def sit_down(self) -> SkillResult:
        if self.state.posture is Posture.SITTING:
            return self._finish(
                "sit_down", {},
                data={"note": "Already sitting; no action needed. Continue with the task."},
            )
        err = self._precheck("sit_down", None, COST_SIT)
        if err:
            return self._finish("sit_down", {}, error=err)
        self.state.posture = Posture.SITTING
        self.state.moving = False
        self.state.following = False
        self._drain(COST_SIT)
        return self._finish("sit_down", {})

    def move_forward(self, distance: float) -> SkillResult:
        return self._move("move_forward", distance, direction=1)

    def move_backward(self, distance: float) -> SkillResult:
        return self._move("move_backward", distance, direction=-1)

    def turn_left(self, angle: float) -> SkillResult:
        return self._turn("turn_left", angle, sign=1)

    def turn_right(self, angle: float) -> SkillResult:
        return self._turn("turn_right", angle, sign=-1)

    def stop(self) -> SkillResult:
        err = self._injected_failures.pop("stop", None)
        if err:
            return self._finish("stop", {}, error=err)
        self.state.moving = False
        self.state.following = False
        return self._finish("stop", {})

    def find_object(self, label: str) -> SkillResult:
        err = self._precheck("find_object", None, COST_SCAN)
        if err:
            return self._finish("find_object", {"label": label}, error=err)
        self._drain(COST_SCAN)
        match = next(
            (o for o in self.visible_objects if o["label"].lower() == label.lower()),
            None,
        )
        data = {"found": match is not None, "label": label}
        if match:
            data.update(position=match["position"], confidence=match["confidence"])
        return self._finish("find_object", {"label": label}, data=data)

    def detect_objects(self) -> SkillResult:
        err = self._precheck("detect_objects", None, COST_SCAN)
        if err:
            return self._finish("detect_objects", {}, error=err)
        self._drain(COST_SCAN)
        return self._finish(
            "detect_objects", {}, data={"objects": deepcopy(self.visible_objects)}
        )

    def follow_person(self) -> SkillResult:
        err = self._precheck("follow_person", Posture.STANDING, COST_FOLLOW)
        if err:
            return self._finish("follow_person", {}, error=err)
        person = next(
            (o for o in self.visible_objects if o["label"] == "person"), None
        )
        if person is None:
            return self._finish("follow_person", {}, error="No person is visible.")
        self.state.moving = True
        self.state.following = True
        self._drain(COST_FOLLOW)
        return self._finish("follow_person", {}, data={"target": person})

    def follow_object(self, label: str) -> SkillResult:
        err = self._precheck("follow_object", Posture.STANDING, COST_FOLLOW)
        if err:
            return self._finish("follow_object", {"label": label}, error=err)
        target = next(
            (o for o in self.visible_objects if o["label"].lower() == label.lower()),
            None,
        )
        if target is None:
            return self._finish(
                "follow_object", {"label": label}, error=f"No {label} is visible."
            )
        self.state.moving = True
        self.state.following = True
        self._drain(COST_FOLLOW)
        return self._finish("follow_object", {"label": label}, data={"target": target})

    def emergency_stop(self) -> None:
        self.state.moving = False
        self.state.following = False

    def get_state(self) -> RobotState:
        return self.state

    # -- internals -----------------------------------------------------------

    def _move(self, action: str, distance: float, direction: int) -> SkillResult:
        cost = COST_PER_METER * distance
        err = self._precheck(action, Posture.STANDING, cost)
        if err:
            return self._finish(action, {"distance": distance}, error=err)
        heading_rad = math.radians(self.state.heading_deg)
        self.state.x += direction * distance * math.cos(heading_rad)
        self.state.y += direction * distance * math.sin(heading_rad)
        self._drain(cost)
        return self._finish(action, {"distance": distance})

    def _turn(self, action: str, angle: float, sign: int) -> SkillResult:
        cost = COST_PER_DEGREE * angle
        err = self._precheck(action, Posture.STANDING, cost)
        if err:
            return self._finish(action, {"angle": angle}, error=err)
        self.state.heading_deg = (self.state.heading_deg + sign * angle) % 360
        self._drain(cost)
        return self._finish(action, {"angle": angle})

    def _precheck(
        self, action: str, required_posture: Posture | None, cost: float
    ) -> str | None:
        injected = self._injected_failures.pop(action, None)
        if injected:
            return injected
        if required_posture and self.state.posture != required_posture:
            return (
                f"Cannot {action}: robot is {self.state.posture.value}, "
                f"must be {required_posture.value}."
            )
        if self.state.battery_pct < max(LOW_BATTERY_PCT, cost):
            return (
                f"Battery too low ({self.state.battery_pct:.1f}%) to {action}."
            )
        return None

    def _drain(self, cost: float) -> None:
        self.state.battery_pct = max(0.0, self.state.battery_pct - cost)

    def _finish(
        self, action: str, params: dict, data: dict | None = None, error: str | None = None
    ) -> SkillResult:
        result = SkillResult(
            success=error is None, action=action, data=data, error=error
        )
        self.log.write(
            type="skill_execution",
            skill=action,
            params=params,
            result=result.to_dict(),
            state=self.state.to_dict(),
        )
        return result
