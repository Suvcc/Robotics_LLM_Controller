"""The controller interface every robot backend implements.

This is the ROS seam: Phase 11 replaces MockRobotController with a ROS-backed
implementation of this same Protocol, and nothing above this layer changes.
Controller method names mirror skill names exactly, which is what makes
`execute_skill` a complete dispatch table.
"""
from dataclasses import dataclass, field
from typing import Protocol

from ..skills.definitions import Posture
from .result import SkillResult


@dataclass
class RobotState:
    posture: Posture = Posture.SITTING
    moving: bool = False
    following: bool = False
    x: float = 0.0
    y: float = 0.0
    heading_deg: float = 0.0
    battery_pct: float = field(default=100.0)

    def to_dict(self) -> dict:
        return {
            "posture": self.posture.value,
            "moving": self.moving,
            "following": self.following,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "heading_deg": round(self.heading_deg, 1),
            "battery_pct": round(self.battery_pct, 1),
        }


class RobotController(Protocol):
    def stand_up(self) -> SkillResult: ...
    def sit_down(self) -> SkillResult: ...
    def move_forward(self, distance: float) -> SkillResult: ...
    def move_backward(self, distance: float) -> SkillResult: ...
    def turn_left(self, angle: float) -> SkillResult: ...
    def turn_right(self, angle: float) -> SkillResult: ...
    def stop(self) -> SkillResult: ...
    def find_object(self, label: str) -> SkillResult: ...
    def follow_person(self) -> SkillResult: ...
    def get_state(self) -> RobotState: ...


def execute_skill(
    controller: RobotController, skill_name: str, params: dict
) -> SkillResult:
    method = getattr(controller, skill_name)
    return method(**params)
