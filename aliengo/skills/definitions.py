"""Single source of truth for every robot skill the LLM may request.

Parameter bounds live here (Pydantic Field constraints) and nowhere else: the
OpenAI tool schemas, the safety validator, and the dispatch table are all
derived from these definitions.
"""
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Posture(str, Enum):
    SITTING = "sitting"
    STANDING = "standing"


class RiskLevel(str, Enum):
    SAFE = "safe"
    REQUIRES_CONFIRMATION = "requires_confirmation"


class NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MoveParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distance: float = Field(
        gt=0, le=3.0, description="Distance to move in meters, between 0 and 3."
    )


class TurnParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    angle: float = Field(
        gt=0, le=180, description="Angle to turn in degrees, between 0 and 180."
    )


class FindObjectParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(
        min_length=1,
        description="Name of the object to look for, e.g. 'bottle' or 'person'.",
    )


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    params_model: type[BaseModel]
    required_posture: Posture | None = None
    # Posture-changing skills are idempotent: when the robot is already in the
    # goal posture the call is a no-op success, never a block, so the LLM can
    # continue a multi-step task instead of treating it as a failure.
    goal_posture: Posture | None = None
    risk: RiskLevel = RiskLevel.SAFE
    # Recovery skills: usable while the emergency stop is active, and exempt
    # from per-command session limits so they can never be gated out.
    allowed_during_estop: bool = False


ALL_SKILLS: tuple[Skill, ...] = (
    Skill(
        name="stand_up",
        description=(
            "Make the robot stand up. Required before any movement or turning. "
            "Safe to call if already standing (does nothing)."
        ),
        params_model=NoParams,
        goal_posture=Posture.STANDING,
    ),
    Skill(
        name="sit_down",
        description=(
            "Make the robot sit down. Safe to call if already sitting "
            "(does nothing)."
        ),
        params_model=NoParams,
        goal_posture=Posture.SITTING,
        allowed_during_estop=True,
    ),
    Skill(
        name="move_forward",
        description=(
            "Walk forward in a straight line. Distance is in meters, maximum 3.0 "
            "per call. The robot must be standing."
        ),
        params_model=MoveParams,
        required_posture=Posture.STANDING,
    ),
    Skill(
        name="move_backward",
        description=(
            "Walk backward in a straight line. Distance is in meters, maximum 3.0 "
            "per call. The robot must be standing."
        ),
        params_model=MoveParams,
        required_posture=Posture.STANDING,
    ),
    Skill(
        name="turn_left",
        description=(
            "Turn left in place. Angle is in degrees, maximum 180 per call. "
            "The robot must be standing."
        ),
        params_model=TurnParams,
        required_posture=Posture.STANDING,
    ),
    Skill(
        name="turn_right",
        description=(
            "Turn right in place. Angle is in degrees, maximum 180 per call. "
            "The robot must be standing."
        ),
        params_model=TurnParams,
        required_posture=Posture.STANDING,
    ),
    Skill(
        name="stop",
        description=(
            "Immediately stop all current motion. Use this whenever the user "
            "says stop or something looks wrong."
        ),
        params_model=NoParams,
        allowed_during_estop=True,
    ),
    Skill(
        name="find_object",
        description=(
            "Scan with the camera for a named object and report whether it is "
            "visible and where (left, center, right)."
        ),
        params_model=FindObjectParams,
    ),
    Skill(
        name="detect_objects",
        description=(
            "Scan with the camera and list every object currently visible, "
            "with position (left, center, right) and confidence. Use this when "
            "asked what you can see or before searching for something."
        ),
        params_model=NoParams,
    ),
    Skill(
        name="follow_person",
        description=(
            "Start following the nearest visible person, walking to keep them in "
            "view. The robot must be standing. Continues until stop is called."
        ),
        params_model=NoParams,
        required_posture=Posture.STANDING,
        risk=RiskLevel.REQUIRES_CONFIRMATION,
    ),
)
