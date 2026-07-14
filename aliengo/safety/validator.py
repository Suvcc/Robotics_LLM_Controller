"""Deterministic safety gate between the LLM and any robot controller.

`validate` is a pure function of its inputs: it never mutates state or the
session (the agent loop updates the session after execution). BLOCK reasons
are written for the LLM to read — they are fed back as tool results so the
model can recover.
"""
from dataclasses import dataclass
from enum import Enum

from pydantic import ValidationError

from ..config import SafetyConfig
from ..robot.interface import RobotState
from ..skills.registry import get_skill


class Decision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    CONFIRM = "confirm"


@dataclass
class SafetyDecision:
    decision: Decision
    reason: str = ""
    params: dict | None = None  # validated params when ALLOW / CONFIRM


@dataclass
class SafetySession:
    """Per-user-command counters, updated by the agent loop after execution."""

    tool_calls_made: int = 0
    distance_moved_m: float = 0.0
    estop_active: bool = False


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "parameters"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def validate(
    skill_name: str,
    raw_args: dict,
    state: RobotState,
    session: SafetySession,
    limits: SafetyConfig,
) -> SafetyDecision:
    skill = get_skill(skill_name)
    if skill is None:
        return SafetyDecision(
            Decision.BLOCK,
            f"'{skill_name}' is not an available skill. Use only the provided tools.",
        )

    try:
        params = skill.params_model.model_validate(raw_args or {})
    except ValidationError as exc:
        return SafetyDecision(
            Decision.BLOCK,
            f"Invalid parameters for {skill_name}: {_format_validation_error(exc)}",
        )
    validated = params.model_dump()

    if skill.required_posture and state.posture != skill.required_posture:
        return SafetyDecision(
            Decision.BLOCK,
            f"Robot is {state.posture.value} but {skill_name} requires it to be "
            f"{skill.required_posture.value}."
            + (
                " Call stand_up first."
                if skill.required_posture.value == "standing"
                else ""
            ),
        )

    # Recovery skills (stop, sit_down) are exempt from session limits so they
    # can never be gated out.
    if not skill.allowed_during_estop:
        if session.tool_calls_made >= limits.max_tool_calls_per_command:
            return SafetyDecision(
                Decision.BLOCK,
                f"Limit of {limits.max_tool_calls_per_command} actions per command "
                "reached. Stop and report to the user.",
            )
        requested_distance = validated.get("distance", 0.0)
        if (
            session.distance_moved_m + requested_distance
            > limits.max_cumulative_distance_m
        ):
            return SafetyDecision(
                Decision.BLOCK,
                f"Moving {requested_distance} m would exceed the "
                f"{limits.max_cumulative_distance_m} m total movement limit for one "
                f"command (already moved {session.distance_moved_m} m). "
                "Ask the user before continuing.",
            )

    if session.estop_active and not skill.allowed_during_estop:
        return SafetyDecision(
            Decision.BLOCK,
            "Emergency stop is active. Only stop and sit_down are allowed until "
            "the operator releases it.",
        )

    if skill.risk.value == "requires_confirmation":
        return SafetyDecision(
            Decision.CONFIRM,
            f"{skill_name} requires operator confirmation before it runs.",
            params=validated,
        )

    return SafetyDecision(Decision.ALLOW, params=validated)
