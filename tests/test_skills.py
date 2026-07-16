import pytest
from pydantic import ValidationError

from aliengo.skills.definitions import (
    ALL_SKILLS,
    FindObjectParams,
    MoveParams,
    TurnParams,
)
from aliengo.skills.registry import get_skill, to_openai_tools

EXPECTED_SKILLS = {
    "stand_up", "sit_down", "move_forward", "move_backward",
    "turn_left", "turn_right", "stop", "find_object", "detect_objects",
    "follow_person",
}


def test_all_skills_registered():
    assert {s.name for s in ALL_SKILLS} == EXPECTED_SKILLS


def test_openai_tools_shape():
    tools = to_openai_tools()
    assert len(tools) == len(EXPECTED_SKILLS)
    for tool in tools:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert fn["name"] in EXPECTED_SKILLS
        assert fn["description"]
        assert fn["parameters"]["type"] == "object"


def test_schema_carries_bounds_from_single_source():
    tools = {t["function"]["name"]: t["function"] for t in to_openai_tools()}
    distance = tools["move_forward"]["parameters"]["properties"]["distance"]
    assert distance["maximum"] == 3.0
    assert distance["exclusiveMinimum"] == 0
    angle = tools["turn_right"]["parameters"]["properties"]["angle"]
    assert angle["maximum"] == 180


@pytest.mark.parametrize("distance", [0, -1, 3.5, 100])
def test_move_params_rejects_out_of_bounds(distance):
    with pytest.raises(ValidationError):
        MoveParams(distance=distance)


def test_move_params_accepts_valid():
    assert MoveParams(distance=2.0).distance == 2.0


@pytest.mark.parametrize("angle", [0, -10, 181, 720])
def test_turn_params_rejects_out_of_bounds(angle):
    with pytest.raises(ValidationError):
        TurnParams(angle=angle)


def test_find_object_rejects_empty_label():
    with pytest.raises(ValidationError):
        FindObjectParams(label="")


def test_extra_parameters_rejected():
    with pytest.raises(ValidationError):
        MoveParams(distance=1.0, speed=99)


def test_unknown_skill_lookup():
    assert get_skill("fly") is None
    assert get_skill("move_forward") is not None
