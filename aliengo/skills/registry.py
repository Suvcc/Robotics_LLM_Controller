from typing import Any

from .definitions import ALL_SKILLS, Skill

SKILLS: dict[str, Skill] = {skill.name: skill for skill in ALL_SKILLS}


def get_skill(name: str) -> Skill | None:
    return SKILLS.get(name)


def _strip_titles(schema: Any) -> Any:
    # Pydantic adds "title" keys the model doesn't need; smaller schemas are
    # easier for local models to follow.
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(item) for item in schema]
    return schema


def to_openai_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": skill.name,
                "description": skill.description,
                "parameters": _strip_titles(skill.params_model.model_json_schema()),
            },
        }
        for skill in ALL_SKILLS
    ]
