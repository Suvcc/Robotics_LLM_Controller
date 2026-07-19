from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    model: str = "qwen3:8b"
    base_url: str = "http://localhost:11434/v1"
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_iterations: int = Field(default=10, ge=1)
    # How many past user commands (with their tool exchanges) to keep in the
    # conversation; older ones are dropped so latency doesn't creep up.
    max_history_commands: int = Field(default=8, ge=1)
    # Prepend a one-line robot-state summary to each command so the model can
    # "see" posture/battery/motion instead of guessing (e.g. defensive stand_up).
    inject_state: bool = True


class SafetyConfig(BaseModel):
    max_cumulative_distance_m: float = Field(default=6.0, gt=0)
    max_tool_calls_per_command: int = Field(default=8, ge=1)


class LoggingConfig(BaseModel):
    actions_path: str | None = "logs/actions.jsonl"


class SpeechConfig(BaseModel):
    model_size: str = "base"
    duration_s: float = Field(default=5.0, gt=0)
    language: str | None = None  # e.g. "en"; None = Whisper auto-detect


class RobotConfig(BaseModel):
    # "mock" = stateful fake (default, no hardware); "jetracer" = real car.
    backend: Literal["mock", "jetracer"] = "mock"


class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    safety: SafetyConfig = SafetyConfig()
    logging: LoggingConfig = LoggingConfig()
    speech: SpeechConfig = SpeechConfig()
    robot: RobotConfig = RobotConfig()


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    p = Path(path)
    if not p.exists():
        return AppConfig()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(data)
