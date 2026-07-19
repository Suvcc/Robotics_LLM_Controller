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


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8443, ge=1, le=65535)
    tls_certfile: str = ".certs/aliengo.pem"
    tls_keyfile: str = ".certs/aliengo-key.pem"
    lease_timeout_s: float = Field(default=300.0, gt=0)
    confirmation_timeout_s: float = Field(default=30.0, gt=0)
    session_idle_timeout_s: float = Field(default=86400.0, gt=0)
    max_command_chars: int = Field(default=1000, ge=1)
    max_audio_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_audio_duration_s: float = Field(default=15.0, gt=0)
    auth_attempts_per_minute: int = Field(default=5, ge=1)


class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    safety: SafetyConfig = SafetyConfig()
    logging: LoggingConfig = LoggingConfig()
    speech: SpeechConfig = SpeechConfig()
    robot: RobotConfig = RobotConfig()
    server: ServerConfig = ServerConfig()


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    p = Path(path)
    if not p.exists():
        return AppConfig()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(data)
