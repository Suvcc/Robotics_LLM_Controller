# AlienGo Local LLM System

Local intelligence stack for the AlienGo robot dog, built mock-first: a local
LLM (via Ollama) selects approved robot skills, every call passes a
deterministic safety layer, and a stateful mock robot executes them. The
`RobotController` interface in [aliengo/robot/interface.py](aliengo/robot/interface.py)
is the seam where ROS/Unitree SDK plugs in later — nothing above it changes.

## Setup

```powershell
# 1. Python env (from the project root)
py -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"

# 2. Run the offline test suite (no GPU needed)
.venv\Scripts\python -m pytest

# 3. Install Ollama (https://ollama.com/download), then:
ollama pull qwen3:8b

# 4. Phase 1 spike — pick your model with data
.venv\Scripts\python scripts\spike_toolcall.py --models qwen3:8b llama3.1:8b qwen3:14b

# 5. Chat with the robot
.venv\Scripts\python -m aliengo.cli
```

## Layout

- `aliengo/skills/` — single source of truth: skill specs + bounds → OpenAI tool schemas
- `aliengo/robot/` — `RobotController` interface, `SkillResult`, stateful mock robot
- `aliengo/safety/` — deterministic validate() gate (exists → params → posture → limits → e-stop → confirm)
- `aliengo/agent/` — Ollama client wrapper + tool-calling loop
- `aliengo/cli.py` — rich REPL (`/state`, `/log`, `/estop`, `/reset`)
- `scripts/spike_toolcall.py` — model tool-calling benchmark
- `tests/` — offline suite incl. scripted-fake-LLM loop tests; `regression_commands.md` for live checks

## Config

Everything lives in [config.yaml](config.yaml): model name, temperature,
iteration cap, and safety limits. Swapping models is a one-line change.

Note: if this folder stays under OneDrive, exclude `.venv/` and `logs/` from
sync (they are already gitignored).
