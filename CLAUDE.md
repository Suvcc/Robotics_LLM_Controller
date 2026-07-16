# CLAUDE.md

## Project

Local LLM intelligence stack for the **Unitree AlienGo robot dog**, built mock-first:
a local model (Ollama) selects approved robot skills via tool calling, every call
passes a deterministic safety layer, and a stateful mock robot executes them.
Core rule: **the LLM never knows whether a mock or the real robot executes** —
ROS/Unitree SDK integration later only reimplements the `RobotController` Protocol.

## Commands

```powershell
uv run aliengo                                   # chat CLI (Ollama, config.yaml)
uv run aliengo --config config.openai.yaml       # same CLI against OpenAI (.env key)
uv run pytest                                    # offline test suite, no GPU needed
uv run python scripts/evaluate.py --runs 3       # benchmark -> terminal + logs/eval_<ts>.md
uv run python scripts/spike_toolcall.py          # raw tool-calling spike (model triage)
```

## Architecture

```
text or /voice → LLM (agent/llm.py) → tool calls → safety.validate() → RobotController → SkillResult fed back → final reply
```

- `aliengo/skills/definitions.py` — **single source of truth**: Pydantic param bounds
  generate the OpenAI tool schemas, the validation, and imply the dispatch. Skills
  change here and nowhere else.
- `aliengo/robot/interface.py` — `RobotController` Protocol; methods mirror skill
  names exactly (dispatch is `getattr(controller, skill_name)`). This is the ROS seam.
- `aliengo/robot/mock.py` — stateful fake robot (posture, pose, battery, error
  injection). Enforces preconditions itself — honest even if safety is bypassed.
- `aliengo/safety/validator.py` — pure 6-stage gate (exists → params → posture →
  session limits → e-stop → confirmation). LLM-independent; BLOCK reasons are
  written for the LLM to read and recover from.
- `aliengo/agent/loop.py` — tool-calling loop: sequential execution, batch-skip after
  block/failure, malformed retries, iteration cap, history cap, per-command stats.
- `aliengo/speech/stt.py` — mic → faster-whisper → text; feeds the same pipeline.
- `aliengo/cli.py` — rich REPL: `/voice /state /log /estop /reset`.
- `prompts/system_prompt.md` — behavior rules incl. anti-stale-plan rules.
- `tests/` — offline suite uses a scripted FakeLLM; `eval_commands.json` is the
  live benchmark dataset (13 cases).

## Conventions

- Use **uv** for everything; deps live in `pyproject.toml` (`[dependency-groups]`
  dev). Never bare pip — plain `uv sync` must stay green (it uninstalls undeclared
  packages).
- `OPENAI_API_KEY` comes from `.env` (gitignored) or shell env; never in files.
- Loop/safety/mock tests must run without a GPU (FakeLLM pattern in
  `tests/test_agent_loop.py`).
- Windows dev box (RTX 5070, 12 GB VRAM); project lives under OneDrive — keep
  `.venv/` and `logs/` out of sync.
- English commands only for now.

## Status — updated 2026-07-16 (update this section as things change)

| Phase | State |
|---|---|
| 1–5 core (LLM, skills, mock, safety, loop/CLI) | ✅ done, tested |
| 6 speech-to-text | ⚠️ works, but see issues below |
| 7 mock vision (`find_object`, `detect_objects`) | ✅ done |
| 8 real YOLO vision | ⬜ not started |
| 9 multi-step planner | ⬜ deferred until eval shows multi-step failures |
| 10 evaluation | ✅ harness done (13 cases, --runs, MD reports); dataset to grow |
| 11 ROS/AlienGo bridge | ⬜ design agreed: HTTP skill-server on robot's Ubuntu PC, `RemoteRobotController` client |
| 12 real-robot safety | ⬜ not started |

Benchmark baseline (qwen3.5:9b, 1 run/case, before fixes): 77% raw ≈ 92% adjusted.
Rerun after the fixes below (`uv run python scripts/evaluate.py --runs 3`) and
update this line.

**Fixes applied 2026-07-16 (post-first-benchmark):**
1. ✅ Robot-state injection — each command is prefixed with `[Robot state: posture,
   battery, motion, heading]` (`format_state_prefix` in agent/loop.py; toggle
   `llm.inject_state`). Stops blind `stand_up`, makes state questions answerable.
2. ✅ Eval matcher `ignore_skills` — strips harmless no-op skills before matching;
   applied to `turn_right_45` / `spin_ambiguous`.
3. ✅ Warm-up call in evaluate.py before timing (removes ~30 s first-case load).
4. ✅ Anti-flail prompt rule — refuse/acknowledge in text, never via stop/movement.
5. ✅ `/voice` crash was already fixed by the merge (stt.py has `listen(cfg=None)`
   + cached Whisper model).

**Open / next:** rerun the benchmark to confirm gains; grow the eval dataset;
then Phase 8 (real YOLO vision) or the Phase 11 ROS bridge when hardware is ready.
