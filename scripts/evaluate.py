"""Automated end-to-end evaluation (Phase 10 seed).

Runs every case in tests/eval_commands.json through a LIVE model with a fresh
mock robot per case, records which skills executed / were blocked / needed
confirmation, and scores the outcome against the case's acceptable results.

Usage:
    python scripts/evaluate.py                          # model from config.yaml
    python scripts/evaluate.py --models qwen3.5:9b llama3.1:8b
    python scripts/evaluate.py --config config.openai.yaml

Case format (tests/eval_commands.json): each case has a `command`, optional
`setup` ({"standing": true, "estop": true}), optional `pre_commands` (run
first, events discarded — used to test stale-plan carryover), optional
`confirm` (default true = auto-approve confirmations), and `any_of`: a list of
acceptable outcomes. An outcome passes if ALL its keys hold:
    executed          exact ordered list of successfully executed skills
    params            {skill: {param: value}} subset match on the first call
    must_block        every listed skill got a safety BLOCK
    must_not_execute  none of the listed skills executed successfully
    must_confirm      every listed skill triggered a CONFIRM decision
    allowed_only      at least one skill executed, all from this set
    no_tools          the model made no tool calls at all
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from aliengo.agent.llm import LLMClient  # noqa: E402
from aliengo.agent.loop import AgentLoop  # noqa: E402
from aliengo.config import load_config  # noqa: E402
from aliengo.robot.mock import MockRobotController  # noqa: E402
from aliengo.skills.registry import to_openai_tools  # noqa: E402

OUTCOME_KEYS = {
    "executed", "params", "must_block", "must_not_execute",
    "must_confirm", "allowed_only", "no_tools",
}


class Recorder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tool_calls: list[tuple[str, dict]] = []
        self.executed: list[str] = []
        self.blocked: list[str] = []
        self.confirmed: list[str] = []

    def __call__(self, kind: str, payload: dict):
        if kind == "tool_call":
            self.tool_calls.append((payload["skill"], payload["args"]))
        elif kind == "safety":
            if payload["decision"] == "block":
                self.blocked.append(payload["skill"])
            elif payload["decision"] == "confirm":
                self.confirmed.append(payload["skill"])
        elif kind == "result" and payload["success"]:
            self.executed.append(payload["action"])


def outcome_matches(spec: dict, rec: Recorder) -> bool:
    if spec.get("no_tools") and rec.tool_calls:
        return False
    if "executed" in spec and rec.executed != spec["executed"]:
        return False
    if "params" in spec:
        for skill, expected in spec["params"].items():
            args = next((a for s, a in rec.tool_calls if s == skill), None)
            if args is None:
                return False
            for key, value in expected.items():
                try:
                    if float(args.get(key)) != float(value):
                        return False
                except (TypeError, ValueError):
                    return False
    if "must_block" in spec and not all(s in rec.blocked for s in spec["must_block"]):
        return False
    if "must_not_execute" in spec and any(s in rec.executed for s in spec["must_not_execute"]):
        return False
    if "must_confirm" in spec and not all(s in rec.confirmed for s in spec["must_confirm"]):
        return False
    if "allowed_only" in spec:
        if not rec.executed or not all(s in spec["allowed_only"] for s in rec.executed):
            return False
    return True


def run_case(case: dict, config, system_prompt: str, tools: list[dict]) -> tuple[bool, str, float]:
    controller = MockRobotController()
    recorder = Recorder()
    approve = case.get("confirm", True)
    loop = AgentLoop(
        llm=LLMClient(config.llm),
        controller=controller,
        config=config,
        system_prompt=system_prompt,
        tools=tools,
        confirm=lambda *a: approve,
        on_event=recorder,
    )

    setup = case.get("setup", {})
    if setup.get("standing"):
        controller.stand_up()
    loop.estop_active = bool(setup.get("estop"))

    for pre in case.get("pre_commands", []):
        loop.run_command(pre)
    recorder.reset()  # score only the final command

    start = time.perf_counter()
    reply = loop.run_command(case["command"])
    seconds = time.perf_counter() - start

    passed = any(outcome_matches(spec, recorder) for spec in case["any_of"])
    detail = (
        f"executed={recorder.executed} blocked={recorder.blocked} "
        f"confirmed={recorder.confirmed} reply={reply[:60]!r}"
    )
    return passed, detail, seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--models", nargs="*", help="Override config model(s)")
    parser.add_argument("--cases", default=str(ROOT / "tests" / "eval_commands.json"))
    args = parser.parse_args()

    base_config = load_config(args.config)
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    system_prompt = (ROOT / "prompts" / "system_prompt.md").read_text(encoding="utf-8")
    tools = to_openai_tools()
    models = args.models or [base_config.llm.model]

    summary = []
    for model in models:
        config = base_config.model_copy(deep=True)
        config.llm.model = model
        print(f"\n=== {model} ===")
        passes, latencies = 0, []
        for case in cases:
            passed, detail, seconds = run_case(case, config, system_prompt, tools)
            latencies.append(seconds)
            passes += passed
            status = "ok " if passed else "FAIL"
            print(f"  [{status}] {case['id']:32} {seconds:5.1f}s  {detail}")
        summary.append((model, passes / len(cases), statistics.median(latencies)))

    print("\n=== summary ===")
    print(f"{'model':25} {'score':>8} {'p50 latency':>12}")
    for model, score, p50 in sorted(summary, key=lambda r: -r[1]):
        print(f"{model:25} {score:>7.0%} {p50:>10.1f}s")


if __name__ == "__main__":
    main()
