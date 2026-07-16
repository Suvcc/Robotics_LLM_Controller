"""Automated end-to-end evaluation (Phase 10 seed).

Runs every case in tests/eval_commands.json through a LIVE model with a fresh
mock robot per case, records which skills executed / were blocked / needed
confirmation, and scores the outcome against the case's acceptable results.

Usage:
    python scripts/evaluate.py                          # model from config.yaml
    python scripts/evaluate.py --models qwen3.5:9b llama3.1:8b
    python scripts/evaluate.py --config config.openai.yaml
    python scripts/evaluate.py --runs 3                 # consistency check

Metrics reported per case:
    n/N   pass rate across --runs (ok = all passed, FLKY = mixed, FAIL = none);
          a flaky case is a coin flip, not a capability
    s     median wall seconds for the command (all LLM round-trips + tools)
    rt    median LLM round-trips needed to finish the command — directly
          multiplies latency
    iv    interventions across all runs: safety BLOCKs + malformed tool calls;
          near-misses that the pass/fail verdict hides
The overall score is the mean of per-case pass rates. Token usage is not shown
here but is logged per command to logs/actions.jsonl (command_complete lines).

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
from datetime import datetime
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
    # ignore_skills drops harmless no-ops (e.g. a redundant stand_up) before
    # matching, so they don't pollute exact-sequence or allowed_only checks.
    ignore = set(spec.get("ignore_skills", []))
    executed = [s for s in rec.executed if s not in ignore]
    tool_calls = [(s, a) for s, a in rec.tool_calls if s not in ignore]

    if spec.get("no_tools") and tool_calls:
        return False
    if "executed" in spec and executed != spec["executed"]:
        return False
    if "params" in spec:
        for skill, expected in spec["params"].items():
            args = next((a for s, a in tool_calls if s == skill), None)
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
    if "must_not_execute" in spec and any(s in executed for s in spec["must_not_execute"]):
        return False
    if "must_confirm" in spec and not all(s in rec.confirmed for s in spec["must_confirm"]):
        return False
    if "allowed_only" in spec:
        if not executed or not all(s in spec["allowed_only"] for s in executed):
            return False
    return True


def run_case(case: dict, config, system_prompt: str, tools: list[dict]) -> dict:
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
    stats = loop.last_command_stats
    return {
        "passed": passed,
        "detail": detail,
        "seconds": seconds,
        "round_trips": stats.get("llm_calls", 0),
        "interventions": len(recorder.blocked) + stats.get("malformed_calls", 0),
    }


def build_report(meta: dict, summary_rows: list, model_sections: list) -> str:
    """Assemble the Markdown report. Pure function — no I/O — so it's testable."""
    lines = [
        f"# AlienGo Eval Report — {meta['timestamp']}",
        "",
        f"**Config:** {meta['config']} · **Cases:** {meta['n_cases']} · "
        f"**Runs per case:** {meta['runs']}",
        "",
        "## Summary",
        "",
        "| Model | Score | p50 latency | Avg round-trips | Interventions |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, score, p50, avg_rt, iv in summary_rows:
        lines.append(
            f"| {model} | {score:.0%} | {p50:.1f} s | {avg_rt:.1f} | {iv} |"
        )
    for section in model_sections:
        lines += [
            "",
            f"## {section['model']}",
            "",
            "| | Case | Pass | Median | RT | IV |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in section["rows"]:
            lines.append(
                f"| {row['status']} | {row['id']} | {row['n_pass']}/{row['runs']} "
                f"| {row['med_s']:.1f} s | {row['med_rt']:.0f} | {row['iv']} |"
            )
        if section["failures"]:
            lines += ["", "### Failure details", ""]
            for failure in section["failures"]:
                lines.append(
                    f"- **{failure['id']}** ({failure['n_pass']}/{failure['runs']}) "
                    f"— first failing run: {failure['detail']}"
                )
    lines += [
        "",
        "## Metric legend",
        "",
        "- **Score** — mean of per-case pass rates (a 2/3 case counts as 0.67).",
        "- **Pass** — runs passed / total runs; a mixed result means the case is flaky.",
        "- **Median** — median wall seconds per command (all LLM round-trips + tool execution).",
        "- **RT** — median LLM round-trips needed to complete the command.",
        "- **IV** — interventions: safety blocks + malformed tool calls across all runs.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--models", nargs="*", help="Override config model(s)")
    parser.add_argument("--cases", default=str(ROOT / "tests" / "eval_commands.json"))
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Times to run each case; >1 exposes flaky (inconsistent) cases",
    )
    parser.add_argument(
        "--out", default=None,
        help="Markdown report path (default: logs/eval_<timestamp>.md)",
    )
    args = parser.parse_args()
    started = datetime.now()

    base_config = load_config(args.config)
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    system_prompt = (ROOT / "prompts" / "system_prompt.md").read_text(encoding="utf-8")
    tools = to_openai_tools()
    models = args.models or [base_config.llm.model]

    summary = []
    model_sections = []
    for model in models:
        config = base_config.model_copy(deep=True)
        config.llm.model = model
        print(f"\n=== {model} ({args.runs} run(s) per case) ===")
        # Warm-up: load the model into VRAM so the first timed case doesn't
        # absorb the multi-second load and skew its latency.
        print("  warming up...", flush=True)
        try:
            LLMClient(config.llm).chat([{"role": "user", "content": "ready?"}], [])
        except Exception as exc:  # noqa: BLE001 — warm-up is best-effort
            print(f"  (warm-up skipped: {exc})")
        case_rates, latencies, round_trips, interventions = [], [], [], []
        section = {"model": model, "rows": [], "failures": []}
        model_sections.append(section)
        for case in cases:
            runs = [
                run_case(case, config, system_prompt, tools)
                for _ in range(args.runs)
            ]
            n_pass = sum(r["passed"] for r in runs)
            if n_pass == len(runs):
                status, emoji = "ok  ", "✅"
            elif n_pass == 0:
                status, emoji = "FAIL", "❌"
            else:
                status, emoji = "FLKY", "⚠️"
            med_s = statistics.median(r["seconds"] for r in runs)
            med_rt = statistics.median(r["round_trips"] for r in runs)
            case_iv = sum(r["interventions"] for r in runs)
            case_rates.append(n_pass / len(runs))
            latencies.extend(r["seconds"] for r in runs)
            round_trips.extend(r["round_trips"] for r in runs)
            interventions.append(case_iv)
            print(
                f"  [{status}] {case['id']:32} {n_pass}/{len(runs)} "
                f"{med_s:5.1f}s  rt={med_rt:.0f} iv={case_iv}  {runs[-1]['detail']}"
            )
            section["rows"].append({
                "status": emoji, "id": case["id"], "n_pass": n_pass,
                "runs": len(runs), "med_s": med_s, "med_rt": med_rt,
                "iv": case_iv,
            })
            fail_detail = next((r["detail"] for r in runs if not r["passed"]), None)
            if fail_detail:
                section["failures"].append({
                    "id": case["id"], "n_pass": n_pass, "runs": len(runs),
                    "detail": fail_detail,
                })
        summary.append((
            model,
            statistics.mean(case_rates),
            statistics.median(latencies),
            statistics.mean(round_trips),
            sum(interventions),
        ))

    summary_sorted = sorted(summary, key=lambda r: -r[1])
    print("\n=== summary ===")
    print(f"{'model':25} {'score':>8} {'p50 latency':>12} {'avg rt':>8} {'interventions':>14}")
    for model, score, p50, avg_rt, iv in summary_sorted:
        print(f"{model:25} {score:>7.0%} {p50:>10.1f}s {avg_rt:>8.1f} {iv:>14}")

    meta = {
        "timestamp": started.strftime("%Y-%m-%d %H:%M"),
        "config": Path(args.config).name,
        "n_cases": len(cases),
        "runs": args.runs,
    }
    out_path = (
        Path(args.out) if args.out
        else ROOT / "logs" / f"eval_{started.strftime('%Y-%m-%d_%H-%M')}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(meta, summary_sorted, model_sections), encoding="utf-8")
    print(f"\nreport: {out_path}")


if __name__ == "__main__":
    main()
