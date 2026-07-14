"""Phase 1 spike: score local models on tool-calling reliability.

Usage (Ollama must be running and the models pulled):
    python scripts/spike_toolcall.py --models qwen3:8b llama3.1:8b qwen3:14b

Each command is sent fresh (no history) with three hand-written dummy tools.
Scores: correct tool chosen, arguments valid, negative/ambiguous cases left
tool-free. Prints a summary table plus p50 latency per model.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SYSTEM = (
    "You are the control brain of a quadruped robot dog. Act only through the "
    "provided tools. Distances are meters, angles degrees. If a command is "
    "ambiguous or has no matching tool, reply in text instead of calling a tool."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_forward",
            "description": "Walk forward. Distance in meters, max 3.0.",
            "parameters": {
                "type": "object",
                "properties": {
                    "distance": {"type": "number", "description": "Meters, 0-3."}
                },
                "required": ["distance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn_left",
            "description": "Turn left in place. Angle in degrees, max 180.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {"type": "number", "description": "Degrees, 0-180."}
                },
                "required": ["angle"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stand_up",
            "description": "Stand up from sitting. No parameters.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# (command, expected tool or None, argument check)
CASES = [
    ("Stand up", "stand_up", lambda a: True),
    ("Get on your feet", "stand_up", lambda a: True),
    ("Move forward two meters", "move_forward", lambda a: a.get("distance") == 2),
    ("Walk ahead 1.5 meters", "move_forward", lambda a: a.get("distance") == 1.5),
    ("Go forward half a meter", "move_forward", lambda a: a.get("distance") == 0.5),
    ("Move forward 3 meters", "move_forward", lambda a: a.get("distance") == 3),
    ("Turn left 90 degrees", "turn_left", lambda a: a.get("angle") == 90),
    ("Turn left 45 degrees", "turn_left", lambda a: a.get("angle") == 45),
    ("Rotate left a quarter turn", "turn_left", lambda a: a.get("angle") == 90),
    ("Take one step forward", "move_forward", lambda a: 0 < a.get("distance", 0) <= 1),
    # Negative / ambiguous: correct behavior is NO tool call.
    ("What's the weather like today?", None, None),
    ("Fly upward", None, None),
    ("Go over there", None, None),
    ("Tell me a joke", None, None),
    ("Turn right 30 degrees", None, None),  # no turn_right tool in this spike
]


def run_model(client: OpenAI, model: str, temperature: float) -> dict:
    correct = 0
    latencies = []
    failures = []
    for command, expected_tool, check in CASES:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": command},
            ],
            tools=TOOLS,
            temperature=temperature,
        )
        latencies.append(time.perf_counter() - start)
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if expected_tool is None:
            ok = not tool_calls
            detail = f"called {tool_calls[0].function.name}" if tool_calls else "no tool (correct)"
        elif not tool_calls:
            ok, detail = False, "no tool call"
        else:
            call = tool_calls[0]
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = None
            if call.function.name != expected_tool:
                ok, detail = False, f"wrong tool {call.function.name}"
            elif args is None:
                ok, detail = False, "invalid JSON arguments"
            elif not check(args):
                ok, detail = False, f"bad args {args}"
            else:
                ok, detail = True, f"{call.function.name}({args})"

        correct += ok
        status = "ok " if ok else "FAIL"
        print(f"  [{status}] {command!r:45} -> {detail}")
        if not ok:
            failures.append(command)

    return {
        "model": model,
        "score": correct / len(CASES),
        "p50_s": statistics.median(latencies),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["qwen3:8b"])
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    client = OpenAI(
        base_url=args.base_url,
        api_key=os.environ.get("OPENAI_API_KEY") or "ollama",
    )
    results = []
    for model in args.models:
        print(f"\n=== {model} ===")
        results.append(run_model(client, model, args.temperature))

    print("\n=== summary ===")
    print(f"{'model':25} {'score':>8} {'p50 latency':>12}")
    for r in sorted(results, key=lambda r: -r["score"]):
        print(f"{r['model']:25} {r['score']:>7.0%} {r['p50_s']:>10.2f}s")
    print("\nTarget: ≥ 90% to pass Phase 1. Put the winner in config.yaml.")


if __name__ == "__main__":
    main()
