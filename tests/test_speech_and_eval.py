"""Contract tests for the STT text output and the eval dataset schema."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from aliengo.speech.stt import combine_segments


def _load_evaluate_module():
    path = Path(__file__).parent.parent / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

EVAL_PATH = Path(__file__).parent / "eval_commands.json"
KNOWN_OUTCOME_KEYS = {
    "executed", "params", "must_block", "must_not_execute",
    "must_confirm", "allowed_only", "no_tools",
}


def test_combine_segments_single_line():
    segments = [
        SimpleNamespace(text="  Stand up "),
        SimpleNamespace(text="and move forward.\n"),
    ]
    assert combine_segments(segments) == "Stand up and move forward."


def test_combine_segments_empty():
    assert combine_segments([]) == ""


def test_build_report_markdown():
    evaluate = _load_evaluate_module()
    meta = {"timestamp": "2026-07-16 14:32", "config": "config.yaml", "n_cases": 2, "runs": 3}
    summary_rows = [("qwen3.5:9b", 0.835, 4.2, 2.8, 3)]
    model_sections = [{
        "model": "qwen3.5:9b",
        "rows": [
            {"status": "✅", "id": "stand", "n_pass": 3, "runs": 3,
             "med_s": 2.1, "med_rt": 2, "iv": 0},
            {"status": "⚠️", "id": "stand_and_move", "n_pass": 2, "runs": 3,
             "med_s": 5.8, "med_rt": 3, "iv": 1},
        ],
        "failures": [
            {"id": "stand_and_move", "n_pass": 2, "runs": 3,
             "detail": "executed=['stand_up'] blocked=[]"},
        ],
    }]
    report = evaluate.build_report(meta, summary_rows, model_sections)

    assert "# AlienGo Eval Report — 2026-07-16 14:32" in report
    assert "| Model | Score | p50 latency | Avg round-trips | Interventions |" in report
    assert "| qwen3.5:9b | 84% | 4.2 s | 2.8 | 3 |" in report
    assert "| ✅ | stand | 3/3 | 2.1 s | 2 | 0 |" in report
    assert "| ⚠️ | stand_and_move | 2/3 | 5.8 s | 3 | 1 |" in report
    assert "### Failure details" in report
    assert "**stand_and_move** (2/3)" in report
    assert "## Metric legend" in report


def test_eval_dataset_schema():
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    assert cases, "eval dataset must not be empty"
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    for case in cases:
        assert case["command"].strip()
        assert case["any_of"], f"{case['id']}: needs at least one outcome"
        for outcome in case["any_of"]:
            unknown = set(outcome) - KNOWN_OUTCOME_KEYS
            assert not unknown, f"{case['id']}: unknown outcome keys {unknown}"
